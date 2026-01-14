import os
import ssl
import cv2
import time
import torch
import shutil
import tempfile
import numpy as np
from PIL import Image
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any, Union
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from ..base import BaseStrategy


def build_mlp(input_size, hidden_sizes, output_size, dropout=0.0, use_batchnorm=False, add_dropout_after=True):
    layers = []
    
    if hidden_sizes:
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0 and add_dropout_after:
            layers.append(nn.Dropout(dropout))
        
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0 and add_dropout_after:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
    else:
        layers.append(nn.Linear(input_size, output_size))
    
    return nn.Sequential(*layers)


class ResNetClassifier(nn.Module):
    def __init__(self, arch_name='resnet18', n_label=10, pretrained=True, dropout=0.2,
                 fine_tune_layers=1, emb_size=256, in_channels=3):
        super(ResNetClassifier, self).__init__()

        self.n_label = n_label
        model = getattr(models, arch_name)
        if pretrained:
            original_ssl_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            try:
                resnet = model(weights='DEFAULT')
            finally:
                ssl._create_default_https_context = original_ssl_context
        else:
            resnet = model(weights=None)

        modules = list(resnet.children())[:-1]
        if modules[0].in_channels != in_channels:
            conv = modules[0]
            modules[0] = nn.Conv2d(in_channels=in_channels, out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size, stride=conv.stride,
                                   padding=conv.padding, bias=conv.bias)
            pretrained = False

        self.resnet = nn.Sequential(*modules)
        if pretrained:
            self.fine_tune(fine_tune_layers)
        input_size = resnet.fc.in_features
        if emb_size <= 0 or emb_size == input_size:
            self.embedding_size = input_size
            self.hidden_layers = None
        else:
            self.embedding_size = emb_size
            self.hidden_layers = build_mlp(input_size, (), emb_size, dropout=dropout, use_batchnorm=False,
                                     add_dropout_after=False)
        self.classifier = build_mlp(self.embedding_size, (), n_label,
                                    dropout=dropout,
                                    use_batchnorm=False,
                                    add_dropout_after=False)

    def forward(self, x, embedding=False):
        if embedding:
            embd = x
        else:
            embd = self.resnet(x)
            batch_size, feature_size, x, y = embd.size()
            embd = embd.view(batch_size, feature_size)
            if self.hidden_layers:
                embd = self.hidden_layers(embd)

        out = self.classifier(embd)
        return out, embd

    def get_embedding_dim(self):
        return self.embedding_size

    def fine_tune(self, fine_tune_layers):
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[
                 0 if fine_tune_layers < 0 else len(list(self.resnet.children())) - (1 + fine_tune_layers):]:
            for p in c.parameters():
                p.requires_grad = True

    def get_classifier(self):
        return self.classifier[-1]


class ObjectCropDataset(Dataset):
    
    def __init__(self, crop_paths: List[str], labels: List[int], transform=None):
        self.crop_paths = crop_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.crop_paths)
    
    def __getitem__(self, idx):
        img_path = self.crop_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label


def train_ddp_worker_standalone(rank: int, labeled_crop_info: Dict[str, Any], ddp_config: Dict[str, Any], model_save_path: str):
    try:
        os.environ['MASTER_ADDR'] = ddp_config['master_addr']
        os.environ['MASTER_PORT'] = str(ddp_config['master_port'])
        
        world_size = ddp_config['world_size']
        backend = ddp_config['backend']
        device_str = ddp_config['devices'][rank]
        
        if device_str.startswith('cuda:'):
            device = torch.device(device_str)
        else:
            device = torch.device(f'cuda:{device_str}')
        
        print(f"Worker {rank}: Initializing DDP with backend={backend}, rank={rank}, world_size={world_size}")
        
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        
        print(f"Worker {rank}: Using device {device}")
        
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        elif backend == 'nccl':
            print(f"Warning: Using CPU device {device} with NCCL backend. Consider using 'gloo' backend for CPU.")
        
        model = ResNetClassifier(
            arch_name=ddp_config['supporter'],
            n_label=ddp_config['num_classes'] or 20,
            emb_size=ddp_config['supporter_embedding_size'],
            pretrained=True,
            dropout=0.2,
            fine_tune_layers=1,
            in_channels=3
        ).to(device)
        
        ddp_model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
        
        transform = transforms.Compose([
            transforms.Resize((ddp_config['supporter_imgsz'], ddp_config['supporter_imgsz'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = ObjectCropDataset(
            labeled_crop_info['crop_paths'],
            labeled_crop_info['crop_labels'],
            transform=transform
        )
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        batch_size = max(1, min(ddp_config['supporter_batch_size'] // world_size, len(train_dataset) // world_size, 32))
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler, 
            num_workers=2,
            pin_memory=True
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
        
        if rank == 0:
            print(f"DDP Training supporter model on {len(labeled_crop_info['crop_paths'])} labeled crops")
            print(f"Batch size per GPU: {batch_size}, Total effective batch size: {batch_size * world_size}")
        
        for epoch in range(ddp_config['supporter_epochs']):
            train_sampler.set_epoch(epoch)
            ddp_model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs, _ = ddp_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_loss = epoch_loss / max(num_batches, 1)
            
            if rank == 0:
                print(f"DDP Supporter epoch {epoch + 1}/{ddp_config['supporter_epochs']}, train_loss: {avg_loss:.4f}")
        
        if rank == 0:
            torch.save(ddp_model.module.state_dict(), model_save_path)
            print(f"Saved DDP model to {model_save_path}")
        
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"Error in DDP worker {rank}: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()


class FDAL(BaseStrategy):
    def __init__(self,
                 model,
                 supporter: str = "resnet18",
                 supporter_epochs: int = 10,
                 supporter_batch_size: int = 1000000,
                 learn_alpha: bool = True,
                 alpha_cap: float = 0.03125,
                 alpha_learning_rate: float = 0.1,
                 lambda_hyp: float = 1,
                 supporter_embedding_size: int = 256,
                 supporter_imgsz: int = 320,
                 one_alpha_cap: bool = False,
                 experiment_dir: Optional[str] = None,
                 round: Optional[int] = None,
                 device: Union[str, List[str]] = 'auto',
                 train_ddp: bool = False,
                 force_single_gpu: bool = False,
                 master_addr: str = "localhost",
                 master_port: int = 12355,
                 backend: str = "nccl",
                 seed: int = 42,
                 prev_score_file: Optional[str] = None,
                 use_disu: bool = False,
                 **kwargs):
        super().__init__(model, **kwargs)
        print(type(self.model))
        self.supporter = supporter
        self.supporter_epochs = supporter_epochs
        self.supporter_batch_size = supporter_batch_size
        self.learn_alpha = learn_alpha
        self.alpha_cap = alpha_cap
        self.alpha_learning_rate = alpha_learning_rate
        self.lambda_hyp = lambda_hyp
        self.supporter_embedding_size = supporter_embedding_size
        self.supporter_imgsz = supporter_imgsz
        self.one_alpha_cap = one_alpha_cap
        self.experiment_dir = experiment_dir
        self.round = round
        self.prev_score_file = prev_score_file
        self.use_disu = use_disu
        
        self.device_config = device
        self.seed = seed
        self.devices = self._setup_devices(device)

        self.train_ddp = train_ddp and not force_single_gpu
        self.force_single_gpu = force_single_gpu
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend
        self.use_ddp = self.train_ddp and len(self.devices) > 1
        self.world_size = len(self.devices) if self.use_ddp else 1
        self.device = self.devices[0] if self.devices else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.supporter_model = None
        self.num_classes = None
        
        if force_single_gpu:
            print("Warning: force_single_gpu=True, disabling DDP training")
        
        if self.train_ddp and len(self.devices) <= 1:
            print(f"Warning: train_ddp=True but only {len(self.devices)} device(s) available. Falling back to single device training.")
        
        if self.use_ddp:
            print(f"FDAL will use DDP training with {len(self.devices)} devices: {self.devices}")
            print(f"DDP config: master_addr={self.master_addr}, master_port={self.master_port}, backend={self.backend}")
        else:
            print(f"FDAL will use single device training on: {self.device}")
        
    def _setup_devices(self, device: Union[str, List[str]]) -> List[torch.device]:
        if isinstance(device, list):
            devices = []
            for d in device:
                d_str = str(d)
                if d_str.lower() in ['auto', 'cpu']:
                    devices.append(torch.device('cpu'))
                else:
                    devices.append(torch.device(f'cuda:{d_str}'))
            return devices
        elif isinstance(device, str):
            if device.lower() == 'auto':
                if torch.cuda.is_available():
                    return [torch.device('cuda:0')]
                else:
                    return [torch.device('cpu')]
            elif device.lower() == 'cpu':
                return [torch.device('cpu')]
            elif ',' in device:
                device_list = [d.strip() for d in device.split(',')]
                return self._setup_devices(device_list)
            else:
                return [torch.device(f'cuda:{device}')]
        elif isinstance(device, int):
            return [torch.device(f'cuda:{device}')]
        else:
            return [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
        
    def query(self, 
              unlabeled_indices: np.ndarray,
              image_paths: List[str],
              n_samples: int,
              **kwargs) -> np.ndarray:
        timelog_file = Path(self.experiment_dir) / os.environ["TIME_LOGFILE"] # type: ignore
        if not timelog_file.exists():
            with open(timelog_file, 'w') as f:
                f.write("Round,TotalTime,NumImages,TimePerImage\n")

        selectionlog_file = Path(self.experiment_dir) / os.environ["SELECTION_LOGFILE"] # type: ignore
        if not selectionlog_file.exists():
            selectionlog_file.touch()

        supporter_trainingtime_logfile = Path(self.experiment_dir) / "supporter_trainingtime_log.txt" # type: ignore
        if not supporter_trainingtime_logfile.exists():
            with open(supporter_trainingtime_logfile, 'w') as f:
                f.write("Round,Epoch,TrainLoss,Accuracy,TrainingTime\n")
        self.supporter_trainingtime_logfile = supporter_trainingtime_logfile

        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        local_kwargs = dict(kwargs)
        num_inf = local_kwargs.pop('num_inference', -1)

        unlabeled_image_paths = self._get_image_paths_for_indices(unlabeled_indices, image_paths)
        unlabeled_image_paths = unlabeled_image_paths if num_inf < 0 else unlabeled_image_paths[:num_inf]
        
        print(f"Running FDAL strategy on {len(unlabeled_image_paths)} unlabeled images...")
        
        print("Step 1: Running YOLO inference for object detection...")
        start_time = time.time()

        results = self.model.inference(
            unlabeled_image_paths,
            return_boxes=True,
            return_classes=True,
            return_probs=False,
            **kwargs
        )
        
        print("Step 2: Cropping detected objects...")
        temp_dir = self._create_temp_crops_dir()
        try:
            crop_info = self._crop_and_save_objects(unlabeled_image_paths, results, temp_dir)
            
            if not crop_info['crop_paths']:
                print("Warning: No objects detected. Falling back to random selection.")
                return np.random.choice(unlabeled_indices, size=min(n_samples, len(unlabeled_indices)), replace=False)
            
            print("Step 3: Collecting labeled data for supporter training...")
            labeled_crop_info = self._get_labeled_crops_for_supporter(temp_dir)
            
            print("Step 4: Training supporter model...")
            train_start_time = time.time()
            self._train_supporter_model(labeled_crop_info, crop_info)
            train_end_time = time.time()
            train_time = train_end_time - train_start_time
            
            print("Step 5: Applying FDAL selection algorithm...")
            selected_local_indices = self._apply_fdal_selection(
                crop_info, unlabeled_image_paths, n_samples, labeled_crop_info=labeled_crop_info, **kwargs
            )
            
            end_time = time.time()
            num_ulbl_images = len(unlabeled_image_paths)
            time_per_image = (end_time - start_time - train_time) / num_ulbl_images if num_ulbl_images > 0 else 0
            print(f"FDAL selection completed in {end_time - start_time:.2f}s, num images: {num_ulbl_images}, time per image: {time_per_image:.4f}s")
            with open(timelog_file, 'a') as f:
                f.write(f"{self.round},{end_time - start_time:.2f},{num_ulbl_images},{time_per_image:.4f}\n")
            print("Write to time log file. ", timelog_file.absolute())
            selected_indices = unlabeled_indices[selected_local_indices]
            
            self._cleanup_labeled_crops(labeled_crop_info['crop_paths'])
            
            selected_image_names = [Path(image_paths[idx]).name for idx in selected_indices]
            with open(selectionlog_file, 'a') as f:
                f.write(','.join(selected_image_names) + '\n')
            print("Write to selection log file. ", selectionlog_file.absolute())
            
            selected_image_paths = [image_paths[idx] for idx in selected_indices]
            self._save_predictions_for_selection(
                experiment_dir=self.experiment_dir,
                round_num=self.round,
                selected_image_paths=selected_image_paths,
                image_paths=image_paths,
                selected_indices=selected_indices,
                results=results,
                unlabeled_indices=unlabeled_indices,
            )
            self._save_selection_symlinks(
                experiment_dir=self.experiment_dir,
                round_num=self.round,
                selected_image_paths=selected_image_paths,
            )
            
            return selected_indices
            
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                
    def _create_temp_crops_dir(self) -> Path:
        if self.experiment_dir and self.round is not None:
            temp_dir = Path(self.experiment_dir) / f"round_{self.round}" / "fdal_crops"
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="fdal_crops_"))
        return temp_dir
    
    def _crop_and_save_objects(self, image_paths: List[str], results: List, temp_dir: Path) -> Dict[str, Any]:
        crop_paths = []
        crop_labels = []
        image_to_crops = {}
        crop_to_image = []
        
        for img_idx, (img_path, result) in enumerate(zip(image_paths, results)):
            if result.boxes is None or len(result.boxes) == 0:
                image_to_crops[img_idx] = []
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                image_to_crops[img_idx] = []
                continue
                
            img_height, img_width = img.shape[:2]
            crop_indices_for_img = []
            
            for det_idx, (box, cls) in enumerate(zip(result.boxes, result.classes)):
                try:
                    x1, y1, x2, y2 = map(int, box)
                    
                    x1 = max(0, min(x1, img_width - 1))
                    y1 = max(0, min(y1, img_height - 1))
                    x2 = max(x1 + 1, min(x2, img_width))
                    y2 = max(y1 + 1, min(y2, img_height))
                    
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        print(f"Warning: Detected object too small in image {img_path}, skipping")
                        continue
                    
                    cropped_obj = img[y1:y2, x1:x2]
                    
                    cropped_obj = cv2.resize(cropped_obj, (self.supporter_imgsz, self.supporter_imgsz))
                    
                    crop_filename = f"img_{img_idx}_det{det_idx}_cls{int(cls)}.jpg"
                    crop_path = temp_dir / crop_filename
                    cv2.imwrite(str(crop_path), cropped_obj)
                    
                    crop_idx = len(crop_paths)
                    crop_paths.append(str(crop_path))
                    crop_labels.append(int(cls))
                    crop_to_image.append(img_idx)
                    crop_indices_for_img.append(crop_idx)
                    
                except Exception as e:
                    print(f"Warning: Failed to crop object {det_idx} from image {img_idx}: {e}")
                    continue
            
            image_to_crops[img_idx] = crop_indices_for_img
        
        print(f"Created {len(crop_paths)} object crops from {len(image_paths)} images")
        
        return {
            'crop_paths': crop_paths,
            'crop_labels': crop_labels,
            'crop_to_image': crop_to_image,
            'image_to_crops': image_to_crops
        }
    
    def _get_labeled_crops_for_supporter(self, temp_dir: Path) -> Dict[str, Any]:
        labeled_crop_paths = []
        labeled_crop_labels = []
        
        if self.experiment_dir and self.round is not None and self.round > 0:
            labeled_crops = self._extract_labeled_crops_from_rounds(temp_dir)
            labeled_crop_paths.extend(labeled_crops['paths'])
            labeled_crop_labels.extend(labeled_crops['labels'])
        
        print(f"Extracted {len(labeled_crop_paths)} labeled crops from previous rounds")
        
        return {
            'crop_paths': labeled_crop_paths,
            'crop_labels': labeled_crop_labels
        }
    
    def _extract_labeled_crops_from_rounds(self, temp_dir: Path) -> Dict[str, List]:
        crop_paths = []
        crop_labels = []
        
        if not (self.experiment_dir and self.round is not None):
            return {'paths': [], 'labels': []}
        
        try:
            exp_path = Path(self.experiment_dir)
            
            for round_idx in range(self.round):
                round_dir = exp_path / f"round_{round_idx}"
                if not round_dir.exists():
                    continue
                
                images_dir = round_dir / "images" / "train"
                labels_dir = round_dir / "labels" / "train"
                
                if images_dir.exists() and labels_dir.exists():
                    round_crops_dir = temp_dir / f"round_{round_idx}_train"
                    round_crops_dir.mkdir(exist_ok=True)
                    
                    for img_ext in ['*.jpg', '*.png', '*.jpeg']:
                        for img_file in images_dir.glob(img_ext):
                            label_file = labels_dir / f"{img_file.stem}.txt"
                            if label_file.exists():
                                crops_from_labeled = self._extract_crops_from_labeled_image(
                                    str(img_file), str(label_file), round_crops_dir
                                )
                                crop_paths.extend(crops_from_labeled['paths'])
                                crop_labels.extend(crops_from_labeled['labels'])
            
        except Exception as e:
            print(f"Warning: Failed to extract crops from previous rounds: {e}")
        
        return {'paths': crop_paths, 'labels': crop_labels}
    
    def _extract_crops_from_labeled_image(self, img_path: str, label_path: str, temp_dir: Path) -> Dict[str, List]:
        crop_paths = []
        crop_labels = []
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                return {'paths': [], 'labels': []}
            
            img_height, img_width = img.shape[:2]
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(x1 + 1, min(x2, img_width))
                y2 = max(y1 + 1, min(y2, img_height))
                
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                
                cropped_obj = img[y1:y2, x1:x2]
                
                cropped_obj = cv2.resize(cropped_obj, (self.supporter_imgsz, self.supporter_imgsz))
                
                img_stem = Path(img_path).stem
                crop_filename = f"labeled_{img_stem}_obj_{line_idx}_cls_{class_id}.jpg"
                crop_path = temp_dir / crop_filename
                cv2.imwrite(str(crop_path), cropped_obj)
                
                crop_paths.append(str(crop_path))
                crop_labels.append(class_id)
                
        except Exception as e:
            print(f"Warning: Failed to extract crops from labeled image {img_path}: {e}")
        
        return {'paths': crop_paths, 'labels': crop_labels}
    
    def _get_val_crops_for_supporter(self) -> Dict[str, Any]:
        val_crop_paths = []
        val_crop_labels = []
        
        if self.experiment_dir and self.round is not None and self.round > 0:
            val_crops = self._extract_val_crops_from_rounds()
            val_crop_paths.extend(val_crops['paths'])
            val_crop_labels.extend(val_crops['labels'])
        
        return {
            'crop_paths': val_crop_paths,
            'crop_labels': val_crop_labels
        }
    
    def _extract_val_crops_from_rounds(self) -> Dict[str, List]:
        crop_paths = []
        crop_labels = []
        
        if not (self.experiment_dir and self.round is not None):
            return {'paths': [], 'labels': []}
        
        try:
            exp_path = Path(self.experiment_dir)
            temp_dir = self._create_temp_crops_dir() / "val_crops"
            temp_dir.mkdir(exist_ok=True)
            
            for round_idx in range(self.round):
                round_dir = exp_path / f"round_{round_idx}"
                if not round_dir.exists():
                    continue
                
                images_dir = round_dir / "images" / "val"
                labels_dir = round_dir / "labels" / "val"
                
                if images_dir.exists() and labels_dir.exists():
                    round_crops_dir = temp_dir / f"round_{round_idx}_val"
                    round_crops_dir.mkdir(exist_ok=True)
                    
                    for img_ext in ['*.jpg', '*.png', '*.jpeg']:
                        for img_file in images_dir.glob(img_ext):
                            label_file = labels_dir / f"{img_file.stem}.txt"
                            if label_file.exists():
                                crops_from_labeled = self._extract_crops_from_labeled_image(
                                    str(img_file), str(label_file), round_crops_dir
                                )
                                crop_paths.extend(crops_from_labeled['paths'])
                                crop_labels.extend(crops_from_labeled['labels'])
            
        except Exception as e:
            print(f"Warning: Failed to extract validation crops from previous rounds: {e}")
        
        return {'paths': crop_paths, 'labels': crop_labels}
    
    def _validate_supporter_model(self, val_dataloader: DataLoader, criterion: nn.Module) -> float:
        if self.supporter_model is None or val_dataloader is None:
            return 0.0
        
        self.supporter_model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, _ = self.supporter_model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                num_batches += 1
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def _train_supporter_model(self, labeled_crop_info: Dict[str, Any], unlabeled_crop_info: Dict[str, Any]):
        if not labeled_crop_info['crop_labels']:
            print("Warning: No labeled objects detected for supporter training")
            return
            
        self.num_classes = len(self.model.model.names) # type: ignore
        
        if self.use_ddp:
            print(f"Training supporter model with DDP on {len(self.devices)} GPUs: {self.devices}")
            try:
                self._train_supporter_ddp(labeled_crop_info, unlabeled_crop_info)
            except Exception as e:
                print(f"DDP training failed with error: {e}")
                print("Falling back to single GPU training")
                self.use_ddp = False
                self._train_supporter_single(labeled_crop_info, unlabeled_crop_info)
        else:
            self._train_supporter_single(labeled_crop_info, unlabeled_crop_info)
    
    def _train_supporter_single(self, labeled_crop_info: Dict[str, Any], unlabeled_crop_info: Dict[str, Any]):
        self.supporter_model = ResNetClassifier(
            arch_name=self.supporter,
            n_label=self.num_classes, # type: ignore
            emb_size=self.supporter_embedding_size,
            pretrained=True,
            dropout=0.2,
            fine_tune_layers=1,
            in_channels=3
        ).to(self.device)
        
        transform = transforms.Compose([
            transforms.Resize((self.supporter_imgsz, self.supporter_imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_crop_paths = labeled_crop_info['crop_paths']
        all_crop_labels = labeled_crop_info['crop_labels']
        
        if not all_crop_paths:
            print("Warning: No training data available for supporter model")
            return
        
        train_dataset = ObjectCropDataset(
            all_crop_paths,
            all_crop_labels,
            transform=transform
        )
        
        batch_size = min(self.supporter_batch_size, len(train_dataset), 64)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        val_crop_info = self._get_val_crops_for_supporter()
        val_dataloader = None
        if val_crop_info['crop_paths']:
            val_dataset = ObjectCropDataset(
                val_crop_info['crop_paths'],
                val_crop_info['crop_labels'],
                transform=transform
            )
            val_batch_size = min(64, len(val_dataset))
            val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=2)
            print(f"Validation set: {len(val_crop_info['crop_paths'])} crops")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.supporter_model.parameters(), lr=0.001)
        
        print(f"Training supporter model on {len(all_crop_paths)} labeled crops")
        
        for epoch in range(self.supporter_epochs):
            start_time = time.time()
            self.supporter_model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.supporter_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_loss = epoch_loss / max(num_batches, 1)
            
            val_acc = self._validate_supporter_model(val_dataloader, criterion) if val_dataloader else None
            end_time = time.time()
            val_str = f", val_acc: {val_acc:.4f}" if val_acc is not None else ""
            print(f"Supporter epoch {epoch + 1}/{self.supporter_epochs}, train_loss: {avg_loss:.4f}{val_str}, time: {end_time - start_time:.2f}s")
            with open(self.supporter_trainingtime_logfile, 'a') as f:
                f.write(f"{self.round},{epoch + 1},{avg_loss:.4f},{val_acc},{end_time - start_time:.2f}\n")
        
        if val_crop_info['crop_paths']:
            self._cleanup_labeled_crops(val_crop_info['crop_paths'])
        
        model_save_path = self._get_ddp_model_save_path(0)
        torch.save(self.supporter_model.state_dict(), model_save_path)
        print(f"Saved supporter model to {model_save_path}")
    
    def _train_supporter_ddp(self, labeled_crop_info: Dict[str, Any], unlabeled_crop_info: Dict[str, Any]):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            print("Warning: spawn method already set")
        
        ddp_config = {
            'supporter': str(self.supporter),
            'supporter_epochs': int(self.supporter_epochs),
            'supporter_batch_size': int(self.supporter_batch_size),
            'supporter_embedding_size': int(self.supporter_embedding_size),
            'supporter_imgsz': int(self.supporter_imgsz),
            'num_classes': int(self.num_classes) if self.num_classes else 20,
            'world_size': int(self.world_size),
            'devices': [str(device) for device in self.devices],
            'master_addr': str(self.master_addr),
            'master_port': int(self.master_port),
            'backend': str(self.backend)
        }
        
        serializable_crop_info = {
            'crop_paths': [str(path) for path in labeled_crop_info['crop_paths']],
            'crop_labels': [int(label) for label in labeled_crop_info['crop_labels']]
        }
        
        try:
            import pickle
            pickle.dumps(serializable_crop_info)
            pickle.dumps(ddp_config)
            print("Data serialization validation passed")
        except Exception as e:
            print(f"Data serialization validation failed: {e}")
            raise e
        
        processes = []
        try:
            for rank in range(self.world_size):
                model_save_path = self._get_ddp_model_save_path(rank)
                p = mp.Process(
                    target=train_ddp_worker_standalone, 
                    args=(rank, serializable_crop_info, ddp_config, model_save_path)
                )
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()
                if p.exitcode != 0:
                    print(f"Warning: DDP worker process exited with code {p.exitcode}")
                    
        except Exception as e:
            print(f"Error during multiprocessing: {e}")
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join()
            raise e
        
        main_model_path = self._get_ddp_model_save_path(0)
        if Path(main_model_path).exists():
            checkpoint = torch.load(main_model_path, map_location=self.device)
            self.supporter_model = ResNetClassifier(
                arch_name=self.supporter,
                n_label=self.num_classes or 20,
                emb_size=self.supporter_embedding_size,
                pretrained=True,
                dropout=0.2,
                fine_tune_layers=1,
                in_channels=3
            ).to(self.device)
            self.supporter_model.load_state_dict(checkpoint)
            print(f"Loaded DDP trained model from {main_model_path}")
        else:
            print("Warning: DDP training completed but model file not found, falling back to single GPU training")
            self._train_supporter_single(labeled_crop_info, unlabeled_crop_info)
    
    def _get_ddp_model_save_path(self, rank: int) -> str:
        if self.experiment_dir and self.round is not None:
            save_dir = Path(self.experiment_dir) / f"round_{self.round}" / "supporter_models"
            save_dir.mkdir(parents=True, exist_ok=True)
            return str(save_dir / f"supporter_model_rank_{rank}.pth")
        else:
            return f"/tmp/supporter_model_rank_{rank}.pth"
    
    def _cleanup_labeled_crops(self, labeled_crop_paths: List[str]):
        try:
            for crop_path in labeled_crop_paths:
                if Path(crop_path).exists():
                    Path(crop_path).unlink()
            print(f"Cleaned up {len(labeled_crop_paths)} labeled crop files")
        except Exception as e:
            print(f"Warning: Failed to clean up some labeled crop files: {e}")
    
    def _apply_fdal_selection(self, crop_info: Dict[str, Any], image_paths: List[str], 
                                n_samples: int, labeled_crop_info: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
        
        if self.supporter_model is None:
            print("Warning: Supporter model not trained. Falling back to random selection.")
            return np.random.choice(len(image_paths), size=min(n_samples, len(image_paths)), replace=False)
        
        supporter_probs, supporter_embeddings = self._get_supporter_predictions(crop_info)
        
        if len(supporter_probs) == 0:
            print("Warning: No supporter predictions available. Falling back to random selection.")
            return np.random.choice(len(image_paths), size=min(n_samples, len(image_paths)), replace=False)
        
        yolo_labels = np.array(crop_info['crop_labels'])
        
        if labeled_crop_info and labeled_crop_info['crop_paths']:
            labeled_probs, labeled_embeddings = self._get_supporter_predictions(labeled_crop_info)
            labeled_labels = np.array(labeled_crop_info['crop_labels'])
        else:
            labeled_embeddings = np.array([])
            labeled_labels = np.array([])
        
        candidate_crops, min_alphas, image_changes_list, second_scores = self._find_candidate_set(
            supporter_embeddings, supporter_probs, yolo_labels, crop_info, len(image_paths), labeled_embeddings, labeled_labels
        )
        
        if self.use_disu:
            unlabeled_indices = np.arange(len(image_paths))
            self._write_scores_for_disu(crop_info, unlabeled_indices, image_changes_list)
        
        print(f"Found {np.sum(candidate_crops)} candidate crops with prediction inconsistencies")
        supporter_probs_all, _ = self._get_supporter_predictions(crop_info)
        entropies = -np.sum(supporter_probs_all * np.log(supporter_probs_all + 1e-8), axis=1)
        print(f"Computed entropy for {len(entropies)} crops (mean entropy: {np.mean(entropies):.4f})")
        second_scores = np.where(candidate_crops, -np.inf, entropies)
        
        if np.sum(candidate_crops) > 0:
            candidate_crop_indices = np.where(candidate_crops)[0]
        else:
            print("No candidate crops found. Falling back to random selection.")
            candidate_crop_indices = np.array([])
        
        image_scores = self._aggregate_to_image_level(
            candidate_crop_indices, crop_info, len(image_paths)
        )
        
        num_uncertain = int(np.sum(image_scores > 0))
        print(f"Found {num_uncertain} images with prediction inconsistencies out of {len(image_paths)} total images")
        
        if num_uncertain >= n_samples:
            print(f"Sufficient uncertain images found ({num_uncertain} >= {n_samples}). Selecting top {n_samples} by uncertainty score.")
            top_indices = np.argsort(image_scores)[-n_samples:]
        else:
            print(f"Insufficient uncertain images ({num_uncertain} < {n_samples}). Using {num_uncertain} uncertain images + {n_samples - num_uncertain} high-entropy images.")
            uncertain_indices = np.argsort(image_scores)[-num_uncertain:]
            remaining = n_samples - num_uncertain
            
            image_max_entropy = np.full(len(image_paths), -np.inf)
            for crop_idx in range(len(second_scores)):
                if second_scores[crop_idx] > -np.inf:
                    img_idx = crop_info['crop_to_image'][crop_idx]
                    image_max_entropy[img_idx] = max(image_max_entropy[img_idx], second_scores[crop_idx])
            
            zero_score_images = np.where(image_scores == 0)[0]
            print(f"Computing entropy for {len(zero_score_images)} images without prediction inconsistencies")
            
            if len(zero_score_images) >= remaining:
                entropies = image_max_entropy[zero_score_images]
                valid_entropy_count = np.sum(entropies > -np.inf)
                print(f"Found {valid_entropy_count} images with valid entropy scores. Selecting top {remaining} by entropy as tie breaker.")
                sorted_idx = np.argsort(-entropies)
                selected_zero = zero_score_images[sorted_idx][:remaining]
            else:
                print(f"Only {len(zero_score_images)} images available for entropy selection (needed {remaining}). Selecting all available.")
                selected_zero = zero_score_images
            
            top_indices = np.concatenate([uncertain_indices, selected_zero[:remaining]])
            print(f"Final selection: {len(uncertain_indices)} uncertain images + {len(selected_zero[:remaining])} entropy-based images = {len(top_indices)} total")
        
        self._save_changes(image_changes_list, top_indices)
        
        return top_indices
    
    def _save_changes(self, image_changes_list: List[np.ndarray], selected_indices: np.ndarray):
        out_path = Path(self.experiment_dir) / f"round_{self.round}" / "flip_classes.txt" if (self.experiment_dir and self.round is not None) else Path("flip_classes.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            if not image_changes_list:
                f.write("Warning: No image changes recorded\n")
                return
            if selected_indices is None or len(selected_indices) == 0:
                f.write("Warning: No selected images provided\n")
                return
            f.write("Image changes per alpha cap for selected images:\n")
            for cap_idx, image_changes in enumerate(image_changes_list):
                alpha_val = (cap_idx + 1) * self.alpha_cap
                changes_list = []
                for idx in np.unique(selected_indices):
                    if idx >= len(image_changes):
                        continue
                    changes = int(image_changes[idx])
                    if changes > 0:
                        changes_list.append(f"{idx}: {changes}")
                if changes_list:
                    f.write(f"Alpha cap {alpha_val:.4f}: {changes_list}\n")
    
    def _get_supporter_predictions(self, crop_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        if self.supporter_model is None or not crop_info['crop_paths']:
            return np.array([]), np.array([])
            
        transform = transforms.Compose([
            transforms.Resize((self.supporter_imgsz, self.supporter_imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = ObjectCropDataset(
            crop_info['crop_paths'],
            crop_info['crop_labels'],
            transform=transform
        )
        
        batch_size = min(64, len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        all_probs = []
        all_embeddings = []
        
        self.supporter_model.eval()
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs, embeddings = self.supporter_model(images)
                probs = F.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_embeddings.append(embeddings.cpu().numpy())
        
        if all_probs:
            return np.vstack(all_probs), np.vstack(all_embeddings)
        else:
            return np.array([]), np.array([])
    
    def _find_candidate_set(self, embeddings: np.ndarray, probs: np.ndarray, 
                           yolo_labels: np.ndarray, crop_info: Dict[str, Any], num_images: int,
                           labeled_embeddings: np.ndarray, labeled_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
        
        if len(embeddings) == 0:
            return np.array([], dtype=bool), np.array([]), [], np.array([])
        
        embeddings_torch = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        probs_torch = torch.tensor(probs, dtype=torch.float32).to(self.device)
        yolo_labels_torch = torch.tensor(yolo_labels, dtype=torch.long).to(self.device)
        
        labeled_embeddings_torch = torch.tensor(labeled_embeddings, dtype=torch.float32).to(self.device) if len(labeled_embeddings) > 0 else None
        labeled_labels_torch = torch.tensor(labeled_labels, dtype=torch.long).to(self.device) if len(labeled_labels) > 0 else None
        
        unlabeled_size = embeddings_torch.size(0)
        embedding_size = embeddings_torch.size(1)
        
        min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float32).to(self.device)
        pred_change = torch.zeros(unlabeled_size, dtype=torch.bool).to(self.device)
        
        class_flip_counts = {}
        total_objects_per_class = {}
        flipped_objects = set()
        
        for label in yolo_labels:
            class_id = int(label)
            total_objects_per_class[class_id] = total_objects_per_class.get(class_id, 0) + 1
            class_flip_counts[class_id] = 0
        
        pred_1 = torch.argmax(probs_torch, dim=1)
        
        alpha_cap = 0.0
        candidate_count = 0
        max_entropy = torch.zeros(unlabeled_size, dtype=torch.float32).to(self.device)
        image_changes_list = []
        
        max_alpha_cap = self.alpha_cap if self.one_alpha_cap else 1.0
        
        while alpha_cap < max_alpha_cap and candidate_count < unlabeled_size:
            alpha_cap += self.alpha_cap
            alpha_cap = min(alpha_cap, 1.0)
            
            changed_so_far = pred_change.clone()
            tmp_pred_change, tmp_min_alphas, entropy = self._find_candidate_set_with_alpha_cap(
                embeddings_torch, probs_torch, yolo_labels_torch, pred_1, alpha_cap, labeled_embeddings_torch, labeled_labels_torch
            )
            
            max_entropy = torch.where(changed_so_far, max_entropy, torch.maximum(max_entropy, entropy))
            changed_so_far |= tmp_pred_change
            
            tmp_pred_change_np = tmp_pred_change.cpu().numpy()
            for crop_idx in range(len(tmp_pred_change_np)):
                if tmp_pred_change_np[crop_idx] and crop_idx not in flipped_objects:
                    flipped_objects.add(crop_idx)
                    original_class = int(yolo_labels[crop_idx])
                    class_flip_counts[original_class] += 1
            
            is_changed = torch.norm(min_alphas, dim=1) >= torch.norm(tmp_min_alphas, dim=1)
            min_alphas[is_changed] = tmp_min_alphas[is_changed]
            pred_change += tmp_pred_change
            
            candidate_count = torch.sum(pred_change).item()
            print(f'With alpha_cap {alpha_cap:.6f}, found {torch.sum(tmp_pred_change).item()} new inconsistencies (total: {candidate_count})')
            
            image_changes = np.zeros(num_images)
            for crop_idx in range(len(tmp_pred_change_np)):
                if tmp_pred_change_np[crop_idx]:
                    img_idx = crop_info['crop_to_image'][crop_idx]
                    image_changes[img_idx] += 1
            image_changes_list.append(image_changes)
            
            if candidate_count > unlabeled_size // 2:
                break
        
            image_changes_list.append(image_changes)
            
            if candidate_count > unlabeled_size // 2:
                break
        
        self._save_class_flip_counts(class_flip_counts, total_objects_per_class)
        
        second_scores = torch.where(pred_change, torch.tensor(float('-inf'), device=self.device), max_entropy)
        
        return pred_change.cpu().numpy(), min_alphas.cpu().numpy(), image_changes_list, second_scores.cpu().numpy()
    
    def _find_candidate_set_with_alpha_cap(self, embeddings_torch: torch.Tensor, probs_torch: torch.Tensor,
                                          yolo_labels_torch: torch.Tensor, pred_1: torch.Tensor,
                                          alpha_cap: float, labeled_embeddings_torch: Optional[torch.Tensor],
                                          labeled_labels_torch: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        unlabeled_size = embeddings_torch.size(0)
        embedding_size = embeddings_torch.size(1)
        
        min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float32).to(self.device)
        pred_change = torch.zeros(unlabeled_size, dtype=torch.bool).to(self.device)
        max_entropy = torch.zeros(unlabeled_size, dtype=torch.float32).to(self.device)
        
        unique_classes = torch.unique(labeled_labels_torch) if labeled_labels_torch is not None else torch.unique(yolo_labels_torch)
        
        for class_id in unique_classes:
            if labeled_labels_torch is not None and labeled_embeddings_torch is not None:
                class_mask = labeled_labels_torch == class_id
                if torch.sum(class_mask) == 0:
                    continue
                class_embeddings = labeled_embeddings_torch[class_mask]
            else:
                class_mask = yolo_labels_torch == class_id
                if torch.sum(class_mask) == 0:
                    continue
                class_embeddings = embeddings_torch[class_mask]
            
            anchor_i = torch.mean(class_embeddings, dim=0, keepdim=True)
            anchor_i = anchor_i.repeat(unlabeled_size, 1)
            
            if self.learn_alpha:
                alpha, pc = self._learn_alpha_simplified(
                    embeddings_torch, pred_1, anchor_i, alpha_cap, class_id
                )
                mixed_embeddings = (1 - alpha) * embeddings_torch + alpha * anchor_i
            else:
                alpha = self._generate_alpha(unlabeled_size, embedding_size, alpha_cap)
                
                mixed_embeddings = (1 - alpha) * embeddings_torch + alpha * anchor_i
                
                pc = self._get_mixed_predictions_change(mixed_embeddings, pred_1)
            
            mixed_preds, probs = self._get_mixed_predictions(mixed_embeddings)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            max_entropy = torch.maximum(max_entropy, entropy)
            
            alpha_norm = torch.norm(alpha, dim=1)
            min_alpha_norm = torch.norm(min_alphas, dim=1)
            
            is_better = pc & (alpha_norm < min_alpha_norm)
            min_alphas[is_better] = alpha[is_better]
            pred_change[pc] = True
        
        return pred_change, min_alphas, max_entropy
    
    def _learn_alpha_simplified(self, embeddings: torch.Tensor, pred_1: torch.Tensor, 
                               anchor_i: torch.Tensor, alpha_cap: float, class_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        unlabeled_size = embeddings.size(0)
        embedding_size = embeddings.size(1)
        
        alpha = self._generate_alpha(unlabeled_size, embedding_size, alpha_cap)
        alpha = Variable(alpha, requires_grad=True)
        
        optimizer = torch.optim.Adam([alpha], lr=self.alpha_learning_rate)
        
        pred_changed = torch.zeros(unlabeled_size, dtype=torch.bool).to(self.device)
        
        for iter in range(min(10, self.supporter_epochs)):
            optimizer.zero_grad()
            
            mixed_embeddings = (1 - alpha) * embeddings + alpha * anchor_i
            
            mixed_preds, _ = self._get_mixed_predictions(mixed_embeddings)
            
            label_change = mixed_preds != pred_1
            pred_changed |= label_change
            
            clf_loss = -torch.mean(label_change.float())
            l2_norm = torch.norm(alpha, dim=1)
            loss = self.lambda_hyp * clf_loss + (1-self.lambda_hyp) * torch.mean(l2_norm)
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward(retain_graph=True)
                optimizer.step()
                
                with torch.no_grad():
                    alpha.data = torch.clamp(alpha.data, min=1e-8, max=alpha_cap)
        
        return alpha.detach(), pred_changed
    
    def _get_mixed_predictions(self, mixed_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.supporter_model is None:
            print("Warning: Supporter model is None, returning random predictions")
            num_classes = self.num_classes or 2
            probs = torch.randn(mixed_embeddings.size(0), num_classes, device=self.device).softmax(dim=1)
            predictions = torch.argmax(probs, dim=1)
            return predictions, probs
        
        self.supporter_model.eval()
        with torch.no_grad():
            outputs, _ = self.supporter_model(mixed_embeddings, embedding=True)
            probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        return predictions, probs
    
    def _get_mixed_predictions_change(self, mixed_embeddings: torch.Tensor, original_preds: torch.Tensor) -> torch.Tensor:
        mixed_preds, _ = self._get_mixed_predictions(mixed_embeddings)
        return mixed_preds != original_preds
    
    def _generate_alpha(self, size: int, embedding_size: int, alpha_cap: float) -> torch.Tensor:
        alpha = torch.normal(
            mean=alpha_cap / 2.0,
            std=alpha_cap / 2.0,
            size=(size, embedding_size),
            device=self.device
        )
        
        alpha = torch.nan_to_num(alpha, nan=alpha_cap / 2.0)
        alpha = torch.clamp(alpha, min=1e-8, max=alpha_cap)
        return alpha
    
    def _aggregate_to_image_level(self, candidate_crop_indices: np.ndarray,
                                 crop_info: Dict[str, Any], num_images: int) -> np.ndarray:
        image_scores = np.zeros(num_images)
        for crop_idx in candidate_crop_indices:
            if crop_idx < len(crop_info['crop_to_image']):
                img_idx = crop_info['crop_to_image'][crop_idx]
                image_scores[img_idx] += 1.0
        return image_scores

    def _save_class_flip_counts(self, class_flip_counts: Dict[int, int], total_objects_per_class: Dict[int, int]):
        if self.experiment_dir and self.round is not None:
            num_classes = len(self.model.model.names) # type: ignore
            
            total_flips = sum(class_flip_counts.values())
            
            classwise_quality = []
            
            for class_id in range(num_classes):
                flip_count = class_flip_counts.get(class_id, 0)
                flip_percentage = (flip_count / total_flips) * 100 if total_flips > 0 else 0.0
                classwise_quality.append(flip_percentage)
        
            save_path = Path(self.experiment_dir) / f"round_{self.round}" / "fdal_classwise_quality.npy"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, np.array(classwise_quality))
            
            print(f"Total objects per class: {total_objects_per_class}")
            print(f"Total flips across all classes: {total_flips}")
            print(f"Classwise quality (flip percentages): {classwise_quality}")
            print(f"Saved classwise quality to {save_path}")

    def _write_scores_for_disu(self, crop_info: Dict[str, Any], unlabeled_indices: np.ndarray, 
                              image_changes_list: List[np.ndarray]) -> None:
        if not self.use_disu or not self.prev_score_file:
            return
        
        if not (self.experiment_dir and self.round is not None):
            print("Warning: Cannot write DiSu scores without experiment_dir and round")
            return
        
        round_dir = Path(self.experiment_dir) / f"round_{self.round}"
        round_dir.mkdir(parents=True, exist_ok=True)
        score_file_path = round_dir / self.prev_score_file
        
        print(f"Writing DiSu scores to {score_file_path}")
        
        with open(score_file_path, 'w') as f:
            f.write("image_id,object_id,score\n")
            
            for img_idx, global_img_idx in enumerate(unlabeled_indices):
                if img_idx in crop_info.get('image_to_crops', {}):
                    crops_for_image = crop_info['image_to_crops'][img_idx]
                    
                    total_flips = 0
                    for changes in image_changes_list:
                        if img_idx < len(changes):
                            total_flips += changes[img_idx]
                    
                    for obj_idx in range(len(crops_for_image)):
                        f.write(f"{global_img_idx},{obj_idx},{total_flips}\n")
                else:
                    f.write(f"{global_img_idx},0,0\n")
                    print(f"Warning: No crops found for image index {img_idx} (global index {global_img_idx})")
        
        print(f"Successfully wrote scores for {len(unlabeled_indices)} images to {score_file_path}")

    def get_strategy_name(self) -> str:
        return "fdal"
