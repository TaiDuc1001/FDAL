import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .dataset import ALDataset


class DataManager:
    
    def __init__(self, 
                 original_dataset_path: str,
                 dataset_name: str,
                 class_names: List[str],
                 experiments_root: str = "experiments"):
        self.original_dataset = ALDataset(original_dataset_path, dataset_name, class_names)
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.experiments_root = Path(experiments_root)
        self.experiments_root.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment_dir = None
        self.current_subdataset_dir = None
        
    def create_experiment(self, 
                         strategy_name: str,
                         model_name: str,
                         initial_labeled_indices: np.ndarray,
                         val_indices: Optional[np.ndarray] = None,
                         test_indices: Optional[np.ndarray] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{self.dataset_name}_{strategy_name}_{model_name}_{timestamp}"
        
        self.current_experiment_dir = self.experiments_root / experiment_name
        self.current_experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.original_dataset.set_labeled_indices(initial_labeled_indices)
        
        subdataset_dir = self._create_subdataset(
            round_num=0,
            train_indices=initial_labeled_indices,
            val_indices=val_indices,
            test_indices=test_indices
        )
        
        self._save_experiment_metadata(
            strategy_name=strategy_name,
            model_name=model_name,
            initial_labeled_count=len(initial_labeled_indices)
        )
        
        print(f"Created experiment: {experiment_name}")
        print(f"Initial labeled samples: {len(initial_labeled_indices)}")
        
        return str(self.current_experiment_dir)
    
    def add_labeled_samples(self, 
                           round_num: int,
                           new_labeled_indices: np.ndarray,
                           new_labels_data: Dict[int, List[List[float]]]) -> str:
        if self.current_experiment_dir is None:
            raise RuntimeError("No active experiment. Call create_experiment first.")
        
        prev_round_dir = self.current_experiment_dir / f"round_{round_num-1}"
        if not prev_round_dir.exists():
            raise FileNotFoundError(f"Previous round directory not found: {prev_round_dir}")
            
        with open(prev_round_dir / "metadata.yaml", 'r') as f:
            prev_metadata = yaml.safe_load(f)
            
        prev_labeled_indices = np.array(prev_metadata['train_indices'])
        
        all_labeled_indices = np.concatenate([prev_labeled_indices, new_labeled_indices])
        
        self.original_dataset.set_labeled_indices(all_labeled_indices)
        
        # self._write_new_label_files(new_labeled_indices, new_labels_data)
        
        subdataset_dir = self._create_subdataset(
            round_num=round_num,
            train_indices=all_labeled_indices,
            val_indices=np.array(prev_metadata['val_indices']),
            test_indices=np.array(prev_metadata['test_indices']) if prev_metadata.get('test_indices') else None
        )
        
        print(f"Round {round_num}: Added {len(new_labeled_indices)} new labeled samples")
        print(f"Total labeled samples: {len(all_labeled_indices)}")
        
        return str(subdataset_dir)
    
    def _create_subdataset(self,
                          round_num: int,
                          train_indices: np.ndarray,
                          val_indices: Optional[np.ndarray] = None,
                          test_indices: Optional[np.ndarray] = None) -> Path:
        subdataset_dir = self.current_experiment_dir / f"round_{round_num}" # type: ignore
        subdataset_dir.mkdir(parents=True, exist_ok=True)
        images_dir = subdataset_dir / "images"
        labels_dir = subdataset_dir / "labels" 
        
        for split in ["train", "val", "test"]:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        self._create_symlinks_for_split(
            subdataset_dir, "train", train_indices
        )
        
        if val_indices is None:
            val_indices = self.original_dataset.val_indices
            
        if len(val_indices) > 0:
            self._create_symlinks_for_split(
                subdataset_dir, "val", val_indices
            )
        
        if test_indices is not None and len(test_indices) > 0:
            self._create_symlinks_for_split(
                subdataset_dir, "test", test_indices
            )
        
        yaml_path = subdataset_dir / "data.yaml"
        self.original_dataset.create_dataset_yaml(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            output_path=str(yaml_path)
        )
        
        self._save_round_metadata(
            subdataset_dir, round_num, train_indices, val_indices, test_indices
        )
        
        self.current_subdataset_dir = subdataset_dir
        return subdataset_dir
    
    def _create_symlinks_for_split(self,
                                  subdataset_dir: Path,
                                  split: str,
                                  indices: np.ndarray) -> None:
        images_split_dir = subdataset_dir / "images" / split
        labels_split_dir = subdataset_dir / "labels" / split
        
        valid_indices = self._validate_indices(indices)
        if len(valid_indices) < len(indices):
            print(f"Warning: {len(indices) - len(valid_indices)} indices have missing source files for {split} split")
        
        for idx in valid_indices:
            img_path = self.original_dataset.all_images[idx]
            label_path = self.original_dataset.all_labels[idx]
            
            img_symlink = images_split_dir / img_path.name
            if not img_symlink.exists():
                try:
                    os.symlink(str(img_path.resolve()), str(img_symlink))
                except Exception:
                    try:
                        os.link(str(img_path.resolve()), str(img_symlink))
                    except Exception:
                        shutil.copy2(str(img_path), str(img_symlink))
            
            if img_symlink.is_symlink() and not img_symlink.exists():
                print(f"Created broken symlink: {img_symlink} -> {img_path}")
                img_symlink.unlink()
                continue
            
            if label_path is not None and label_path.exists():
                label_symlink = labels_split_dir / label_path.name
                if not label_symlink.exists():
                    try:
                        os.symlink(str(label_path.resolve()), str(label_symlink))
                    except Exception:
                        try:
                            os.link(str(label_path.resolve()), str(label_symlink))
                        except Exception:
                            shutil.copy2(str(label_path), str(label_symlink))
    
    def _validate_indices(self, indices: np.ndarray) -> np.ndarray:
        valid_indices = []
        for idx in indices:
            if idx < len(self.original_dataset.all_images):
                img_path = self.original_dataset.all_images[idx]
                if img_path.exists():
                    valid_indices.append(idx)
                else:
                    print(f"Skipping index {idx}: image does not exist at {img_path}")
            else:
                print(f"Skipping index {idx}: out of range (max: {len(self.original_dataset.all_images)-1})")
        
        print(f"Validated {len(valid_indices)}/{len(indices)} indices")
        return np.array(valid_indices)
    
    def _write_new_label_files(self, 
                              indices: np.ndarray,
                              labels_data: Dict[int, List[List[float]]]) -> None:
        temp_labels_dir = self.current_experiment_dir / "new_labels"  # type: ignore
        temp_labels_dir.mkdir(parents=True, exist_ok=True)
        def _write_one(idx: int, annotations: List[List[float]]):
            label_lines = [" ".join(map(str, ann)) for ann in annotations]
            img_path = self.original_dataset.all_images[idx]
            label_filename = img_path.stem + ".txt"
            temp_label_path = temp_labels_dir / label_filename
            with open(temp_label_path, 'w', newline='\n') as f:
                f.write("\n".join(label_lines))
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    print(f"Warning: fsync failed for {temp_label_path}")
                    pass
            return idx, temp_label_path

        tasks = []
        for idx in indices:
            if idx in labels_data:
                tasks.append((idx, labels_data[idx]))

        if not tasks:
            print("No tasks to process")
            return

        max_workers = min(32, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {ex.submit(_write_one, idx, ann): idx for idx, ann in tasks}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    _, path = future.result()
                    self.original_dataset.all_labels[idx] = path
                except Exception as e:
                    print(f"Error occurred while writing labels for index {idx}: {e}")
                    raise
    
    def _save_experiment_metadata(self,
                                strategy_name: str,
                                model_name: str,
                                initial_labeled_count: int) -> None:
        metadata = {
            'dataset_name': self.dataset_name,
            'strategy_name': strategy_name,
            'model_name': model_name,
            'class_names': self.class_names,
            'initial_labeled_count': initial_labeled_count,
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.current_experiment_dir / "experiment_metadata.yaml", 'w') as f: # type: ignore
            yaml.dump(metadata, f, default_flow_style=False)
    
    def _save_round_metadata(self,
                           subdataset_dir: Path,
                           round_num: int,
                           train_indices: np.ndarray,
                           val_indices: Optional[np.ndarray],
                           test_indices: Optional[np.ndarray]) -> None:
        metadata = {
            'round': round_num,
            'train_count': len(train_indices),
            'val_count': len(val_indices) if val_indices is not None else 0,
            'test_count': len(test_indices) if test_indices is not None else 0,
            'train_indices': train_indices.tolist(),
            'val_indices': val_indices.tolist() if val_indices is not None else [],
            'test_indices': test_indices.tolist() if test_indices is not None else [],
            'created_at': datetime.now().isoformat()
        }
        
        with open(subdataset_dir / "metadata.yaml", 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
    
    def get_current_dataset_yaml(self) -> str:
        if self.current_subdataset_dir is None:
            raise RuntimeError("No active subdataset")
        return str(self.current_subdataset_dir / "data.yaml")
    
    def get_unlabeled_indices(self) -> np.ndarray:
        return self.original_dataset.get_unlabeled_indices()
    
    def get_all_image_paths(self) -> List[str]:
        return self.original_dataset.get_image_paths()
