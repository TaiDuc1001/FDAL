import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
from loguru import logger


class ALDataset:
    def __init__(self, 
                 dataset_root: str,
                 dataset_name: str,
                 class_names: List[str]):
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        self._validate_dataset_structure()
        self.all_images, self.all_labels = self._discover_images_and_labels()
        self.total_images = len(self.all_images)
        
        self.train_indices, self.val_indices, self.test_indices = self._parse_dataset_yaml()
        
        self.active_labeled_indices = set()
        
    def _parse_dataset_yaml(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_indices = []
        val_indices = []
        test_indices = []
        
        if hasattr(self, 'split_assignments') and self.split_assignments:
            for image_index, split_name in self.split_assignments.items():
                if split_name == 'train':
                    train_indices.append(image_index)
                elif split_name == 'val':
                    val_indices.append(image_index)
                elif split_name == 'test':
                    test_indices.append(image_index)
                else:
                    train_indices.append(image_index)
        else:
            train_indices = list(range(self.total_images))
        
        logger.info(f"Dataset splits from YAML - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        return np.array(train_indices), np.array(val_indices), np.array(test_indices)
        
    def _validate_dataset_structure(self) -> None:
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
            
        images_dir = self.dataset_root / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
            
        labels_dir = self.dataset_root / "labels"
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        logger.info(f"Dataset validation passed for: {self.dataset_root}")

    def _discover_images_and_labels(self) -> Tuple[List[Path], List[Optional[Path]]]:
        images_dir = self.dataset_root / "images"
        labels_dir = self.dataset_root / "labels"
        
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        self.split_assignments = {}
        all_images = []
        all_labels = []
        
        yaml_file = self.dataset_root / "data.yaml"
        if not yaml_file.exists():
            yaml_file = self.dataset_root.parent / "data.yaml"
        
        if yaml_file.exists():
            with open(yaml_file, 'r') as f:
                data_config = yaml.safe_load(f)
            
            for split_name in ['train', 'val', 'test']:
                split_config = data_config.get(split_name, [])
                split_paths = []
                
                if isinstance(split_config, str):
                    split_paths = [split_config]
                elif isinstance(split_config, list):
                    split_paths = split_config
                
                for split_path in split_paths:
                    sp = Path(split_path)
                    candidates = [sp, self.dataset_root / split_path, self.dataset_root.parent / split_path]
                    split_dir = None
                    for c in candidates:
                        if c.exists():
                            split_dir = c
                            break
                    if split_dir is None:
                        cwd_candidate = Path.cwd() / split_path
                        if cwd_candidate.exists():
                            split_dir = cwd_candidate

                    if split_dir is None:
                        logger.warning(f"Split path not found: {split_path}")
                        continue

                    for img_path in split_dir.rglob('*'):
                        if img_path.suffix.lower() in img_extensions and img_path.exists():
                            image_index = len(all_images)
                            self.split_assignments[image_index] = split_name
                            all_images.append(img_path)
                            
                            if images_dir in img_path.parents:
                                rel_path = img_path.relative_to(images_dir)
                            else:
                                try:
                                    rel_path = img_path.relative_to(split_dir.parent / "images")
                                except Exception:
                                    rel_path = img_path.name

                            label_path = labels_dir / Path(rel_path).with_suffix('.txt')
                            if label_path.exists():
                                all_labels.append(label_path)
                            else:
                                all_labels.append(None)
        else:
            if images_dir.exists():
                for img_path in tqdm(images_dir.rglob('*'), desc="Discovering images"):
                    if img_path.suffix.lower() in img_extensions and img_path.exists():
                        image_index = len(all_images)
                        self.split_assignments[image_index] = 'train'
                        all_images.append(img_path)
                        rel_path = img_path.relative_to(images_dir)
                        label_path = labels_dir / rel_path.with_suffix('.txt')
                        if label_path.exists():
                            all_labels.append(label_path)
                        else:
                            all_labels.append(None)
        
        logger.info(f"Discovered {len(all_images)} images in dataset")
        logger.info(f"Total images available: {len(all_images)}")

        self._validate_discovered_images(all_images)

        return all_images, all_labels
    
    def _validate_discovered_images(self, all_images: List[Path]) -> None:
        invalid_indices = []
        for i, img_path in enumerate(all_images):
            if not img_path.exists():
                invalid_indices.append(i)
        
        if invalid_indices:
            logger.warning(f"Found {len(invalid_indices)} invalid image paths")
            for idx in invalid_indices[:5]:
                logger.warning(f"Invalid: {all_images[idx]}")
            if len(invalid_indices) > 5:
                logger.warning(f"... and {len(invalid_indices) - 5} more")
        else:
            logger.info("All discovered images validated successfully")
    
    def get_labeled_indices(self) -> np.ndarray:
        return np.array(list(self.active_labeled_indices))
    
    def get_unlabeled_indices(self) -> np.ndarray:
        training_indices_set = set(self.train_indices)
        unlabeled_training_indices = training_indices_set - self.active_labeled_indices
        return np.array(list(unlabeled_training_indices))
    
    def set_labeled_indices(self, indices: np.ndarray) -> None:
        self.active_labeled_indices = set(indices)
    
    def get_image_paths(self, indices: Optional[np.ndarray] = None) -> List[str]:
        if indices is None:
            return [str(path) for path in self.all_images]
        else:
            return [str(self.all_images[i]) for i in indices]
    
    def get_label_paths(self, indices: Optional[np.ndarray] = None) -> List[Optional[str]]:
        if indices is None:
            return [str(path) if path else None for path in self.all_labels]
        else:
            return [str(self.all_labels[i]) if self.all_labels[i] else None for i in indices]
    
    def create_dataset_yaml(self, 
                           train_indices: np.ndarray,
                           val_indices: np.ndarray,
                           test_indices: Optional[np.ndarray] = None,
                           output_path: str = "data.yaml") -> str:
        yaml_dir = Path(output_path).parent
        yaml_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_config = {
            'path': str(yaml_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(self.class_names)},
            'nc': self.num_classes
        }
        
        if test_indices is not None:
            dataset_config['test'] = 'images/test'
        
        with open(output_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
            
        return output_path
