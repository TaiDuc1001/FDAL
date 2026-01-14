from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image

from ..models.base import BaseModel


class BaseStrategy(ABC):
    
    def __init__(self, 
                 model: BaseModel,
                 **kwargs):
        self.model = model
        self.strategy_params = kwargs
        
    def _save_predictions_for_selection(
        self,
        experiment_dir: str,
        round_num: int,
        selected_image_paths: List[str],
        image_paths: List[str],
        selected_indices: np.ndarray,
        results: Optional[List] = None,
        unlabeled_indices: Optional[np.ndarray] = None,
    ) -> None:
        if experiment_dir is None or round_num is None:
            return
            
        pred_dir = Path(experiment_dir) / "selection_prediction" / f"round_{round_num}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        if results is not None and unlabeled_indices is not None:
            idx_to_result = {unlabeled_indices[i]: results[i] for i in range(len(results))}
        else:
            idx_to_result = {}
        
        for global_idx, img_path in zip(selected_indices, selected_image_paths):
            img_path_obj = Path(img_path)
            pred_file = pred_dir / f"{img_path_obj.stem}.txt"
            
            result = idx_to_result.get(global_idx)
            if result is None:
                pred_file.touch()
                continue
            
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
            except Exception:
                img_width, img_height = 640, 640
            
            lines = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                classes = result.classes if result.classes is not None else [0] * len(boxes)
                
                for box, cls in zip(boxes, classes):
                    x1, y1, x2, y2 = box[:4]
                    
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    lines.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            with open(pred_file, 'w') as f:
                f.write('\n'.join(lines))
        
        print(f"Saved predictions for {len(selected_image_paths)} selected images to {pred_dir}")
    
    def _save_selection_symlinks(
        self,
        experiment_dir: str,
        round_num: int,
        selected_image_paths: List[str],
    ) -> None:
        import os
        if experiment_dir is None or round_num is None:
            return
            
        selection_dir = Path(experiment_dir) / f"round_{round_num}" / "selection"
        selection_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in selected_image_paths:
            img_path_obj = Path(img_path)
            symlink_path = selection_dir / img_path_obj.name
            if symlink_path.exists():
                symlink_path.unlink()
            os.symlink(img_path_obj.resolve(), symlink_path)
        
        print(f"Created {len(selected_image_paths)} symlinks in {selection_dir}")
        
    @abstractmethod
    def query(self, 
              unlabeled_indices: np.ndarray,
              image_paths: List[str],
              n_samples: int,
              **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        pass
    
    def _validate_inputs(self, 
                        unlabeled_indices: np.ndarray,
                        image_paths: List[str],
                        n_samples: int) -> None:
        if len(unlabeled_indices) == 0:
            raise ValueError("No unlabeled samples provided")
            
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
            
        if n_samples > len(unlabeled_indices):
            raise ValueError(f"Cannot select {n_samples} samples from {len(unlabeled_indices)} unlabeled samples")
            
        if max(unlabeled_indices) >= len(image_paths):
            raise ValueError("unlabeled_indices contains invalid indices for image_paths")
    
    def _get_image_paths_for_indices(self, 
                                   indices: np.ndarray,
                                   all_image_paths: List[str]) -> List[str]:
        return [all_image_paths[idx] for idx in indices]
