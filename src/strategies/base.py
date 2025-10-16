from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from ..models.base import BaseModel


class BaseStrategy(ABC):
    
    def __init__(self, 
                 model: BaseModel,
                 **kwargs):
        self.model = model
        self.strategy_params = kwargs
        
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
