import numpy as np
from typing import List

from .base import BaseStrategy


class RandomStrategy(BaseStrategy):
    
    def query(self, 
              unlabeled_indices: np.ndarray,
              image_paths: List[str],
              n_samples: int,
              **kwargs) -> np.ndarray:
        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        
        selected_indices = np.random.choice(
            unlabeled_indices, 
            size=n_samples, 
            replace=False
        )
        
        return selected_indices
    
    def get_strategy_name(self) -> str:
        return "random"
