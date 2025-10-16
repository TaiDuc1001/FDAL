import numpy as np
from typing import List

from .base import BaseStrategy


class UncertaintyStrategy(BaseStrategy):
    
    def __init__(self, model, uncertainty_method: str = "entropy", **kwargs):
        super().__init__(model, **kwargs)
        self.uncertainty_method = uncertainty_method
        
    def query(self, 
              unlabeled_indices: np.ndarray,
              image_paths: List[str],
              n_samples: int,
              **kwargs) -> np.ndarray:
        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        
        unlabeled_image_paths = self._get_image_paths_for_indices(unlabeled_indices, image_paths)
        
        results = self.model.inference(
            unlabeled_image_paths,
            return_boxes=True,
            return_probs=True,
            num_inference=kwargs.get('num_inference', -1),
            **kwargs
        )
        
        uncertainty_scores = []
        
        for result in results:
            if result.probs is not None and len(result.probs) > 0:
                if self.uncertainty_method == "entropy":
                    probs = result.probs
                    entropy = -np.mean(probs * np.log(probs + 1e-8))
                    uncertainty_scores.append(entropy)
                    
                elif self.uncertainty_method == "confidence":
                    max_conf = np.max(result.probs)
                    uncertainty_scores.append(1.0 - max_conf)
                    
                elif self.uncertainty_method == "margin":
                    if len(result.probs) >= 2:
                        sorted_probs = np.sort(result.probs)
                        margin = sorted_probs[-1] - sorted_probs[-2]
                        uncertainty_scores.append(1.0 - margin)
                    else:
                        uncertainty_scores.append(1.0 - result.probs[0])
                        
                else:
                    raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
            else:
                uncertainty_scores.append(1.0)
        
        uncertainty_scores = np.array(uncertainty_scores)
        
        top_uncertain_idx = np.argsort(uncertainty_scores)[-n_samples:]
        selected_indices = unlabeled_indices[top_uncertain_idx]
        
        return selected_indices
    
    def get_strategy_name(self) -> str:
        return f"uncertainty_{self.uncertainty_method}"
