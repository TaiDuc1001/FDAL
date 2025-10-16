import numpy as np
from typing import List

from ..base import BaseStrategy


class MarginStrategy(BaseStrategy):
    
    def query(self, 
              unlabeled_indices: np.ndarray,
              image_paths: List[str],
              n_samples: int,
              **kwargs) -> np.ndarray:
        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        
        unlabeled_image_paths = self._get_image_paths_for_indices(unlabeled_indices, image_paths)
        num_inf = kwargs.get('num_inference', -1)
        unlabeled_image_paths = unlabeled_image_paths[:num_inf] if num_inf > 0 else unlabeled_image_paths
        results = self.model.inference(
            unlabeled_image_paths,
            return_boxes=True,
            return_probs=True,
        )
        
        uncertainty_scores = []
        
        for result in results:
            probs = result.probs
            if probs is None:
                uncertainty_scores.append(1.0)
                continue

            probs_arr = np.asarray(probs)
            if probs_arr.size == 0:
                uncertainty_scores.append(1.0)
                continue

            probs_flat = probs_arr.ravel()
            if probs_flat.size >= 2:
                sorted_probs = np.sort(probs_flat)
                margin = sorted_probs[-1] - sorted_probs[-2]
                uncertainty_scores.append(float(1.0 - margin))
            else:
                print('Single probability value detected.')
                uncertainty_scores.append(float(1.0 - probs_flat[0]))
        
        uncertainty_scores = np.array(uncertainty_scores)
        top_uncertain_idx = np.argsort(uncertainty_scores)[-n_samples:]
        selected_indices = unlabeled_indices[top_uncertain_idx]
        
        return selected_indices
    
    def get_strategy_name(self) -> str:
        return "margin"
