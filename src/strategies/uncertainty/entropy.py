import os
import time
import numpy as np
from typing import List
from pathlib import Path
from typing import Optional
from ..base import BaseStrategy

class EntropyStrategy(BaseStrategy):
    def __init__(self, 
                 model,
                 round: Optional[int] = None,
                 experiment_dir: Optional[str] = None,
                 **kwargs):
        super().__init__(model, **kwargs)
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.round = round
        self.experiment_dir = experiment_dir

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

        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        unlabeled_image_paths = self._get_image_paths_for_indices(unlabeled_indices, image_paths)
        
        local_kwargs = dict(kwargs)
        num_inf = local_kwargs.pop('num_inference', -1)
        unlabeled_image_paths = unlabeled_image_paths[:num_inf] if num_inf > 0 else unlabeled_image_paths

        start_time = time.time()
        results = self.model.inference(
            unlabeled_image_paths,
            return_boxes=True,
            return_probs=True,
            num_inference=num_inf,
            **local_kwargs
        )
        
        uncertainty_scores = []
        defects = 0
        for result in results:
            if result.probs is not None and len(result.probs) > 0:
                probs = np.asarray(result.probs)
                probs = probs.astype(float)
                if probs.ndim == 1:
                    probs = probs.reshape(1, -1)
                
                per_det_entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
                
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    boxes = np.array(result.boxes)
                    if boxes.shape[1] >= 5:
                        confidences = boxes[:, 4]
                        conf_weights = 1.0 - confidences
                        if len(conf_weights) == len(per_det_entropy):
                            weighted_entropy = per_det_entropy * (1.0 + conf_weights)
                            entropy = float(np.mean(weighted_entropy))
                        else:
                            entropy = float(np.mean(per_det_entropy))
                    else:
                        entropy = float(np.mean(per_det_entropy))
                else:
                    entropy = float(np.mean(per_det_entropy))
                
                uncertainty_scores.append(entropy)
            else:
                uncertainty_scores.append(np.inf)
                defects += 1
        
        print(f"Defects: {defects}/{len(results)}")
        uncertainty_scores = np.array(uncertainty_scores)
        top_uncertain_idx = np.argsort(uncertainty_scores)[-n_samples:]
        selected_indices = unlabeled_indices[top_uncertain_idx]
        end_time = time.time()
        total_time = end_time - start_time
        time_per_image = total_time / len(unlabeled_image_paths)
        with open(timelog_file, 'a') as f:
            f.write(f"{self.round},{total_time:.4f},{len(unlabeled_image_paths)},{time_per_image:.4f}\n")
        print("Write time log to", timelog_file.absolute())
        selected_image_paths = [image_paths[i] for i in selected_indices]
        selected_image_names = [Path(p).name for p in selected_image_paths]
        with open(selectionlog_file, 'a') as f:
            f.write(','.join(selected_image_names) + '\n')
        print("Write to selection log file. ", selectionlog_file.absolute())
        
        return selected_indices
    
    def get_strategy_name(self) -> str:
        return "entropy"
