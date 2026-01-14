import os
import time
import torch
import threading
import numpy as np
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from typing import List, Optional
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

from ..base import BaseStrategy


class BADGEStrategy(BaseStrategy):
    
    def __init__(self, 
                 model,
                 round: Optional[int] = None,
                 experiment_dir: Optional[str] = None,
                 uncertainty_method: str = "entropy",
                 distance_metric: str = "cosine",
                 beta: float = 1.0,
                 **kwargs):
        super().__init__(model, **kwargs)
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.uncertainty_method = uncertainty_method
        self.distance_metric = distance_metric
        self.beta = beta
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

        unlabeled_image_paths = self._get_image_paths_for_indices(unlabeled_indices, image_paths)
        start_time = time.time()
        results = self.model.inference(
            unlabeled_image_paths,
            return_boxes=True,
            return_probs=True,
            **kwargs
        )
        
        gradient_embeddings = []
        uncertainty_scores = []
        
        for result in results:
            if result.layer_gradients is not None:
                grad_embedding = self._compute_gradient_embedding(result.layer_gradients)
                gradient_embeddings.append(grad_embedding)
            else:
                gradient_embeddings.append(None)
            
            if result.probs is not None and len(result.probs) > 0:
                uncertainty = self._compute_uncertainty(result.probs)
                uncertainty_scores.append(uncertainty)
            else:
                uncertainty_scores.append(0.0)
        sorted_by_unc = np.argsort(uncertainty_scores)[::-1]
        oversample_factor = 3
        candidate_count = min(len(sorted_by_unc), max(n_samples * oversample_factor, n_samples))
        candidates_local = sorted_by_unc[:candidate_count].tolist()

        if len(candidates_local) == 0:
            return np.random.choice(unlabeled_indices, size=min(n_samples, len(unlabeled_indices)), replace=False)

        gpu_lock = threading.Lock()

        def _compute_for_candidate(local_idx):
            img_path = unlabeled_image_paths[local_idx]
            inf_res = results[local_idx]
            compute_fn = getattr(self.model, "_compute_layer_gradients", None)
            if not callable(compute_fn):
                return local_idx, None
            with gpu_lock:
                try:
                    grad = compute_fn(img_path, inf_res)
                except Exception:
                    grad = None
            return local_idx, grad

        max_workers = 8
        computed_embeddings = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(_compute_for_candidate, li) for li in candidates_local]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="gradients", unit="img"):
                try:
                    local_idx, grad = fut.result()
                    if grad is not None:
                        computed_embeddings[local_idx] = self._compute_gradient_embedding(grad) # type: ignore
                except Exception:
                    pass

        valid_local_idxs = [i for i in candidates_local if computed_embeddings.get(i) is not None]
        if len(valid_local_idxs) == 0:
            top_local = sorted_by_unc[:n_samples].tolist()
            selected_global = [unlabeled_indices[i] for i in top_local]
            return np.array(selected_global)

        valid_embeddings = np.array([computed_embeddings[i] for i in valid_local_idxs])
        valid_uncertainties = np.array([uncertainty_scores[i] for i in valid_local_idxs])
        selected_local_rel = self._badge_selection(valid_embeddings, valid_uncertainties, n_samples)
        selected_local = [valid_local_idxs[i] for i in selected_local_rel]
        selected_global = [unlabeled_indices[i] for i in selected_local]

        total_time = time.time() - start_time
        num_images = len(unlabeled_image_paths)
        time_per_image = total_time / num_images
        with open(timelog_file, 'a') as f:
            f.write(f"{self.round},{total_time:.4f},{num_images},{time_per_image:.6f}\n")
        print("Write time log to", timelog_file.absolute())
        selected_image_paths = [image_paths[i] for i in selected_global]
        selected_image_names = [Path(p).name for p in selected_image_paths]
        with open(selectionlog_file, 'a') as f:
            f.write(','.join(selected_image_names) + '\n')
        print("Write to selection log file. ", selectionlog_file.absolute())

        self._save_predictions_for_selection(
            experiment_dir=self.experiment_dir,
            round_num=self.round,
            selected_image_paths=selected_image_paths,
            image_paths=image_paths,
            selected_indices=np.array(selected_global),
            results=results,
            unlabeled_indices=unlabeled_indices,
        )
        self._save_selection_symlinks(
            experiment_dir=self.experiment_dir,
            round_num=self.round,
            selected_image_paths=selected_image_paths,
        )

        return np.array(selected_global)
    
    def _compute_gradient_embedding(self, layer_gradients: np.ndarray) -> np.ndarray:
        if layer_gradients.ndim == 1:
            return layer_gradients
        elif layer_gradients.ndim == 2:
            return layer_gradients.flatten()
        else:
            return layer_gradients.reshape(-1)
    
    def _compute_uncertainty(self, probs: np.ndarray) -> float:
        if self.uncertainty_method == "entropy":
            probs = np.clip(probs, 1e-8, 1.0)
            entropy = -np.mean(probs * np.log(probs))
            return float(entropy)
            
        elif self.uncertainty_method == "confidence":
            max_conf = np.max(probs)
            return 1.0 - float(max_conf)
            
        elif self.uncertainty_method == "margin":
            if len(probs) >= 2:
                sorted_probs = np.sort(probs)
                margin = sorted_probs[-1] - sorted_probs[-2]
                return 1.0 - float(margin)
            else:
                return 1.0 - float(probs[0])
        else:
            return float(np.std(probs))
    
    def _badge_selection(self, 
                        embeddings: np.ndarray, 
                        uncertainties: np.ndarray, 
                        n_samples: int) -> List[int]:
        n_points = len(embeddings)
        
        if n_samples >= n_points:
            return list(range(n_points))
        
        if n_samples == 1:
            max_uncertainty_idx = np.argmax(uncertainties)
            return [int(max_uncertainty_idx)]
        
        try:
            selected_indices = self._kmeans_plus_plus_with_uncertainty(
                embeddings, uncertainties, n_samples
            )
        except Exception:
            top_uncertain_indices = np.argsort(uncertainties)[-n_samples:]
            selected_indices = top_uncertain_indices.tolist()
        
        return selected_indices
    
    def _kmeans_plus_plus_with_uncertainty(self, 
                                          embeddings: np.ndarray, 
                                          uncertainties: np.ndarray, 
                                          n_samples: int) -> List[int]:
        n_points = len(embeddings)
        selected = []
        
        weighted_uncertainties = uncertainties ** self.beta
        weighted_uncertainties = weighted_uncertainties / np.sum(weighted_uncertainties)
        
        first_idx = np.random.choice(n_points, p=weighted_uncertainties)
        selected.append(first_idx)
        
        for _ in range(1, n_samples):
            selected_embeddings = embeddings[selected]
            
            distances = pairwise_distances(
                embeddings, selected_embeddings, metric=self.distance_metric
            )
            min_distances = np.min(distances, axis=1)
            
            combined_scores = min_distances * weighted_uncertainties
            
            combined_scores[selected] = 0
            
            if np.sum(combined_scores) == 0:
                remaining = [i for i in range(n_points) if i not in selected]
                if remaining:
                    next_idx = np.random.choice(remaining)
                else:
                    break
            else:
                probs = combined_scores / np.sum(combined_scores)
                next_idx = np.random.choice(n_points, p=probs)
            
            selected.append(next_idx)
        
        return selected
    
    def get_strategy_name(self) -> str:
        return f"BADGE_{self.uncertainty_method}_{self.distance_metric}_beta{self.beta}"
