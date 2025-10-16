import os
import time
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Optional

from ..base import BaseStrategy

class CDALStrategy(BaseStrategy):
    
    def __init__(self, 
                 model,
                 experiment_dir: Optional[str] = None,
                 round: Optional[int] = None,
                 epsilon: float = 1e-8,
                 conf_threshold: float = 0.25,
                 **kwargs):
        super().__init__(model, **kwargs)
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.round = round
        self.experiment_dir = experiment_dir
        self.epsilon = float(epsilon)
        self.conf_threshold = float(conf_threshold)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        conf = local_kwargs.get('conf', self.conf_threshold)
        
        start_time = time.time()
        results = self.model.inference(
            unlabeled_image_paths,
            return_boxes=True,
            return_classes=True,
            return_probs=True,
            num_inference=num_inf,
            conf=conf,
            **local_kwargs
        )
        
        if len(results) == 0:
            print("Warning: No inference results available")
            return np.random.choice(unlabeled_indices, size=min(n_samples, len(unlabeled_indices)), replace=False)
        
        probs_dict = {}
        pseudo_labels_dict = {}
        all_classes = set()
        
        for i, result in enumerate(results):
            image_idx = i
            probs_dict[image_idx] = []
            pseudo_labels_dict[image_idx] = []
            
            if (result.probs is not None and result.classes is not None and 
                len(result.probs) > 0 and len(result.classes) > 0):
                
                probs_array = np.asarray(result.probs)
                classes_array = np.asarray(result.classes)

                probs_dict[image_idx] = probs_array
                pseudo_labels_dict[image_idx] = classes_array

                all_classes.update(classes_array.astype(int))
            else:
                print(f"Warning: Image {image_idx} has invalid or empty inference results, skipping")
                probs_dict[image_idx] = np.array([])
                pseudo_labels_dict[image_idx] = np.array([])
        
        if len(all_classes) == 0:
            print("Warning: No class predictions found")
            return np.random.choice(unlabeled_indices, size=min(n_samples, len(unlabeled_indices)), replace=False)
        
        all_classes = sorted(list(all_classes))
        num_classes = len(self.model.model.names)  # type: ignore
        max_class_id = max(all_classes) if all_classes else 0
        matrix_size = max(num_classes, max_class_id + 1)
        print(f"CDAL: Using matrix size {matrix_size} (model has {num_classes} classes, max class ID is {max_class_id})")
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        
        P_per_image = self._compute_confusion(
            list(range(len(results))), probs_dict, pseudo_labels_dict, 
            all_classes, matrix_size
        )
        
        dist_matrix = np.zeros((len(results), len(results)))
        for c in all_classes:
            for i in range(len(results)):
                if c not in P_per_image[i]:
                    continue
                P_i = P_per_image[i][c]
                P_i = np.clip(P_i, self.epsilon, 1.0)
                P_i = P_i / P_i.sum()
                for j in range(i + 1, len(results)):
                    if c not in P_per_image[j]:
                        continue
                    P_j = P_per_image[j][c]
                    P_j = np.clip(P_j, self.epsilon, 1.0)
                    P_j = P_j / P_j.sum()
                    try:
                        kl1 = np.sum(P_i * np.log(P_i / P_j))
                        kl2 = np.sum(P_j * np.log(P_j / P_i))
                        d = 0.5 * (kl1 + kl2)
                        dist_matrix[i, j] += d
                        dist_matrix[j, i] += d
                    except Exception as e:
                        print(f"Warning: Failed to compute KL divergence for class {c}, images {i},{j}: {e}")
                        continue
        print("Distance matrix computed.")
        
        selected_local_indices = self._select_batch(
            list(range(len(results))), P_per_image, all_classes, n_samples, dist_matrix
        )
        
        print(f"CDAL: Selected {len(selected_local_indices)} out of {len(results)} images for batch")
        
        selected_global_indices = [unlabeled_indices[i] for i in selected_local_indices]
        selected_indices = np.array(selected_global_indices)
        total_time = time.time() - start_time
        num_images = len(unlabeled_image_paths)
        time_per_image = total_time / num_images
        with open(timelog_file, 'a') as f:
            f.write(f"{self.round},{total_time:.4f},{num_images},{time_per_image:.6f}\n")
        print("Write time log to", timelog_file.absolute())
        selected_image_paths = [image_paths[i] for i in selected_indices]
        selected_image_names = [Path(p).name for p in selected_image_paths]
        with open(selectionlog_file, 'a') as f:
            f.write(','.join(selected_image_names) + '\n')
        print("Write to selection log file. ", selectionlog_file.absolute())
        
        print(f"CDAL Strategy: Selected {len(selected_indices)} samples from {len(unlabeled_indices)} unlabeled images")
        print(f"First 5 selected image indices: {selected_indices[:5]}")
        
        if len(selected_local_indices) > 1:
            pairs = [(i, j) for i in range(len(selected_local_indices)) 
                    for j in range(i+1, len(selected_local_indices))]
            
            diversity_scores = []
            
            for i, j in tqdm(pairs, desc="Computing diversity scores"):
                score = dist_matrix[selected_local_indices[i], selected_local_indices[j]]
                diversity_scores.append(score)
            
            if diversity_scores:
                diversity_scores = np.array(diversity_scores)
                print(f"Diversity scores statistics:")
                print(f"  Mean: {diversity_scores.mean():.4f}")
                print(f"  Std: {diversity_scores.std():.4f}")
                print(f"  Min: {diversity_scores.min():.4f}")
                print(f"  Max: {diversity_scores.max():.4f}")
                print(f"  Median: {np.median(diversity_scores):.4f}")
            else:
                print("Warning: No diversity scores could be computed")
        
        return selected_indices
    
    def _compute_confusion(self, 
                              U: List[int], 
                              probs: Dict[int, np.ndarray], 
                              pseudo_labels: Dict[int, np.ndarray], 
                              C: List[int],
                              matrix_size: int) -> Dict[int, Dict[int, np.ndarray]]:
        P_per_image = {}
        
        for I in U:
            P_per_image[I] = {}
            
            if len(pseudo_labels[I]) == 0 or len(probs[I]) == 0:
                print(f"Warning: Image {I} has no pseudo-labels or probabilities, skipping confusion computation")
                continue
                
            unique_classes = np.unique(pseudo_labels[I])
            
            probs_tensor = torch.tensor(probs[I], dtype=torch.float32, device=self.device)
            labels_tensor = torch.tensor(pseudo_labels[I], dtype=torch.long, device=self.device)
            
            for c in unique_classes:
                if c not in C:
                    print(f"Warning: Class {c} not in allowed classes list, skipping")
                    continue
                    
                mask = (labels_tensor == c)
                if not mask.any():
                    print(f"Warning: No regions found for class {c} in image {I}, skipping")
                    continue
                
                regions_probs = probs_tensor[mask]
                
                if regions_probs.dim() == 1:
                    try:
                        prob_matrices = torch.zeros(len(regions_probs), matrix_size, device=self.device)
                        prob_matrices[:, int(c)] = regions_probs
                        remaining = 1.0 - regions_probs.unsqueeze(1)
                        if matrix_size > 1:
                            other_prob = remaining / (matrix_size - 1)
                            for i in range(matrix_size):
                                if i != int(c):
                                    prob_matrices[:, i] = other_prob.squeeze()
                    except Exception as e:
                        print(f"Warning: Failed to create probability matrix for class {c} in image {I}: {e}")
                        continue
                else:
                    prob_matrices = regions_probs
                    if prob_matrices.size(1) < matrix_size:
                        padded = torch.zeros(prob_matrices.size(0), matrix_size, device=self.device)
                        padded[:, :prob_matrices.size(1)] = prob_matrices
                        prob_matrices = padded
                    elif prob_matrices.size(1) > matrix_size:
                        prob_matrices = prob_matrices[:, :matrix_size]
                
                prob_matrices = torch.clamp(prob_matrices, self.epsilon, 1.0)
                prob_matrices = prob_matrices / prob_matrices.sum(dim=1, keepdim=True)
                
                try:
                    entropies = -torch.sum(prob_matrices * torch.log2(prob_matrices + self.epsilon), dim=1)
                    weights = entropies + self.epsilon
                    
                    weighted_probs = (weights.unsqueeze(1) * prob_matrices).sum(dim=0)
                    total_weight = weights.sum()
                except Exception as e:
                    print(f"Warning: Failed to compute entropies for class {c} in image {I}: {e}")
                    continue
                
                if total_weight > 0:
                    P_per_image[I][c] = (weighted_probs / total_weight).cpu().numpy()
                else:
                    print(f"Warning: All entropies were invalid for class {c} in image {I}, using uniform distribution")
                    P_per_image[I][c] = np.ones(matrix_size) / matrix_size
        
        return P_per_image
    
    def _compute_diversity(self, 
                              P1: Dict[int, np.ndarray],
                              P2: Dict[int, np.ndarray], 
                              C: List[int]) -> float:
        if not P1 or not P2:
            print("Warning: One or both probability distributions are empty, returning 0 diversity")
            return 0.0
            
        common_classes = set(P1.keys()) & set(P2.keys()) & set(C)
        if not common_classes:
            print("Warning: No common classes found between probability distributions, returning 0 diversity")
            return 0.0
        
        total_div = 0.0
        for c in common_classes:
            p1_tensor = torch.tensor(P1[c], dtype=torch.float32, device=self.device)
            p2_tensor = torch.tensor(P2[c], dtype=torch.float32, device=self.device)
            
            p1_tensor = torch.clamp(p1_tensor, self.epsilon, 1.0)
            p2_tensor = torch.clamp(p2_tensor, self.epsilon, 1.0)
            
            p1_tensor = p1_tensor / p1_tensor.sum()
            p2_tensor = p2_tensor / p2_tensor.sum()
            
            kl1 = torch.sum(p1_tensor * torch.log(p1_tensor / p2_tensor))
            kl2 = torch.sum(p2_tensor * torch.log(p2_tensor / p1_tensor))
            
            total_div += 0.5 * (kl1 + kl2).item()
        
        return total_div
    
    @staticmethod
    def _compute_entropy_static(probs: np.ndarray, epsilon: float) -> float:
        probs = np.asarray(probs, dtype=np.float64)
        
        if probs.ndim == 0:
            prob_val = float(probs)
            if prob_val <= 0 or prob_val >= 1:
                return 0.0
            p = prob_val
            q = 1.0 - p
            if p > 0 and q > 0:
                return float(-(p * np.log2(p + epsilon) + q * np.log2(q + epsilon)))
            else:
                return 0.0
        
        probs = np.clip(probs, epsilon, 1.0)
        
        entropy_sum = 0.0
        for p in probs:
            if p > 0:
                entropy_sum += p * np.log2(p + epsilon)
        
        return float(0.0 - entropy_sum)
    
    def _compute_entropy(self, probs: np.ndarray) -> float:
        probs = np.asarray(probs, dtype=np.float64)
        
        if probs.ndim == 0:
            prob_val = float(probs)
            if prob_val <= 0 or prob_val >= 1:
                print(f"Warning: Invalid probability value {prob_val}, returning 0 entropy")
                return 0.0
            p = prob_val
            q = 1.0 - p
            if p > 0 and q > 0:
                return float(-(p * np.log2(p + self.epsilon) + q * np.log2(q + self.epsilon)))
            else:
                print(f"Warning: Probability value {prob_val} leads to zero entropy, returning 0")
                return 0.0
        
        probs = np.clip(probs, self.epsilon, 1.0)
        
        entropy_sum = 0.0
        for p in probs:
            if p > 0:
                entropy_sum += p * np.log2(p + self.epsilon)
        
        return float(0.0 - entropy_sum)
    
    def _uniform_prob(self, num_classes: int) -> np.ndarray:
        return np.ones(num_classes) / num_classes
    
    def _kl_divergence(self, P: np.ndarray, Q: np.ndarray) -> float:
        P = np.clip(P, self.epsilon, 1.0)
        Q = np.clip(Q, self.epsilon, 1.0)
        return float(np.sum(P * np.log(P / Q)))
    
    def _select_batch(self, 
                          U: List[int], 
                          P_per_image: Dict[int, Dict[int, np.ndarray]], 
                          C: List[int], 
                          budget: int,
                          dist_matrix: np.ndarray) -> List[int]:
        if budget >= len(U):
            return U
        
        if budget <= 0:
            return []
        
        valid_images = [i for i in U if len(P_per_image.get(i, {})) > 0]
        
        if len(valid_images) == 0:
            print("Warning: No valid images with class predictions found")
            return U[:budget] if len(U) >= budget else U
        
        if budget >= len(valid_images):
            return valid_images
        
        S = []
        remaining = valid_images.copy()
        
        if len(remaining) > 0:
            first_image = np.random.choice(remaining)
            S.append(first_image)
            remaining.remove(first_image)
        
        with tqdm(total=budget-1, desc="Selecting diverse batch") as pbar:
            
            for i in range(1, budget):
                if len(remaining) == 0:
                    break
                    
                best_image = None
                max_min_dist = -np.inf
                
                for candidate in remaining:
                    min_dist = np.inf
                    
                    for selected in S:
                        dist = dist_matrix[candidate, selected]
                        min_dist = min(min_dist, dist)
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_image = candidate
                
                if best_image is not None:
                    S.append(best_image)
                    remaining.remove(best_image)
                else:
                    if remaining:
                        print("Warning: No best image found, selecting first remaining image")
                        S.append(remaining[0])
                        remaining.remove(remaining[0])
                
                pbar.update(1)
        
        while len(S) < budget and len(remaining) > 0:
            S.append(remaining.pop(0))
        
        if len(S) < budget:
            print(f"Warning: Could only select {len(S)} images out of requested {budget}")
        
        return S

    def get_strategy_name(self) -> str:
        return f"CDAL"