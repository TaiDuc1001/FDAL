import os
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
from loguru import logger
from typing import Optional
from ..base import BaseStrategy
import torch.nn.functional as nnf
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


class CoreSetStrategy(BaseStrategy):
    
    def __init__(self, 
                 model,
                 experiment_dir: Optional[str] = None,
                 round: Optional[int] = None,
                 use_embeddings: bool = True, 
                 distance_metric: str = "cosine", 
                 **kwargs):
        super().__init__(model, **kwargs)
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.round = round
        self.experiment_dir = experiment_dir
        self.use_embeddings = use_embeddings
        self.distance_metric = distance_metric
    
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
        if self.use_embeddings:
            results = self.model.inference(
                unlabeled_image_paths,
                return_boxes=True,
                return_embeddings=True,
                num_inference=num_inf,
                **local_kwargs
            )
            features = self._extract_embedding_features(results)
        else:
            results = self.model.inference(
                unlabeled_image_paths,
                return_boxes=True,
                num_inference=num_inf,
                **local_kwargs
            )
            features = self._extract_box_features(results)
        
        logger.info(f"Extracted features shape: {features.shape}")
        selected_indices = self._coreset_sampling(features, unlabeled_indices, n_samples)
        total_time = time.time() - start_time
        num_images = len(unlabeled_indices)
        time_per_image = total_time / num_images
        with open(timelog_file, 'a') as f:
            f.write(f"{self.round},{total_time:.4f},{num_images},{time_per_image:.6f}\n")
        print("Write time log to", timelog_file.absolute())
        selected_image_paths = [image_paths[i] for i in selected_indices]
        selected_image_names = [Path(p).name for p in selected_image_paths]
        with open(selectionlog_file, 'a') as f:
            f.write(','.join(selected_image_names) + '\n')
        print("Write to selection log file. ", selectionlog_file.absolute())
        
        self._save_predictions_for_selection(
            experiment_dir=self.experiment_dir,
            round_num=self.round,
            selected_image_paths=selected_image_paths,
            image_paths=image_paths,
            selected_indices=selected_indices,
            results=results,
            unlabeled_indices=unlabeled_indices,
        )
        self._save_selection_symlinks(
            experiment_dir=self.experiment_dir,
            round_num=self.round,
            selected_image_paths=selected_image_paths,
        )
            
        return selected_indices
    
    def _extract_embedding_features(self, results) -> np.ndarray:
        logger.info(f"Processing {len(results)} results for embedding extraction")
        features_raw = []
        num_defects = 0

        for i, result in enumerate(results):
            emb = getattr(result, 'embeddings', None)
            if emb is not None and len(emb) > 0:
                emb_np = np.asarray(emb)
                if emb_np.ndim == 1:
                    avg_embedding = emb_np
                else:
                    avg_embedding = np.mean(emb_np, axis=0)
                features_raw.append(np.asarray(avg_embedding))
            else:
                features_raw.append(None)
                num_defects += 1
        
        if num_defects > int(0.5 * len(results)):
            num_defects = 0
            for i, result in enumerate(results):
                emb = getattr(result, 'features', None)
                if emb is not None and len(emb) > 0:
                    emb_np = np.asarray(emb)
                    if emb_np.ndim == 1:
                        avg_embedding = emb_np
                    else:
                        avg_embedding = np.mean(emb_np, axis=0)
                    features_raw.append(np.asarray(avg_embedding))
                else:
                    features_raw.append(None)
                    num_defects += 1

        logger.info(f"Found {num_defects}/{len(results)} defects.")

        lengths = [f.shape[0] for f in features_raw if f is not None]
        if len(lengths) == 0:
            max_dim = 512
        else:
            max_dim = int(max(lengths))

        features_array = np.zeros((len(features_raw), max_dim), dtype=float)
        for i, f in enumerate(features_raw):
            if f is None:
                print('Missing feature vector')
                continue
            d = f.shape[0]
            if d == max_dim:
                features_array[i] = f
            elif d < max_dim:
                features_array[i, :d] = f
            else:
                print('Truncating feature vector')
                features_array[i] = f[:max_dim]

        return features_array
    
    def _extract_box_features(self, results) -> np.ndarray:
        features = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                n_detections = len(result.boxes)
                box_areas = []
                for box in result.boxes:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    box_areas.append(area)
                avg_box_area = np.mean(box_areas)
                
                centers_x = (result.boxes[:, 0] + result.boxes[:, 2]) / 2
                centers_y = (result.boxes[:, 1] + result.boxes[:, 3]) / 2
                center_spread_x = np.std(centers_x) if len(centers_x) > 1 else 0
                center_spread_y = np.std(centers_y) if len(centers_y) > 1 else 0
                
                feature_vec = [n_detections, avg_box_area, center_spread_x, center_spread_y]
            else:
                feature_vec = [0, 0, 0, 0]
                
            features.append(feature_vec)
        
        features_array = np.array(features)
        logger.info(f"Box features extracted: {features_array.shape}")
        return features_array
    
    def _coreset_sampling(self,
                         features: np.ndarray,
                         unlabeled_indices: np.ndarray,
                         n_samples: int) -> np.ndarray:
        if len(features) <= n_samples:
            return unlabeled_indices
            
        X = np.asarray(features, dtype=np.float32)

        if torch.cuda.is_available():
            device = torch.device('cuda')
            X_t = torch.from_numpy(X).to(device)
            n = X_t.shape[0]
            first_idx = int(torch.randint(0, n, (1,)).item())
            selected_idx = [first_idx]
            selected_feats = X_t[first_idx:first_idx+1]
            mask = torch.ones(n, dtype=torch.bool, device=device)
            mask[first_idx] = False

            for _ in tqdm(range(n_samples - 1), desc="Coreset Sampling"):
                rem = torch.nonzero(mask, as_tuple=False).squeeze(1)
                if rem.numel() == 0:
                    break
                candidates = X_t[rem]
                if self.distance_metric == "cosine":
                    cand_norm = nnf.normalize(candidates, dim=1)
                    sel_norm = nnf.normalize(selected_feats, dim=1)
                    dists = 1.0 - torch.matmul(cand_norm, sel_norm.T)
                else:
                    dists = torch.cdist(candidates, selected_feats, p=2)
                min_dists, _ = dists.min(dim=1)
                best_pos = int(min_dists.argmax().item())
                best_idx = int(rem[best_pos].item())
                selected_idx.append(best_idx)
                selected_feats = torch.cat([selected_feats, X_t[best_idx:best_idx+1]], dim=0)
                mask[best_idx] = False

            selected_indices = [int(unlabeled_indices[i]) for i in selected_idx]
            return np.array(selected_indices)

        selected_indices = []
        remaining_indices = list(range(len(X)))
        first_idx = np.random.randint(len(X))
        selected_indices.append(unlabeled_indices[first_idx])
        selected_features = [X[first_idx]]
        remaining_indices.remove(first_idx)
        for iteration in tqdm(range(n_samples - 1), desc="Coreset Sampling"):
            if not remaining_indices:
                break
            max_min_distance = -1
            best_idx = None
            for idx in remaining_indices:
                candidate_feature = X[idx]
                min_distance = float('inf')
                for selected_feature in selected_features:
                    if self.distance_metric == "cosine":
                        distance = cosine_distances(
                            candidate_feature.reshape(1, -1),
                            selected_feature.reshape(1, -1)
                        )[0, 0]
                    else:
                        distance = euclidean_distances(
                            candidate_feature.reshape(1, -1),
                            selected_feature.reshape(1, -1)
                        )[0, 0]
                    if distance < min_distance:
                        min_distance = distance
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx
            if best_idx is not None:
                selected_indices.append(unlabeled_indices[best_idx])
                selected_features.append(X[best_idx])
                remaining_indices.remove(best_idx)
        return np.array(selected_indices)
    
    def get_strategy_name(self) -> str:
        return "coreset"
