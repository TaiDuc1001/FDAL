import numpy as np
from typing import List
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

from .base import BaseStrategy


class DiversityStrategy(BaseStrategy):
    def __init__(self, model, diversity_method: str = "kmeans", **kwargs):
        super().__init__(model, **kwargs)
        self.diversity_method = diversity_method
        
    def query(self, 
              unlabeled_indices: np.ndarray,
              image_paths: List[str],
              n_samples: int,
              **kwargs) -> np.ndarray:
        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        unlabeled_image_paths = self._get_image_paths_for_indices(unlabeled_indices, image_paths)
        results = self.model.inference(
            unlabeled_image_paths,
            return_features=True,
            return_embeddings=True,
            num_inference=kwargs.get('num_inference', -1),
            **kwargs
        )
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
        
        features = np.array(features)
        if self.diversity_method == "kmeans":
            selected_indices = self._kmeans_sampling(features, unlabeled_indices, n_samples)
        elif self.diversity_method == "farthest":
            selected_indices = self._farthest_first_sampling(features, unlabeled_indices, n_samples)
        else:
            raise ValueError(f"Unknown diversity method: {self.diversity_method}")
            
        return selected_indices
    
    def _kmeans_sampling(self, 
                        features: np.ndarray,
                        unlabeled_indices: np.ndarray,
                        n_samples: int) -> np.ndarray:
        if len(features) <= n_samples:
            return unlabeled_indices
        kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        selected_indices = []
        for cluster_id in range(n_samples):
            cluster_mask = cluster_labels == cluster_id
            if np.any(cluster_mask):
                cluster_features = features[cluster_mask]
                cluster_indices = unlabeled_indices[cluster_mask]
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.sum((cluster_features - centroid) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                
                selected_indices.append(cluster_indices[closest_idx])
        
        return np.array(selected_indices)
    
    def _farthest_first_sampling(self,
                                features: np.ndarray,
                                unlabeled_indices: np.ndarray,
                                n_samples: int) -> np.ndarray:
        if len(features) <= n_samples:
            return unlabeled_indices
            
        selected_indices = []
        first_idx = np.random.randint(len(features))
        selected_indices.append(unlabeled_indices[first_idx])
        selected_features = [features[first_idx]]
        for _ in range(n_samples - 1):
            distances = []
            for i, feature in enumerate(features):
                if unlabeled_indices[i] in selected_indices:
                    distances.append(-1)
                else:
                    min_dist = float('inf')
                    for selected_feature in selected_features:
                        dist = np.sum((feature - selected_feature) ** 2)
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)
            
            farthest_idx = np.argmax(distances)
            selected_indices.append(unlabeled_indices[farthest_idx])
            selected_features.append(features[farthest_idx])
        
        return np.array(selected_indices)
    
    def get_strategy_name(self) -> str:
        return f"diversity_{self.diversity_method}"
