import os
import time
import numpy as np
from pathlib import Path
from loguru import logger
from ..base import BaseStrategy
from typing import List, Optional, Tuple
from scipy.spatial.distance import cdist


class CCMSStrategy(BaseStrategy):
    
    def __init__(self, 
                 model,
                 round: Optional[int] = None,
                 experiment_dir: Optional[str] = None,
                 n_clusters: Optional[int] = None,
                 num_clusters: Optional[int] = None,
                 distance_metric: str = 'euclidean',
                 n_iterations: int = 10,
                 k_medoids_iters: int = 10,
                 feature_layers: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(model, **kwargs)
        if isinstance(n_clusters, str) and n_clusters.lower() == 'none':
            n_clusters = None
        if isinstance(num_clusters, str) and num_clusters.lower() == 'none':
            num_clusters = None
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.round = round
        self.experiment_dir = experiment_dir
        self.n_clusters_param = n_clusters or num_clusters
        self.distance_metric = distance_metric
        self.n_iterations = k_medoids_iters or n_iterations
        self.feature_layers = feature_layers
        self.medoid_indices_ = None
        self.cluster_assignments_ = None
        
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

        inference_kwargs = {
            'return_boxes': True,
            'return_features': True,
            'num_inference': num_inf,
            **local_kwargs
        }
        if self.feature_layers:
            inference_kwargs['feature_layers'] = self.feature_layers
            logger.info(f"Using custom feature layers: {self.feature_layers}")
        
        start_time = time.time()
        results = self.model.inference(
            unlabeled_image_paths,
            **inference_kwargs
        )
        features = self._extract_features(results)
        
        if len(unlabeled_indices) != len(features):
            processed_indices = unlabeled_indices[:len(features)]
            logger.warning(f"Mismatch between unlabeled_indices ({len(unlabeled_indices)}) and features ({len(features)}). Using first {len(features)} indices.")
        else:
            processed_indices = unlabeled_indices

        selected_indices = self._ccms_sampling(features, processed_indices, n_samples)
        selected_image_paths = self._get_image_paths_for_indices(selected_indices, image_paths)
        selected_image_names = [Path(p).name for p in selected_image_paths]
        with open(selectionlog_file, 'a') as f:
            f.write(','.join(selected_image_names) + '\n')
        print("Write to selection log file. ", selectionlog_file.absolute())
        
        total_time = time.time() - start_time
        time_per_image = total_time / len(unlabeled_image_paths)
        with open(timelog_file, 'a') as f:
            f.write(f"{self.round},{total_time:.4f},{len(unlabeled_image_paths)},{time_per_image:.4f}\n")
        print("Write time log to", timelog_file.absolute())
        return selected_indices
    
    def _extract_features(self, results) -> np.ndarray:
        features = []
        feature_dim = None
        
        for result in results:
            if result.features is not None and len(result.features) > 0:
                feature_vec = result.features
                if feature_dim is None:
                    feature_dim = len(feature_vec)
                features.append(feature_vec)
            else:
                if feature_dim is None:
                    feature_dim = 512
                dummy_feature = np.zeros(feature_dim)
                features.append(dummy_feature)
                logger.warning(f"No features found for a result. Using dummy feature vector of size {feature_dim}.")
        
        features_array = np.array(features)
        logger.info(f"Feature vectors extracted: {features_array.shape}")
        
        if len(features_array) > 0:
            feature_stats = {
                'mean': np.mean(features_array),
                'std': np.std(features_array),
                'min': np.min(features_array),
                'max': np.max(features_array)
            }
            logger.info(f"Feature statistics: mean={feature_stats['mean']:.4f}, std={feature_stats['std']:.4f}, min={feature_stats['min']:.4f}, max={feature_stats['max']:.4f}")
            
            pairwise_similarities = np.corrcoef(features_array)
            avg_similarity = np.mean(pairwise_similarities[np.triu_indices_from(pairwise_similarities, k=1)])
            logger.info(f"Average pairwise correlation: {avg_similarity:.4f}")
        
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
                logger.warning("No bounding boxes found for a result. Using default box features [0, 0, 0, 0].")
                
            features.append(feature_vec)
        
        features_array = np.array(features)
        logger.info(f"Box features extracted: {features_array.shape}")
        return features_array
    
    @staticmethod
    def _k_centroid_greedy(distance_matrix: np.ndarray, K: int) -> List[int]:
        N = distance_matrix.shape[0]
        centroids = []
        K = min(K, N)
        if K <= 0:
            logger.warning(f"Invalid K value ({K}). Returning empty list.")
            return []

        c = np.random.randint(0, N)
        centroids.append(c)

        for _ in range(1, K):
            min_dist_to_centroids = distance_matrix[:, centroids].min(axis=1)
            min_dist_to_centroids[centroids] = 0  
            max_dist_idx = np.argmax(min_dist_to_centroids)
            centroids.append(max_dist_idx)

        return centroids

    @staticmethod
    def _k_medoids_refine(distance_matrix: np.ndarray, K: int, n_iter: int = 10) -> Tuple[List[int], np.ndarray]:
        N = distance_matrix.shape[0]
        if N == 0:
            logger.warning("No data points in distance matrix. Returning empty medoids and assignments.")
            return [], np.array([])
        K = min(K, N)
        if K <= 0:
            logger.warning(f"Invalid K value ({K}). Returning empty medoids and assignments.")
            return [], np.array([])

        medoids = np.array(CCMSStrategy._k_centroid_greedy(distance_matrix, K))
        data_indices = np.arange(N)
        cluster_assign = np.zeros(N, dtype=int)
        
        best_cost = float('inf')
        best_medoids = medoids.copy()
        best_assignments = cluster_assign.copy()

        for iteration in range(n_iter):
            centroid_dis = distance_matrix[:, medoids]
            new_cluster_assign = np.argmin(centroid_dis, axis=1)

            total_cost = np.sum([distance_matrix[i, medoids[new_cluster_assign[i]]] for i in range(N)])
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_medoids = medoids.copy()
                best_assignments = new_cluster_assign.copy()

            if np.array_equal(new_cluster_assign, cluster_assign):
                logger.info(f"K-medoids converged after {iteration + 1} iterations")
                break
            cluster_assign = new_cluster_assign

            new_medoids = []
            for i in range(K):
                cluster_i_indices = data_indices[cluster_assign == i]
                if len(cluster_i_indices) == 0:
                    if i < len(medoids):
                        new_medoids.append(medoids[i])
                        logger.warning(f"Cluster {i} has no points. Reusing previous medoid.")
                    else:
                        new_medoids.append(np.random.choice(data_indices))
                        logger.warning(f"Cluster {i} has no points. Choosing random medoid.")
                    continue

                dist_mat_i = distance_matrix[cluster_i_indices][:, cluster_i_indices]
                within_cluster_dist_sum = dist_mat_i.sum(axis=1)
                min_sum_idx_in_cluster = np.argmin(within_cluster_dist_sum)
                new_medoid_i = cluster_i_indices[min_sum_idx_in_cluster]
                new_medoids.append(new_medoid_i)

            medoids = np.array(new_medoids)
            medoids = np.unique(medoids)
            K = len(medoids)

        final_centroid_dis = distance_matrix[:, best_medoids]
        final_cluster_assign = np.argmin(final_centroid_dis, axis=1)

        return best_medoids.tolist(), final_cluster_assign

    def _ccms_sampling(self, 
                      features: np.ndarray,
                      unlabeled_indices: np.ndarray,
                      n_samples: int) -> np.ndarray:
        num_available = features.shape[0]
        if num_available <= n_samples:
            self.medoid_indices_ = np.arange(num_available)
            self.cluster_assignments_ = np.arange(num_available)
            return unlabeled_indices[:num_available]

        if self.n_clusters_param is not None:
            k = min(self.n_clusters_param, num_available, n_samples)
        else:
            k = min(max(n_samples // 2, 10), num_available)
        
        k = max(1, k)
        logger.info(f"Using k={k} clusters for {num_available} samples to select {n_samples} diverse samples.")

        logger.info(f"Normalizing features and calculating pairwise distances ({self.distance_metric}).")
        try:
            features_float = features.astype(np.float64)
            
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_normalized = scaler.fit_transform(features_float)
                logger.info("Using sklearn StandardScaler for feature normalization.")
            except ImportError:
                logger.warning("sklearn not available, using manual standardization.")
                features_mean = np.mean(features_float, axis=0)
                features_std = np.std(features_float, axis=0)
                features_std[features_std == 0] = 1  
                features_normalized = (features_float - features_mean) / features_std
            
            distance_matrix = cdist(features_normalized, features_normalized, metric=self.distance_metric) # type: ignore
        except Exception as e:
            logger.error(f"Distance matrix calculation failed: {e}. Returning first 'n_samples' indices.")
            self.medoid_indices_, self.cluster_assignments_ = None, None
            return unlabeled_indices[:n_samples]

        logger.info(f"Performing K-Medoids with k={k} (iterations={self.n_iterations}).")
        try:
            medoid_indices, cluster_assignments = self._k_medoids_refine(distance_matrix, k, self.n_iterations)
            self.medoid_indices_ = np.array(medoid_indices)
            self.cluster_assignments_ = cluster_assignments
            logger.info(f"K-Medoids complete. Found {len(self.medoid_indices_)} medoids.")
            
            if len(self.medoid_indices_) > 0:
                logger.info(f"First 10 medoid indices: {self.medoid_indices_[:10]}")
                distances_between_medoids = distance_matrix[np.ix_(self.medoid_indices_, self.medoid_indices_)]
                avg_medoid_distance = np.mean(distances_between_medoids[np.triu_indices_from(distances_between_medoids, k=1)])
                logger.info(f"Average distance between medoids: {avg_medoid_distance:.4f}")
        except Exception as e:
            logger.error(f"K-Medoids failed: {e}. Returning first 'n_samples' indices.")
            self.medoid_indices_, self.cluster_assignments_ = None, None
            return unlabeled_indices[:n_samples]

        if len(self.medoid_indices_) >= n_samples:
            selected_indices = self.medoid_indices_[:n_samples]
        else:
            remaining_needed = n_samples - len(self.medoid_indices_)
            non_medoid_indices = np.setdiff1d(np.arange(num_available), self.medoid_indices_)
            
            if len(non_medoid_indices) >= remaining_needed:
                distances_to_medoids = distance_matrix[non_medoid_indices][:, self.medoid_indices_]
                min_distances_to_medoids = distances_to_medoids.min(axis=1)
                farthest_indices = np.argsort(-min_distances_to_medoids)[:remaining_needed]
                additional_indices = non_medoid_indices[farthest_indices]
                selected_indices = np.concatenate([self.medoid_indices_, additional_indices])
            else:
                selected_indices = np.concatenate([self.medoid_indices_, non_medoid_indices])
        
        selected_global_indices = unlabeled_indices[selected_indices]
        
        logger.info(f"Final selection: {len(selected_global_indices)} samples")
        logger.info(f"First 10 selected indices: {selected_indices[:10]}")
        if len(selected_indices) > 1:
            selected_distances = distance_matrix[np.ix_(selected_indices, selected_indices)]
            avg_selected_distance = np.mean(selected_distances[np.triu_indices_from(selected_distances, k=1)])
            logger.info(f"Average distance between selected samples: {avg_selected_distance:.4f}")
        
        return selected_global_indices
    
    def get_strategy_name(self) -> str:
        return "ccms"
