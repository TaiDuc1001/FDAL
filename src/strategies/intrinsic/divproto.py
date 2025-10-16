import os
import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ..base import BaseStrategy
from typing import List, Dict, Tuple, Set, Optional
from sklearn.metrics.pairwise import cosine_similarity


class DivProtoStrategy(BaseStrategy):
    
    def __init__(self, 
                 model,
                 round: Optional[int] = None,
                 experiment_dir: Optional[str] = None,
                 tenms: float = 0.5,  
                 tintra: float = 0.7,  
                 tinter: float = 0.3,  
                 alpha: float = 0.5,   
                 beta: float = 0.75,   
                 conf_threshold: float = 0.25,
                 feature_layers: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(model, **kwargs)
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.round = round
        self.experiment_dir = experiment_dir
        self.tenms = float(tenms)
        self.tintra = float(tintra)
        self.tinter = float(tinter)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.num_classes = len(self.model.model.names) # type: ignore
        self.conf_threshold = float(conf_threshold)
        self.feature_layers = feature_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def query(self, 
              unlabeled_indices: np.ndarray,
              image_paths: List[str],
              n_samples: int,
              labeled_indices: Optional[np.ndarray] = None,
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
        conf = local_kwargs.get('conf', self.conf_threshold)
        unlabeled_image_paths = unlabeled_image_paths[:num_inf] if num_inf > 0 else unlabeled_image_paths
        
        inference_kwargs = {
            'return_boxes': True,
            'return_classes': True,
            'return_probs': True,
            'return_features': True,
            'num_inference': num_inf,
            'conf': conf,
            **local_kwargs
        }
        if self.feature_layers:
            inference_kwargs['feature_layers'] = self.feature_layers
            
        print("Running inference on unlabeled images...")
        start_time = time.time()
        results = self.model.inference(unlabeled_image_paths, **inference_kwargs)
        
        if len(results) == 0:
            print("Warning: No inference results available")
            return np.random.choice(unlabeled_indices, size=min(n_samples, len(unlabeled_indices)), replace=False)
        
        
        print("Computing entropies and prototypes...")
        entropies = {}
        prototypes = {}
        
        for i, result in enumerate(results):
            entropy, proto_dict = self._enms_and_prototypes(result)
            entropies[i] = entropy
            prototypes[i] = proto_dict
            
        
        class_frequencies = self._compute_class_frequencies(results) if results else {}
        print("Applying DivProto selection...")
        selected_local_indices = self._div_proto_selection(
            list(range(len(results))), 
            entropies, 
            prototypes, 
            results,
            n_samples, 
            class_frequencies
        )
        
        selected_global_indices = [unlabeled_indices[i] for i in selected_local_indices]
        selected_indices = np.array(selected_global_indices)
        end_time = time.time()
        total_time = end_time - start_time
        num_images = len(unlabeled_image_paths)
        time_per_image = total_time / num_images if num_images > 0 else 0
        with open(timelog_file, 'a') as f:
            f.write(f"{self.round},{total_time:.4f},{num_images},{time_per_image:.6f}\n")
        print("Write time log to", timelog_file.absolute())

        selected_image_paths = [image_paths[i] for i in selected_indices]
        selected_image_names = [Path(p).name for p in selected_image_paths]
        with open(selectionlog_file, 'a') as f:
            f.write(','.join(selected_image_names) + '\n')
        print("Write to selection log file. ", selectionlog_file.absolute())
        
        print(f"DivProto Strategy: Selected {len(selected_indices)} samples from {len(unlabeled_indices)} unlabeled images")
        return selected_indices
    
    def _enms_and_prototypes(self, result) -> Tuple[float, Dict[int, Optional[np.ndarray]]]:
        if (result.probs is None or result.classes is None or result.features is None or
            len(result.probs) == 0 or len(result.classes) == 0):
            return 0.0, {}
            
        probs = np.asarray(result.probs)
        classes = np.asarray(result.classes, dtype=int)
        features = np.asarray(result.features)
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
            features = np.tile(features, (len(probs), 1))
        elif features.ndim == 2 and features.shape[0] != len(probs):
            if features.shape[0] == 1:
                features = np.tile(features, (len(probs), 1))
            else:
                features = features[:1]
                features = np.tile(features, (len(probs), 1))
        
        t = len(probs)  
        instance_entropies = np.random.uniform(0, 1, size=t)
        image_entropy = float(np.sum(instance_entropies))
        proto_dict = {}
        for c in range(1, self.num_classes + 1):
            class_mask = (classes == c)
            if not np.any(class_mask):
                proto_dict[c] = None
                continue
            class_features = features[class_mask]
            proto_dict[c] = np.mean(class_features, axis=0) if len(class_features) > 0 else None
        return image_entropy, proto_dict
    
    def _compute_class_frequencies(self, results: List) -> Dict[int, int]:
        class_frequencies = {}
        for result in results:
            if result.classes is not None:
                for cls in result.classes:
                    class_frequencies[cls] = class_frequencies.get(cls, 0) + 1
        return class_frequencies
    
    def _div_proto_selection(self, 
                           unlabeled_indices: List[int],
                           entropies: Dict[int, float],
                           prototypes: Dict[int, Dict[int, Optional[np.ndarray]]],
                           results: List,
                           budget: int,
                           class_frequencies: Dict[int, int]) -> List[int]:
        sorted_indices = sorted(unlabeled_indices, key=lambda i: entropies.get(i, 0), reverse=True)
        selected_images = []
        selected_prototypes = []  
        
        for img_idx in sorted_indices:
            if len(selected_images) >= budget:
                break
            mg_score = self._compute_mg_score(prototypes[img_idx], selected_prototypes)
            if mg_score < self.tintra:
                selected_images.append(img_idx)
                selected_prototypes.append(prototypes[img_idx])
        
        for img_idx in sorted_indices:
            if len(selected_images) >= budget:
                break
            if img_idx not in selected_images:
                selected_images.append(img_idx)
        
        return selected_images[:budget]
    
    def _compute_mg_score(self, 
                         img_prototypes: Dict[int, Optional[np.ndarray]], 
                         selected_prototypes: List[Dict[int, Optional[np.ndarray]]]) -> float:
        if not selected_prototypes:
            return 0.0  
        
        mg_score = float('-inf')
        
        for c in range(1, self.num_classes + 1):
            img_proto = img_prototypes.get(c)
            if img_proto is None:
                continue
                
            min_sim_c = 0.0
            for sel_protos in selected_prototypes:
                sel_proto = sel_protos.get(c)
                if sel_proto is not None:
                    sim = cosine_similarity(
                        img_proto.reshape(1, -1),
                        sel_proto.reshape(1, -1)
                    )[0, 0]
                    min_sim_c = max(min_sim_c, sim)
            
            mg_score = max(mg_score, min_sim_c)
        return mg_score if mg_score != float('-inf') else 0.0
    
    def _compute_mp_score(self, result, minority_classes: Set[int]) -> float:
        if not minority_classes:
            return 1.0  
            
        if (result.classes is None or result.probs is None or 
            len(result.classes) == 0 or len(result.probs) == 0):
            return 0.0
            
        classes = np.asarray(result.classes, dtype=int)
        probs = np.asarray(result.probs)
        
        mp_score = 0.0
        for c in minority_classes:
            p_ic = self._compute_class_presence_with_probs(classes, probs, c)
            mp_score = max(mp_score, p_ic)
            
        return mp_score
    
    def _compute_class_presence(self, result, class_id: int) -> float:
        if (result.classes is None or result.probs is None or 
            len(result.classes) == 0 or len(result.probs) == 0):
            return 0.0
            
        classes = np.asarray(result.classes, dtype=int)
        probs = np.asarray(result.probs)
        
        return self._compute_class_presence_with_probs(classes, probs, class_id)
    
    def _compute_class_presence_with_probs(self, classes: np.ndarray, probs: np.ndarray, class_id: int) -> float:
        class_scores = []
        for k in range(len(classes)):
            if classes[k] == class_id:
                class_scores.append(probs[k])
            else:
                class_scores.append(0.0)
        
        return max(class_scores) if class_scores else 0.0
    
    def get_strategy_name(self) -> str:
        return f"divproto"
