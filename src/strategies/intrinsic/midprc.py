import os
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch.nn.functional as F

from ..base import BaseStrategy


class DPRLCalculator:
    def __init__(self, epsilon: float = 5.0, lambda_weight: float = 1.0, noise_std: float = 0.1):
        self.epsilon = int(epsilon)
        self.lambda_weight = lambda_weight
        self.noise_std = noise_std
        self._reg_weight_vector: Optional[torch.Tensor] = None

    def compute(
        self, 
        model, 
        box_features: torch.Tensor,
        scale_idx: int = -1
    ) -> torch.Tensor:
        if box_features is None or len(box_features) == 0:
            return torch.tensor([0.0])
        
        device = box_features.device
        losses = []
        
        for _ in range(self.epsilon):
            noise = torch.randn_like(box_features) * self.noise_std
            perturbed = box_features + noise
            
            with torch.no_grad():
                _, orig_bbox = model.forward_head_on_features(box_features, scale_idx)
                _, pert_bbox = model.forward_head_on_features(perturbed, scale_idx)
            
            if orig_bbox is None or pert_bbox is None:
                continue
            
            odlv = orig_bbox - pert_bbox
            odlv_norm = torch.norm(odlv, dim=-1, p=2) ** 2
            
            cos_alpha = self._compute_cos_alpha(model, box_features)
            
            dprl = self.lambda_weight * cos_alpha * odlv_norm
            losses.append(dprl)
        
        if len(losses) == 0:
            return torch.zeros(box_features.shape[0], device=device)
        
        return torch.stack(losses).mean(dim=0)

    def _compute_cos_alpha(self, model, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        if self._reg_weight_vector is None:
            self._reg_weight_vector = model.get_regression_weight_vector()
        
        if self._reg_weight_vector is None:
            return torch.ones(features.shape[0], device=features.device)
        
        weight = self._reg_weight_vector.to(features.device)
        
        if weight.shape[0] != features.shape[1]:
            min_dim = min(weight.shape[0], features.shape[1])
            weight = weight[:min_dim]
            features = features[:, :min_dim]
        
        feat_norm = F.normalize(features, dim=1)
        weight_norm = F.normalize(weight.unsqueeze(0), dim=1)
        
        cos_alpha = torch.abs(torch.mm(feat_norm, weight_norm.t())).squeeze(-1)
        return cos_alpha.clamp(0.01, 1.0)


class RegressionEntropyCalculator:
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def compute(self, dprl_losses: torch.Tensor) -> torch.Tensor:
        if dprl_losses is None or len(dprl_losses) == 0:
            return torch.tensor(0.0)
        
        if dprl_losses.dim() == 0:
            return dprl_losses
        
        similarities = torch.exp(-dprl_losses)
        total = similarities.sum() + self.epsilon
        probs = similarities / total
        
        entropy = -torch.sum(probs * torch.log2(probs + self.epsilon))
        l_reg = dprl_losses.mean()
        
        i_reg = l_reg / (entropy + self.epsilon)
        return i_reg


class PrototypeManager:
    def __init__(self, num_classes: int, feature_dim: int = 256):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.prototypes: Optional[torch.Tensor] = None
        self.bg_prototype: Optional[torch.Tensor] = None
        self.counts: Optional[torch.Tensor] = None

    def initialize(self, device: torch.device):
        self.prototypes = torch.zeros(self.num_classes, self.feature_dim, device=device)
        self.bg_prototype = torch.zeros(self.feature_dim, device=device)
        self.counts = torch.zeros(self.num_classes, device=device)

    def update(self, features: torch.Tensor, labels: torch.Tensor):
        if self.prototypes is None:
            self.initialize(features.device)
        
        self.prototypes = self.prototypes.to(features.device)
        self.counts = self.counts.to(features.device)
        
        feat_dim = features.shape[1]
        if feat_dim != self.feature_dim:
            if feat_dim > self.feature_dim:
                features = features[:, :self.feature_dim]
            else:
                padded = torch.zeros(features.shape[0], self.feature_dim, device=features.device)
                padded[:, :feat_dim] = features
                features = padded
        
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_features = features[mask].mean(dim=0)
                self.prototypes[c] = (
                    self.prototypes[c] * self.counts[c] + class_features
                ) / (self.counts[c] + 1)
                self.counts[c] += 1

    def update_background(self, features: torch.Tensor):
        if self.bg_prototype is None:
            self.initialize(features.device)
        
        self.bg_prototype = self.bg_prototype.to(features.device)
        
        feat_dim = features.shape[1]
        if feat_dim != self.feature_dim:
            if feat_dim > self.feature_dim:
                features = features[:, :self.feature_dim]
            else:
                padded = torch.zeros(features.shape[0], self.feature_dim, device=features.device)
                padded[:, :feat_dim] = features
                features = padded
        
        self.bg_prototype = features.mean(dim=0)

    def get_distance_entropy(self, features: torch.Tensor) -> torch.Tensor:
        if self.prototypes is None:
            return torch.ones(features.shape[0], device=features.device)
        
        self.prototypes = self.prototypes.to(features.device)
        
        feat_dim = features.shape[1]
        if feat_dim != self.feature_dim:
            if feat_dim > self.feature_dim:
                features = features[:, :self.feature_dim]
            else:
                padded = torch.zeros(features.shape[0], self.feature_dim, device=features.device)
                padded[:, :feat_dim] = features
                features = padded
        
        distances = torch.cdist(features, self.prototypes)
        probs = F.softmax(-distances, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy

    def get_offset_weight(self, features: torch.Tensor) -> torch.Tensor:
        if self.bg_prototype is None:
            return torch.ones(features.shape[0], device=features.device)
        
        self.bg_prototype = self.bg_prototype.to(features.device)
        
        feat_dim = features.shape[1]
        if feat_dim != self.feature_dim:
            if feat_dim > self.feature_dim:
                features = features[:, :self.feature_dim]
            else:
                padded = torch.zeros(features.shape[0], self.feature_dim, device=features.device)
                padded[:, :feat_dim] = features
                features = padded
        
        distances = torch.norm(features - self.bg_prototype.unsqueeze(0), dim=1, p=2)
        return distances


class MultiInstanceAggregator:
    def __init__(self, prototype_manager: PrototypeManager):
        self.prototype_manager = prototype_manager

    def compute_image_score(
        self, 
        instance_i_reg: torch.Tensor,
        instance_i_cls: torch.Tensor,
        instance_features: torch.Tensor,
        instance_labels: torch.Tensor
    ) -> torch.Tensor:
        if len(instance_i_reg) == 0:
            return torch.tensor(0.0)
        
        w_i = self.prototype_manager.get_distance_entropy(instance_features)
        kappa_i = self.prototype_manager.get_offset_weight(instance_features)
        
        w_i = w_i.to(instance_i_reg.device)
        kappa_i = kappa_i.to(instance_i_reg.device)
        instance_i_cls = instance_i_cls.to(instance_i_reg.device)
        
        im_score = (w_i * (instance_i_reg + kappa_i * instance_i_cls)).sum()
        return im_score


class DiversitySampler:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def filter_redundant(
        self, 
        image_scores: torch.Tensor, 
        image_features: torch.Tensor, 
        budget: int
    ) -> List[int]:
        if len(image_scores) == 0:
            return []
        
        sorted_indices = torch.argsort(image_scores, descending=True)
        selected = []
        selected_features = []
        
        for idx in sorted_indices:
            if len(selected) >= budget:
                break
            
            feat = image_features[idx]
            
            if len(selected_features) == 0:
                selected.append(idx.item())
                selected_features.append(feat)
                continue
            
            feat_stack = torch.stack(selected_features)
            similarities = F.cosine_similarity(feat.unsqueeze(0), feat_stack)
            
            if similarities.max() < self.similarity_threshold:
                selected.append(idx.item())
                selected_features.append(feat)
        
        return selected


class MIDPRCStrategy(BaseStrategy):
    
    def __init__(
        self,
        model,
        round: Optional[int] = None,
        experiment_dir: Optional[str] = None,
        epsilon: float = 5.0,
        lambda_weight: float = 1.0,
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        similarity_threshold: float = 0.85,
        feature_dim: int = 256,
        noise_std: float = 0.1,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        for key, val in kwargs.items():
            setattr(self, key, val)
        
        self.round = round
        self.experiment_dir = experiment_dir
        self.epsilon = float(epsilon)
        self.lambda_weight = float(lambda_weight)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.feature_dim = int(feature_dim)
        self.noise_std = float(noise_std)
        
        self.num_classes = len(self.model.model.names) if hasattr(self.model, 'model') and hasattr(self.model.model, 'names') else 80
        
        self.dprl_calculator = DPRLCalculator(
            epsilon=self.epsilon,
            lambda_weight=self.lambda_weight,
            noise_std=self.noise_std
        )
        self.entropy_calculator = RegressionEntropyCalculator()
        self.prototype_manager = PrototypeManager(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim
        )
        self.aggregator = MultiInstanceAggregator(self.prototype_manager)
        self.diversity_sampler = DiversitySampler(
            similarity_threshold=self.similarity_threshold
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def query(
        self,
        unlabeled_indices: np.ndarray,
        image_paths: List[str],
        n_samples: int,
        **kwargs
    ) -> np.ndarray:
        timelog_file = Path(self.experiment_dir) / os.environ.get("TIME_LOGFILE", "time_log.csv")
        if not timelog_file.exists():
            with open(timelog_file, 'w') as f:
                f.write("Round,TotalTime,NumImages,TimePerImage\n")

        selectionlog_file = Path(self.experiment_dir) / os.environ.get("SELECTION_LOGFILE", "selection_log.txt")
        if not selectionlog_file.exists():
            selectionlog_file.touch()

        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        
        unlabeled_image_paths = self._get_image_paths_for_indices(unlabeled_indices, image_paths)
        
        local_kwargs = dict(kwargs)
        num_inf = local_kwargs.pop('num_inference', -1)
        unlabeled_image_paths = unlabeled_image_paths[:num_inf] if num_inf > 0 else unlabeled_image_paths
        conf = local_kwargs.get('conf', self.conf_threshold)

        print(f"MI-DPRC: Running inference on {len(unlabeled_image_paths)} unlabeled images...")
        start_time = time.time()
        
        results = self.model.inference(
            unlabeled_image_paths,
            return_boxes=True,
            return_classes=True,
            return_probs=True,
            return_features=True,
            num_inference=num_inf,
            conf=conf,
            **local_kwargs
        )
        
        if len(results) == 0:
            print("Warning: No inference results available")
            return np.random.choice(unlabeled_indices, size=min(n_samples, len(unlabeled_indices)), replace=False)
        
        print("MI-DPRC: Computing image scores...")
        image_scores = []
        image_features = []
        
        for i, result in enumerate(tqdm(results, desc="Scoring images")):
            boxes = result.boxes
            classes = result.classes
            
            if boxes is None or len(boxes) == 0:
                image_scores.append(torch.tensor(0.0))
                image_features.append(torch.zeros(self.feature_dim))
                continue
            
            box_features = self.model.get_box_features(
                boxes, 
                image_size=(640, 640),
                feature_layer="model.15"
            )
            
            if len(box_features) == 0:
                image_scores.append(torch.tensor(0.0))
                image_features.append(torch.zeros(self.feature_dim))
                continue
            
            box_features_t = box_features.to(self.device)
            classes_t = torch.tensor(classes, dtype=torch.long, device=self.device)
            
            dprl_losses = self.dprl_calculator.compute(
                self.model, box_features_t, scale_idx=-1
            )
            
            i_reg = self.entropy_calculator.compute(dprl_losses)
            
            cls_output, _ = self.model.forward_head_on_features(box_features_t, scale_idx=-1)
            if cls_output is not None:
                cls_probs = F.softmax(cls_output, dim=1)
                i_cls = -torch.sum(cls_probs * torch.log(cls_probs + 1e-8), dim=1)
            else:
                i_cls = torch.zeros(len(box_features_t), device=self.device)
            
            self.prototype_manager.update(box_features_t, classes_t)
            
            im_score = self.aggregator.compute_image_score(
                i_reg.expand(len(box_features_t)),
                i_cls,
                box_features_t,
                classes_t
            )
            
            if i < 3:
                print(f"\n[DEBUG] Image {i}:")
                print(f"  boxes.shape: {boxes.shape}")
                print(f"  box_features.shape: {box_features_t.shape}")
                print(f"  classes: {classes_t.cpu().numpy()}")
                print(f"  dprl_losses: {dprl_losses[:5].cpu().numpy() if len(dprl_losses) > 5 else dprl_losses.cpu().numpy()}")
                print(f"  i_reg: {i_reg.item():.6f}")
                print(f"  i_cls (first 5): {i_cls[:5].cpu().numpy() if len(i_cls) > 5 else i_cls.cpu().numpy()}")
                print(f"  im_score: {im_score.item():.6f}")
            
            image_scores.append(im_score.cpu())
            image_features.append(box_features_t.mean(dim=0).cpu())
        
        image_scores = torch.stack(image_scores)
        image_features = torch.stack(image_features)
        
        print("MI-DPRC: Applying diversity sampling...")
        selected_local_indices = self.diversity_sampler.filter_redundant(
            image_scores, image_features, n_samples
        )
        
        selected_global_indices = [unlabeled_indices[i] for i in selected_local_indices]
        selected_indices = np.array(selected_global_indices)
        
        total_time = time.time() - start_time
        num_images = len(unlabeled_image_paths)
        time_per_image = total_time / num_images if num_images > 0 else 0
        
        with open(timelog_file, 'a') as f:
            f.write(f"{self.round},{total_time:.4f},{num_images},{time_per_image:.6f}\n")
        print("Write time log to", timelog_file.absolute())
        
        selected_image_paths = [image_paths[i] for i in selected_indices]
        selected_image_names = [Path(p).name for p in selected_image_paths]
        with open(selectionlog_file, 'a') as f:
            f.write(','.join(selected_image_names) + '\n')
        print("Write to selection log file.", selectionlog_file.absolute())
        
        self._save_predictions_for_selection(
            experiment_dir=self.experiment_dir,
            round_num=self.round,
            selected_image_paths=selected_image_paths,
            image_paths=image_paths,
            selected_indices=selected_indices,
            results=results,
            unlabeled_indices=unlabeled_indices,
        )
        
        print(f"MI-DPRC Strategy: Selected {len(selected_indices)} samples from {len(unlabeled_indices)} unlabeled images")
        
        return selected_indices

    def get_strategy_name(self) -> str:
        return f"midprc_eps{self.epsilon}_lambda{self.lambda_weight}"
