import os
import time
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Optional, Dict
from src.strategies.base import BaseStrategy

def restore_classwise_quality(model, last_ckpt_path=None):
    if last_ckpt_path is None:
        raise ValueError("last_ckpt_path is required to restore classwise_quality")
    else:
        last_ckpt_path = Path(last_ckpt_path)
    
    classwise_quality_path = last_ckpt_path.parent / "classwise_quality.npy"
    
    if classwise_quality_path.exists():
        classwise_quality = np.load(classwise_quality_path)
        model_to_set = model.module if hasattr(model, "module") else model
        setattr(model_to_set, "classwise_quality", torch.tensor(classwise_quality, dtype=torch.float32))
        logger.info(f"Loaded classwise_quality from numpy file: {classwise_quality_path}")
        return model
    
    if not last_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {last_ckpt_path}")

    ckpt = torch.load(last_ckpt_path, map_location="cpu")
    model_to_set = model.module if hasattr(model, "module") else model
    if "classwise_quality" in ckpt:
        setattr(model_to_set, "classwise_quality", ckpt["classwise_quality"].clone())
        logger.info(f"Loaded classwise_quality from checkpoint: {last_ckpt_path}")
        return model
    else:
        raise ValueError(f"'classwise_quality' not found in checkpoint: {last_ckpt_path} or numpy file: {classwise_quality_path}")

class DCUSStrategy(BaseStrategy):
    def __init__(self,
                model,
                alpha: float = 0.2,
                beta: float = 0.2,
                momentum: float = 0.9,
                xi: float = 0.5,
                experiment_dir: Optional[str] = None,
                round: Optional[int] = None,
                **kwargs):
        super().__init__(model, **kwargs)
        self.momentum = momentum
        self.xi = xi
        self.alpha = alpha
        self.beta = beta
        self.experiment_dir = experiment_dir
        self.round = round - 1 # type: ignore

    def query(self, 
              unlabeled_indices: np.ndarray,
              image_paths: List[str],
              n_samples: int,
              **kwargs) -> np.ndarray:
        
        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        timelog_file = Path(self.experiment_dir) / os.environ["TIME_LOGFILE"] # type: ignore
        if not timelog_file.exists():
            with open(timelog_file, 'w') as f:
                f.write("Round,TotalTime,NumImages,TimePerImage\n")

        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        
        if self.experiment_dir:
            logger.info(f"DCUS strategy using experiment directory: {self.experiment_dir}")
        else:
            logger.warning("DCUS strategy running without experiment_dir - may not work properly if classwise_quality is not available")
        
        self._ensure_classwise_quality()
        
        unlabeled_image_paths = self._get_image_paths_for_indices(unlabeled_indices, image_paths)
        local_kwargs = dict(kwargs)
        num_inf = local_kwargs.pop('num_inference', -1)
        unlabeled_image_paths = unlabeled_image_paths[:num_inf] if num_inf > 0 else unlabeled_image_paths
        
        print(f"Computing DCUS uncertainty scores for {len(unlabeled_image_paths)} images...")
        
        start_time = time.time()
        results = self.model.inference(
            unlabeled_image_paths,
            return_boxes=True,
            return_probs=True,
            return_classes=True,
            num_inference=num_inf,
            **local_kwargs
        )
        
        uncertainty_scores = []
        valid_results = 0
        
        for i, result in enumerate(results):
            try:
                score = self._compute_dcus_score(result)
                uncertainty_scores.append(score)
                valid_results += 1
                
            except Exception as e:
                print(f"Warning: Failed to compute DCUS uncertainty for image {i}: {e}")
                uncertainty_scores.append(0.0)
        
        print(f"Successfully computed DCUS uncertainty for {valid_results}/{len(results)} images")
        
        uncertainty_scores = np.array(uncertainty_scores)
        
        top_uncertain_idx = np.argsort(uncertainty_scores)[-n_samples:]
        selected_indices = unlabeled_indices[top_uncertain_idx]
        end_time = time.time()
        total_time = end_time - start_time
        time_per_image = total_time / len(unlabeled_image_paths)
        with open(timelog_file, 'a') as f:
            f.write(f"{self.round+1},{total_time:.4f},{len(unlabeled_image_paths)},{time_per_image:.4f}\n")
        
        if len(selected_indices) > 0:
            selected_local_indices = []
            for sel_idx in selected_indices:
                local_idx = np.where(unlabeled_indices == sel_idx)[0]
                if len(local_idx) > 0:
                    selected_local_indices.append(local_idx[0])
            
            if selected_local_indices:
                top_scores = uncertainty_scores[selected_local_indices]
                print(f"Top 5 DCUS uncertainty scores: {top_scores[-5:]}")
        
        print(f"DCUS score statistics - Mean: {np.mean(uncertainty_scores):.4f}, "
              f"Std: {np.std(uncertainty_scores):.4f}, "
              f"Min: {np.min(uncertainty_scores):.4f}, "
              f"Max: {np.max(uncertainty_scores):.4f}")
        
        return selected_indices

    def _ensure_classwise_quality(self):
        model_to_read = self.model.model.module if hasattr(self.model.model, "module") else self.model.model # type: ignore
        if self.experiment_dir:
            fedal_quality_path = Path(self.experiment_dir) / f"round_{self.round}" / "fedal_classwise_quality.npy"
            last_ckpt_path = next((Path(self.experiment_dir) / f"round_{self.round}" / 'train').glob("*/weights/last.pt"))
            classwise_quality_npy_path = next((Path(self.experiment_dir) / f"round_{self.round}" / 'train').glob("*/weights/classwise_quality.npy"))
            
            if fedal_quality_path.exists():
                classwise_quality = np.load(fedal_quality_path)
                setattr(model_to_read, "classwise_quality", torch.tensor(classwise_quality, dtype=torch.float32))
            elif classwise_quality_npy_path.exists():
                classwise_quality = np.load(classwise_quality_npy_path)
                setattr(model_to_read, "classwise_quality", torch.tensor(classwise_quality, dtype=torch.float32))
            elif last_ckpt_path.exists():
                restore_classwise_quality(self.model.model, last_ckpt_path)
            else:
                logger.warning(f"Neither FeDAL quality file ({fedal_quality_path}), "
                              f"numpy quality file ({classwise_quality_npy_path}), nor "
                              f"checkpoint ({last_ckpt_path}) found. Initializing with zeros.")
                nc = getattr(model_to_read, 'nc', 20)
                setattr(model_to_read, "classwise_quality", torch.zeros(nc, dtype=torch.float32))
        else:
            raise ValueError("No experiment_dir provided. Cannot restore classwise_quality. "
                            "Please provide experiment_dir or ensure the model has been trained with DCUS enabled.")

    def _get_classwise_weights(self) -> Dict[int, float]:
        model_to_read = self.model.model.module if hasattr(self.model.model, "module") else self.model.model # type: ignore
        class_qualities = getattr(model_to_read, "classwise_quality", None)
        if class_qualities is None:
            raise ValueError("Model does not have classwise_quality attribute. Train with DCUS enabled first.")

        class_qualities = class_qualities.cpu().numpy()
        reverse_q = 1 - class_qualities
        gamma = np.exp(1. / self.alpha) - 1
        _weights = 1 + self.alpha * np.log(gamma * reverse_q + 1) * self.alpha * self.beta

        return {i: float(_weights[i]) for i in range(len(_weights))}

    def _compute_dcus_score(self, result) -> float:
        if result.probs is None or len(result.probs) == 0:
            return 0.0

        probs = np.array(result.probs)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)

        probs = probs + 1e-10
        detection_entropies = -np.sum(probs * np.log(probs), axis=1)

        class_weights = self._get_classwise_weights()
        
        if result.classes is not None and len(result.classes) > 0:
            classes = np.array(result.classes).astype(int)
            detection_weights = np.array([class_weights.get(int(c), 1.0) for c in classes])
            weighted_uncertainties = detection_entropies * detection_weights
        else:
            weighted_uncertainties = detection_entropies

        return float(weighted_uncertainties.sum())

    def get_strategy_name(self) -> str:
        return "dcus"
