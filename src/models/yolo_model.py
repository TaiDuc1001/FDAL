from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

from .base import BaseModel, InferenceResult


class YOLOModel(BaseModel):
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_name: str = "yolo11n.pt",
                 feature_layers: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(model_path, **kwargs)
        self.model_name = model_name
        self.feature_layers = feature_layers or ["model.9", "model.12", "model.15", "model.18", "model.21"]
        
        self._feature_maps: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = [] # type: ignore

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            if model_path:
                print(f"Warning: Model path {model_path} does not exist. Loading default model {model_name}.")
            self.model = YOLO(model_name)
            
        self.is_trained = model_path is not None
        
    def get_available_layers(self) -> List[str]:
        if not self.model:
            return []
        return [name for name, _ in self.model.model.named_modules() if not name.endswith(('.act', '.conv', '.bn'))] # type: ignore
        
    def set_feature_layers(self, layers: List[str]) -> None:
        self.feature_layers = layers
        
    def load(self, model_path: str) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = YOLO(model_path)
        self.is_trained = True
        
    def _register_hooks(self):
        self._remove_hooks()
        if not self.model: 
            print("Warning: Model not initialized, cannot register hooks.")
            return False
        
        model_dict = dict(self.model.model.named_modules()) # type: ignore
        found_hooks = 0
        
        for name in self.feature_layers:
            if name in model_dict:
                try:
                    hook = model_dict[name].register_forward_hook(self._create_hook(name))
                    self._hooks.append(hook)
                    found_hooks += 1
                except Exception:
                    pass
            else:
                print(f"Warning: Feature layer {name} not found in model.")
        
        return found_hooks > 0
        
    def _create_hook(self, name: str):
        def hook(module, input, output):
            data = None
            if isinstance(output, torch.Tensor):
                data = output
            elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                data = output[0]
            if data is not None:
                self._feature_maps[name] = data.detach().clone()
            else:
                print(f"Warning: Could not extract data from layer {name}")
        return hook

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
    def _extract_features_from_feature_maps(self, image_shape: tuple) -> Optional[np.ndarray]:
        if not self._feature_maps:
            print(f"Warning: No feature maps available for extraction. Feature layers: {self.feature_layers}")
            return None
            
        combined_features = []
        img_h, img_w = image_shape[:2]
        
        for layer_name in self.feature_layers:
            if layer_name not in self._feature_maps:
                continue
                
            feature_map = self._feature_maps[layer_name]
            if feature_map is None or feature_map.numel() == 0:
                continue
                
            if feature_map.dim() == 4:
                pooled = F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze()
            elif feature_map.dim() == 3:
                pooled = feature_map.mean(dim=-1)
            elif feature_map.dim() == 2:
                pooled = feature_map
            else:
                pooled = feature_map.flatten()
                
            if pooled.dim() == 0:
                pooled = pooled.unsqueeze(0)
            elif pooled.dim() > 1:
                pooled = pooled.flatten()
                
            pooled = F.normalize(pooled, p=2, dim=0)
            combined_features.append(pooled)
            
        if combined_features:
            final_vector = torch.cat(combined_features, dim=0)
            final_vector = F.normalize(final_vector, p=2, dim=0)
            return final_vector.cpu().numpy()
            
        return None
        
    def _load_image_for_gradient(self, image_path: str, imgsz: int = 640) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((imgsz, imgsz))
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return t
        
    def _make_batch_from_image(self, image_path: str, imgsz: int = 640):
        img_t = self._load_image_for_gradient(image_path, imgsz)
        device = next(self.model.model.parameters()).device if self.model and self.model.model else torch.device('cpu') # type: ignore
        img = img_t.unsqueeze(0).to(device)
        
        batch = {
            "batch_idx": torch.zeros(1, dtype=torch.long, device=device),
            "cls": torch.zeros(1, dtype=torch.long, device=device),  
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32, device=device),
        }
        return batch, img
    
    def _pool_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        if gradients.dim() == 4:
            return F.adaptive_avg_pool2d(gradients, (1, 1)).squeeze()
        elif gradients.dim() == 3:
            return F.adaptive_avg_pool2d(gradients.unsqueeze(0), (1, 1)).squeeze()
        elif gradients.dim() == 2:
            return F.adaptive_avg_pool2d(gradients.unsqueeze(0).unsqueeze(0), (1, 1)).squeeze()
        else:
            return gradients.mean() if gradients.numel() > 1 else gradients
    
    def _pool_layer_gradients(self, param_grads: List[torch.Tensor]) -> torch.Tensor:
        if not param_grads:
            return torch.tensor([])
        
        pooled_grads = []
        
        for param_grad in param_grads:
            if param_grad.numel() == 0:
                continue
            if param_grad.dim() == 4:
                pooled = F.adaptive_avg_pool2d(param_grad, (1, 1)).mean()
            elif param_grad.dim() == 2:
                pooled = param_grad.mean()
            elif param_grad.dim() == 1:
                pooled = param_grad.mean()
            else:
                pooled = param_grad.flatten().mean()
                
            pooled_grads.append(pooled.unsqueeze(0))
        
        if pooled_grads:
            result = torch.cat(pooled_grads)
            return result
        else:
            return torch.tensor([])

    def _compute_layer_gradients(self, image_path: str, inference_result, use_pool: bool = True) -> Optional[np.ndarray]:
        if self.model is None:
            print("Warning: Model not initialized")
            return None
            
        try:
            imgsz = 640
            if hasattr(self.model, 'args') and hasattr(self.model.args, 'imgsz'):
                imgsz = self.model.args.imgsz # type: ignore
                if isinstance(imgsz, (list, tuple)):
                    imgsz = imgsz[0]
            imgsz = int(imgsz) # type: ignore
            
            batch, img = self._make_batch_from_image(image_path, imgsz)
            
            original_mode = self.model.model.training # type: ignore
            
            try:
                self.model.model.training = True # type: ignore
                for param in self.model.model.parameters(): # type: ignore
                    param.requires_grad_(True)
                
                param_list = []
                model_dict = dict(self.model.model.named_modules()) # type: ignore
                
                for layer_name in self.feature_layers:
                    if layer_name in model_dict:
                        layer = model_dict[layer_name]
                        for param in layer.parameters():
                            if param.requires_grad:
                                param_list.append(param)
                
                if not param_list:
                    print("Warning: No trainable parameters found in feature layers")
                    return None
                    
                img_clone = img.clone().detach().requires_grad_(True)
                raw_outputs = self.model.model(img_clone) # type: ignore
                
                try:
                    loss_fn = v8DetectionLoss(self.model.model)
                    hyp = SimpleNamespace()
                    hyp.box = 7.5
                    hyp.cls = 0.5
                    hyp.dfl = 1.5
                    loss_fn.hyp = hyp
                    
                    loss, _ = loss_fn(raw_outputs, batch)
                    total_loss = loss.sum() if hasattr(loss, 'sum') else loss
                except Exception as e:
                    print(f"Warning: Failed to compute loss: {e}")
                    return None
                
                total_loss.backward()
                
                param_grads = []
                for p in param_list:
                    g = p.grad
                    if g is None:
                        param_grads.append(torch.zeros_like(p))
                    else:
                        param_grads.append(g.detach().clone())
                
                if param_grads:
                    if use_pool:
                        result = self._pool_layer_gradients(param_grads)
                        return result.cpu().numpy()
                    else:
                        result = torch.cat([p.flatten() for p in param_grads])
                        return result.cpu().numpy()
                
                return None
                
            finally:
                try:
                    if original_mode:
                        self.model.model.training = True # type: ignore
                    else:
                        self.model.model.eval() # type: ignore
                except:
                    pass
                    
        except Exception as e:
            print(f"Warning: Failed to compute layer gradients for {image_path}: {e}")
            try:
                if hasattr(self.model, 'model') and self.model.model is not None:
                    self.model.model.eval() # type: ignore
            except:
                pass
            return None
    
    def _compute_feature_gradients(self, image_path: str, inference_result, use_pool: bool = True) -> Optional[np.ndarray]:
        if self.model is None:
            print("Warning: Model not initialized")
            return None
            
        try:
            imgsz = 640
            if hasattr(self.model, 'args') and hasattr(self.model.args, 'imgsz'):
                imgsz = self.model.args.imgsz # type: ignore
                if isinstance(imgsz, (list, tuple)):
                    imgsz = imgsz[0]
            imgsz = int(imgsz) # type: ignore
            
            batch, img = self._make_batch_from_image(image_path, imgsz)
            
            original_mode = self.model.model.training # type: ignore
            
            stored_features = {}
            hook_handles = []
            
            try:
                self.model.model.training = True # type: ignore
                for param in self.model.model.parameters(): # type: ignore
                    param.requires_grad_(True)
                
                model_dict = dict(self.model.model.named_modules()) # type: ignore
                
                def make_hook(layer_name):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            output.retain_grad()
                            stored_features[layer_name] = output
                        elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                            output[0].retain_grad()
                            stored_features[layer_name] = output[0]
                    return hook
                
                for layer_name in self.feature_layers:
                    if layer_name in model_dict:
                        h = model_dict[layer_name].register_forward_hook(make_hook(layer_name))
                        hook_handles.append(h)
                
                img_clone = img.clone().detach().requires_grad_(True)
                raw_outputs = self.model.model(img_clone) # type: ignore
                
                try:
                    loss_fn = v8DetectionLoss(self.model.model)
                    hyp = SimpleNamespace()
                    hyp.box = 7.5
                    hyp.cls = 0.5
                    hyp.dfl = 1.5
                    loss_fn.hyp = hyp
                    
                    loss, _ = loss_fn(raw_outputs, batch)
                    total_loss = loss.sum() if hasattr(loss, 'sum') else loss
                except Exception as e:
                    print(f"Warning: Failed to compute loss: {e}")
                    return None
                
                total_loss.backward()
                
                final_grads = []
                for layer_name in self.feature_layers:
                    if layer_name in stored_features:
                        feature = stored_features[layer_name]
                        if feature.grad is not None:
                            g = feature.grad
                            if use_pool:
                                # Apply proper adaptive pooling for each gradient tensor
                                pooled = self._pool_gradients(g)
                                final_grads.append(pooled.detach())
                            else:
                                # Minimal pooling method for when use_pool=False
                                if g.dim() == 4:
                                    pooled = F.adaptive_avg_pool2d(g, (1, 1)).squeeze()
                                elif g.dim() == 3:
                                    pooled = g.mean(dim=-1)
                                elif g.dim() == 2:
                                    pooled = g.mean(dim=0)
                                else:
                                    pooled = g.flatten()
                                
                                if pooled.dim() == 0:
                                    pooled = pooled.unsqueeze(0)
                                elif pooled.dim() > 1:
                                    pooled = pooled.flatten()
                                
                                final_grads.append(pooled.detach())
                        else:
                            print(f"Warning: No gradient found for layer {layer_name}")
                
                if final_grads:
                    result = torch.cat(final_grads)
                    return result.cpu().numpy()
                
                return None
                
            finally:
                for h in hook_handles:
                    try:
                        h.remove()
                    except:
                        pass
                
                try:
                    if original_mode:
                        self.model.model.training = True # type: ignore
                    else:
                        self.model.model.eval() # type: ignore
                except:
                    pass
                    
        except Exception as e:
            print(f"Warning: Failed to compute feature gradients for {image_path}: {e}")
            try:
                if hasattr(self.model, 'model') and self.model.model is not None:
                    self.model.model.eval() # type: ignore
            except:
                pass
            return None
        
    def save(self, save_path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
            
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, 'save'):
            self.model.save(save_path)
        
    def inference(self,
                  image_paths: List[str],
                  return_boxes: bool = True,
                  return_classes: bool = True,
                  return_logits: bool = False,
                  return_probs: bool = False,
                  return_features: bool = False,
                  return_embeddings: bool = False,
                  return_gradients: bool = False,
                  gradient_type: str = "layer",
                  use_pool: bool = True,
                  num_inference: int = -1,
                  conf: float = 0.25,
                  iou: float = 0.7,
                  **kwargs) -> List[InferenceResult]:

        embedding_or_feature = return_embeddings or return_features
        return_embeddings = embedding_or_feature
        return_features = embedding_or_feature
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        if 'feature_layers' in kwargs:
            feature_layers = kwargs.pop('feature_layers')
            if isinstance(feature_layers, list):
                self.set_feature_layers(feature_layers)
        
        inference_results = []
        num_inf = num_inference if num_inference > 0 else len(image_paths)
        image_paths = image_paths[:num_inf]
        if return_features:
            hooks_registered = self._register_hooks()
            if not hooks_registered:
                return_features = False
            
        try:
            for i, image_path in enumerate(image_paths):
                self._feature_maps = {}
                
                results = self.model(image_path, 
                                   conf=conf, 
                                   iou=iou,
                                   verbose=False,
                                   **kwargs)
                
                result = results[0] if results else None
                if result is None:
                    print(f"Warning: No result for image {image_path}")
                    continue
                
                boxes = None
                if return_boxes and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                classes = None
                if return_classes and result.boxes is not None:
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                probs = None
                if return_probs and result.boxes is not None:
                    probs = result.boxes.conf.cpu().numpy()
                    
                logits = None
                features = None
                layer_gradients = None
                embedding_gradients = None
                
                if return_logits:
                    pass
                    
                if return_features:
                    if hasattr(result, 'orig_shape'):
                        features = self._extract_features_from_feature_maps(result.orig_shape)
                    else:
                        features = self._extract_features_from_feature_maps((640, 640))
                
                inference_result = InferenceResult(
                    boxes=boxes,
                    classes=classes,
                    logits=logits,
                    probs=probs,
                    features=features,
                    embeddings=features
                )
                
                if return_gradients:
                    if gradient_type == "layer":
                        layer_gradients = self._compute_layer_gradients(image_path, inference_result, use_pool)
                    elif gradient_type == "feature":
                        embedding_gradients = self._compute_feature_gradients(image_path, inference_result, use_pool)
                    else:
                        print(f"Warning: Unknown gradient type '{gradient_type}'. Expected 'layer' or 'feature'.")
                    
                    inference_result.layer_gradients = layer_gradients
                    inference_result.embedding_gradients = embedding_gradients
                
                inference_results.append(inference_result)
                
        finally:
            if return_features:
                self._remove_hooks()
            
        return inference_results
    
    def train(self,
              data_yaml: str,
              epochs: int = 100,
              batch_size: int = 16,
              imgsz: int = 640,
              save_dir: str = "runs/train",
              **kwargs) -> 'YOLOModel':
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            val=False,
            **kwargs
        )
        best_path = Path(save_dir) / "weights" / "best.pt" # type: ignore
        print(f"Training completed. Best model path: {best_path}")
        self.model_path = str(best_path)
        self.is_trained = True
        
        return self
    
    def val(self,
            data_yaml: str,
            batch_size: int = 32,
            imgsz: int = 640,
            save_dir: str = "runs/val",
            **kwargs) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        results = self.model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=imgsz,
            project=save_dir,
            name="exp",
            **kwargs
        )
        
        metrics = {}
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
        else:
            if hasattr(results, 'box'):
                metrics['map50-95'] = results.box.map
                metrics['map50'] = results.box.map50
                metrics['precision'] = results.box.p.mean()
                metrics['recall'] = results.box.r.mean()
                
        return metrics
