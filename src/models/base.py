from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import cv2

class InferenceResult:
    
    def __init__(self, 
                 boxes: Optional[np.ndarray] = None,
                 classes: Optional[np.ndarray] = None, 
                 logits: Optional[np.ndarray] = None,
                 probs: Optional[np.ndarray] = None,
                 features: Optional[np.ndarray] = None,
                 embeddings: Optional[np.ndarray] = None,
                 layer_gradients: Optional[np.ndarray] = None,
                 embedding_gradients: Optional[np.ndarray] = None):
        self.boxes = boxes
        self.classes = classes
        self.logits = logits
        self.probs = probs
        self.features = features
        self.embeddings = embeddings
        self.layer_gradients = layer_gradients
        self.embedding_gradients = embedding_gradients

    def __repr__(self):
        return f"InferenceResult(boxes={self.boxes.shape if self.boxes is not None else None})"


class BaseModel(ABC):
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def load(self, model_path: str) -> None:
        pass
    
    @abstractmethod
    def save(self, save_path: str) -> None:
        pass
    
    @abstractmethod
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
                  **kwargs) -> List[InferenceResult]:
        assert gradient_type in ["layer", "embedding"], "Invalid gradient type"
        pass
    
    @abstractmethod
    def train(self, 
              data_yaml: str,
              epochs: int = 100,
              batch_size: int = 16,
              imgsz: int = 640,
              save_dir: str = "runs/train",
              **kwargs) -> 'BaseModel':
        pass
    
    @abstractmethod
    def val(self, 
            data_yaml: str,
            batch_size: int = 32,
            imgsz: int = 640,
            save_dir: str = "runs/val",
            **kwargs) -> Dict[str, float]:
        pass
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    def batch_preprocess(self, image_paths: List[str]) -> List[np.ndarray]:
        return [self.preprocess_image(path) for path in image_paths]
