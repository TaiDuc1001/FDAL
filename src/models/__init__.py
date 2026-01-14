"""
Models package for active learning framework
"""

from .base import BaseModel, InferenceResult
from .yolo_model import YOLOModel

__all__ = ["BaseModel", "InferenceResult", "YOLOModel"]
