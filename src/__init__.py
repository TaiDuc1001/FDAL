"""
AIDA: Active learning framework for object detection
"""

__version__ = "0.1.0"
__author__ = "AIDA Team"

from .models.base import BaseModel
from .strategies.base import BaseStrategy
from .data.manager import DataManager
from .data.dataset import ALDataset

__all__ = [
    "BaseModel",
    "BaseStrategy", 
    "DataManager",
    "ALDataset"
]
