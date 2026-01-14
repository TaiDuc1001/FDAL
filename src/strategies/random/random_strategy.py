import os
import time
import numpy as np
from typing import List
from pathlib import Path
from typing import Optional
from ..base import BaseStrategy


class RandomStrategy(BaseStrategy):
    def __init__(self, 
                 model,
                 round: Optional[int] = None,
                 experiment_dir: Optional[str] = None,
                 **kwargs):
        super().__init__(model, **kwargs)
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.round = round
        self.experiment_dir = experiment_dir

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

        selectionlog_file = Path(self.experiment_dir) / os.environ["SELECTION_LOGFILE"] # type: ignore
        if not selectionlog_file.exists():
            selectionlog_file.touch()

        start_time = time.time()
        selected_indices = np.random.choice(
            unlabeled_indices, 
            size=n_samples, 
            replace=False
        )
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
            results=None,
            unlabeled_indices=unlabeled_indices,
        )
        self._save_selection_symlinks(
            experiment_dir=self.experiment_dir,
            round_num=self.round,
            selected_image_paths=selected_image_paths,
        )

        total_time = time.time() - start_time
        num_images = len(unlabeled_indices)
        time_per_image = total_time / num_images
        with open(timelog_file, 'a') as f:
            f.write(f"{self.round},{total_time:.4f},{num_images},{time_per_image:.6f}\n")
        print("Write time log to", timelog_file.absolute())

        return selected_indices
    
    def get_strategy_name(self) -> str:
        return "random"
