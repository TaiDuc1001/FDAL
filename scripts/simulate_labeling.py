import argparse
import sys
from pathlib import Path
import json
import numpy as np
import yaml
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env.training')

sys.path.append(str(Path(__file__).parent.parent))

from src.data.manager import DataManager
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Simulate labeling process')
    parser.add_argument('--dataset_yaml', type=str,
                       help='Path to dataset YAML configuration')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--indices', type=str,
                       help='Comma-separated indices (deprecated, use --indices_file)')
    parser.add_argument('--indices_file', type=str,
                       help='Path to numpy file containing selected indices')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--round', type=int, required=True)
    return parser.parse_args()


def simulate_labeling(selected_indices: List[int], data_manager: DataManager) -> Dict[int, List[List[float]]]:
    labels_data = {}
    
    for idx in selected_indices:
        original_label_path = data_manager.original_dataset.all_labels[idx]
        
        if original_label_path is not None:
            with open(original_label_path, 'r') as f:
                annotations = []
                for line in f:
                    line = line.strip()
                    if line:
                        parts = list(map(float, line.split()))
                        annotations.append(parts)
                labels_data[idx] = annotations
        else:
            labels_data[idx] = []
    
    return labels_data


def main():
    args = parse_args()
    
    if args.indices_file:
        selected_indices = np.load(args.indices_file, allow_pickle=True).tolist()
    elif args.indices:
        selected_indices = [int(x) for x in args.indices.split('-')]
    else:
        raise ValueError("Either --indices or --indices_file must be provided")
    
    dataset_path = args.dataset
    dataset_name = Path(args.dataset).name
    class_names = []
    
    if args.dataset_yaml:
        try:
            with open(args.dataset_yaml, 'r') as f:
                dy = yaml.safe_load(f)
            names = dy.get('names')
            if isinstance(names, dict):
                class_names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            elif isinstance(names, list):
                class_names = names
            dataset_name = Path(args.dataset_yaml).stem
        except Exception as e:
            logger.warning(f"Could not read dataset YAML {args.dataset_yaml}: {e}")
    
    if not class_names:
        class_names = ['class_0']
    
    data_manager = DataManager(
        original_dataset_path=dataset_path,
        dataset_name=dataset_name,
        class_names=class_names,
        experiments_root=str(Path(args.experiment_dir).parent)
    )
    
    experiment_name = Path(args.experiment_dir).name
    data_manager.current_experiment_dir = Path(args.experiment_dir)
    
    if args.round > 0:
        prev_round_dir = data_manager.current_experiment_dir / f"round_{args.round-1}"
        if prev_round_dir.exists():
            metadata_file = prev_round_dir / "metadata.yaml"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                current_labeled_indices = np.array(metadata.get('train_indices', []))
                data_manager.original_dataset.set_labeled_indices(current_labeled_indices)
                logger.info(f"Restored {len(current_labeled_indices)} labeled indices from previous round")
    
    new_labels = simulate_labeling(selected_indices, data_manager)
    data_manager.add_labeled_samples(args.round, np.array(selected_indices), new_labels)
    
    return 0


if __name__ == "__main__":
    exit(main())
