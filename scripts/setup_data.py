import os
import argparse
import sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env.training')

sys.path.append(str(Path(__file__).parent.parent))

from aida.data.manager import DataManager


def parse_args():
    parser = argparse.ArgumentParser(description='Setup experiment data')
    parser.add_argument('--dataset_yaml', type=str,
                        help='Path to dataset YAML (preferred). If provided, --dataset_path/--dataset_name/--class_names* are optional')
    parser.add_argument('--initial_count', type=int, required=True)
    parser.add_argument('--experiments_root', type=str, default='experiments')
    parser.add_argument('--seed', type=str, default='42')
    parser.add_argument('--model_name', type=str, default='yolo')
    parser.add_argument('--strategy', type=str, default='experiment',
                       help='Strategy name (can be comma-separated for chaining)')
    return parser.parse_args()


def main():
    args = parse_args()
    import yaml
    try:
        with open(args.dataset_yaml, 'r') as f:
            dy = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to read dataset YAML {args.dataset_yaml}: {e}")

    dataset_path = str(dy.get('path') or dy.get('root') or '')
    names = dy.get('names')
    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    elif isinstance(names, list):
        class_names = names
    else:
        class_names = []

    if isinstance(dy.get('name'), str) and dy.get('name').strip():
        dataset_name = dy.get('name').strip()
    else:
        parent_name = Path(args.dataset_yaml).parent.name
        if parent_name and parent_name.strip():
            dataset_name = parent_name
        elif dataset_path:
            dataset_name = Path(dataset_path).stem
        else:
            dataset_name = Path(args.dataset_yaml).stem

    data_manager = DataManager(
        original_dataset_path=dataset_path,
        dataset_name=dataset_name,
        class_names=class_names,
        experiments_root=args.experiments_root
    )
    
    training_indices = data_manager.original_dataset.train_indices
    print(f"Found {len(training_indices)} training images available for active learning")
    
    if len(training_indices) < args.initial_count:
        raise ValueError(f"Not enough training images. Found {len(training_indices)}, need {args.initial_count}")
    
    if args.seed and os.path.isdir(args.seed):
        seed_folder = Path(args.seed)
        image_names = {p.stem for p in seed_folder.glob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
        print(f"Using {len(image_names)} images from seed folder: {seed_folder}")
        
        initial_indices = []
        for i in training_indices:
            img_name = data_manager.original_dataset.all_images[i].stem
            if img_name in image_names:
                initial_indices.append(i)
        
        initial_indices = np.array(initial_indices[:args.initial_count])
        print(f"Matched {len(initial_indices)} images from dataset")
    else:
        np.random.seed(int(args.seed))
        initial_indices = np.random.choice(training_indices, size=args.initial_count, replace=False)
    
    # Parse strategy name for directory naming
    if '-' in args.strategy:
        strategy_name = args.strategy.replace('-', '_')
    else:
        strategy_name = args.strategy
    
    experiment_dir = data_manager.create_experiment(
        strategy_name=strategy_name,
        model_name=args.model_name,
        initial_labeled_indices=initial_indices
    )
    
    print(experiment_dir)
    return 0


if __name__ == "__main__":
    exit(main())
