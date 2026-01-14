import argparse
import numpy as np
from pathlib import Path
import sys
import os
import yaml
from typing import List
from loguru import logger
from dotenv import load_dotenv

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env.training')
if cuda_visible_devices is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

logger.info(f"[strategy.py] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

sys.path.append(str(Path(__file__).parent.parent))

from aida.data.manager import DataManager
from aida.models.base import BaseModel
from aida.strategies import (
    RandomStrategy,
    EntropyStrategy, 
    MarginStrategy,
    CoreSetStrategy,
    BADGEStrategy,
    FDAL,
    CCMSStrategy,
    DCUSStrategy,
    CDALStrategy,
    DivProtoStrategy,
    MIDPRCStrategy
)
from scripts.utils import create_model


def parse_args():
    parser = argparse.ArgumentParser(description='Run active learning strategy')
    parser.add_argument('--dataset_yaml', type=str,
                       help='Path to dataset YAML configuration')
    parser.add_argument('--strategy', type=str, default='random',
                       help='Active learning strategy to use (can be comma-separated for chaining, e.g., "dcus,ccms")')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='Model path or name')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to select')
    parser.add_argument('--num_inference', type=str, default='-1',
                       help='Number of images to run inference, -1 for full dataset, or path to folder of images')
    parser.add_argument('--experiment_dir', type=str,
                       help='Path to existing experiment directory')
    parser.add_argument('--round', type=int, default=0,
                       help='Current round number')
    parser.add_argument('--output_file', type=str, 
                       help='File to write selected indices to')
    parser.add_argument('--config', type=str,
                       help='Path to config YAML file for strategy-specific arguments')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for inference (e.g., "0", "1", "cpu", "auto")')
    
    return parser.parse_args()


def load_model(model_path: str, class_names: List[str]):
    logger.info(f"Loading model: {model_path}")
    return create_model(model_path, categories=class_names)


def load_strategy_args(config_path: str, strategy_name: str) -> dict:
    if not config_path or not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        strategy_args = config.get('strategy_args', {})
        return strategy_args.get(strategy_name, {})
    
    except Exception as e:
        logger.warning(f"Failed to load strategy args from config {config_path}: {e}")
        return {}


def load_config(config_path: str) -> dict:
    if not config_path or not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


def create_strategy(strategy_name: str, model: BaseModel, **kwargs):
    if strategy_name == 'random':
        return RandomStrategy(model, **kwargs)
    elif strategy_name == 'entropy':
        return EntropyStrategy(model, **kwargs)
    elif strategy_name == 'margin':
        return MarginStrategy(model, **kwargs)
    elif strategy_name == 'coreset':
        return CoreSetStrategy(model, **kwargs)
    elif strategy_name == 'badge':
        return BADGEStrategy(model, **kwargs)
    elif strategy_name == 'fdal':
        return FDAL(model, **kwargs)
    elif strategy_name == 'ccms':
        return CCMSStrategy(model, **kwargs)
    elif strategy_name == 'dcus':
        return DCUSStrategy(model, **kwargs)
    elif strategy_name == 'cdal':
        return CDALStrategy(model, **kwargs)
    elif strategy_name == 'divproto':
        return DivProtoStrategy(model, **kwargs)
    elif strategy_name == 'midprc':
        return MIDPRCStrategy(model, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def execute_chained_strategies(
    strategies: List[str],
    expand_ratios: List[float],
    model: BaseModel,
    unlabeled_indices: np.ndarray,
    all_image_paths: List[str],
    final_n_samples: int,
    num_inference: int,
    config: dict
) -> np.ndarray:
    if len(strategies) == 1:
        strategy_name = strategies[0]
        strategy_specific_args = config.get('strategy_args', {}).get(strategy_name, {})
        strategy_kwargs = {
            'experiment_dir': config.get('experiment_dir'),
            'round': config.get('round'),
            'num_inference': num_inference,
            'device': config.get('device', 'auto'),
            'train_ddp': config.get('train_ddp', False),
            'master_addr': config.get('master_addr', 'localhost'),
            'master_port': config.get('master_port', 12355),
            'backend': config.get('backend', 'nccl'),
            'seed': config.get('seed', 42),
        }
        strategy_kwargs.update(strategy_specific_args)
        print(strategy_specific_args)
        
        logger.info(f"Creating single strategy '{strategy_name}' with args: {strategy_kwargs}")
        strategy = create_strategy(strategy_name, model, **strategy_kwargs)
        
        return strategy.query(
            unlabeled_indices=unlabeled_indices,
            image_paths=all_image_paths,
            n_samples=min(final_n_samples, len(unlabeled_indices)),
            num_inference=num_inference
        )
    
    if len(expand_ratios) != len(strategies) - 1:
        raise ValueError(f"Expected {len(strategies) - 1} expand_ratios for {len(strategies)} strategies, got {len(expand_ratios)}")
    
    current_indices = unlabeled_indices.copy()
    
    for i, strategy_name in enumerate(strategies[:-1]):
        expand_ratio = expand_ratios[i] * expand_ratios[i+1] if i < len(expand_ratios) - 1 else 1
        n_samples_for_stage = int(final_n_samples * expand_ratio)
        
        strategy_specific_args = config.get('strategy_args', {}).get(strategy_name, {})
        strategy_kwargs = {
            'experiment_dir': config.get('experiment_dir'),
            'round': config.get('round'),
            'num_inference': num_inference,
            'device': config.get('device', 'auto'),
            'train_ddp': config.get('train_ddp', False),
            'master_addr': config.get('master_addr', 'localhost'),
            'master_port': config.get('master_port', 12355),
            'backend': config.get('backend', 'nccl'),
            'seed': config.get('seed', 42),
        }
        strategy_kwargs.update(strategy_specific_args)
        
        logger.info(f"Stage {i+1}: Creating strategy '{strategy_name}' with args: {strategy_kwargs}")
        logger.info(f"Stage {i+1}: Selecting {n_samples_for_stage} samples from {len(current_indices)} candidates")
        
        strategy = create_strategy(strategy_name, model, **strategy_kwargs)
        
        stage_selected = strategy.query(
            unlabeled_indices=current_indices,
            image_paths=all_image_paths,
            n_samples=min(n_samples_for_stage, len(current_indices)),
            num_inference=num_inference
        )
        
        current_indices = stage_selected
        logger.info(f"Stage {i+1}: Selected {len(stage_selected)} samples")
    
    final_strategy_name = strategies[-1]
    final_strategy_specific_args = config.get('strategy_args', {}).get(final_strategy_name, {})
    final_strategy_kwargs = {
        'experiment_dir': config.get('experiment_dir'),
        'round': config.get('round'),
        'num_inference': num_inference,
        'device': config.get('device', 'auto'),
        'train_ddp': config.get('train_ddp', False),
        'master_addr': config.get('master_addr', 'localhost'),
        'master_port': config.get('master_port', 12355),
        'backend': config.get('backend', 'nccl')
    }
    final_strategy_kwargs.update(final_strategy_specific_args)
    
    logger.info(f"Final stage: Creating strategy '{final_strategy_name}' with args: {final_strategy_kwargs}")
    logger.info(f"Final stage: Selecting {final_n_samples} samples from {len(current_indices)} candidates")
    
    final_strategy = create_strategy(final_strategy_name, model, **final_strategy_kwargs)
    
    final_selected = final_strategy.query(
        unlabeled_indices=current_indices,
        image_paths=all_image_paths,
        n_samples=min(final_n_samples, len(current_indices)),
        num_inference=num_inference
    )
    
    logger.info(f"Final stage: Selected {len(final_selected)} samples")
    return final_selected


def main():
    args = parse_args()

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
        class_names=class_names
    )
    
    if args.experiment_dir and args.round > 0:
        prev_round_dir = Path(args.experiment_dir) / f"round_{args.round-1}"
        if prev_round_dir.exists():
            metadata_file = prev_round_dir / "metadata.yaml"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                current_labeled_indices = np.array(metadata.get('train_indices', []))
                data_manager.original_dataset.set_labeled_indices(current_labeled_indices)
                logger.info(f"Restored {len(current_labeled_indices)} labeled indices from previous round")
    
    model = load_model(args.model, class_names)
    
    config = load_config(args.config)
    
    if args.device != 'auto':
        # When CUDA_VISIBLE_DEVICES is set, we need to map device indices
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_visible and cuda_visible not in ['auto', 'cpu']:
            # Map the device index to 0-based indexing
            visible_devices = [d.strip() for d in cuda_visible.split(',')]
            if args.device in visible_devices:
                mapped_device = str(visible_devices.index(args.device))
                config['device'] = mapped_device
                logger.info(f"Mapped device {args.device} to {mapped_device} (CUDA_VISIBLE_DEVICES={cuda_visible})")
            else:
                config['device'] = '0'  # Default to first visible device
                logger.warning(f"Device {args.device} not in CUDA_VISIBLE_DEVICES={cuda_visible}, using device 0")
        else:
            config['device'] = args.device
        logger.info(f"Overriding device from config with command line arg: {args.device}")
    else:
        logger.info(f"Using device from config: {config.get('device', 'auto')}")
    
    logger.info(f"Final device configuration: {config.get('device', 'auto')}")
    
    strategies = config.get('strategy', args.strategy)
    if isinstance(strategies, str):
        if '-' in strategies:
            strategies = [s.strip() for s in strategies.split('-')]
        else:
            strategies = [strategies]
    
    expand_ratios = config.get('expand_ratios', [])
    
    config['experiment_dir'] = args.experiment_dir
    config['round'] = args.round
    
    unlabeled_indices = data_manager.get_unlabeled_indices()
    all_image_paths = data_manager.get_all_image_paths()
    
    num_inference = args.num_inference
    if num_inference and os.path.isdir(num_inference):
        inference_folder = Path(num_inference)
        inference_image_names = {p.stem for p in inference_folder.glob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
        logger.info(f"Using {len(inference_image_names)} images from inference folder: {inference_folder}")
        
        filtered_indices = []
        for idx in unlabeled_indices:
            img_name = Path(all_image_paths[idx]).stem
            if img_name in inference_image_names:
                filtered_indices.append(idx)
        
        unlabeled_indices = np.array(filtered_indices)
        logger.info(f"Filtered to {len(unlabeled_indices)} matching unlabeled images")
        num_inference_int = -1
    else:
        num_inference_int = int(num_inference)
    
    if len(unlabeled_indices) == 0:
        logger.info("No unlabeled samples available")
        return
    
    logger.info(f"Found {len(unlabeled_indices)} unlabeled samples")
    
    if len(strategies) == 1:
        logger.info(f"Using single strategy: {strategies[0]}")
    else:
        logger.info(f"Using chained strategies: {' -> '.join(strategies)}")
        logger.info(f"Expand ratios: {expand_ratios}")
        total_budget = args.num_samples
        for i, ratio in enumerate(expand_ratios):
            stage_samples = int(total_budget * ratio)
            logger.info(f"  Stage {i+1} ({strategies[i]}): {stage_samples} samples")
        logger.info(f"  Final stage ({strategies[-1]}): {total_budget} samples")
    
    try:
        selected_indices = execute_chained_strategies(
            strategies=strategies,
            expand_ratios=expand_ratios,
            model=model,
            unlabeled_indices=unlabeled_indices,
            all_image_paths=all_image_paths,
            final_n_samples=args.num_samples,
            num_inference=num_inference_int,
            config=config
        )
        
        logger.info(f"Selected {len(selected_indices)} samples")
        logger.info(f"Selected indices: {selected_indices}")
        
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.save(output_path, selected_indices)
            logger.info(f"Saved selected indices to: {output_path}")
            
            selected_image_paths = [all_image_paths[idx] for idx in selected_indices]
            paths_file = output_path.with_suffix('.txt')
            with open(paths_file, 'w') as f:
                for path in selected_image_paths:
                    f.write(f"{path}\\n")
            logger.info(f"Saved selected image paths to: {paths_file}")
    except Exception as e:
        logger.exception(f"Error during strategy execution: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
