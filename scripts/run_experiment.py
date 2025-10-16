import csv
import sys
import json
import yaml
import time
import pandas
import argparse
import subprocess
import numpy as np
import os
import tempfile
from pathlib import Path
from loguru import logger
from typing import Dict, Union
from dotenv import load_dotenv
from setproctitle import setproctitle

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env.training')


logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def parse_args():
    parser = argparse.ArgumentParser(description='Run active learning experiment')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent.parent / 'configs' / 'config.yaml'),
                        help='Path to config YAML (default: configs/config.yaml)')
    parser.add_argument('--dataset_yaml', type=str, help='Override dataset_yaml')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch_size')
    parser.add_argument('--val_batch_size', type=int, help='Override val_batch_size')
    parser.add_argument('--imgsz', type=int, help='Override imgsz')
    parser.add_argument('--device', type=str, help='Override device (can be single device or comma-separated list for multiple GPUs)')
    parser.add_argument('--patience', type=int, help='Override patience')
    parser.add_argument('--experiments_root', type=str, help='Override experiments_root')
    parser.add_argument('--seed', type=int, help='Override seed')
    parser.add_argument('--strategy', type=str, help='Override strategy (can be single strategy or comma-separated for chaining)')
    parser.add_argument('--model_name', type=str, help='Override model_name')
    parser.add_argument('--initial_labeled_count', type=int, help='Override initial_labeled_count')
    parser.add_argument('--max_rounds', type=int, help='Override max_rounds')
    parser.add_argument('--samples_per_round', type=int, help='Override samples_per_round')
    parser.add_argument('--num_inference', type=int, help='Num images to run inference, -1 for full dataset')
    parser.add_argument('--override', action='append', help='Override nested config keys, e.g., --override strategy_args.alfi.supporter_embedding_size=512')

    return parser.parse_args()


def load_config(config_path: Union[str, Path]) -> Dict:
    config_path = Path(config_path) if config_path else Path(__file__).parent.parent / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['_config_path'] = str(config_path)
    return config


def set_nested(d, key, value):
    keys = key.split('.')
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def apply_overrides(config: Dict, args) -> Dict:
    if args.dataset_yaml:
        config['dataset_yaml'] = args.dataset_yaml
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.val_batch_size is not None:
        config['val_batch_size'] = args.val_batch_size
    if args.imgsz is not None:
        config['imgsz'] = args.imgsz
    if args.device:
        if ',' in args.device:
            config['device'] = [d.strip() for d in args.device.split(',')]
        else:
            config['device'] = args.device
    if args.patience is not None:
        config['patience'] = args.patience
    if args.experiments_root:
        config['experiments_root'] = args.experiments_root
    if args.seed is not None:
        config['seed'] = args.seed
    if args.strategy:
        if '-' in args.strategy:
            config['strategy'] = [s.strip() for s in args.strategy.split('-')]
        else:
            config['strategy'] = args.strategy
    if args.model_name:
        config['model_name'] = args.model_name
    if args.initial_labeled_count is not None:
        config['initial_labeled_count'] = args.initial_labeled_count
    if args.max_rounds is not None:
        config['max_rounds'] = args.max_rounds
    if args.samples_per_round is not None:
        config['samples_per_round'] = args.samples_per_round
    if args.num_inference is not None:
        config['num_inference'] = args.num_inference
    
    if args.override:
        for ov in args.override:
            if '=' not in ov:
                logger.warning(f"Invalid override format: {ov}")
                continue
            key, value_str = ov.split('=', 1)
            try:
                if value_str.isdigit():
                    value = int(value_str)
                elif '.' in value_str and value_str.replace('.', '').replace('-', '').isdigit():
                    value = float(value_str)
                else:
                    value = value_str
            except:
                value = value_str
            set_nested(config, key, value)
    
    return config


def set_process_title(config: Dict):
    proctitle_startstr = os.environ['PROCTITLE_STARTSTR']
    if not proctitle_startstr:
        raise ValueError("Environment variable PROCTITLE_STARTSTR is not set or empty")
    proctitle_midstr = config['proctitle_midstr']
    proctitle_endstr_of = config.get('proctitle_endstr_of', '')
    
    endstr_value = str(config[proctitle_endstr_of]) if proctitle_endstr_of and proctitle_endstr_of in config else ''
    process_title = f"{proctitle_startstr}_{proctitle_midstr}_{endstr_value}"
    setproctitle(process_title)


def run_data_setup(config: Dict) -> str:
    dataset_yaml = config.get('dataset_yaml')
    dataset_path = None
    dataset_name = None
    class_names = []
    
    if dataset_yaml:
        try:
            with open(dataset_yaml, 'r') as f:
                dy = yaml.safe_load(f)
            dataset_path = str(dy.get('path') or dy.get('root') or '')
            names = dy.get('names')
            if isinstance(names, dict):
                class_names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            elif isinstance(names, list):
                class_names = names

            dataset_name = None
            if isinstance(dy.get('name'), str) and dy.get('name').strip():
                dataset_name = dy.get('name').strip()
            else:
                parent_name = Path(dataset_yaml).parent.name
                if parent_name and parent_name.strip():
                    dataset_name = parent_name
                elif dataset_path:
                    dataset_name = Path(dataset_path).stem
                else:
                    dataset_name = Path(dataset_yaml).stem
        except Exception as e:
            raise ValueError(f"Failed to read dataset YAML {dataset_yaml}: {e}")
    
    if not dataset_path:
        dataset_path = config.get('dataset_path', '/default/path')
    if not dataset_name:
        dataset_name = config.get('dataset_name', 'default_dataset')
    if not class_names:
        class_names = config.get('class_names', ['default_class'])

    strategy = config.get('strategy', 'experiment')
    if isinstance(strategy, list):
        strategy_str = '-'.join(strategy)
    else:
        strategy_str = str(strategy)
    
    setup_args = [
        sys.executable, str(Path(__file__).parent / 'setup_data.py'),
        '--dataset_yaml', str(dataset_yaml) if dataset_yaml else '',
        '--initial_count', str(config.get('initial_labeled_count', 0)),
        '--experiments_root', config.get('experiments_root', 'experiments'),
        '--seed', str(config.get('seed', 42)),
        '--model_name', config.get('model_name', 'yolo'),
        '--strategy', strategy_str
    ]
    
    result = subprocess.run(setup_args, check=True, capture_output=True, text=True)
    experiment_dir = result.stdout.strip().split('\n')[-1]

    return experiment_dir


def run_training(config: Dict, experiment_dir: str, round_num: int = 0) -> Dict:
    dataset_yaml_path = Path(experiment_dir) / f"round_{round_num}" / "data.yaml"
    model_path_file = Path(experiment_dir) / f"round_{round_num}" / "model_path.txt"
    
    model_name = config.get('model_name', 'yolo11n.pt')
    strategy = config.get('strategy', 'random')
    
    if isinstance(strategy, list):
        strategy_str = '-'.join(strategy)
    else:
        strategy_str = str(strategy)
    device = config.get('device', 'auto')
    device_str = ','.join(str(d) for d in device) if isinstance(device, list) else str(device)
    
    train_args = [
        sys.executable, str(Path(__file__).parent / 'train.py'),
        '--dataset_yaml', str(dataset_yaml_path),
        '--model', model_name,
        '--epochs', str(config.get('epochs', 100)),
        '--batch_size', str(config.get('batch_size', 16)),
        '--imgsz', str(config.get('imgsz', 640)),
        '--device', device_str,
        '--project', str(Path(experiment_dir) / f"round_{round_num}" / "train"),
        '--save_dir', str(Path(experiment_dir) / f"round_{round_num}" / "train" / model_name.split('/')[-1].replace('.pt','').replace('.yaml','')),
        '--name', 'model',
        '--model_path_file', str(model_path_file),
        '--strategy', strategy_str
    ]
    
    env = os.environ.copy()
    
    try:
        subprocess.run(train_args, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Training failed for round {round_num}: {e}")
        raise
    metrics_file = Path(experiment_dir) / f"round_{round_num}" / "train" / model_name / "results.csv"
    if metrics_file.exists():
        df = pandas.read_csv(metrics_file)
        last_row = df.iloc[-1]
        metrics = {
            'train/box_loss': float(last_row['train/box_loss']),
            'train/cls_loss': float(last_row['train/cls_loss']),
            'train/dfl_loss': float(last_row['train/dfl_loss']),
            'metrics/precision': float(last_row['metrics/precision(B)']),
            'metrics/recall': float(last_row['metrics/recall(B)']),
            'map50': float(last_row['metrics/mAP50(B)']),
            'map50-95': float(last_row['metrics/mAP50-95(B)']),
            'val/box_loss': float(last_row['val/box_loss']),
            'val/cls_loss': float(last_row['val/cls_loss']),
            'val/dfl_loss': float(last_row['val/dfl_loss']),
        }
        return metrics
    return {}

def run_strategy(config: Dict, experiment_dir: str, round_num: int):
    model_path_file = Path(experiment_dir) / f"round_{round_num-1}" / "model_path.txt"
    if model_path_file.exists():
        with open(model_path_file, 'r') as f:
            model_path = f.read().strip()
    else:
        model_name = config.get('model_name', 'model')
        if model_name.endswith('.pt'):
            model_name = model_name[:-3]
        model_path = Path(experiment_dir) / f"round_{round_num-1}" / "train" / model_name / "weights" / "best.pt"
        print(f"Model path file not found, using default path: {model_path}")
    
    output_file = Path(experiment_dir) / f"round_{round_num}" / "selected_indices.npy"
    dataset_yaml = config.get('dataset_yaml')
    dataset_path = None
    dataset_name = None
    class_names = []
    
    if dataset_yaml:
        try:
            with open(dataset_yaml, 'r') as f:
                dy = yaml.safe_load(f)
            dataset_path = str(dy.get('path') or dy.get('root') or '')
            names = dy.get('names')
            if isinstance(names, dict):
                class_names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            elif isinstance(names, list):
                class_names = names

            dataset_name = None
            if isinstance(dy.get('name'), str) and dy.get('name').strip():
                dataset_name = dy.get('name').strip()
            else:
                parent_name = Path(dataset_yaml).parent.name
                if parent_name and parent_name.strip():
                    dataset_name = parent_name
                elif dataset_path:
                    dataset_name = Path(dataset_path).stem
                else:
                    dataset_name = Path(dataset_yaml).stem
        except Exception as e:
            raise ValueError(f"Failed to read dataset YAML {dataset_yaml}: {e}")
    
    if not dataset_path:
        dataset_path = config.get('dataset_path', '/default/path')
    if not dataset_name:
        dataset_name = config.get('dataset_name', 'default_dataset')
    if not class_names:
        class_names = config.get('class_names', ['default_class'])

    strategy = config.get('strategy', 'random')
    if isinstance(strategy, list):
        strategy_str = '-'.join(strategy)
    else:
        strategy_str = str(strategy)
    
    device = config.get('device', 'auto')
    device_str = ','.join(str(d) for d in device) if isinstance(device, list) else str(device)
    
    strategy_args = [
        sys.executable, str(Path(__file__).parent / 'strategy.py'),
        '--dataset_yaml', str(dataset_yaml) if dataset_yaml else '',
        '--strategy', strategy_str,
        '--model', str(model_path),
        '--num_samples', str(config.get('samples_per_round', 50)),
        '--num_inference', str(config['num_inference']),
        '--experiment_dir', experiment_dir,
        '--round', str(round_num),
        '--output_file', str(output_file),
        '--config', str(Path(config.get('_temp_config_path', config.get('_config_path', Path(__file__).parent.parent / 'configs' / 'config.yaml')))),
        '--device', device_str
    ]
    
    env = os.environ.copy()
    cuda_visible = env.get('CUDA_VISIBLE_DEVICES', 'not set')
    
    subprocess.run(strategy_args, check=True, env=env)
    
    if output_file.exists():
        return np.load(output_file, allow_pickle=True)
    return np.array([])


def simulate_labeling_cli(selected_indices: np.ndarray, config: Dict, experiment_dir: str, round_num: int):
    dataset_yaml = config.get('dataset_yaml')
    dataset_path = None
    
    if dataset_yaml:
        try:
            with open(dataset_yaml, 'r') as f:
                dy = yaml.safe_load(f)
            dataset_path = str(dy.get('path') or dy.get('root') or '')
        except Exception as e:
            raise ValueError(f"Failed to read dataset YAML {dataset_yaml}: {e}")
    
    if not dataset_path:
        dataset_path = config.get('dataset_path', '/default/path')

    indices_file = Path(experiment_dir) / f"round_{round_num}" / "selected_indices.npy"
    indices_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(indices_file, selected_indices)

    label_args = [
        sys.executable, str(Path(__file__).parent / 'simulate_labeling.py'),
        '--dataset_yaml', str(dataset_yaml) if dataset_yaml else '',
        '--dataset', dataset_path,
        '--indices_file', str(indices_file),
        '--experiment_dir', experiment_dir,
        '--round', str(round_num)
    ]
    
    subprocess.run(label_args, check=True)


def main():
    args = parse_args()
    
    config = load_config(args.config)
    config = apply_overrides(config, args)
    
    temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_config_file)
    temp_config_file.close()
    config['_temp_config_path'] = temp_config_file.name
    
    device_config = config.get('device', ['0'])
    
    # Normalize device_config to handle both string and list inputs consistently
    if isinstance(device_config, str):
        if ',' in device_config:
            # Handle comma-separated device string like "0,1,2"
            device_list = [d.strip() for d in device_config.split(',')]
            device_str = device_config
            is_list_config = True
        else:
            # Single device string like "3"
            device_list = [device_config]
            device_str = device_config
            is_list_config = False
    elif isinstance(device_config, list):
        device_list = [str(d) for d in device_config]
        device_str = ','.join(device_list)
        is_list_config = True
    else:
        device_str = str(device_config)
        device_list = [device_str]
        is_list_config = False
    
    if device_str not in ['auto', 'cpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        logger.info(f"Set CUDA_VISIBLE_DEVICES = {device_str}")
    
    set_process_title(config)
    
    logger.info("Loaded configuration")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    if not config.get('dataset_yaml'):
        logger.error('Dataset YAML path is required')
        return 1
    if config.get('max_rounds', 0) <= 0:
        logger.error('max_rounds must be > 0')
        return 1
    
    try:
        experiment_dir = run_data_setup(config)
        logger.info(f"Initialized experiment in: {experiment_dir}")
    except Exception as e:
        logger.exception(f"Failed to initialize experiment: {e}")
        return 1
    
    max_rounds = config.get('max_rounds', 1)
    results = {'rounds': [], 'metrics_history': []}
    model_name = config.get('model_name', 'yolo11n.pt')
    is_yoloe = 'yoloe' in model_name.lower()
    
    if not is_yoloe:
        logger.info("=== Initial Training (Round 0) ===")
        try:
            start_time = time.time()
            metrics = run_training(config, experiment_dir, 0)
            results['rounds'].append(0)
            results['metrics_history'].append(metrics)
            end_time = time.time()
            logger.info(f"Initial training completed in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.exception(f"Initial training failed: {e}")
            return 1
    else:
        logger.info("=== YOLOE Model: Skipping initial training ===")
    
    for round_num in range(1, max_rounds + 1):
        try:
            logger.info(f"=== Round {round_num} ===")
            start_time = time.time()
            selected_indices = run_strategy(config, experiment_dir, round_num)
            
            if len(selected_indices) == 0:
                logger.info("No more samples to select. Stopping experiment.")
                break
            
            logger.info(f"Selected {len(selected_indices)} samples for labeling")
            end_time = time.time()
            logger.info(f"Strategy completed in {end_time - start_time:.2f} seconds")
            simulate_labeling_cli(selected_indices, config, experiment_dir, round_num)

            start_time = time.time()
            logger.info(f"Training model for round {round_num}...")
            metrics = run_training(config, experiment_dir, round_num)
            
            results['rounds'].append(round_num)
            results['metrics_history'].append(metrics)

            results_file = Path(experiment_dir) / "results.csv"
            with open(results_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=results.keys())
                writer.writeheader()
                for i in range(len(results['rounds'])):
                    writer.writerow({k: v[i] for k, v in results.items() if i < len(v)})

            end_time = time.time()
            logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

            if metrics:
                logger.info(f"Metrics for round {round_num}:\n{json.dumps(metrics, indent=2)}")
            else:
                logger.info(f"No metrics produced for round {round_num}")

        except Exception as e:
            logger.exception(f"Round {round_num} failed with error: {e}")
            continue
    
    logger.info("=== Experiment Complete ===")
    logger.info(f"Completed {len(results['rounds'])} rounds")
    logger.info(f"Results saved in: {experiment_dir}")
    
    if results['metrics_history']:
        if not is_yoloe and len(results['metrics_history']) > 1:
            initial_map = results['metrics_history'][0].get('map50', 0)
            final_map = results['metrics_history'][-1].get('map50', 0)
            logger.info(f"Initial mAP: {initial_map:.4f}")
            logger.info(f"Final mAP: {final_map:.4f}")
            logger.info(f"Improvement: {final_map - initial_map:.4f}")
        elif is_yoloe:
            final_map = results['metrics_history'][-1].get('map50', 0)
            logger.info(f"Final mAP: {final_map:.4f}")
    
    # Clean up temp config file
    if '_temp_config_path' in config:
        os.unlink(config['_temp_config_path'])
    
    return 0


if __name__ == "__main__":
    exit(main())
