import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env.training')
if cuda_visible_devices is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import create_model
from src.strategies.uncertainty.dcus_patching import patching

def parse_args():
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--dataset_yaml', type=str, required=True, help='Path to dataset YAML configuration')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Model path or name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--save_dir', type=str, default='runs/train', help='Directory to save training results')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, 0, 1, etc.) or comma-separated list for multiple GPUs')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loader workers')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--model_path_file', type=str, help='File to save the trained model path')
    parser.add_argument('--strategy', type=str, help='Name of strategy (can be single or comma-separated chain)')

    return parser.parse_args()


def main():
    args = parse_args()
    
    if not Path(args.dataset_yaml).exists():
        print(f"Dataset YAML not found: {args.dataset_yaml}")
        return 1
    
    print("Training Configuration:")
    print(f"  Dataset YAML: {args.dataset_yaml}")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print("="*50)
    
    device = args.device
    if ',' in device:
        device = [d.strip() for d in device.split(',')]
    
    try:
        model = create_model(args.model)
        if args.strategy:
            strategies = []
            if '-' in args.strategy:
                strategies = [s.strip() for s in args.strategy.split('-')]
            else:
                strategies = [args.strategy]
            
            if any('dcus' in strategy.lower() for strategy in strategies):
                model = patching(model) # type: ignore
        
        print("Starting training...")
        trained_model = model.train( # type: ignore
            data_yaml=args.dataset_yaml,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            save_dir=args.save_dir,
            patience=args.patience,
            device=device,
            project=args.project,
            name=args.name
        )
        print(trained_model.model.model.names) # type: ignore
        
        print("Training completed successfully!")
        if trained_model.model_path: # type: ignore
            print(f"Best model saved at: {trained_model.model_path}") # type: ignore
            
            if args.model_path_file:
                model_path_file = Path(args.model_path_file)
                model_path_file.parent.mkdir(parents=True, exist_ok=True)
                with open(model_path_file, 'w') as f:
                    f.write(trained_model.model_path) # type: ignore
        else:
            print("Warning: Model path not set after training")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
