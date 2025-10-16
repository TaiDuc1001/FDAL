import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.models.base import BaseModel
from src.models.yolo_model import YOLOModel


def create_model(model_path: str, categories: Optional[List[str]] = None) -> BaseModel:
    return YOLOModel(
        model_path=model_path if Path(model_path).exists() else None,
        model_name=model_path
    )

def plot_learning_curves(results_file: str, save_path: str = ""):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    rounds = results['rounds']
    metrics_history = results['metrics_history']
    
    map_values = []
    for metrics in metrics_history:
        map_val = metrics.get('map50-95', 0)
        if isinstance(map_val, str):
            map_val = float(map_val) if map_val != 'N/A' else 0
        map_values.append(map_val)
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, map_values, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Active Learning Round')
    plt.ylabel('mAP@0.5-0.95')
    plt.title('Active Learning Progress')
    plt.grid(True, alpha=0.3)
    if len(map_values) > 1:
        improvement = map_values[-1] - map_values[0]
        plt.text(0.02, 0.98, f'Improvement: {improvement:.4f}', 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to: {save_path}")
    else:
        plt.show()


def compare_strategies(experiment_dirs: List[str], 
                      strategy_names: List[str],
                      save_path: str = ""):
    plt.figure(figsize=(12, 8))
    
    for exp_dir, strategy_name in zip(experiment_dirs, strategy_names):
        results_file = Path(exp_dir) / "results.json"
        
        if not results_file.exists():
            print(f"Results file not found: {results_file}")
            continue
            
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        rounds = results['rounds']
        metrics_history = results['metrics_history']
        
        map_values = []
        for metrics in metrics_history:
            map_val = metrics.get('map50-95', 0)
            if isinstance(map_val, str):
                map_val = float(map_val) if map_val != 'N/A' else 0
            map_values.append(map_val)
        
        plt.plot(rounds, map_values, marker='o', linewidth=2, 
                label=strategy_name, markersize=6)
    
    plt.xlabel('Active Learning Round')
    plt.ylabel('mAP@0.5-0.95')
    plt.title('Active Learning Strategy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()


def analyze_dataset_statistics(dataset_path: str) -> Dict:
    from src.data.dataset import ALDataset
    dataset = ALDataset(dataset_path, "analysis", ["dummy"])
    
    stats = {
        'total_images': dataset.total_images,
        'labeled_images': len(dataset.get_labeled_indices()),
        'unlabeled_images': len(dataset.get_unlabeled_indices()),
    }
    
    return stats


def create_summary_report(experiment_dir: str, output_file: str = ""):
    exp_path = Path(experiment_dir)
    
    metadata_file = exp_path / "experiment_metadata.yaml"
    results_file = exp_path / "results.json"
    
    if not metadata_file.exists() or not results_file.exists():
        print("Required files not found for summary report")
        return
    
    import yaml
    
    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    report = []
    report.append("# Active Learning Experiment Report")
    report.append("")
    report.append("## Experiment Configuration")
    report.append(f"- Dataset: {metadata['dataset_name']}")
    report.append(f"- Strategy: {metadata['strategy_name']}")
    report.append(f"- Model: {metadata['model_name']}")
    report.append(f"- Initial Labeled: {metadata['initial_labeled_count']}")
    report.append(f"- Created: {metadata['created_at']}")
    report.append("")
    
    report.append("## Results Summary")
    if results['metrics_history']:
        initial_map = results['metrics_history'][0].get('map50-95', 0)
        final_map = results['metrics_history'][-1].get('map50-95', 0)
        improvement = final_map - initial_map if isinstance(final_map, (int, float)) and isinstance(initial_map, (int, float)) else 0
        
        report.append(f"- Rounds Completed: {len(results['rounds'])}")
        report.append(f"- Initial mAP: {initial_map:.4f}")
        report.append(f"- Final mAP: {final_map:.4f}")
        report.append(f"- Total Improvement: {improvement:.4f}")
    
    report.append("")
    report.append("## Round-by-Round Results")
    report.append("| Round | mAP@0.5-0.95 |")
    report.append("|-------|--------------|")
    
    for i, (round_num, metrics) in enumerate(zip(results['rounds'], results['metrics_history'])):
        map_val = metrics.get('map50-95', 'N/A')
        if isinstance(map_val, (int, float)):
            map_val = f"{map_val:.4f}"
        report.append(f"| {round_num} | {map_val} |")
    
    report_text = "\\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    else:
        print(report_text)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python utils.py <command> [args...]")
        print("Commands:")
        print("  plot <results_file> [save_path]")
        print("  compare <exp_dir1> <exp_dir2> ... --names <name1> <name2> ...")
        print("  report <experiment_dir> [output_file]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "plot":
        results_file = sys.argv[2]
        save_path = sys.argv[3] if len(sys.argv) > 3 else None
        plot_learning_curves(results_file, save_path) # type: ignore
        
    elif command == "report":
        exp_dir = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        create_summary_report(exp_dir, output_file) # type: ignore
        
    else:
        print(f"Unknown command: {command}")
