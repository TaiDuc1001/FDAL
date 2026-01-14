import argparse
import shutil
from pathlib import Path

import yaml


def parse_selection_log(log_path: Path) -> list[str]:
    images = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                images.extend([img.strip() for img in line.split(",") if img.strip()])
    return images


def get_image_source_dirs(config_path: Path) -> list[Path]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_yaml_path = Path(config["dataset_yaml"])
    project_root = config_path.parent
    while project_root.name and not (project_root / "pyproject.toml").exists():
        project_root = project_root.parent
    if not dataset_yaml_path.is_absolute():
        dataset_yaml_path = project_root / dataset_yaml_path
    
    with open(dataset_yaml_path, "r") as f:
        dataset_config = yaml.safe_load(f)
    
    train_value = dataset_config.get("train", "")
    if isinstance(train_value, str):
        train_value = [train_value]
    
    train_paths = []
    for tv in train_value:
        train_path = Path(tv)
        if not train_path.is_absolute():
            train_path = project_root / train_path
        train_paths.append(train_path)
    
    return train_paths


def find_image(img_name: str, source_dirs: list[Path]) -> Path | None:
    for source_dir in source_dirs:
        src_path = source_dir / img_name
        if src_path.exists():
            return src_path
    return None


def copy_images(
    selection_log: Path,
    source_dirs: list[Path],
    output_dir: Path,
    by_round: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(selection_log, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    copied_count = 0
    missing_count = 0
    
    for round_idx, line in enumerate(lines):
        images = [img.strip() for img in line.split(",") if img.strip()]
        
        if by_round:
            round_dir = output_dir / f"round_{round_idx}"
            round_dir.mkdir(parents=True, exist_ok=True)
            target_dir = round_dir
        else:
            target_dir = output_dir
        
        for img_name in images:
            src_path = find_image(img_name, source_dirs)
            dst_path = target_dir / img_name
            
            if src_path:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                print(f"Warning: Image not found in any source dir: {img_name}")
                missing_count += 1
    
    print(f"Copied {copied_count} images to {output_dir}")
    if missing_count > 0:
        print(f"Missing: {missing_count} images")


def main():
    parser = argparse.ArgumentParser(
        description="Copy selected images from selection_log.txt to a destination folder"
    )
    parser.add_argument(
        "selection_log",
        type=Path,
        help="Path to selection_log.txt",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to experiment config YAML (e.g., configs/voc/config_fdal.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: same folder as selection_log with name 'selected_images')",
    )
    parser.add_argument(
        "--by-round",
        action="store_true",
        help="Organize images into subdirectories by round",
    )
    
    args = parser.parse_args()
    
    selection_log = args.selection_log.resolve()
    config_path = args.config.resolve()
    
    if args.output:
        output_dir = args.output.resolve()
    else:
        output_dir = selection_log.parent / "selected_images"
    
    source_dirs = get_image_source_dirs(config_path)
    print(f"Source image directories: {source_dirs}")
    print(f"Output directory: {output_dir}")
    
    copy_images(selection_log, source_dirs, output_dir, args.by_round)


if __name__ == "__main__":
    main()
