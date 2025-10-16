# FeDAL: Active Learning from Feature Instability

**Official implementation** of the paper "From Object Instability to Image Uncertainty: A Strategy for Active Learning in Object Detection"

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

> 🎯 **TL;DR**: A comprehensive active learning framework for object detection that intelligently selects the most informative samples to annotate, reducing labeling costs while maintaining high model performance.

## 📖 About

FeDAL (Active Learning from Feature Instability) introduces a novel uncertainty estimation approach based on feature instability for active learning in object detection. This repository provides the complete implementation along with support for multiple established active learning strategies and datasets.

### Key Features

- 🚀 **State-of-the-art FeDAL Strategy**: Our novel feature instability-based approach
- 📊 **Multiple AL Strategies**: Uncertainty-based, diversity-based, and hybrid methods
- 🎯 **Multi-Dataset Support**: COCO, VOC, Cityscapes, KITTI
- ⚡ **YOLO Integration**: Built on ultralytics YOLO models
- 🔧 **Easy Configuration**: YAML-based experiment setup
- 📈 **Comprehensive Evaluation**: Built-in metrics and visualization tools

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.9 or higher (as specified in `.python-version`)
- CUDA-compatible GPU (recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/TaiDuc1001/FeDAL.git
cd FeDAL
```

2. **Install with uv** (recommended):
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys (Kaggle, Weights & Biases)
```

### Download Dataset

```bash
# Download COCO dataset
uv run python scripts/down_data.py --dataset coco

# Or download VOC dataset
uv run python scripts/down_data.py --dataset VOC
```

### Run Your First Experiment

```bash
# Run FeDAL strategy on VOC dataset
uv run python scripts/run_experiment.py --config configs/voc/config_fedal.yaml

# Run baseline random sampling for comparison
uv run python scripts/run_experiment.py --config configs/voc/config_random.yaml
```

## 🧠 Supported Active Learning Strategies

### Our Method
- **FeDAL** 🌟: Active Learning from Feature Instability (our paper's contribution)

### Uncertainty-based Strategies
- **Entropy**: Prediction entropy-based selection
- **BADGE**: Batch Active learning by Diverse Gradient Embeddings
- **CDAL**: Contextual Diversity Active Learning
- **DCUS**: Difficulty Calibrated Uncertainty Sampling

### Diversity-based Strategies
- **CoreSet**: Core-set based active learning
- **DivProto**: Diversity Prototype
- **CCMS**: Category Conditioned Matching Similarity

### Baseline & Hybrid
- **Random**: Random sample selection baseline
- **PPAL**: Plug and Play Active Learning (DCUS + CCMS)

## 📂 Project Structure

```
FeDAL/
├── configs/                    # Experiment configurations
│   ├── coco/                  # COCO dataset configs
│   ├── voc/                   # VOC dataset configs
│   ├── cityscapes/            # Cityscapes configs
│   └── kitti/                 # KITTI configs
├── scripts/                   # Main execution scripts
│   ├── run_experiment.py      # 🎯 Main experiment runner
│   ├── train.py              # Model training
│   ├── strategy.py           # Strategy execution
│   ├── down_data.py          # Dataset downloader
│   ├── setup_data.py         # Data setup
│   ├── simulate_labeling.py  # Labeling simulation
│   └── utils.py              # Utility functions
├── src/                      # Source code
│   ├── data/                 # Data loading and management
│   ├── models/               # YOLO model implementations
│   └── strategies/           # Active learning strategies
│       ├── uncertainty/      # Uncertainty-based methods
│       ├── diversity/        # Diversity-based methods
│       ├── intrinsic/        # Intrinsic methods
│       └── random/           # Random baseline
└── pyproject.toml            # uv/pip dependencies
```

## 🔧 Usage

### Basic Experiment

```bash
# Run an experiment with default settings
uv run python scripts/run_experiment.py --config configs/voc/config_fedal.yaml
```

### Custom Configuration

```bash
# Override specific parameters
uv run python scripts/run_experiment.py \
    --config configs/voc/config_fedal.yaml \
    --epochs 50 \
    --batch_size 32 \
    --max_rounds 5 \
    --device 0
```

### Strategy Comparison

```bash
# Run multiple strategies for comparison
uv run python scripts/run_experiment.py --config configs/voc/config_fedal.yaml
uv run python scripts/run_experiment.py --config configs/voc/config_entropy.yaml
uv run python scripts/run_experiment.py --config configs/voc/config_random.yaml

# Generate comparison plots
uv run python scripts/utils.py compare \
    experiments/fedal_exp experiments/entropy_exp experiments/random_exp \
    --names FeDAL Entropy Random
```

### Individual Components

```bash
# Train a model
uv run python scripts/train.py \
    --dataset_yaml datasets/VOC/data.yaml \
    --model yolo11s.pt \
    --epochs 26

# Run strategy selection only
uv run python scripts/strategy.py \
    --dataset_yaml datasets/VOC/data.yaml \
    --strategy fedal \
    --model path/to/model.pt \
    --num_samples 414
```

## ⚙️ Configuration

Each strategy has its own configuration file in `configs/`. Here's an example from the actual codebase:

```yaml
# configs/voc/config_fedal.yaml
dataset_yaml: "datasets/VOC/data.yaml"
epochs: 26
batch_size: 16
val_batch_size: 256
imgsz: 640
device: ['0']
strategy: "fedal"
model_name: "yolo11s.pt"
initial_labeled_count: 828
max_rounds: 7
samples_per_round: 414
num_inference: 6000

strategy_args:
  fedal:
    supporter: "resnet18"
    supporter_epochs: 5
    supporter_batch_size: 1000000
    learn_alpha: true
    alpha_cap: 0.03125
    alpha_learning_rate: 0.1
    lambda_hyp: 0.99
    supporter_embedding_size: 32
    supporter_imgsz: 320
    one_alpha_cap: false
```

### Key Parameters

- `initial_labeled_count`: Number of initially labeled samples
- `samples_per_round`: Samples to select in each active learning round
- `max_rounds`: Maximum number of active learning rounds
- `strategy`: Active learning strategy to use
- `strategy_args`: Strategy-specific hyperparameters

## 📊 Evaluation & Results

### Metrics
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5-0.95**: Mean Average Precision averaged over IoU thresholds
- **Learning Curves**: Performance vs. annotation budget

### Visualization
```bash
# Plot learning curves
uv run python scripts/utils.py plot experiments/your_exp/results.json

# Generate experiment report
uv run python scripts/utils.py report experiments/your_exp
```

## 🎯 FeDAL Method Details

Our FeDAL strategy leverages **feature instability** to identify the most informative samples:

1. **Feature Extraction**: Extract features from multiple layers of the trained model
2. **Instability Measurement**: Compute feature instability using supporter networks
3. **Uncertainty Quantification**: Convert instability to uncertainty scores
4. **Sample Selection**: Select samples with highest uncertainty for annotation

Key advantages:
- ✅ Better correlation with true uncertainty than entropy-based methods
- ✅ Captures model's confusion about both classification and localization
- ✅ Robust across different datasets and model architectures

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

<!-- ## 📚 Citation

If you use FeDAL in your research, please cite our paper:

```bibtex
@article{fedal2024,
  title={Active Learning from Feature Instability},
  author={TaiDuc1001},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
``` -->

## 🙏 Acknowledgments

- Built on top of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Inspired by recent advances in active learning research
- Special thanks to the open-source computer vision community


⭐ **Star this repository if you find it useful!**
