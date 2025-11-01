# Vision Transformer for CIFAR-100 Classification

This repository contains a complete implementation of Vision Transformer (ViT) for image classification on the CIFAR-100 dataset. The implementation follows the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" and is built using PyTorch.

## Overview

The Vision Transformer model treats images as sequences of patches and processes them using a transformer encoder architecture. This implementation includes:

- Complete Vision Transformer architecture with configurable model sizes
- Data preprocessing and augmentation suitable for ViT training
- Comprehensive training pipeline with learning rate scheduling and regularization
- Extensive evaluation metrics including confusion matrix, F1 score, precision, recall, and ROC-AUC
- Visualization tools for results analysis

## Requirements

### Python Version
- Python 3.10 or higher

### Dependencies

Create a virtual environment and install the required packages:

```bash
# Create virtual environment
python -m venv vit_env

# Activate virtual environment
# On Linux/Mac:
source vit_env/bin/activate
# On Windows:
# vit_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
tqdm>=4.65.0
tensorboard>=2.13.0
Pillow>=9.5.0
```

## Project Structure

```
vision_transformer/
│
├── dataset.py          # Dataset loading and preprocessing
├── model.py           # Vision Transformer architecture
├── train.py           # Training procedures and utilities
├── evaluate.py        # Evaluation metrics and analysis
├── main.py           # Main training script
├── requirements.txt   # Package dependencies
└── README.md         # This file
```

## Installation and Setup

1. **Clone or download this repository**

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv vit_env
   source vit_env/bin/activate  # Linux/Mac
   # or
   vit_env\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
   # or
   pip install torch torchvision  # For CPU only
   
   pip install numpy scikit-learn matplotlib seaborn pandas tqdm tensorboard Pillow
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Usage

### Quick Start

To train the Vision Transformer model with default settings:

```bash
python main.py
```

### Custom Training

You can customize the training with various command-line arguments:

```bash
python main.py \
    --model_type vit_small \
    --batch_size 128 \
    --epochs 100 \
    --learning_rate 3e-4 \
    --img_size 224 \
    --data_dir ./data \
    --seed 42
```

### Available Arguments

- `--model_type`: Choose model size (`vit_tiny`, `vit_small`, `vit_base`)
  - `vit_tiny`: Fastest training, lower accuracy (192 dim, 3 heads)
  - `vit_small`: Balanced performance (384 dim, 6 heads) - **Recommended**
  - `vit_base`: Highest accuracy, slower training (768 dim, 12 heads)

- `--batch_size`: Batch size for training (default: 128)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Initial learning rate (default: 3e-4)
- `--img_size`: Input image size (default: 224)
- `--data_dir`: Directory for CIFAR-100 dataset (default: ./data)
- `--seed`: Random seed for reproducibility (default: 42)

## Training Process

### Step-by-Step Execution

1. **Dataset Preparation** (Part 1)
   - Automatically downloads CIFAR-100 if not present
   - Splits data into train (45,000), validation (5,000), and test (10,000)

2. **Data Preprocessing** (Part 2)
   - Resizes images from 32x32 to 224x224 (or specified size)
   - Applies data augmentation:
     - Random horizontal flip
     - Random rotation (±15°)
     - Color jittering
     - Random erasing (cutout)
   - Normalizes with CIFAR-100 statistics

3. **Model Setup** (Part 3)
   - Initializes Vision Transformer architecture
   - Patch size: 16x16
   - Adds [CLS] token and positional embeddings
   - Configures transformer encoder blocks

4. **Training** (Part 4)
   - Uses AdamW optimizer with weight decay
   - Cosine annealing learning rate schedule with warmup
   - Label smoothing for better generalization
   - Gradient clipping for stability
   - Saves checkpoints every 10 epochs

5. **Evaluation** (Part 5 & 6)
   - Calculates comprehensive metrics on test set:
     - Overall accuracy
     - Confusion matrix (100x100)
     - F1 scores (macro, micro, weighted)
     - Precision and recall
     - ROC-AUC scores (one-vs-rest)
     - Per-class performance analysis

## Output Files

After training, the following files and directories will be created:

### Checkpoints (`./checkpoints/`)
- `best_model.pt`: Best model based on validation accuracy
- `checkpoint_epoch_N.pt`: Checkpoints at every 10 epochs

### Logs (`./logs/`)
- `training_log_TIMESTAMP.txt`: Detailed training logs with loss and accuracy for each epoch

### Evaluation Results (`./evaluation_results/`)
- `evaluation_metrics.json`: All metrics in JSON format
- `confusion_matrix.png`: Visualization of 100x100 confusion matrix
- `per_class_performance.png`: Bar charts of per-class metrics
- `classification_report.txt`: Detailed classification report

### TensorBoard Logs (`./runs/`)
- Real-time training metrics viewable with:
  ```bash
  tensorboard --logdir ./runs
  ```

### Configuration (`./configs/`)
- `config_TIMESTAMP.json`: Training configuration for reproducibility

## Model Architecture Details

The Vision Transformer consists of:

1. **Patch Embedding Layer**
   - Splits image into 16x16 patches
   - Projects patches to embedding dimension

2. **Transformer Encoder**
   - Multi-head self-attention mechanism
   - MLP blocks with GELU activation
   - Layer normalization and residual connections
   - Configurable depth (12 blocks by default)

3. **Classification Head**
   - Uses [CLS] token representation
   - Linear projection to 100 classes

### Model Configurations

| Model      | Embed Dim | Heads | Depth | Parameters | Memory  |
|------------|-----------|-------|-------|------------|---------|
| ViT-Tiny   | 192       | 3     | 12    | ~5.5M      | ~1.5GB  |
| ViT-Small  | 384       | 6     | 12    | ~21.7M     | ~3GB    |
| ViT-Base   | 768       | 12    | 12    | ~86.4M     | ~8GB    |

## Performance Expectations

With the default configuration (ViT-Small, 100 epochs):

- **Training time**: 4-6 hours on GPU (V100/A100)
- **Validation accuracy**: 70-75%
- **Test accuracy**: 68-73%
- **F1 Score (Macro)**: 0.68-0.72
- **ROC-AUC (Macro)**: 0.94-0.96

**Note**: The assignment requires a minimum of 65% test accuracy.

## Tips for Better Performance

1. **For faster training**:
   - Use `vit_tiny` model
   - Reduce image size to 112 or 56
   - Decrease batch size if memory limited

2. **For higher accuracy**:
   - Train for more epochs (150-200)
   - Use `vit_base` model if GPU memory allows
   - Experiment with learning rate (1e-4 to 5e-4)
   - Increase image size to 256

3. **For limited GPU memory**:
   - Reduce batch size (64 or 32)
   - Use gradient accumulation
   - Use `vit_tiny` model
   - Enable mixed precision training

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./runs
```
Then open http://localhost:6006 in your browser

### Training Logs
Training progress is printed to console and saved to log files showing:
- Epoch-wise train/validation loss
- Train/validation accuracy
- Current learning rate
- Estimated time remaining

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Use smaller model (`vit_tiny`)
- Reduce image size
- Enable gradient checkpointing

### Low Accuracy
- Train for more epochs
- Adjust learning rate
- Check data augmentation
- Ensure proper normalization

### Slow Training
- Use GPU if available
- Reduce image size
- Use smaller model
- Increase number of workers in DataLoader

## Code Quality

All code has been:
- Properly commented with docstrings
- Structured with object-oriented design
- Type-hinted for clarity
- Linted to meet quality standards

To check code quality:
```bash
pylint vision_transformer/*.py
```

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [PyTorch Vision Transformer Documentation](https://pytorch.org/vision/main/models/vision_transformer.html)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## License

This project is created for educational purposes as part of an assignment.

## Contact

For questions or issues with the code, please refer to the documentation or create an issue in the repository.