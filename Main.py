"""
Main script for training Vision Transformer on CIFAR-100.

This script orchestrates the entire training pipeline including data preparation,
model initialization, training, and evaluation.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# Import custom modules
from dataset import prepare_cifar100_data, create_data_loaders, get_cifar100_class_names
from model import create_vit_tiny, create_vit_small, create_vit_base
from train import Trainer, TrainingConfig, save_training_logs
from evaluate import Evaluator


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config: TrainingConfig) -> nn.Module:
    """
    Create Vision Transformer model based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Vision Transformer model
    """
    model_creators = {
        'vit_tiny': create_vit_tiny,
        'vit_small': create_vit_small,
        'vit_base': create_vit_base
    }
    
    if config.model_type not in model_creators:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    model = model_creators[config.model_type](
        num_classes=config.num_classes,
        img_size=config.img_size
    )
    
    return model


def main(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Create configuration
    config = TrainingConfig(
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        img_size=args.img_size,
        seed=args.seed
    )
    
    # Set random seed
    set_seed(config.seed)
    
    # Print configuration
    print("=" * 80)
    print("Vision Transformer Training on CIFAR-100")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Model: {config.model_type}")
    print(f"  Image Size: {config.img_size}x{config.img_size}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Device: {config.device}")
    print("=" * 80)
    
    # Save configuration
    os.makedirs('configs', exist_ok=True)
    config_path = f"configs/config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    config.save(config_path)
    print(f"\nConfiguration saved to {config_path}")
    
    # Step 1: Prepare dataset
    print("\n" + "=" * 80)
    print("Step 1: Dataset Preparation")
    print("=" * 80)
    data_splits = prepare_cifar100_data(
        data_dir=args.data_dir,
        val_split=0.1,
        random_seed=config.seed
    )
    
    # Step 2: Create data loaders
    print("\n" + "=" * 80)
    print("Step 2: Data Preprocessing")
    print("=" * 80)
    data_loaders = create_data_loaders(
        data_splits,
        batch_size=config.batch_size,
        img_size=config.img_size,
        num_workers=config.num_workers
    )
    print(f"Data loaders created with batch size {config.batch_size}")
    
    # Step 3: Initialize model
    print("\n" + "=" * 80)
    print("Step 3: Model Initialization")
    print("=" * 80)
    model = create_model(config)
    print(f"Model initialized: {config.model_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 4: Training
    print("\n" + "=" * 80)
    print("Step 4: Training Procedure")
    print("=" * 80)
    trainer = Trainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        config=config
    )
    
    # Train the model
    history = trainer.train()
    
    # Step 5: Save training logs
    print("\n" + "=" * 80)
    print("Step 8: Training Logs")
    print("=" * 80)
    os.makedirs('logs', exist_ok=True)
    log_path = f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    save_training_logs(history, log_path)
    
    # Step 6: Evaluation on test set
    print("\n" + "=" * 80)
    print("Step 5: Evaluation Procedure")
    print("=" * 80)
    
    # Load best model
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(best_model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
    
    # Create evaluator
    class_names = get_cifar100_class_names()
    evaluator = Evaluator(
        model=model,
        test_loader=data_loaders['test'],
        class_names=class_names,
        device=config.device
    )
    
    # Run evaluation
    print("\n" + "=" * 80)
    print("Step 6: Metrics and Analysis")
    print("=" * 80)
    metrics = evaluator.evaluate()
    
    # Generate comprehensive report
    evaluator.generate_report(metrics, save_dir='evaluation_results')
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training and Evaluation Complete!")
    print("=" * 80)
    print(f"Best Validation Accuracy: {checkpoint['best_val_acc']:.2f}%")
    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    print(f"F1 Score (Macro): {metrics['f1_scores']['macro']:.4f}")
    print(f"ROC-AUC (Macro): {metrics['roc_auc_scores']['macro']:.4f}")
    print("\nResults saved in:")
    print(f"  - Checkpoints: {config.checkpoint_dir}/")
    print(f"  - Logs: logs/")
    print(f"  - Evaluation: evaluation_results/")
    print("=" * 80)
    
    # Check if we met the minimum accuracy requirement
    if metrics['accuracy'] >= 65.0:
        print("\n✓ SUCCESS: Achieved minimum required accuracy of 65%!")
    else:
        print(f"\n✗ WARNING: Accuracy {metrics['accuracy']:.2f}% is below the required 65%")
        print("  Consider training for more epochs or adjusting hyperparameters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Vision Transformer on CIFAR-100"
    )
    
    # Model arguments
    parser.add_argument(
        '--model_type',
        type=str,
        default='vit_small',
        choices=['vit_tiny', 'vit_small', 'vit_base'],
        help='Type of Vision Transformer model'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Directory to download/load CIFAR-100 dataset'
    )
    
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help='Image size for Vision Transformer input'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3e-4,
        help='Initial learning rate'
    )
    
    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Run training
    main(args)