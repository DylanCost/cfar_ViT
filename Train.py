"""
Training module for Vision Transformer on CIFAR-100.

This module contains the training loop, validation procedures, and utilities
for training a Vision Transformer model on the CIFAR-100 dataset.
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """
    Configuration for training Vision Transformer.
    
    Hyperparameter Justification:
    - learning_rate: 3e-4 is standard for ViT training, provides stable convergence
    - batch_size: 128 balances memory usage and gradient stability
    - epochs: 100 allows sufficient training for CIFAR-100 complexity
    - weight_decay: 0.05 helps prevent overfitting on smaller dataset
    - warmup_epochs: 10 epochs for learning rate warmup helps stabilize training
    - label_smoothing: 0.1 improves generalization for 100-class classification
    """
    # Model configuration
    model_type: str = 'vit_small'  # Options: vit_tiny, vit_small, vit_base
    img_size: int = 224
    num_classes: int = 100
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    batch_size: int = 128
    epochs: int = 100
    warmup_epochs: int = 10
    
    # Regularization
    weight_decay: float = 0.05
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    
    # Optimizer settings
    optimizer: str = 'adamw'
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    # Scheduler settings
    scheduler: str = 'cosine'  # Options: cosine, linear, none
    
    # Training settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    seed: int = 42
    
    # Logging settings
    log_interval: int = 50
    save_checkpoint: bool = True
    checkpoint_dir: str = './checkpoints'
    tensorboard_dir: str = './runs'
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class Trainer:
    """
    Trainer class for Vision Transformer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: TrainingConfig
    ):
        """
        Initialize trainer.
        
        Args:
            model: Vision Transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        """
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup logging
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(config.tensorboard_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.epoch = 0
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        # Weight decay only for non-bias and non-normalization parameters
        params_with_wd = []
        params_without_wd = []
        
        for name, param in self.model.named_parameters():
            if 'bias' in name or 'norm' in name:
                params_without_wd.append(param)
            else:
                params_with_wd.append(param)
        
        param_groups = [
            {'params': params_with_wd, 'weight_decay': self.config.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}
        ]
        
        if self.config.optimizer == 'adamw':
            return optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                eps=self.config.eps
            )
        elif self.config.optimizer == 'adam':
            return optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                eps=self.config.eps
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                param_groups,
                lr=self.config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler."""
        if self.config.scheduler == 'none':
            return None
        
        # Calculate steps for warmup and main scheduler
        warmup_steps = self.config.warmup_epochs * len(self.train_loader)
        total_steps = self.config.epochs * len(self.train_loader)
        
        # Create warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        if self.config.scheduler == 'cosine':
            # Cosine annealing after warmup
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.min_learning_rate
            )
            
            # Combine warmup and main scheduler
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = warmup_scheduler
            
        return scheduler
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.config.epochs} [Train]"
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move data to device
            images = images.to(self.config.device)
            labels = labels.to(self.config.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update learning rate (per step)
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%',
                'lr': f'{self.get_current_lr():.6f}'
            })
            
            # Log to tensorboard
            if batch_idx % self.config.log_interval == 0:
                global_step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/Accuracy', accuracy, global_step)
                self.writer.add_scalar('Train/LearningRate', self.get_current_lr(), global_step)
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f"Epoch {self.epoch + 1}/{self.config.epochs} [Val]"
            )
            
            for batch_idx, (images, labels) in enumerate(pbar):
                # Move data to device
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update metrics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{accuracy:.2f}%'
                })
        
        return avg_loss, accuracy
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': asdict(self.config)
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{self.epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation accuracy: {self.best_val_acc:.2f}%")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.config.device}")
        print(f"Model: {self.config.model_type}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.get_current_lr())
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            self.writer.add_scalar('Epoch/TrainAcc', train_acc, epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
            self.writer.add_scalar('Epoch/ValAcc', val_acc, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {self.get_current_lr():.6f}")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            if self.config.save_checkpoint and (epoch + 1) % 10 == 0:
                self.save_checkpoint(is_best)
            elif is_best:
                self.save_checkpoint(is_best)
            
            print("-" * 80)
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time / 3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save final model
        self.save_checkpoint(is_best=False)
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.history


def save_training_logs(history: Dict[str, List[float]], filepath: str) -> None:
    """
    Save training logs to file.
    
    Args:
        history: Training history dictionary
        filepath: Path to save the logs
    """
    with open(filepath, 'w') as f:
        f.write("Vision Transformer Training Logs\n")
        f.write("=" * 80 + "\n\n")
        
        for epoch in range(len(history['train_loss'])):
            f.write(f"Epoch {epoch + 1}:\n")
            f.write(f"  Train Loss: {history['train_loss'][epoch]:.4f}\n")
            f.write(f"  Train Accuracy: {history['train_acc'][epoch]:.2f}%\n")
            f.write(f"  Val Loss: {history['val_loss'][epoch]:.4f}\n")
            f.write(f"  Val Accuracy: {history['val_acc'][epoch]:.2f}%\n")
            f.write(f"  Learning Rate: {history['learning_rates'][epoch]:.6f}\n")
            f.write("-" * 40 + "\n")
        
        # Summary statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("Summary Statistics:\n")
        f.write(f"  Best Val Accuracy: {max(history['val_acc']):.2f}%\n")
        f.write(f"  Best Val Accuracy Epoch: {np.argmax(history['val_acc']) + 1}\n")
        f.write(f"  Final Train Accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"  Final Val Accuracy: {history['val_acc'][-1]:.2f}%\n")
    
    print(f"Training logs saved to {filepath}")