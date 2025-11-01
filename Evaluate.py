"""
Evaluation module for Vision Transformer on CIFAR-100.

This module provides comprehensive evaluation metrics including accuracy,
confusion matrix, F1 score, precision, recall, and ROC-AUC scores.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import pandas as pd


class Evaluator:
    """
    Comprehensive evaluator for Vision Transformer on CIFAR-100.
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        class_names: List[str],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Vision Transformer model
            test_loader: Test data loader
            class_names: List of class names
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.num_classes = len(class_names)
        
        # Storage for predictions and labels
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
        
    def evaluate(self) -> Dict:
        """
        Perform comprehensive evaluation on test set.
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        print("Starting model evaluation...")
        self.model.eval()
        
        # Collect predictions
        self._collect_predictions()
        
        # Convert lists to numpy arrays
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        y_prob = np.array(self.all_probabilities)
        
        # Calculate metrics
        metrics = {
            'accuracy': self._calculate_accuracy(y_true, y_pred),
            'confusion_matrix': self._calculate_confusion_matrix(y_true, y_pred),
            'f1_scores': self._calculate_f1_scores(y_true, y_pred),
            'precision_scores': self._calculate_precision_scores(y_true, y_pred),
            'recall_scores': self._calculate_recall_scores(y_true, y_pred),
            'roc_auc_scores': self._calculate_roc_auc_scores(y_true, y_prob),
            'per_class_metrics': self._calculate_per_class_metrics(y_true, y_pred),
            'classification_report': self._generate_classification_report(y_true, y_pred)
        }
        
        return metrics
    
    def _collect_predictions(self) -> None:
        """Collect all predictions from the test set."""
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Collecting predictions"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get model outputs
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Store results
                self.all_predictions.extend(predicted.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
                self.all_probabilities.extend(probabilities.cpu().numpy())
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate overall accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Overall accuracy
        """
        accuracy = accuracy_score(y_true, y_pred) * 100
        print(f"Overall Accuracy: {accuracy:.2f}%")
        return accuracy
    
    def _calculate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix calculated")
        return cm
    
    def _calculate_f1_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate F1 scores.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with macro, micro, and weighted F1 scores
        """
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        f1_scores = {
            'macro': f1_macro,
            'micro': f1_micro,
            'weighted': f1_weighted
        }
        
        print(f"F1 Scores - Macro: {f1_macro:.4f}, Micro: {f1_micro:.4f}, Weighted: {f1_weighted:.4f}")
        
        return f1_scores
    
    def _calculate_precision_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate precision scores.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with macro, micro, and weighted precision scores
        """
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        
        precision_scores = {
            'macro': precision_macro,
            'micro': precision_micro,
            'weighted': precision_weighted
        }
        
        print(f"Precision Scores - Macro: {precision_macro:.4f}, "
              f"Micro: {precision_micro:.4f}, Weighted: {precision_weighted:.4f}")
        
        return precision_scores
    
    def _calculate_recall_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate recall scores.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with macro, micro, and weighted recall scores
        """
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        
        recall_scores = {
            'macro': recall_macro,
            'micro': recall_micro,
            'weighted': recall_weighted
        }
        
        print(f"Recall Scores - Macro: {recall_macro:.4f}, "
              f"Micro: {recall_micro:.4f}, Weighted: {recall_weighted:.4f}")
        
        return recall_scores
    
    def _calculate_roc_auc_scores(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate ROC-AUC scores using one-vs-rest approach.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary with macro and weighted ROC-AUC scores
        """
        # Convert labels to one-hot encoding for ROC-AUC calculation
        y_true_onehot = np.zeros((len(y_true), self.num_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        
        # Calculate ROC-AUC scores
        roc_auc_macro = roc_auc_score(
            y_true_onehot, y_prob, average='macro', multi_class='ovr'
        )
        roc_auc_weighted = roc_auc_score(
            y_true_onehot, y_prob, average='weighted', multi_class='ovr'
        )
        
        roc_auc_scores = {
            'macro': roc_auc_macro,
            'weighted': roc_auc_weighted
        }
        
        print(f"ROC-AUC Scores (OvR) - Macro: {roc_auc_macro:.4f}, "
              f"Weighted: {roc_auc_weighted:.4f}")
        
        return roc_auc_scores
    
    def _calculate_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            DataFrame with per-class metrics
        """
        # Calculate per-class metrics
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Calculate support (number of samples) per class
        support = np.bincount(y_true, minlength=self.num_classes)
        
        # Create DataFrame
        per_class_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class,
            'Support': support
        })
        
        # Sort by F1 score
        per_class_df = per_class_df.sort_values('F1-Score', ascending=False)
        
        print("\nTop 10 Best Performing Classes:")
        print(per_class_df.head(10).to_string(index=False))
        
        print("\nTop 10 Worst Performing Classes:")
        print(per_class_df.tail(10).to_string(index=False))
        
        return per_class_df
    
    def _generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4
        )
        return report
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: str = 'confusion_matrix.png',
        figsize: Tuple[int, int] = (20, 20)
    ) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the figure
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=False,  # Don't annotate cells for 100x100 matrix
            fmt='.2f',
            cmap='Blues',
            square=True,
            cbar_kws={'label': 'Normalized Count'},
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        
        plt.title('Normalized Confusion Matrix - CIFAR-100', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_per_class_performance(
        self,
        per_class_df: pd.DataFrame,
        save_path: str = 'per_class_performance.png'
    ) -> None:
        """
        Plot per-class performance metrics.
        
        Args:
            per_class_df: DataFrame with per-class metrics
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sort by F1 score for consistent ordering
        df_sorted = per_class_df.sort_values('F1-Score', ascending=True)
        
        # Plot F1 scores
        axes[0, 0].barh(range(len(df_sorted)), df_sorted['F1-Score'])
        axes[0, 0].set_xlabel('F1-Score')
        axes[0, 0].set_title('F1-Score by Class')
        axes[0, 0].set_ylim(-1, len(df_sorted))
        
        # Plot Precision
        axes[0, 1].barh(range(len(df_sorted)), df_sorted['Precision'])
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_title('Precision by Class')
        axes[0, 1].set_ylim(-1, len(df_sorted))
        
        # Plot Recall
        axes[1, 0].barh(range(len(df_sorted)), df_sorted['Recall'])
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_title('Recall by Class')
        axes[1, 0].set_ylim(-1, len(df_sorted))
        
        # Plot Support
        axes[1, 1].barh(range(len(df_sorted)), df_sorted['Support'])
        axes[1, 1].set_xlabel('Support (Number of Samples)')
        axes[1, 1].set_title('Support by Class')
        axes[1, 1].set_ylim(-1, len(df_sorted))
        
        # Remove y-axis labels for clarity (too many classes)
        for ax in axes.flat:
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Per-Class Performance Metrics - CIFAR-100', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Per-class performance plot saved to {save_path}")
    
    def save_metrics(self, metrics: Dict, filepath: str = 'evaluation_metrics.json') -> None:
        """
        Save evaluation metrics to JSON file.
        
        Args:
            metrics: Dictionary of evaluation metrics
            filepath: Path to save the metrics
        """
        # Convert numpy arrays and DataFrames to serializable format
        metrics_serializable = {}
        
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                metrics_serializable[key] = value.to_dict()
            elif isinstance(value, (dict, float, str)):
                metrics_serializable[key] = value
            else:
                metrics_serializable[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=4)
        
        print(f"Metrics saved to {filepath}")
    
    def generate_report(
        self,
        metrics: Dict,
        save_dir: str = './evaluation_results'
    ) -> None:
        """
        Generate comprehensive evaluation report with visualizations.
        
        Args:
            metrics: Dictionary of evaluation metrics
            save_dir: Directory to save report files
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics to JSON
        self.save_metrics(
            metrics,
            os.path.join(save_dir, 'evaluation_metrics.json')
        )
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            metrics['confusion_matrix'],
            os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # Plot per-class performance
        self.plot_per_class_performance(
            metrics['per_class_metrics'],
            os.path.join(save_dir, 'per_class_performance.png')
        )
        
        # Save classification report
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write("CIFAR-100 Vision Transformer Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.2f}%\n\n")
            f.write("F1 Scores:\n")
            f.write(f"  Macro: {metrics['f1_scores']['macro']:.4f}\n")
            f.write(f"  Micro: {metrics['f1_scores']['micro']:.4f}\n")
            f.write(f"  Weighted: {metrics['f1_scores']['weighted']:.4f}\n\n")
            f.write("Precision Scores:\n")
            f.write(f"  Macro: {metrics['precision_scores']['macro']:.4f}\n")
            f.write(f"  Micro: {metrics['precision_scores']['micro']:.4f}\n")
            f.write(f"  Weighted: {metrics['precision_scores']['weighted']:.4f}\n\n")
            f.write("Recall Scores:\n")
            f.write(f"  Macro: {metrics['recall_scores']['macro']:.4f}\n")
            f.write(f"  Micro: {metrics['recall_scores']['micro']:.4f}\n")
            f.write(f"  Weighted: {metrics['recall_scores']['weighted']:.4f}\n\n")
            f.write("ROC-AUC Scores (One-vs-Rest):\n")
            f.write(f"  Macro: {metrics['roc_auc_scores']['macro']:.4f}\n")
            f.write(f"  Weighted: {metrics['roc_auc_scores']['weighted']:.4f}\n\n")
            f.write("-" * 80 + "\n\n")
            f.write("Detailed Classification Report:\n\n")
            f.write(metrics['classification_report'])
        
        print(f"\nEvaluation report generated in {save_dir}/")


def evaluate_model(
    model_path: str,
    test_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    model_class: nn.Module,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Convenience function to evaluate a saved model.
    
    Args:
        model_path: Path to saved model checkpoint
        test_loader: Test data loader
        class_names: List of class names
        model_class: Model class to instantiate
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator and run evaluation
    evaluator = Evaluator(model, test_loader, class_names, device)
    metrics = evaluator.evaluate()
    evaluator.generate_report(metrics)
    
    return metrics