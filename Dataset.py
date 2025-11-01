"""
Dataset preparation and preprocessing for CIFAR-100 with Vision Transformers.

This module handles downloading, splitting, and preprocessing of the CIFAR-100 dataset,
including appropriate data augmentation techniques for Vision Transformer training.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from typing import Tuple, Optional, Dict, Any
import urllib.request
import tarfile
from PIL import Image


class CIFAR100Dataset(Dataset):
    """
    Custom CIFAR-100 dataset class for Vision Transformer training.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize CIFAR-100 dataset.
        
        Args:
            data: Numpy array of images
            labels: Numpy array of labels
            transform: Optional transform to be applied on images
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, label)
        """
        img = self.data[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


class DataPreprocessor:
    """
    Handles data preprocessing and augmentation for Vision Transformers.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        mean: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408),
        std: Tuple[float, float, float] = (0.2675, 0.2565, 0.2761)
    ):
        """
        Initialize data preprocessor.
        
        Args:
            img_size: Target image size for ViT
            mean: Mean values for normalization (CIFAR-100 statistics)
            std: Standard deviation values for normalization
        """
        self.img_size = img_size
        self.mean = mean
        self.std = std
        
    def get_train_transforms(self) -> transforms.Compose:
        """
        Get training data transforms with augmentation.
        
        Returns:
            Composition of transforms for training
        """
        return transforms.Compose([
            # Resize to ViT input size
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.BICUBIC),
            
            # Data augmentation techniques suitable for ViT
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            
            # Random erasing (similar to cutout)
            transforms.RandomApply([
                transforms.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value='random'
                )
            ], p=0.25),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
    
    def get_val_test_transforms(self) -> transforms.Compose:
        """
        Get validation/test data transforms (no augmentation).
        
        Returns:
            Composition of transforms for validation/testing
        """
        return transforms.Compose([
            # Resize to ViT input size
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.BICUBIC),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])


def download_cifar100(data_dir: str = './data') -> None:
    """
    Download CIFAR-100 dataset if not already present.
    
    Args:
        data_dir: Directory to save the dataset
    """
    os.makedirs(data_dir, exist_ok=True)
    
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = os.path.join(data_dir, 'cifar-100-python.tar.gz')
    
    if not os.path.exists(filename):
        print(f"Downloading CIFAR-100 dataset from {url}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete!")
        
    # Extract if not already extracted
    extracted_dir = os.path.join(data_dir, 'cifar-100-python')
    if not os.path.exists(extracted_dir):
        print("Extracting dataset...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(data_dir)
        print("Extraction complete!")


def load_cifar100_batch(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a CIFAR-100 batch file.
    
    Args:
        file_path: Path to the batch file
        
    Returns:
        Tuple of (data, labels)
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    data = batch[b'data']
    labels = batch[b'fine_labels']
    
    # Reshape data to (num_samples, 3, 32, 32) then transpose to (num_samples, 32, 32, 3)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return data, np.array(labels)


def prepare_cifar100_data(
    data_dir: str = './data',
    val_split: float = 0.1,
    random_seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare CIFAR-100 dataset with train/val/test splits.
    
    Args:
        data_dir: Directory containing the dataset
        val_split: Fraction of training data to use for validation
        random_seed: Random seed for reproducible splits
        
    Returns:
        Dictionary with 'train', 'val', and 'test' keys containing (data, labels) tuples
    """
    # Download dataset if needed
    download_cifar100(data_dir)
    
    # Load training data
    train_file = os.path.join(data_dir, 'cifar-100-python', 'train')
    train_data, train_labels = load_cifar100_batch(train_file)
    
    # Load test data
    test_file = os.path.join(data_dir, 'cifar-100-python', 'test')
    test_data, test_labels = load_cifar100_batch(test_file)
    
    # Split training data into train and validation
    np.random.seed(random_seed)
    num_train = len(train_data)
    indices = np.random.permutation(num_train)
    
    val_size = int(num_train * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Create splits
    splits = {
        'train': (train_data[train_indices], train_labels[train_indices]),
        'val': (train_data[val_indices], train_labels[val_indices]),
        'test': (test_data, test_labels)
    }
    
    print(f"Dataset prepared:")
    print(f"  Training samples: {len(splits['train'][0])}")
    print(f"  Validation samples: {len(splits['val'][0])}")
    print(f"  Test samples: {len(splits['test'][0])}")
    
    return splits


def create_data_loaders(
    data_splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.
    
    Args:
        data_splits: Dictionary with data splits
        batch_size: Batch size for training
        img_size: Target image size for ViT
        num_workers: Number of worker processes for data loading
        
    Returns:
        Dictionary with DataLoaders for each split
    """
    preprocessor = DataPreprocessor(img_size=img_size)
    
    # Create datasets
    train_dataset = CIFAR100Dataset(
        data_splits['train'][0],
        data_splits['train'][1],
        transform=preprocessor.get_train_transforms()
    )
    
    val_dataset = CIFAR100Dataset(
        data_splits['val'][0],
        data_splits['val'][1],
        transform=preprocessor.get_val_test_transforms()
    )
    
    test_dataset = CIFAR100Dataset(
        data_splits['test'][0],
        data_splits['test'][1],
        transform=preprocessor.get_val_test_transforms()
    )
    
    # Create data loaders
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return loaders


def get_cifar100_class_names() -> list:
    """
    Get CIFAR-100 class names.
    
    Returns:
        List of class names
    """
    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]


if __name__ == "__main__":
    # Test dataset preparation
    print("Testing dataset preparation...")
    data_splits = prepare_cifar100_data()
    loaders = create_data_loaders(data_splits, batch_size=32, img_size=224)
    
    # Test a batch
    for batch_idx, (images, labels) in enumerate(loaders['train']):
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image min: {images.min():.3f}, max: {images.max():.3f}")
        break