"""
Vision Transformer (ViT) Model Implementation for CIFAR-100 Classification.

This module implements the Vision Transformer architecture as described in
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".
The implementation is adapted for CIFAR-100 classification with 100 classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math


class PatchEmbedding(nn.Module):
    """
    Converts input images into patch embeddings.
    
    Splits the image into fixed-size patches and linearly embeds them.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        """
        Initialize patch embedding layer.
        
        Args:
            img_size: Size of input image (assumes square images)
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Dimension of patch embeddings
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through patch embedding.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim)
        """
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0
    ):
        """
        Initialize multi-head self-attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(
            batch_size, seq_len, embed_dim
        )
        
        # Final projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with multi-head attention and MLP.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0
    ):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout probability
            attn_dropout: Attention dropout probability
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for CIFAR-100 classification.
    
    Architecture Overview:
    1. Split image into fixed-size patches
    2. Linearly embed patches
    3. Add position embeddings and optional [CLS] token
    4. Process through transformer encoder blocks
    5. Use [CLS] token or average pooling for classification
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 100,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0
    ):
        """
        Initialize Vision Transformer.
        
        Args:
            img_size: Input image size
            patch_size: Size of image patches
            in_channels: Number of input channels
            num_classes: Number of classification classes (100 for CIFAR-100)
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout probability
            attn_dropout: Attention dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers and layer norms
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through Vision Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm
        x = self.norm(x)
        
        # Extract [CLS] token representation
        cls_output = x[:, 0]
        
        # Classification head
        logits = self.head(cls_output)
        
        return logits


def create_vit_base(num_classes: int = 100, img_size: int = 224) -> VisionTransformer:
    """
    Create ViT-Base model for CIFAR-100.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        
    Returns:
        VisionTransformer model
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes
    )


def create_vit_small(num_classes: int = 100, img_size: int = 224) -> VisionTransformer:
    """
    Create smaller ViT model for CIFAR-100 (more suitable for smaller datasets).
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        
    Returns:
        VisionTransformer model
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes
    )


def create_vit_tiny(num_classes: int = 100, img_size: int = 224) -> VisionTransformer:
    """
    Create tiny ViT model for CIFAR-100 (faster training).
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        
    Returns:
        VisionTransformer model
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes
    )