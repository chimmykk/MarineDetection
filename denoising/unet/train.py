"""
U-Net Training Script for Underwater Image Enhancement

CPU-optimized training with MSE + SSIM loss and PSNR/SSIM metrics.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Callable
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime

from .model import get_model


class UnderwaterDataset(Dataset):
    """
    Dataset for paired underwater images (degraded -> clean).
    
    Expects two directories with matching filenames:
    - input_dir: Degraded underwater images
    - target_dir: Clean/enhanced reference images
    """
    
    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = True
    ):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Find matching pairs
        self.pairs = self._find_pairs()
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(image_size)
    
    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching input-target image pairs."""
        pairs = []
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for input_path in self.input_dir.iterdir():
            if input_path.suffix.lower() not in extensions:
                continue
            
            # Look for matching target
            target_path = self.target_dir / input_path.name
            if target_path.exists():
                pairs.append((input_path, target_path))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_path, target_path = self.pairs[idx]
        
        # Load images
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        # Resize
        input_img = self.resize(input_img)
        target_img = self.resize(target_img)
        
        # Augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
                target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random vertical flip
            if np.random.random() > 0.5:
                input_img = input_img.transpose(Image.FLIP_TOP_BOTTOM)
                target_img = target_img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Convert to tensor [0, 1]
        input_tensor = self.to_tensor(input_img)
        target_tensor = self.to_tensor(target_img)
        
        return input_tensor, target_tensor


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss.
    
    SSIM measures perceptual similarity between images.
    """
    
    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)
    
    def _gaussian(self, window_size: int, sigma: float = 1.5) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        x = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create 2D Gaussian window for SSIM calculation."""
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)."""
        window = self.window.to(img1.device).type_as(img1)
        
        mu1 = nn.functional.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = nn.functional.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """Combined MSE + SSIM loss for image enhancement."""
    
    def __init__(self, mse_weight: float = 0.5, ssim_weight: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        ssim_loss = self.ssim(pred, target)
        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Structural Similarity Index."""
    ssim_loss = SSIMLoss()
    with torch.no_grad():
        ssim_value = 1 - ssim_loss(pred, target).item()
    return ssim_value


class UNetTrainer:
    """
    Trainer class for U-Net image enhancement.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model_type: str = 'standard',
        base_filters: int = 64,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        num_workers: int = 0,
        checkpoint_dir: str = 'checkpoints',
        device: Optional[str] = None
    ):
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Model
        self.model = get_model(model_type, base_filters=base_filters)
        self.model = self.model.to(self.device)
        
        # Training settings
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Loss and optimizer
        self.criterion = CombinedLoss(mse_weight=0.5, ssim_weight=0.5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_psnr': [],
            'val_psnr': [],
            'train_ssim': [],
            'val_ssim': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            
            # Loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                psnr = calculate_psnr(outputs, targets)
                ssim = calculate_ssim(outputs, targets)
            
            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr:.2f}',
                'ssim': f'{ssim:.4f}'
            })
        
        return {
            'loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches,
            'ssim': total_ssim / num_batches
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            
            loss = self.criterion(outputs, targets)
            psnr = calculate_psnr(outputs, targets)
            ssim = calculate_ssim(outputs, targets)
            
            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches,
            'ssim': total_ssim / num_batches
        }
    
    def train(
        self,
        train_input_dir: str,
        train_target_dir: str,
        val_input_dir: Optional[str] = None,
        val_target_dir: Optional[str] = None,
        epochs: int = 100,
        image_size: Tuple[int, int] = (256, 256),
        save_frequency: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_input_dir: Directory with training input images
            train_target_dir: Directory with training target images
            val_input_dir: Directory with validation input images (optional)
            val_target_dir: Directory with validation target images (optional)
            epochs: Number of training epochs
            image_size: Size to resize images to
            save_frequency: Save checkpoint every N epochs
        
        Returns:
            Training history dictionary
        """
        # Create datasets
        train_dataset = UnderwaterDataset(
            train_input_dir, train_target_dir,
            image_size=image_size, augment=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        
        # Validation loader
        val_loader = None
        if val_input_dir and val_target_dir:
            val_dataset = UnderwaterDataset(
                val_input_dir, val_target_dir,
                image_size=image_size, augment=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            print(f"Validation samples: {len(val_dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_psnr'].append(train_metrics['psnr'])
            self.history['train_ssim'].append(train_metrics['ssim'])
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"PSNR: {train_metrics['psnr']:.2f}, "
                  f"SSIM: {train_metrics['ssim']:.4f}")
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_psnr'].append(val_metrics['psnr'])
                self.history['val_ssim'].append(val_metrics['ssim'])
                
                print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"PSNR: {val_metrics['psnr']:.2f}, "
                      f"SSIM: {val_metrics['ssim']:.4f}")
                
                # Learning rate scheduler
                self.scheduler.step(val_metrics['loss'])
                
                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pth')
                    print("Saved best model!")
            
            # Periodic checkpoint
            if (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
    
    def save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train U-Net for underwater image enhancement')
    parser.add_argument('--train-input', required=True, help='Training input directory')
    parser.add_argument('--train-target', required=True, help='Training target directory')
    parser.add_argument('--val-input', help='Validation input directory')
    parser.add_argument('--val-target', help='Validation target directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model-type', choices=['standard', 'lightweight', 'residual'],
                        default='standard', help='Model type')
    parser.add_argument('--base-filters', type=int, default=64, help='Base filter count')
    parser.add_argument('--image-size', type=int, default=256, help='Image size')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    trainer = UNetTrainer(
        model_type=args.model_type,
        base_filters=args.base_filters,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir
    )
    
    trainer.train(
        train_input_dir=args.train_input,
        train_target_dir=args.train_target,
        val_input_dir=args.val_input,
        val_target_dir=args.val_target,
        epochs=args.epochs,
        image_size=(args.image_size, args.image_size)
    )
