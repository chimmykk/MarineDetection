"""
U-Net Architecture for Underwater Image Enhancement

Implements both standard and lightweight U-Net variants
optimized for CPU execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BN -> ReLU) * 2"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, 
                      padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block: Upsample/ConvTranspose -> Concat -> DoubleConv"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, 
                                   mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch due to pooling
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution: 1x1 conv to reduce channels"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for image enhancement.
    
    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output channels (3 for RGB)
        base_filters: Number of filters in first layer (doubled each level)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
    """
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 3,
        base_filters: int = 64,
        bilinear: bool = True
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        
        self.outc = OutConv(base_filters, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output with residual connection
        logits = self.outc(x)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net for CPU-optimized inference.
    
    Uses fewer filters and only 3 downsampling levels.
    
    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output channels (3 for RGB)
        base_filters: Number of filters in first layer
    """
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 3,
        base_filters: int = 32
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder (3 levels instead of 4)
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        
        # Decoder
        self.up1 = Up(base_filters * 8 + base_filters * 4, base_filters * 4, bilinear=True)
        self.up2 = Up(base_filters * 4 + base_filters * 2, base_filters * 2, bilinear=True)
        self.up3 = Up(base_filters * 2 + base_filters, base_filters, bilinear=True)
        
        self.outc = OutConv(base_filters, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)      # 32
        x2 = self.down1(x1)   # 64
        x3 = self.down2(x2)   # 128
        x4 = self.down3(x3)   # 256
        
        # Decoder with skip connections
        x = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.up1.conv(x)
        
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.up2.conv(x)
        
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.up3.conv(x)
        
        return self.outc(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualUNet(nn.Module):
    """
    U-Net with residual learning for image enhancement.
    
    Learns the difference between input and target, which is
    easier for enhancement tasks where output is similar to input.
    """
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 3,
        base_filters: int = 64,
        bilinear: bool = True
    ):
        super().__init__()
        self.unet = UNet(n_channels, n_classes, base_filters, bilinear)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Learn residual and add to input
        residual = self.unet(x)
        return torch.clamp(x + residual, 0, 1)


def get_model(
    model_type: str = 'standard',
    n_channels: int = 3,
    n_classes: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create U-Net models.
    
    Args:
        model_type: 'standard', 'lightweight', or 'residual'
        n_channels: Number of input channels
        n_classes: Number of output channels
        **kwargs: Additional model arguments
    
    Returns:
        Initialized model
    """
    models = {
        'standard': UNet,
        'lightweight': LightweightUNet,
        'residual': ResidualUNet
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return models[model_type](n_channels, n_classes, **kwargs)


if __name__ == '__main__':
    # Test model creation and forward pass
    print("Testing U-Net models...\n")
    
    device = torch.device('cpu')
    x = torch.randn(1, 3, 256, 256, device=device)
    
    for model_type in ['standard', 'lightweight', 'residual']:
        print(f"\n{model_type.upper()} U-Net:")
        
        if model_type == 'lightweight':
            model = get_model(model_type, base_filters=32)
        else:
            model = get_model(model_type, base_filters=64)
        
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            y = model(x)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Parameters:   {params:,}")
