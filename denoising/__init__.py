"""
Underwater Image Denoising and Enhancement Module

This module provides various image enhancement techniques optimized
for underwater imagery, including:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- White Balance correction
- Underwater dehazing
- U-Net based deep learning enhancement
"""

from .clahe import apply_clahe, batch_clahe
from .white_balance import gray_world, shades_of_gray, white_patch_retinex
from .dehazing import dark_channel_prior, underwater_dehaze

__all__ = [
    'apply_clahe',
    'batch_clahe',
    'gray_world',
    'shades_of_gray',
    'white_patch_retinex',
    'dark_channel_prior',
    'underwater_dehaze',
]
