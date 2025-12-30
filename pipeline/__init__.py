"""
Pipeline Module

Provides end-to-end processing combining image enhancement
and marine life detection.
"""

from .preprocess import ImagePreprocessor, VideoPreprocessor
from .run_pipeline import UnderwaterPipeline

__all__ = [
    'ImagePreprocessor',
    'VideoPreprocessor',
    'UnderwaterPipeline',
]
