"""
Tests for Denoising Module

Run with: pytest tests/test_denoising.py -v
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from denoising.clahe import apply_clahe, adaptive_clahe, apply_clahe_rgb
from denoising.white_balance import gray_world, shades_of_gray, white_patch_retinex, combined_white_balance
from denoising.dehazing import dark_channel_prior, underwater_dehaze, simple_dehaze


# Fixtures

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create 100x100 image with blue-green cast (underwater simulation)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = 150  # Blue
    image[:, :, 1] = 100  # Green
    image[:, :, 2] = 50   # Red (attenuated)
    
    # Add some variation
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


@pytest.fixture
def dark_image():
    """Create a dark test image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 30


@pytest.fixture
def bright_image():
    """Create a bright test image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 220


# CLAHE Tests

class TestCLAHE:
    
    def test_apply_clahe_output_shape(self, sample_image):
        """Test that CLAHE preserves image shape."""
        result = apply_clahe(sample_image)
        assert result.shape == sample_image.shape
    
    def test_apply_clahe_output_type(self, sample_image):
        """Test that CLAHE returns uint8."""
        result = apply_clahe(sample_image)
        assert result.dtype == np.uint8
    
    def test_apply_clahe_increases_contrast(self, sample_image):
        """Test that CLAHE increases contrast."""
        result = apply_clahe(sample_image, clip_limit=4.0)
        
        # Calculate standard deviation (proxy for contrast)
        original_std = np.std(sample_image)
        result_std = np.std(result)
        
        # CLAHE should generally increase or maintain contrast
        assert result_std >= original_std * 0.8
    
    def test_apply_clahe_lab_vs_hsv(self, sample_image):
        """Test CLAHE with different color spaces."""
        lab_result = apply_clahe(sample_image, color_space='LAB')
        hsv_result = apply_clahe(sample_image, color_space='HSV')
        
        # Both should produce valid images
        assert lab_result.shape == sample_image.shape
        assert hsv_result.shape == sample_image.shape
        
        # Results should be different
        assert not np.array_equal(lab_result, hsv_result)
    
    def test_apply_clahe_invalid_colorspace(self, sample_image):
        """Test that invalid color space raises error."""
        with pytest.raises(ValueError):
            apply_clahe(sample_image, color_space='INVALID')
    
    def test_adaptive_clahe_dark_image(self, dark_image):
        """Test adaptive CLAHE on dark image."""
        result = adaptive_clahe(dark_image)
        
        # Should brighten dark image
        assert np.mean(result) >= np.mean(dark_image)
    
    def test_adaptive_clahe_bright_image(self, bright_image):
        """Test adaptive CLAHE on bright image."""
        result = adaptive_clahe(bright_image)
        
        # Should not over-enhance bright image
        assert result.max() <= 255


# White Balance Tests

class TestWhiteBalance:
    
    def test_gray_world_output_shape(self, sample_image):
        """Test that Gray World preserves shape."""
        result = gray_world(sample_image)
        assert result.shape == sample_image.shape
    
    def test_gray_world_reduces_color_cast(self, sample_image):
        """Test that Gray World reduces color cast."""
        result = gray_world(sample_image)
        
        # Original has strong blue cast
        orig_diff = abs(np.mean(sample_image[:, :, 0]) - np.mean(sample_image[:, :, 2]))
        result_diff = abs(np.mean(result[:, :, 0]) - np.mean(result[:, :, 2]))
        
        # Color difference should be reduced
        assert result_diff <= orig_diff
    
    def test_shades_of_gray_output_shape(self, sample_image):
        """Test Shades of Gray preserves shape."""
        result = shades_of_gray(sample_image, p=6.0)
        assert result.shape == sample_image.shape
    
    def test_white_patch_retinex(self, sample_image):
        """Test White Patch Retinex."""
        result = white_patch_retinex(sample_image)
        
        # At least one channel should reach near max
        assert result.max() > 200
    
    def test_combined_white_balance_auto(self, sample_image):
        """Test automatic method selection."""
        result = combined_white_balance(sample_image, method='auto')
        assert result.shape == sample_image.shape
    
    def test_combined_white_balance_invalid_method(self, sample_image):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            combined_white_balance(sample_image, method='invalid')


# Dehazing Tests

class TestDehazing:
    
    def test_dark_channel_prior_output_shape(self, sample_image):
        """Test DCP preserves shape."""
        result = dark_channel_prior(sample_image)
        assert result.shape == sample_image.shape
    
    def test_dark_channel_prior_output_type(self, sample_image):
        """Test DCP returns uint8."""
        result = dark_channel_prior(sample_image)
        assert result.dtype == np.uint8
    
    def test_underwater_dehaze_output_shape(self, sample_image):
        """Test underwater dehazing preserves shape."""
        result = underwater_dehaze(sample_image)
        assert result.shape == sample_image.shape
    
    def test_underwater_dehaze_red_enhancement(self, sample_image):
        """Test that underwater dehazing enhances red channel."""
        result = underwater_dehaze(sample_image, enhance_red=True)
        
        # Red channel should be boosted
        assert np.mean(result[:, :, 2]) >= np.mean(sample_image[:, :, 2])
    
    def test_simple_dehaze_output_shape(self, sample_image):
        """Test simple dehazing preserves shape."""
        result = simple_dehaze(sample_image, strength=0.5)
        assert result.shape == sample_image.shape
    
    def test_simple_dehaze_strength(self, sample_image):
        """Test simple dehazing strength parameter."""
        weak = simple_dehaze(sample_image, strength=0.2)
        strong = simple_dehaze(sample_image, strength=0.8)
        
        # Stronger dehazing should produce different result
        assert not np.array_equal(weak, strong)


# Integration Tests

class TestIntegration:
    
    def test_full_enhancement_pipeline(self, sample_image):
        """Test complete enhancement pipeline."""
        # Step 1: White balance
        balanced = combined_white_balance(sample_image, method='auto')
        
        # Step 2: CLAHE
        contrast = adaptive_clahe(balanced)
        
        # Step 3: Dehaze
        enhanced = underwater_dehaze(contrast)
        
        # Final image should be valid
        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == np.uint8
        assert enhanced.min() >= 0
        assert enhanced.max() <= 255
    
    def test_all_methods_on_random_image(self):
        """Test all methods work on random image."""
        # Create random image
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        # Test all methods
        _ = apply_clahe(image)
        _ = adaptive_clahe(image)
        _ = gray_world(image)
        _ = shades_of_gray(image)
        _ = white_patch_retinex(image)
        _ = dark_channel_prior(image)
        _ = underwater_dehaze(image)
        _ = simple_dehaze(image)
        
        # If we got here without errors, pass
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
