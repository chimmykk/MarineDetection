"""
White Balance Correction for Underwater Images

Underwater images suffer from color cast due to selective absorption
of light wavelengths. This module provides multiple algorithms to
correct color balance:
- Gray World
- Shades of Gray
- White Patch Retinex
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional


def gray_world(image: np.ndarray) -> np.ndarray:
    """
    Apply Gray World white balance algorithm.
    
    Assumes that the average color of a scene should be gray.
    Particularly effective for underwater images with strong color cast.
    
    Args:
        image: Input BGR image (uint8)
    
    Returns:
        White-balanced BGR image
    """
    # Convert to float for calculations
    img_float = image.astype(np.float32)
    
    # Calculate mean of each channel
    b_mean = np.mean(img_float[:, :, 0])
    g_mean = np.mean(img_float[:, :, 1])
    r_mean = np.mean(img_float[:, :, 2])
    
    # Calculate overall mean
    overall_mean = (b_mean + g_mean + r_mean) / 3.0
    
    # Calculate scaling factors
    b_scale = overall_mean / (b_mean + 1e-6)
    g_scale = overall_mean / (g_mean + 1e-6)
    r_scale = overall_mean / (r_mean + 1e-6)
    
    # Apply scaling
    result = img_float.copy()
    result[:, :, 0] = np.clip(result[:, :, 0] * b_scale, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * g_scale, 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * r_scale, 0, 255)
    
    return result.astype(np.uint8)


def shades_of_gray(image: np.ndarray, p: float = 6.0) -> np.ndarray:
    """
    Apply Shades of Gray white balance algorithm.
    
    A generalization of Gray World that uses Minkowski norm.
    p=1: Gray World, p=6: recommended for natural images, p→∞: Max-RGB
    
    Args:
        image: Input BGR image (uint8)
        p: Minkowski norm power (6.0 is commonly used)
    
    Returns:
        White-balanced BGR image
    """
    img_float = image.astype(np.float64) + 1e-6  # Avoid division by zero
    
    # Calculate Minkowski mean for each channel
    b_mean = np.power(np.mean(np.power(img_float[:, :, 0], p)), 1/p)
    g_mean = np.power(np.mean(np.power(img_float[:, :, 1], p)), 1/p)
    r_mean = np.power(np.mean(np.power(img_float[:, :, 2], p)), 1/p)
    
    # Calculate overall mean
    overall_mean = np.power((b_mean**p + g_mean**p + r_mean**p) / 3.0, 1/p)
    
    # Calculate scaling factors
    b_scale = overall_mean / b_mean
    g_scale = overall_mean / g_mean
    r_scale = overall_mean / r_mean
    
    # Apply scaling
    result = image.astype(np.float64)
    result[:, :, 0] = np.clip(result[:, :, 0] * b_scale, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * g_scale, 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * r_scale, 0, 255)
    
    return result.astype(np.uint8)


def white_patch_retinex(
    image: np.ndarray,
    percentile: float = 99.0
) -> np.ndarray:
    """
    Apply White Patch Retinex white balance algorithm.
    
    Assumes the brightest pixels in the image should be white.
    Uses percentile to avoid outliers affecting the result.
    
    Args:
        image: Input BGR image (uint8)
        percentile: Percentile to use as white reference (default 99%)
    
    Returns:
        White-balanced BGR image
    """
    img_float = image.astype(np.float32)
    
    # Find the white reference using percentile
    b_max = np.percentile(img_float[:, :, 0], percentile)
    g_max = np.percentile(img_float[:, :, 1], percentile)
    r_max = np.percentile(img_float[:, :, 2], percentile)
    
    # Avoid division by zero
    b_max = max(b_max, 1.0)
    g_max = max(g_max, 1.0)
    r_max = max(r_max, 1.0)
    
    # Scale to make white reference = 255
    result = img_float.copy()
    result[:, :, 0] = np.clip(result[:, :, 0] * (255.0 / b_max), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (255.0 / g_max), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (255.0 / r_max), 0, 255)
    
    return result.astype(np.uint8)


def underwater_color_correction(
    image: np.ndarray,
    red_boost: float = 1.5,
    blue_reduction: float = 0.9
) -> np.ndarray:
    """
    Apply underwater-specific color correction.
    
    Underwater images typically lose red tones and have excess blue.
    This function specifically targets these issues.
    
    Args:
        image: Input BGR image (uint8)
        red_boost: Factor to boost red channel (>1 increases red)
        blue_reduction: Factor to reduce blue channel (<1 decreases blue)
    
    Returns:
        Color-corrected BGR image
    """
    img_float = image.astype(np.float32)
    
    # Adjust channels
    result = img_float.copy()
    result[:, :, 0] = np.clip(result[:, :, 0] * blue_reduction, 0, 255)  # Blue
    result[:, :, 2] = np.clip(result[:, :, 2] * red_boost, 0, 255)  # Red
    
    return result.astype(np.uint8)


def compensate_red_channel(image: np.ndarray) -> np.ndarray:
    """
    Compensate for red channel attenuation in underwater images.
    
    Uses green channel as reference to reconstruct red channel,
    based on the observation that green attenuates less than red.
    
    Args:
        image: Input BGR image (uint8)
    
    Returns:
        Red-compensated BGR image
    """
    img_float = image.astype(np.float32)
    
    b, g, r = cv2.split(img_float)
    
    # Calculate mean values
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    
    # Compensate red based on green channel
    if r_mean < g_mean:
        # Red is attenuated, compensate using green as reference
        compensation_factor = g_mean / (r_mean + 1e-6)
        compensation_factor = min(compensation_factor, 2.5)  # Limit compensation
        r_compensated = np.clip(r * compensation_factor, 0, 255)
    else:
        r_compensated = r
    
    result = cv2.merge([b, g, r_compensated])
    return result.astype(np.uint8)


def combined_white_balance(
    image: np.ndarray,
    method: str = 'auto'
) -> np.ndarray:
    """
    Apply the most suitable white balance method based on image analysis.
    
    Args:
        image: Input BGR image
        method: 'gray_world', 'shades_of_gray', 'white_patch', or 'auto'
    
    Returns:
        White-balanced BGR image
    """
    if method == 'gray_world':
        return gray_world(image)
    elif method == 'shades_of_gray':
        return shades_of_gray(image)
    elif method == 'white_patch':
        return white_patch_retinex(image)
    elif method == 'auto':
        # Analyze image to choose best method
        img_float = image.astype(np.float32)
        
        # Calculate color cast
        b_mean = np.mean(img_float[:, :, 0])
        g_mean = np.mean(img_float[:, :, 1])
        r_mean = np.mean(img_float[:, :, 2])
        
        # Strong blue cast (typical underwater) -> Gray World works well
        if b_mean > g_mean * 1.2 and b_mean > r_mean * 1.3:
            return gray_world(image)
        
        # Low contrast image -> Shades of Gray
        if np.std(img_float) < 50:
            return shades_of_gray(image, p=6.0)
        
        # Default to Gray World for underwater
        return gray_world(image)
    else:
        raise ValueError(f"Unknown method: {method}")


def batch_white_balance(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    method: str = 'auto',
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
) -> None:
    """
    Apply white balance to all images in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        method: White balance method to use
        extensions: Image file extensions to process
    """
    from tqdm import tqdm
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    for img_path in tqdm(image_files, desc="White Balance"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        balanced = combined_white_balance(image, method=method)
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), balanced)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply white balance to underwater images')
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('output', help='Output image or directory')
    parser.add_argument('--method', 
                        choices=['gray_world', 'shades_of_gray', 'white_patch', 'auto'],
                        default='auto', help='White balance method')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_dir():
        batch_white_balance(input_path, output_path, method=args.method)
    else:
        image = cv2.imread(str(input_path))
        balanced = combined_white_balance(image, method=args.method)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), balanced)
        print(f"White-balanced image saved to {output_path}")
