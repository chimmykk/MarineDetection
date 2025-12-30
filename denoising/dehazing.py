"""
Underwater Image Dehazing

Implements dehazing algorithms adapted for underwater conditions,
including Dark Channel Prior and underwater-specific methods.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path


def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Compute the dark channel of an image.
    
    The dark channel is the minimum of RGB values in a local patch.
    
    Args:
        image: Input BGR image (float, normalized to [0, 1])
        patch_size: Size of local patch for minimum operation
    
    Returns:
        Dark channel image
    """
    # Get minimum across color channels
    min_channel = np.min(image, axis=2)
    
    # Apply erosion to get local minimum
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (patch_size, patch_size)
    )
    dark = cv2.erode(min_channel, kernel)
    
    return dark


def estimate_atmospheric_light(
    image: np.ndarray,
    dark: np.ndarray,
    top_percent: float = 0.1
) -> np.ndarray:
    """
    Estimate atmospheric light from the brightest pixels in the dark channel.
    
    Args:
        image: Input BGR image (float, normalized to [0, 1])
        dark: Dark channel image
        top_percent: Percentage of brightest pixels to consider
    
    Returns:
        Atmospheric light vector (3 elements for BGR)
    """
    h, w = dark.shape
    num_pixels = int(h * w * top_percent / 100)
    num_pixels = max(num_pixels, 1)
    
    # Find brightest pixels in dark channel
    flat_dark = dark.ravel()
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    
    # Get corresponding pixels from original image
    flat_image = image.reshape(-1, 3)
    brightest_pixels = flat_image[indices]
    
    # Use the brightest pixel as atmospheric light
    atmosphere = np.max(brightest_pixels, axis=0)
    
    return atmosphere


def estimate_transmission(
    image: np.ndarray,
    atmosphere: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15
) -> np.ndarray:
    """
    Estimate transmission map using dark channel prior.
    
    Args:
        image: Input BGR image (float, normalized to [0, 1])
        atmosphere: Estimated atmospheric light
        omega: Parameter to keep some haze for distant objects (0.95 typical)
        patch_size: Patch size for dark channel
    
    Returns:
        Transmission map
    """
    # Normalize image by atmospheric light
    normalized = image / (atmosphere + 1e-6)
    
    # Compute dark channel of normalized image
    dark = dark_channel(normalized, patch_size)
    
    # Estimate transmission
    transmission = 1 - omega * dark
    
    return transmission


def refine_transmission(
    image: np.ndarray,
    transmission: np.ndarray,
    radius: int = 60,
    eps: float = 1e-3
) -> np.ndarray:
    """
    Refine transmission map using guided filter.
    
    Args:
        image: Guidance image (original BGR)
        transmission: Initial transmission estimate
        radius: Guided filter radius
        eps: Regularization parameter
    
    Returns:
        Refined transmission map
    """
    # Convert to grayscale for guidance
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
    else:
        gray = image
    
    # Apply guided filter (using box filter approximation if cv2 version doesn't have it)
    try:
        refined = cv2.ximgproc.guidedFilter(
            gray.astype(np.float32),
            transmission.astype(np.float32),
            radius, eps
        )
    except AttributeError:
        # Fallback to bilateral filter if guided filter not available
        trans_uint8 = (transmission * 255).astype(np.uint8)
        refined = cv2.bilateralFilter(trans_uint8, 9, 75, 75)
        refined = refined.astype(np.float32) / 255.0
    
    return refined


def dark_channel_prior(
    image: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15,
    t_min: float = 0.1
) -> np.ndarray:
    """
    Apply Dark Channel Prior dehazing.
    
    Args:
        image: Input BGR image (uint8)
        omega: Haze retention parameter (0.95 keeps some haze)
        patch_size: Patch size for dark channel computation
        t_min: Minimum transmission to avoid division instability
    
    Returns:
        Dehazed BGR image (uint8)
    """
    # Normalize to [0, 1]
    img_float = image.astype(np.float32) / 255.0
    
    # Compute dark channel
    dark = dark_channel(img_float, patch_size)
    
    # Estimate atmospheric light
    atmosphere = estimate_atmospheric_light(img_float, dark)
    
    # Estimate transmission
    transmission = estimate_transmission(img_float, atmosphere, omega, patch_size)
    
    # Refine transmission
    transmission = refine_transmission(img_float, transmission)
    
    # Clamp transmission
    transmission = np.maximum(transmission, t_min)
    
    # Recover scene radiance
    result = np.zeros_like(img_float)
    for c in range(3):
        result[:, :, c] = (
            (img_float[:, :, c] - atmosphere[c]) / transmission + atmosphere[c]
        )
    
    # Clip and convert back to uint8
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


def underwater_light_estimation(image: np.ndarray) -> np.ndarray:
    """
    Estimate background light for underwater images.
    
    Unlike atmospheric haze, underwater light estimation considers
    color-dependent attenuation.
    
    Args:
        image: Input BGR image (float, [0, 1])
    
    Returns:
        Background light vector
    """
    h, w = image.shape[:2]
    
    # Use different strategies for each channel
    # Blue attenuates least, red attenuates most
    b_light = np.percentile(image[:, :, 0], 99)
    g_light = np.percentile(image[:, :, 1], 99)
    r_light = np.percentile(image[:, :, 2], 95)  # Red is often very low
    
    return np.array([b_light, g_light, r_light])


def underwater_transmission(
    image: np.ndarray,
    background_light: np.ndarray,
    patch_size: int = 15
) -> np.ndarray:
    """
    Estimate transmission for underwater images.
    
    Uses red channel recovery approach specific to underwater.
    
    Args:
        image: Input BGR image (float, [0, 1])
        background_light: Estimated background light
        patch_size: Patch size for local operations
    
    Returns:
        Transmission map
    """
    # Red channel is most attenuated underwater
    # Use difference between blue/green and red to estimate transmission
    
    b, g, r = cv2.split(image)
    
    # Normalize by background light
    b_norm = b / (background_light[0] + 1e-6)
    g_norm = g / (background_light[1] + 1e-6)
    r_norm = r / (background_light[2] + 1e-6)
    
    # Estimate transmission from red channel (most affected by scattering)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    
    # Use minimum of normalized red channel
    t_red = 1 - cv2.erode(r_norm.astype(np.float32), kernel)
    
    # Combine with blue-green attenuation estimate
    bg_avg = (b_norm + g_norm) / 2
    t_bg = 1 - cv2.erode(bg_avg.astype(np.float32), kernel) * 0.5
    
    # Weighted combination
    transmission = 0.7 * t_red + 0.3 * t_bg
    
    return np.clip(transmission, 0.1, 1.0)


def underwater_dehaze(
    image: np.ndarray,
    t_min: float = 0.1,
    patch_size: int = 15,
    enhance_red: bool = True
) -> np.ndarray:
    """
    Apply underwater-specific dehazing algorithm.
    
    Combines dehazing with color restoration specific to underwater conditions.
    
    Args:
        image: Input BGR image (uint8)
        t_min: Minimum transmission value
        patch_size: Patch size for local operations
        enhance_red: Whether to enhance red channel
    
    Returns:
        Dehazed and color-corrected BGR image (uint8)
    """
    # Normalize to [0, 1]
    img_float = image.astype(np.float32) / 255.0
    
    # Estimate underwater background light
    bg_light = underwater_light_estimation(img_float)
    
    # Estimate transmission
    transmission = underwater_transmission(img_float, bg_light, patch_size)
    
    # Refine transmission
    transmission = refine_transmission(img_float, transmission, radius=30)
    transmission = np.maximum(transmission, t_min)
    
    # Recover scene
    result = np.zeros_like(img_float)
    for c in range(3):
        result[:, :, c] = (
            (img_float[:, :, c] - bg_light[c]) / transmission + bg_light[c]
        )
    
    # Enhance red channel if needed
    if enhance_red:
        r_mean = np.mean(result[:, :, 2])
        g_mean = np.mean(result[:, :, 1])
        if r_mean < g_mean * 0.8:
            boost = min(g_mean / (r_mean + 1e-6), 2.0)
            result[:, :, 2] = np.clip(result[:, :, 2] * boost, 0, 1)
    
    # Clip and convert
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


def simple_dehaze(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Apply simple dehazing using contrast stretching.
    
    A lightweight alternative when full dehazing is too slow.
    
    Args:
        image: Input BGR image (uint8)
        strength: Dehazing strength (0-1)
    
    Returns:
        Dehazed image (uint8)
    """
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Stretch L channel contrast
    l_min, l_max = np.percentile(l, [2, 98])
    l_stretched = np.clip((l - l_min) * (255 / (l_max - l_min + 1e-6)), 0, 255)
    
    # Blend with original
    l_final = (l * (1 - strength) + l_stretched * strength).astype(np.uint8)
    
    # Merge and convert back
    lab_enhanced = cv2.merge([l_final, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def batch_dehaze(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    method: str = 'underwater',
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
) -> None:
    """
    Apply dehazing to all images in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        method: 'dcp' for Dark Channel Prior, 'underwater' for underwater-specific
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
    
    dehaze_func = underwater_dehaze if method == 'underwater' else dark_channel_prior
    
    for img_path in tqdm(image_files, desc="Dehazing"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        dehazed = dehaze_func(image)
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), dehazed)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Dehaze underwater images')
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('output', help='Output image or directory')
    parser.add_argument('--method', choices=['dcp', 'underwater', 'simple'],
                        default='underwater', help='Dehazing method')
    parser.add_argument('--patch-size', type=int, default=15, help='Patch size')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_dir():
        batch_dehaze(input_path, output_path, method=args.method)
    else:
        image = cv2.imread(str(input_path))
        
        if args.method == 'dcp':
            dehazed = dark_channel_prior(image, patch_size=args.patch_size)
        elif args.method == 'underwater':
            dehazed = underwater_dehaze(image, patch_size=args.patch_size)
        else:
            dehazed = simple_dehaze(image)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), dehazed)
        print(f"Dehazed image saved to {output_path}")
