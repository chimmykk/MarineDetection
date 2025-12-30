"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) Enhancement

Optimized for underwater images where contrast is typically reduced
due to light absorption and scattering.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Optional
from tqdm import tqdm


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    color_space: str = 'LAB'
) -> np.ndarray:
    """
    Apply CLAHE enhancement to an underwater image.
    
    Args:
        image: Input BGR image (numpy array)
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_grid_size: Size of grid for histogram equalization
        color_space: Color space for processing ('LAB' or 'HSV')
    
    Returns:
        Enhanced BGR image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image")
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if color_space.upper() == 'LAB':
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_enhanced = clahe.apply(l)
        
        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
    elif color_space.upper() == 'HSV':
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Apply CLAHE to V channel
        v_enhanced = clahe.apply(v)
        
        # Merge and convert back
        hsv_enhanced = cv2.merge([h, s, v_enhanced])
        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    
    return result


def apply_clahe_rgb(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE to each RGB channel independently.
    
    This can help correct color cast issues common in underwater images.
    
    Args:
        image: Input BGR image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Enhanced BGR image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Split channels
    b, g, r = cv2.split(image)
    
    # Apply CLAHE to each channel
    b_enhanced = clahe.apply(b)
    g_enhanced = clahe.apply(g)
    r_enhanced = clahe.apply(r)
    
    # Merge channels
    return cv2.merge([b_enhanced, g_enhanced, r_enhanced])


def batch_clahe(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    color_space: str = 'LAB',
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
) -> List[Path]:
    """
    Apply CLAHE enhancement to all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save enhanced images
        clip_limit: CLAHE clip limit
        tile_grid_size: CLAHE tile grid size
        color_space: Color space for processing
        extensions: Image file extensions to process
    
    Returns:
        List of paths to enhanced images
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    enhanced_paths = []
    
    for img_path in tqdm(image_files, desc="Applying CLAHE"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Apply CLAHE
        enhanced = apply_clahe(
            image,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            color_space=color_space
        )
        
        # Save enhanced image
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), enhanced)
        enhanced_paths.append(output_path)
    
    return enhanced_paths


def adaptive_clahe(
    image: np.ndarray,
    dark_threshold: float = 0.3,
    bright_threshold: float = 0.7
) -> np.ndarray:
    """
    Apply adaptive CLAHE with parameters adjusted based on image brightness.
    
    Underwater images often have varying lighting conditions, so this
    function adapts the CLAHE parameters accordingly.
    
    Args:
        image: Input BGR image
        dark_threshold: Mean brightness below which image is considered dark
        bright_threshold: Mean brightness above which image is considered bright
    
    Returns:
        Enhanced BGR image
    """
    # Calculate mean brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray) / 255.0
    
    # Adapt parameters based on brightness
    if mean_brightness < dark_threshold:
        # Dark image: use higher clip limit for more enhancement
        clip_limit = 4.0
        tile_size = (16, 16)
    elif mean_brightness > bright_threshold:
        # Bright image: use lower clip limit to avoid over-enhancement
        clip_limit = 1.5
        tile_size = (8, 8)
    else:
        # Normal brightness
        clip_limit = 2.5
        tile_size = (8, 8)
    
    return apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_size)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply CLAHE enhancement to underwater images')
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('output', help='Output image or directory')
    parser.add_argument('--clip-limit', type=float, default=2.0, help='CLAHE clip limit')
    parser.add_argument('--tile-size', type=int, default=8, help='CLAHE tile grid size')
    parser.add_argument('--color-space', choices=['LAB', 'HSV'], default='LAB')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive CLAHE')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_dir():
        batch_clahe(
            input_path, output_path,
            clip_limit=args.clip_limit,
            tile_grid_size=(args.tile_size, args.tile_size),
            color_space=args.color_space
        )
    else:
        image = cv2.imread(str(input_path))
        if args.adaptive:
            enhanced = adaptive_clahe(image)
        else:
            enhanced = apply_clahe(
                image,
                clip_limit=args.clip_limit,
                tile_grid_size=(args.tile_size, args.tile_size),
                color_space=args.color_space
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), enhanced)
        print(f"Enhanced image saved to {output_path}")
