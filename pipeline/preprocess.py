"""
Preprocessing Module

Handles loading, resizing, and preparing images and videos
for the enhancement and detection pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Generator, Optional
from dataclasses import dataclass
from PIL import Image


@dataclass
class ImageInfo:
    """Container for image information."""
    path: Optional[Path]
    image: np.ndarray
    original_size: Tuple[int, int]  # (width, height)
    filename: str


class ImagePreprocessor:
    """
    Image preprocessing for underwater enhancement pipeline.
    
    Handles:
    - Image loading from various sources
    - Resizing and padding
    - Normalization
    - Batch processing
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
        preserve_aspect: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target (width, height) for resizing
            normalize: Whether to normalize to [0, 1]
            preserve_aspect: Preserve aspect ratio when resizing
        """
        self.target_size = target_size
        self.normalize = normalize
        self.preserve_aspect = preserve_aspect
    
    def load_image(self, source: Union[str, Path, np.ndarray]) -> ImageInfo:
        """
        Load and preprocess a single image.
        
        Args:
            source: Image path or numpy array
        
        Returns:
            ImageInfo with processed image
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not load image: {path}")
            filename = path.name
        else:
            path = None
            image = source
            filename = "array_input"
        
        original_size = (image.shape[1], image.shape[0])
        
        # Resize if needed
        if self.target_size:
            image = self._resize(image)
        
        # Normalize if needed
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        return ImageInfo(
            path=path,
            image=image,
            original_size=original_size,
            filename=filename
        )
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        if self.preserve_aspect:
            return self._resize_with_padding(image)
        else:
            return cv2.resize(image, self.target_size)
    
    def _resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """Resize while preserving aspect ratio with padding."""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
        
        # Center the image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False
    ) -> List[ImageInfo]:
        """
        Load all images from a directory.
        
        Args:
            directory: Directory path
            recursive: Search recursively
        
        Returns:
            List of ImageInfo objects
        """
        directory = Path(directory)
        
        if recursive:
            image_files = [
                f for f in directory.rglob('*')
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
        else:
            image_files = [
                f for f in directory.iterdir()
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
        
        images = []
        for img_path in sorted(image_files):
            try:
                images.append(self.load_image(img_path))
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
        
        return images
    
    def iterate_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False
    ) -> Generator[ImageInfo, None, None]:
        """
        Iterate over images in a directory (memory efficient).
        
        Args:
            directory: Directory path
            recursive: Search recursively
        
        Yields:
            ImageInfo objects one at a time
        """
        directory = Path(directory)
        
        if recursive:
            image_files = sorted([
                f for f in directory.rglob('*')
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ])
        else:
            image_files = sorted([
                f for f in directory.iterdir()
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ])
        
        for img_path in image_files:
            try:
                yield self.load_image(img_path)
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")


class VideoPreprocessor:
    """
    Video preprocessing for underwater enhancement pipeline.
    
    Handles:
    - Video loading
    - Frame extraction
    - Resizing
    """
    
    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        skip_frames: int = 0
    ):
        """
        Initialize video preprocessor.
        
        Args:
            target_size: Target frame size (width, height)
            skip_frames: Number of frames to skip between extractions
        """
        self.target_size = target_size
        self.skip_frames = skip_frames
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'path': str(video_path),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            output_dir: Optional directory to save frames
            max_frames: Maximum number of frames to extract
        
        Returns:
            List of frame arrays
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        frames = []
        frame_idx = 0
        saved_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % (self.skip_frames + 1) == 0:
                # Resize if needed
                if self.target_size:
                    frame = cv2.resize(frame, self.target_size)
                
                frames.append(frame)
                
                # Save if output directory specified
                if output_dir:
                    frame_path = output_dir / f"frame_{saved_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                
                saved_idx += 1
                
                if max_frames and saved_idx >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def iterate_frames(
        self,
        video_path: Union[str, Path]
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterate over video frames (memory efficient).
        
        Args:
            video_path: Path to video file
        
        Yields:
            Tuple of (frame_index, frame_array)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % (self.skip_frames + 1) == 0:
                if self.target_size:
                    frame = cv2.resize(frame, self.target_size)
                yield frame_idx, frame
            
            frame_idx += 1
        
        cap.release()


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    preserve_aspect: bool = True
) -> np.ndarray:
    """
    Resize image utility function.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        preserve_aspect: Preserve aspect ratio
    
    Returns:
        Resized image
    """
    if preserve_aspect:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Convert [0, 1] image back to uint8."""
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocessing utilities')
    parser.add_argument('input', help='Input image/video/directory')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--size', type=int, nargs=2, default=[640, 640],
                        help='Target size (width height)')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Frames to skip for video')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        preprocessor = ImagePreprocessor(
            target_size=tuple(args.size) if args.size else None
        )
        images = preprocessor.load_directory(input_path)
        print(f"Loaded {len(images)} images")
        
    elif input_path.suffix.lower() in VideoPreprocessor.SUPPORTED_EXTENSIONS:
        preprocessor = VideoPreprocessor(
            target_size=tuple(args.size) if args.size else None,
            skip_frames=args.skip_frames
        )
        info = preprocessor.get_video_info(input_path)
        print(f"Video info: {info}")
        
        if args.output:
            frames = preprocessor.extract_frames(input_path, args.output)
            print(f"Extracted {len(frames)} frames to {args.output}")
    else:
        preprocessor = ImagePreprocessor(
            target_size=tuple(args.size) if args.size else None
        )
        info = preprocessor.load_image(input_path)
        print(f"Loaded image: {info.filename}, size: {info.original_size}")
