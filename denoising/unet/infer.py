"""
U-Net Inference for Underwater Image Enhancement

Load pretrained models and run inference on images.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Optional, Tuple
from torchvision import transforms
import cv2
from tqdm import tqdm

from .model import get_model


class UNetInference:
    """
    Inference class for U-Net image enhancement.
    
    Handles model loading, preprocessing, and batch inference.
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'standard',
        base_filters: int = 64,
        device: Optional[str] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model weights
            model_type: Type of model ('standard', 'lightweight', 'residual')
            base_filters: Number of base filters (must match trained model)
            device: Device to use ('cpu' or 'cuda')
        """
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Inference device: {self.device}")
        
        # Load model
        self.model = get_model(model_type, base_filters=base_filters)
        self.model = self.model.to(self.device)
        
        # Load weights
        self._load_weights(model_path)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
    
    def _load_weights(self, model_path: str) -> None:
        """Load model weights from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Loaded weights from {model_path}")
    
    def preprocess(
        self,
        image: Union[np.ndarray, Image.Image, str],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (array, PIL Image, or path)
            target_size: Optional resize dimensions
        
        Returns:
            Tuple of (preprocessed tensor, original size)
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Assume BGR from OpenCV
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        original_size = image.size  # (width, height)
        
        # Resize if specified
        if target_size is not None:
            image = image.resize(target_size, Image.BILINEAR)
        
        # Convert to tensor
        tensor = self.to_tensor(image).unsqueeze(0)  # Add batch dimension
        
        return tensor, original_size
    
    def postprocess(
        self,
        output: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Postprocess model output to image.
        
        Args:
            output: Model output tensor
            original_size: Original image size (width, height)
        
        Returns:
            BGR numpy array
        """
        # Remove batch dimension and move to CPU
        output = output.squeeze(0).cpu()
        
        # Clamp to valid range
        output = torch.clamp(output, 0, 1)
        
        # Convert to numpy array (C, H, W) -> (H, W, C)
        image = output.permute(1, 2, 0).numpy()
        
        # Convert to uint8
        image = (image * 255).astype(np.uint8)
        
        # Resize to original size if different
        if image.shape[1] != original_size[0] or image.shape[0] != original_size[1]:
            image = cv2.resize(image, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert RGB to BGR for OpenCV compatibility
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    @torch.no_grad()
    def enhance(
        self,
        image: Union[np.ndarray, Image.Image, str],
        target_size: Optional[Tuple[int, int]] = None,
        preserve_size: bool = True
    ) -> np.ndarray:
        """
        Enhance a single image.
        
        Args:
            image: Input image (array, PIL Image, or path)
            target_size: Resize for inference (helps with memory)
            preserve_size: Resize output back to original size
        
        Returns:
            Enhanced BGR image as numpy array
        """
        # Preprocess
        tensor, original_size = self.preprocess(image, target_size)
        tensor = tensor.to(self.device)
        
        # Inference
        output = self.model(tensor)
        
        # Postprocess
        if preserve_size:
            result = self.postprocess(output, original_size)
        else:
            result = self.postprocess(output, 
                                      (tensor.shape[3], tensor.shape[2]))
        
        return result
    
    def enhance_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        target_size: Optional[Tuple[int, int]] = (512, 512),
        preserve_size: bool = True
    ) -> List[np.ndarray]:
        """
        Enhance multiple images.
        
        Args:
            images: List of input images
            target_size: Resize for inference
            preserve_size: Resize outputs back to original sizes
        
        Returns:
            List of enhanced images
        """
        results = []
        
        for image in tqdm(images, desc="Enhancing"):
            enhanced = self.enhance(image, target_size, preserve_size)
            results.append(enhanced)
        
        return results
    
    def enhance_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        target_size: Optional[Tuple[int, int]] = (512, 512),
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ) -> List[Path]:
        """
        Enhance all images in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            target_size: Resize for inference
            extensions: Image file extensions to process
        
        Returns:
            List of output paths
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        output_paths = []
        
        for img_path in tqdm(image_files, desc="Enhancing directory"):
            try:
                enhanced = self.enhance(str(img_path), target_size, preserve_size=True)
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), enhanced)
                output_paths.append(output_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return output_paths


def enhance_image_cli():
    """Command-line interface for image enhancement."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance underwater images using U-Net')
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('output', help='Output image or directory')
    parser.add_argument('--model', required=True, help='Path to model weights')
    parser.add_argument('--model-type', choices=['standard', 'lightweight', 'residual'],
                        default='standard', help='Model architecture type')
    parser.add_argument('--base-filters', type=int, default=64, help='Base filter count')
    parser.add_argument('--size', type=int, default=512, help='Inference size')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = UNetInference(
        model_path=args.model,
        model_type=args.model_type,
        base_filters=args.base_filters
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_dir():
        engine.enhance_directory(
            input_path, output_path,
            target_size=(args.size, args.size)
        )
    else:
        enhanced = engine.enhance(
            str(input_path),
            target_size=(args.size, args.size)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), enhanced)
        print(f"Enhanced image saved to {output_path}")


if __name__ == '__main__':
    enhance_image_cli()
