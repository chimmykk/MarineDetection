"""
YOLOv5 Training Wrapper for Marine Life Detection

Provides a simplified interface to train YOLOv5 models
on custom underwater marine life datasets.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import shutil


def setup_yolov5(install_dir: str = 'yolov5') -> Path:
    """
    Clone and setup YOLOv5 repository.
    
    Args:
        install_dir: Directory to install YOLOv5
    
    Returns:
        Path to YOLOv5 directory
    """
    yolov5_path = Path(install_dir)
    
    if not yolov5_path.exists():
        print("Cloning YOLOv5 repository...")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/ultralytics/yolov5.git',
            str(yolov5_path)
        ], check=True)
        
        print("Installing YOLOv5 requirements...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            '-r', str(yolov5_path / 'requirements.txt')
        ], check=True)
    
    return yolov5_path


def create_dataset_yaml(
    train_path: str,
    val_path: str,
    classes: list,
    output_path: str = 'dataset.yaml'
) -> str:
    """
    Create YOLOv5 dataset configuration file.
    
    Args:
        train_path: Path to training images
        val_path: Path to validation images
        classes: List of class names
        output_path: Output YAML file path
    
    Returns:
        Path to created YAML file
    """
    config = {
        'path': str(Path(train_path).parent.parent),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created dataset config at {output_path}")
    return output_path


class YOLOv5Trainer:
    """
    Wrapper class for YOLOv5 training.
    
    Provides simplified interface for training on custom datasets.
    """
    
    def __init__(
        self,
        yolov5_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            yolov5_path: Path to YOLOv5 directory (will be cloned if None)
            device: Device to train on ('cpu', '0', '0,1', etc.)
        """
        if yolov5_path is None:
            self.yolov5_path = setup_yolov5()
        else:
            self.yolov5_path = Path(yolov5_path)
        
        self.device = device
        
        # Add YOLOv5 to path
        sys.path.insert(0, str(self.yolov5_path))
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        weights: str = 'yolov5s.pt',
        project: str = 'runs/train',
        name: str = 'marine_detection',
        **kwargs
    ) -> str:
        """
        Train YOLOv5 model.
        
        Args:
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs
            batch_size: Batch size (reduce for CPU training)
            img_size: Image size
            weights: Pretrained weights ('yolov5n.pt', 'yolov5s.pt', etc.)
            project: Project directory for saving results
            name: Experiment name
            **kwargs: Additional training arguments
        
        Returns:
            Path to best trained weights
        """
        # Build command
        cmd = [
            sys.executable,
            str(self.yolov5_path / 'train.py'),
            '--data', data_yaml,
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--img', str(img_size),
            '--weights', weights,
            '--project', project,
            '--name', name,
            '--device', self.device,
        ]
        
        # Add additional arguments
        for key, value in kwargs.items():
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
        
        print(f"Starting training with command:")
        print(' '.join(cmd))
        
        # Run training
        subprocess.run(cmd, check=True)
        
        # Return path to best weights
        return str(Path(project) / name / 'weights' / 'best.pt')
    
    def validate(
        self,
        weights: str,
        data_yaml: str,
        batch_size: int = 16,
        img_size: int = 640
    ) -> Dict[str, float]:
        """
        Validate trained model.
        
        Args:
            weights: Path to trained weights
            data_yaml: Path to dataset YAML
            batch_size: Batch size
            img_size: Image size
        
        Returns:
            Dictionary of validation metrics
        """
        cmd = [
            sys.executable,
            str(self.yolov5_path / 'val.py'),
            '--weights', weights,
            '--data', data_yaml,
            '--batch-size', str(batch_size),
            '--img', str(img_size),
            '--device', self.device,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        # Parse metrics from output (simplified)
        return {'status': 'complete'}
    
    def export(
        self,
        weights: str,
        format: str = 'onnx',
        img_size: int = 640
    ) -> str:
        """
        Export model to different format.
        
        Args:
            weights: Path to trained weights
            format: Export format ('onnx', 'torchscript', 'coreml', etc.)
            img_size: Image size
        
        Returns:
            Path to exported model
        """
        cmd = [
            sys.executable,
            str(self.yolov5_path / 'export.py'),
            '--weights', weights,
            '--include', format,
            '--img', str(img_size),
            '--device', self.device,
        ]
        
        subprocess.run(cmd, check=True)
        
        # Return exported model path
        weights_path = Path(weights)
        return str(weights_path.with_suffix(f'.{format}'))


def train_marine_detector(
    data_dir: str,
    classes: list = None,
    epochs: int = 100,
    batch_size: int = 8,
    img_size: int = 640,
    model_size: str = 's',  # n, s, m, l, x
    output_dir: str = 'runs/train'
) -> str:
    """
    High-level function to train marine life detector.
    
    This is the recommended entry point for training.
    
    Args:
        data_dir: Root directory with images/ and labels/ subdirs
        classes: List of class names (uses default if None)
        epochs: Number of epochs
        batch_size: Batch size (use 2-4 for CPU)
        img_size: Image size
        model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        output_dir: Output directory for training runs
    
    Returns:
        Path to trained weights
    """
    if classes is None:
        classes = [
            'fish', 'shark', 'jellyfish', 'starfish',
            'coral', 'diseased', 'damaged'
        ]
    
    data_dir = Path(data_dir)
    
    # Create dataset YAML
    yaml_path = data_dir / 'dataset.yaml'
    create_dataset_yaml(
        train_path=str(data_dir / 'images' / 'train'),
        val_path=str(data_dir / 'images' / 'val'),
        classes=classes,
        output_path=str(yaml_path)
    )
    
    # Initialize trainer
    trainer = YOLOv5Trainer(device='cpu')
    
    # Train
    weights = trainer.train(
        data_yaml=str(yaml_path),
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        weights=f'yolov5{model_size}.pt',
        project=output_dir,
        name='marine_detection'
    )
    
    return weights


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv5 for marine detection')
    parser.add_argument('--data', required=True, help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--model', choices=['n', 's', 'm', 'l', 'x'], default='s',
                        help='Model size (n=nano for fastest CPU inference)')
    parser.add_argument('--output', default='runs/train', help='Output directory')
    
    args = parser.parse_args()
    
    weights = train_marine_detector(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        model_size=args.model,
        output_dir=args.output
    )
    
    print(f"\nTraining complete! Best weights saved to: {weights}")
