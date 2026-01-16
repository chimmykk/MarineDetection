"""
Train Unified Marine Detection Model

Trains a single YOLOv5 model on the merged dataset containing:
- Marine life detection (7 classes)
- Fish species identification (12 classes)  
- Fish disease detection (4 classes)

Total: 23 classes
"""

import argparse
import subprocess
import sys
from pathlib import Path


def train_unified_model(
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    model_size: str = 's',
    device: str = 'cpu',
    resume: bool = False
):
    """
    Train the unified marine detection model.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size (reduce for CPU/limited GPU memory)
        img_size: Image size for training
        model_size: YOLOv5 model size (n, s, m, l, x)
        device: Training device ('cpu', '0', '0,1', etc.)
        resume: Resume from last checkpoint
    """
    # Paths
    data_yaml = Path("data/unified_dataset/dataset.yaml")
    project_dir = Path("runs/train")
    experiment_name = "unified_marine_detector"
    
    if not data_yaml.exists():
        print("Error: Unified dataset not found!")
        print("Run 'python scripts/merge_datasets.py' first.")
        sys.exit(1)
    
    # Check if YOLOv5 is cloned
    yolov5_path = Path("yolov5")
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
    
    # Build training command
    weights = f'yolov5{model_size}.pt'
    
    cmd = [
        sys.executable,
        str(yolov5_path / 'train.py'),
        '--data', str(data_yaml.absolute()),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--img', str(img_size),
        '--weights', weights,
        '--project', str(project_dir),
        '--name', experiment_name,
        '--device', device,
        '--cache',  # Cache images for faster training
    ]
    
    if resume:
        cmd.append('--resume')
    
    print("="*60)
    print("UNIFIED MARINE DETECTOR TRAINING")
    print("="*60)
    print(f"Dataset: {data_yaml}")
    print(f"Model: YOLOv5{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    print(f"Output: {project_dir / experiment_name}")
    print("="*60)
    print()
    
    # Run training
    print("Starting training...")
    print("Command:", ' '.join(cmd))
    print()
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        weights_path = project_dir / experiment_name / 'weights' / 'best.pt'
        print()
        print("="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Best weights: {weights_path}")
        print()
        print("To use the model:")
        print(f"  python detection/detect.py <image> --model {weights_path}")
        print()
        print("To copy to models directory:")
        print(f"  cp {weights_path} models/unified_marine_detector.pt")
        return str(weights_path)
    else:
        print("Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train unified marine detection model'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Batch size (default: 16, use 4-8 for CPU)'
    )
    parser.add_argument(
        '--img-size', type=int, default=640,
        help='Image size (default: 640)'
    )
    parser.add_argument(
        '--model', choices=['n', 's', 'm', 'l', 'x'], default='s',
        help='Model size: n=nano, s=small, m=medium, l=large, x=xlarge'
    )
    parser.add_argument(
        '--device', default='cpu',
        help='Device: cpu, 0, 0,1, etc. (default: cpu)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from last checkpoint'
    )
    
    args = parser.parse_args()
    
    train_unified_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        model_size=args.model,
        device=args.device,
        resume=args.resume
    )
