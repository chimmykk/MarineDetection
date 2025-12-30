"""
Dataset Download Script for Underwater Project

This script helps download datasets from Kaggle for training
the underwater image enhancement and marine life detection models.

RECOMMENDED DATASETS:

1. Underwater Object Detection (Best for your project!)
   - 638 images, 7 classes (fish, jellyfish, shark, starfish, etc.)
   - Already in YOLO v5 format with train/val/test splits
   - Kaggle: https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots

2. Fish Species Detection Dataset
   - 8,242 images, 13 fish species
   - YOLO format annotations
   - Kaggle: https://www.kaggle.com/datasets/khairunneesa/fish-species-detection-yolov11

3. DeepFish Object Detection
   - High-resolution fish images
   - YOLO format bounding boxes
   - Kaggle: https://www.kaggle.com/datasets/alzayats/deepfish

4. RUOD (Real Underwater Object Detection)
   - Real underwater imagery
   - Multiple object classes
   - Kaggle: https://www.kaggle.com/datasets/landrykezebou/ruod-underwater-object-detection

For Image Enhancement (degraded -> clean pairs):
5. Underwater Image Enhancement Benchmark (UIEB)
   - 890 underwater images with reference images
   - Kaggle: https://www.kaggle.com/datasets/larjeck/uieb-dataset

SETUP INSTRUCTIONS:
1. Install Kaggle API: pip install kaggle
2. Get API credentials from https://www.kaggle.com/settings
3. Place kaggle.json in ~/.kaggle/ (chmod 600 ~/.kaggle/kaggle.json)
4. Run this script: python download_datasets.py

"""

import os
import sys
import subprocess
import shutil
import zipfile
from pathlib import Path


# Dataset configurations
DATASETS = {
    'underwater_detection': {
        'name': 'Underwater Object Detection',
        'kaggle_id': 'slavkoprytula/aquarium-data-cots',
        'description': '638 images, 7 classes, YOLO v5 format',
        'recommended': True,
        'type': 'detection'
    },
    'fish_species': {
        'name': 'Fish Species Detection',
        'kaggle_id': 'khairunneesa/fish-species-detection-yolov11',
        'description': '8,242 images, 13 species, YOLO format',
        'recommended': False,
        'type': 'detection'
    },
    'deepfish': {
        'name': 'DeepFish Dataset',
        'kaggle_id': 'alzayats/deepfish',
        'description': 'High-resolution fish images with YOLO boxes',
        'recommended': False,
        'type': 'detection'
    },
    'ruod': {
        'name': 'RUOD Underwater Detection',
        'kaggle_id': 'landrykezebou/ruod-underwater-object-detection',
        'description': 'Real underwater object detection dataset',
        'recommended': False,
        'type': 'detection'
    },
    'uieb': {
        'name': 'UIEB Enhancement Dataset',
        'kaggle_id': 'larjeck/uieb-dataset',
        'description': '890 underwater images with reference (for U-Net training)',
        'recommended': True,
        'type': 'enhancement'
    }
}


def check_kaggle_setup():
    """Check if Kaggle API is properly configured."""
    try:
        import kaggle
        return True
    except ImportError:
        print(" Kaggle package not installed.")
        print("   Install with: pip install kaggle")
        return False
    except Exception as e:
        if "Could not find kaggle.json" in str(e):
            print(" Kaggle API credentials not found.")
            print("\n   Setup instructions:")
            print("   1. Go to https://www.kaggle.com/settings")
            print("   2. Click 'Create New API Token'")
            print("   3. Move downloaded kaggle.json to ~/.kaggle/")
            print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        raise


def list_datasets():
    """List available datasets."""
    print("\n" + "=" * 60)
    print("Available Datasets for Underwater Project")
    print("=" * 60)
    
    print("\n DETECTION DATASETS:")
    for key, info in DATASETS.items():
        if info['type'] == 'detection':
            star = "*" if info['recommended'] else "  "
            print(f"\n  {star} {key}")
            print(f"     {info['name']}")
            print(f"     {info['description']}")
            print(f"     Kaggle: kaggle.com/datasets/{info['kaggle_id']}")
    
    print("\nENHANCEMENT DATASETS:")
    for key, info in DATASETS.items():
        if info['type'] == 'enhancement':
            star = "*" if info['recommended'] else "  "
            print(f"\n  {star} {key}")
            print(f"     {info['name']}")
            print(f"     {info['description']}")
            print(f"     Kaggle: kaggle.com/datasets/{info['kaggle_id']}")
    
    print("\n* = Recommended for this project")
    print()


def download_dataset(dataset_key: str, output_dir: str = None):
    """Download a specific dataset from Kaggle."""
    if dataset_key not in DATASETS:
        print(f" Unknown dataset: {dataset_key}")
        print(f"   Available: {', '.join(DATASETS.keys())}")
        return False
    
    if not check_kaggle_setup():
        return False
    
    info = DATASETS[dataset_key]
    
    if output_dir is None:
        # Default to project data directory
        script_dir = Path(__file__).parent.parent
        if info['type'] == 'detection':
            output_dir = script_dir / 'data' / 'detection'
        else:
            output_dir = script_dir / 'data' / 'enhancement'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading: {info['name']}")
    print(f"   From: kaggle.com/datasets/{info['kaggle_id']}")
    print(f"   To: {output_dir}")
    
    try:
        # Use Kaggle API
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', info['kaggle_id'],
            '-p', str(output_dir),
            '--unzip'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully downloaded {info['name']}")
            print(f"   Location: {output_dir}")
            return True
        else:
            print(f"Download failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def download_recommended():
    """Download all recommended datasets."""
    print("\n📦 Downloading recommended datasets...")
    
    for key, info in DATASETS.items():
        if info['recommended']:
            download_dataset(key)


def setup_detection_data(source_dir: str, project_data_dir: str = None):
    """
    Organize downloaded detection data into project structure.
    
    Expected project structure:
    data/detection/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """
    source_dir = Path(source_dir)
    
    if project_data_dir is None:
        project_data_dir = Path(__file__).parent.parent / 'data' / 'detection'
    else:
        project_data_dir = Path(project_data_dir)
    
    print(f"\nSetting up detection data structure...")
    print(f"   Source: {source_dir}")
    print(f"   Target: {project_data_dir}")
    
    # Create directory structure
    for split in ['train', 'val']:
        (project_data_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (project_data_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Look for common dataset structures and copy files
    # This handles different dataset organization patterns
    
    # Pattern 1: images/train, labels/train structure
    if (source_dir / 'images' / 'train').exists():
        print("   Found standard YOLO structure")
        for split in ['train', 'val', 'test']:
            src_img = source_dir / 'images' / split
            src_lbl = source_dir / 'labels' / split
            
            if src_img.exists():
                for f in src_img.iterdir():
                    shutil.copy2(f, project_data_dir / 'images' / ('val' if split == 'test' else split))
            if src_lbl.exists():
                for f in src_lbl.iterdir():
                    shutil.copy2(f, project_data_dir / 'labels' / ('val' if split == 'test' else split))
    
    # Pattern 2: train/images, train/labels structure
    elif (source_dir / 'train' / 'images').exists():
        print("   Found alternate structure (train/images)")
        for split in ['train', 'valid', 'val', 'test']:
            src_dir = source_dir / split
            if not src_dir.exists():
                continue
            
            target_split = 'val' if split in ['valid', 'val', 'test'] else 'train'
            
            for img in (src_dir / 'images').glob('*'):
                shutil.copy2(img, project_data_dir / 'images' / target_split)
            for lbl in (src_dir / 'labels').glob('*'):
                shutil.copy2(lbl, project_data_dir / 'labels' / target_split)
    
    print("Detection data organized")
    
    # Count files
    train_imgs = len(list((project_data_dir / 'images' / 'train').glob('*')))
    val_imgs = len(list((project_data_dir / 'images' / 'val').glob('*')))
    print(f"   Training images: {train_imgs}")
    print(f"   Validation images: {val_imgs}")


def setup_enhancement_data(source_dir: str, project_data_dir: str = None):
    """
    Organize downloaded enhancement data for U-Net training.
    
    Expected structure:
    data/enhancement/
    ├── train/
    │   ├── input/    # Degraded images
    │   └── target/   # Clean reference images
    └── val/
        ├── input/
        └── target/
    """
    source_dir = Path(source_dir)
    
    if project_data_dir is None:
        project_data_dir = Path(__file__).parent.parent / 'data' / 'enhancement'
    else:
        project_data_dir = Path(project_data_dir)
    
    print(f"\nSetting up enhancement data structure...")
    
    # Create directories
    for split in ['train', 'val']:
        (project_data_dir / split / 'input').mkdir(parents=True, exist_ok=True)
        (project_data_dir / split / 'target').mkdir(parents=True, exist_ok=True)
    
    # UIEB dataset has 'raw-890' and 'reference-890' folders
    raw_dir = source_dir / 'raw-890'
    ref_dir = source_dir / 'reference-890'
    
    if raw_dir.exists() and ref_dir.exists():
        print("   Found UIEB structure")
        
        # Get matching files
        raw_files = sorted(list(raw_dir.glob('*.png')) + list(raw_dir.glob('*.jpg')))
        
        # Split 80/20
        split_idx = int(len(raw_files) * 0.8)
        train_files = raw_files[:split_idx]
        val_files = raw_files[split_idx:]
        
        for files, split in [(train_files, 'train'), (val_files, 'val')]:
            for raw_file in files:
                # Copy raw (input)
                shutil.copy2(raw_file, project_data_dir / split / 'input' / raw_file.name)
                
                # Find and copy reference (target)
                ref_file = ref_dir / raw_file.name
                if ref_file.exists():
                    shutil.copy2(ref_file, project_data_dir / split / 'target' / raw_file.name)
        
        print("Enhancement data organized")
        print(f"   Training pairs: {len(train_files)}")
        print(f"   Validation pairs: {len(val_files)}")
    else:
        print("   Could not find expected UIEB structure")
        print("   Please organize manually into input/target folders")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download datasets for Underwater project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available datasets')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a dataset')
    download_parser.add_argument('dataset', 
                                 choices=list(DATASETS.keys()) + ['recommended', 'all'],
                                 help='Dataset to download')
    download_parser.add_argument('--output', '-o', help='Output directory')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Organize downloaded data')
    setup_parser.add_argument('source', help='Source directory with downloaded data')
    setup_parser.add_argument('--type', choices=['detection', 'enhancement'],
                              required=True, help='Type of dataset')
    
    args = parser.parse_args()
    
    if args.command == 'list' or args.command is None:
        list_datasets()
        
    elif args.command == 'download':
        if args.dataset == 'recommended':
            download_recommended()
        elif args.dataset == 'all':
            for key in DATASETS:
                download_dataset(key, args.output)
        else:
            download_dataset(args.dataset, args.output)
            
    elif args.command == 'setup':
        if args.type == 'detection':
            setup_detection_data(args.source)
        else:
            setup_enhancement_data(args.source)


if __name__ == '__main__':
    main()
