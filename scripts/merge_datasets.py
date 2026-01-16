"""
Dataset Merger for Unified Marine Detection Model

Merges three datasets:
1. Existing marine life detection (fish, jellyfish, penguin, puffin, shark, starfish, stingray)
2. Fish Species (13 species from Roboflow)
3. Fish Disease (4 disease classes from Roboflow)

Creates a unified dataset with remapped class IDs.
"""

import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm


# Define unified class mapping
UNIFIED_CLASSES = [
    # Original marine classes (0-6)
    'fish',           # 0 - generic fish
    'jellyfish',      # 1
    'penguin',        # 2
    'puffin',         # 3
    'shark',          # 4
    'starfish',       # 5
    'stingray',       # 6
    
    # Fish Species (7-18) - cleaned names
    'surgeonfish',    # 7 - Acanthuridae
    'triggerfish',    # 8 - Balistidae
    'jack',           # 9 - Carangidae
    'spadefish',      # 10 - Ephippidae
    'wrasse',         # 11 - Labridae
    'snapper',        # 12 - Lutjanidae
    'angelfish',      # 13 - Pomacanthidae
    'damselfish',     # 14 - Pomacentridae
    'parrotfish',     # 15 - Scaridae
    'tuna',           # 16 - Scombridae
    'grouper',        # 17 - Serranidae
    # shark already at 4
    'moorish_idol',   # 18 - Zanclidae
    
    # Fish Disease (19-22)
    'bacterial_gill_disease',  # 19
    'bacterial_red_disease',   # 20
    'bacterial_disease',       # 21
    'healthy_fish',            # 22
]

# Mapping from original dataset class IDs to unified IDs
EXISTING_MAPPING = {
    0: 0,   # fish -> fish
    1: 1,   # jellyfish -> jellyfish
    2: 2,   # penguin -> penguin
    3: 3,   # puffin -> puffin
    4: 4,   # shark -> shark
    5: 5,   # starfish -> starfish
    6: 6,   # stingray -> stingray
}

FISH_SPECIES_MAPPING = {
    0: 7,   # Acanthuridae -> surgeonfish
    1: 8,   # Balistidae -> triggerfish
    2: 9,   # Carangidae -> jack
    3: 10,  # Ephippidae -> spadefish
    4: 11,  # Labridae -> wrasse
    5: 12,  # Lutjanidae -> snapper
    6: 13,  # Pomacanthidae -> angelfish
    7: 14,  # Pomacentridae -> damselfish
    8: 15,  # Scaridae -> parrotfish
    9: 16,  # Scombridae -> tuna
    10: 17, # Serranidae -> grouper
    11: 4,  # Shark -> shark (reuse existing)
    12: 18, # Zanclidae -> moorish_idol
}

FISH_DISEASE_MAPPING = {
    0: 19,  # bacterial gill -> bacterial_gill_disease
    1: 20,  # bacterial red -> bacterial_red_disease
    2: 21,  # bacterial-disease -> bacterial_disease
    3: 22,  # healthy fish -> healthy_fish
}


def remap_labels(src_label_path: Path, dst_label_path: Path, class_mapping: dict):
    """Remap class IDs in a YOLO label file."""
    if not src_label_path.exists():
        return False
    
    with open(src_label_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            old_class_id = int(parts[0])
            if old_class_id in class_mapping:
                new_class_id = class_mapping[old_class_id]
                parts[0] = str(new_class_id)
                new_lines.append(' '.join(parts) + '\n')
    
    if new_lines:
        dst_label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label_path, 'w') as f:
            f.writelines(new_lines)
        return True
    return False


def copy_dataset(
    src_images_dir: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    class_mapping: dict,
    prefix: str = ""
):
    """Copy images and remap labels from source to destination."""
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    if not src_images_dir.exists():
        print(f"  Warning: {src_images_dir} does not exist")
        return 0
    
    images = list(src_images_dir.iterdir())
    for img_path in tqdm(images, desc=f"  Copying {prefix}", leave=False):
        if img_path.suffix.lower() not in image_extensions:
            continue
        
        # Generate unique filename with prefix
        new_name = f"{prefix}_{img_path.name}" if prefix else img_path.name
        dst_img_path = dst_images_dir / new_name
        
        # Copy image
        shutil.copy2(img_path, dst_img_path)
        
        # Find and remap label
        label_name = img_path.stem + '.txt'
        src_label_path = src_labels_dir / label_name
        dst_label_path = dst_labels_dir / (dst_img_path.stem + '.txt')
        
        if remap_labels(src_label_path, dst_label_path, class_mapping):
            copied += 1
    
    return copied


def merge_datasets(output_dir: str = "data/unified_dataset"):
    """Merge all datasets into a unified dataset."""
    output_path = Path(output_dir)
    
    # Clean output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create directory structure
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("Merging datasets into unified format...")
    print(f"Output directory: {output_path}")
    print(f"Total unified classes: {len(UNIFIED_CLASSES)}")
    print()
    
    total_train = 0
    total_val = 0
    
    # 1. Copy existing marine dataset
    print("1. Processing existing marine dataset...")
    existing_base = Path("data/detection")
    if existing_base.exists():
        train_count = copy_dataset(
            existing_base / "images" / "train",
            existing_base / "labels" / "train",
            output_path / "images" / "train",
            output_path / "labels" / "train",
            EXISTING_MAPPING,
            prefix="marine"
        )
        val_count = copy_dataset(
            existing_base / "images" / "val",
            existing_base / "labels" / "val",
            output_path / "images" / "val",
            output_path / "labels" / "val",
            EXISTING_MAPPING,
            prefix="marine"
        )
        total_train += train_count
        total_val += val_count
        print(f"   Added {train_count} train, {val_count} val images")
    else:
        print("   Skipped (not found)")
    
    # 2. Copy fish species dataset
    print("2. Processing fish species dataset...")
    species_base = Path("data/datasets/fish_species")
    if species_base.exists():
        train_count = copy_dataset(
            species_base / "train" / "images",
            species_base / "train" / "labels",
            output_path / "images" / "train",
            output_path / "labels" / "train",
            FISH_SPECIES_MAPPING,
            prefix="species"
        )
        val_count = copy_dataset(
            species_base / "valid" / "images",
            species_base / "valid" / "labels",
            output_path / "images" / "val",
            output_path / "labels" / "val",
            FISH_SPECIES_MAPPING,
            prefix="species"
        )
        total_train += train_count
        total_val += val_count
        print(f"   Added {train_count} train, {val_count} val images")
    else:
        print("   Skipped (not found)")
    
    # 3. Copy fish disease dataset
    print("3. Processing fish disease dataset...")
    disease_base = Path("data/datasets/fish_disease")
    if disease_base.exists():
        train_count = copy_dataset(
            disease_base / "train" / "images",
            disease_base / "train" / "labels",
            output_path / "images" / "train",
            output_path / "labels" / "train",
            FISH_DISEASE_MAPPING,
            prefix="disease"
        )
        val_count = copy_dataset(
            disease_base / "valid" / "images",
            disease_base / "valid" / "labels",
            output_path / "images" / "val",
            output_path / "labels" / "val",
            FISH_DISEASE_MAPPING,
            prefix="disease"
        )
        total_train += train_count
        total_val += val_count
        print(f"   Added {train_count} train, {val_count} val images")
    else:
        print("   Skipped (not found)")
    
    # Create unified dataset.yaml
    print("\n4. Creating unified dataset.yaml...")
    dataset_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(UNIFIED_CLASSES),
        'names': {i: name for i, name in enumerate(UNIFIED_CLASSES)}
    }
    
    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"   Created {yaml_path}")
    
    # Summary
    print("\n" + "="*50)
    print("MERGE COMPLETE")
    print("="*50)
    print(f"Total training images: {total_train}")
    print(f"Total validation images: {total_val}")
    print(f"Total classes: {len(UNIFIED_CLASSES)}")
    print(f"\nDataset ready at: {output_path}")
    print(f"Config file: {yaml_path}")
    
    # Print class summary
    print("\nUnified Classes:")
    print("-" * 40)
    for i, name in enumerate(UNIFIED_CLASSES):
        category = "Marine" if i < 7 else "Species" if i < 19 else "Disease"
        print(f"  {i:2d}: {name:25s} [{category}]")
    
    return str(yaml_path)


if __name__ == "__main__":
    merge_datasets()
