"""
Tests for Detection Module

Run with: pytest tests/test_detection.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.utils import (
    calculate_iou,
    calculate_metrics,
    validate_labels,
    split_dataset
)


# Fixtures

@pytest.fixture
def sample_box():
    """Create a sample bounding box."""
    return {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}


@pytest.fixture
def overlapping_box():
    """Create an overlapping bounding box."""
    return {'x1': 30, 'y1': 30, 'x2': 70, 'y2': 70}


@pytest.fixture
def non_overlapping_box():
    """Create a non-overlapping bounding box."""
    return {'x1': 100, 'y1': 100, 'x2': 150, 'y2': 150}


@pytest.fixture
def identical_box():
    """Create an identical bounding box."""
    return {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directories
        images_dir = Path(tmpdir) / 'images'
        labels_dir = Path(tmpdir) / 'labels'
        images_dir.mkdir()
        labels_dir.mkdir()
        
        # Create sample images (empty files for testing)
        for i in range(5):
            (images_dir / f'image_{i}.jpg').touch()
            
            # Create corresponding label
            with open(labels_dir / f'image_{i}.txt', 'w') as f:
                f.write(f"0 0.5 0.5 0.3 0.3\n")  # Valid YOLO format
        
        # Add an orphan label
        with open(labels_dir / 'orphan.txt', 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
        
        # Add an image without label
        (images_dir / 'no_label.jpg').touch()
        
        yield {
            'tmpdir': tmpdir,
            'images_dir': images_dir,
            'labels_dir': labels_dir
        }


# IoU Tests

class TestIoU:
    
    def test_iou_identical_boxes(self, sample_box, identical_box):
        """Test IoU of identical boxes is 1.0."""
        iou = calculate_iou(sample_box, identical_box)
        assert iou == pytest.approx(1.0)
    
    def test_iou_non_overlapping(self, sample_box, non_overlapping_box):
        """Test IoU of non-overlapping boxes is 0.0."""
        iou = calculate_iou(sample_box, non_overlapping_box)
        assert iou == pytest.approx(0.0)
    
    def test_iou_partial_overlap(self, sample_box, overlapping_box):
        """Test IoU of partially overlapping boxes."""
        iou = calculate_iou(sample_box, overlapping_box)
        
        # IoU should be between 0 and 1
        assert 0 < iou < 1
        
        # Calculate expected IoU
        # Intersection: (30,30) to (50,50) = 20x20 = 400
        # Box1 area: 40x40 = 1600
        # Box2 area: 40x40 = 1600
        # Union: 1600 + 1600 - 400 = 2800
        # IoU: 400/2800 ≈ 0.143
        assert iou == pytest.approx(400/2800, rel=0.01)
    
    def test_iou_contained_box(self):
        """Test IoU when one box contains another."""
        outer = {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}
        inner = {'x1': 25, 'y1': 25, 'x2': 75, 'y2': 75}
        
        iou = calculate_iou(outer, inner)
        
        # IoU = inner_area / outer_area = 2500 / 10000 = 0.25
        assert iou == pytest.approx(0.25)


# Metrics Tests

class TestMetrics:
    
    def test_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        predictions = [
            {'class_id': 0, 'confidence': 0.9, 'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}}
        ]
        ground_truth = [
            {'class_id': 0, 'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}}
        ]
        
        metrics = calculate_metrics(predictions, ground_truth)
        
        assert metrics['precision'] == pytest.approx(1.0)
        assert metrics['recall'] == pytest.approx(1.0)
        assert metrics['f1'] == pytest.approx(1.0)
    
    def test_metrics_no_predictions(self):
        """Test metrics with no predictions."""
        predictions = []
        ground_truth = [
            {'class_id': 0, 'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}}
        ]
        
        metrics = calculate_metrics(predictions, ground_truth)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
    
    def test_metrics_no_ground_truth(self):
        """Test metrics with no ground truth."""
        predictions = [
            {'class_id': 0, 'confidence': 0.9, 'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}}
        ]
        ground_truth = []
        
        metrics = calculate_metrics(predictions, ground_truth)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
    
    def test_metrics_wrong_class(self):
        """Test metrics with wrong class prediction."""
        predictions = [
            {'class_id': 1, 'confidence': 0.9, 'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}}
        ]
        ground_truth = [
            {'class_id': 0, 'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}}
        ]
        
        metrics = calculate_metrics(predictions, ground_truth)
        
        # Wrong class should be false positive
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0


# Label Validation Tests

class TestLabelValidation:
    
    def test_validate_labels_finds_issues(self, temp_dataset):
        """Test that validation finds all issues."""
        issues = validate_labels(
            str(temp_dataset['labels_dir']),
            str(temp_dataset['images_dir']),
            num_classes=7
        )
        
        # Should find missing label for 'no_label.jpg'
        assert len(issues['missing_labels']) == 1
        
        # Should find orphan label
        assert len(issues['orphan_labels']) == 1
    
    def test_validate_labels_invalid_class(self, temp_dataset):
        """Test detection of invalid class IDs."""
        # Create label with invalid class ID
        with open(temp_dataset['labels_dir'] / 'image_0.txt', 'w') as f:
            f.write("99 0.5 0.5 0.3 0.3\n")  # Invalid class ID
        
        issues = validate_labels(
            str(temp_dataset['labels_dir']),
            str(temp_dataset['images_dir']),
            num_classes=7
        )
        
        assert len(issues['invalid_class_id']) >= 1
    
    def test_validate_labels_invalid_bbox(self, temp_dataset):
        """Test detection of invalid bounding boxes."""
        # Create label with invalid bbox (values > 1)
        with open(temp_dataset['labels_dir'] / 'image_0.txt', 'w') as f:
            f.write("0 1.5 0.5 0.3 0.3\n")  # x_center > 1
        
        issues = validate_labels(
            str(temp_dataset['labels_dir']),
            str(temp_dataset['images_dir']),
            num_classes=7
        )
        
        assert len(issues['invalid_bbox']) >= 1


# Dataset Split Tests

class TestDatasetSplit:
    
    def test_split_creates_directories(self, temp_dataset):
        """Test that split creates proper directory structure."""
        output_dir = Path(temp_dataset['tmpdir']) / 'split'
        
        split_dataset(
            str(temp_dataset['images_dir']),
            str(temp_dataset['labels_dir']),
            str(output_dir),
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Check directories exist
        assert (output_dir / 'images' / 'train').exists()
        assert (output_dir / 'images' / 'val').exists()
        assert (output_dir / 'images' / 'test').exists()
        assert (output_dir / 'labels' / 'train').exists()
        assert (output_dir / 'labels' / 'val').exists()
        assert (output_dir / 'labels' / 'test').exists()


# Integration Tests

class TestDetectionIntegration:
    
    def test_detector_import(self):
        """Test that detector can be imported (may require ultralytics)."""
        try:
            from detection.detect import MarineDetector
            assert True
        except ImportError:
            # Skip if ultralytics not installed
            pytest.skip("ultralytics not installed")
    
    def test_utils_import(self):
        """Test that all utilities can be imported."""
        from detection.utils import (
            convert_coco_to_yolo,
            validate_labels,
            split_dataset,
            calculate_metrics,
            calculate_iou
        )
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
