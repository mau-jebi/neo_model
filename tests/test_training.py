"""Unit tests for training infrastructure.

Tests data loading, transforms, training loop, evaluation, and checkpointing.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, ConfigDict
from src.data.transforms import (
    RandomHorizontalFlip, RandomResize, Resize, Normalize,
    ColorJitter, build_transforms
)
from src.data.data_utils import (
    box_cxcywh_to_xyxy, box_xyxy_to_cxcywh,
    compute_aspect_ratio_16_9, get_valid_16_9_sizes,
    clip_boxes, normalize_boxes
)
from src.data.collate import collate_fn
from src.engine.optimizer import build_optimizer, get_parameter_groups
from src.engine.scheduler import WarmupMultiStepLR, build_scheduler
from src.utils.checkpoint import CheckpointManager
from src.utils.metrics import COCOMetrics, convert_predictions_to_coco_format
from src.utils.misc import AverageMeter, Timer, set_seed


class TestConfig:
    """Test configuration loading and validation."""

    def test_load_config(self):
        """Test loading configuration file."""
        config_path = Path(__file__).parent.parent / "configs" / "rtdetr_r50_1920x1080.yml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        config = load_config(config_path)

        assert config.input.width == 1920
        assert config.input.height == 1080
        assert config.data.num_classes == 80
        assert config.training.epochs == 72

    def test_aspect_ratio_validation(self):
        """Test 16:9 aspect ratio validation."""
        assert compute_aspect_ratio_16_9(1920, 1080)
        assert compute_aspect_ratio_16_9(1600, 900)
        assert not compute_aspect_ratio_16_9(640, 640)
        assert not compute_aspect_ratio_16_9(800, 600)

    def test_valid_16_9_sizes(self):
        """Test valid 16:9 resolution list."""
        sizes = get_valid_16_9_sizes()

        assert len(sizes) == 5
        assert (1920, 1080) in sizes

        for w, h in sizes:
            assert compute_aspect_ratio_16_9(w, h)


class TestDataUtils:
    """Test data utility functions."""

    def test_box_conversions(self):
        """Test bounding box format conversions."""
        # Create test boxes in xyxy format
        boxes_xyxy = torch.tensor([
            [10, 20, 50, 60],
            [100, 150, 200, 250]
        ], dtype=torch.float32)

        # Convert to cxcywh
        boxes_cxcywh = box_xyxy_to_cxcywh(boxes_xyxy)

        # Check center and size
        assert torch.allclose(boxes_cxcywh[0, 0], torch.tensor(30.0))  # cx
        assert torch.allclose(boxes_cxcywh[0, 1], torch.tensor(40.0))  # cy
        assert torch.allclose(boxes_cxcywh[0, 2], torch.tensor(40.0))  # w
        assert torch.allclose(boxes_cxcywh[0, 3], torch.tensor(40.0))  # h

        # Convert back to xyxy
        boxes_xyxy_back = box_cxcywh_to_xyxy(boxes_cxcywh)

        # Should match original
        assert torch.allclose(boxes_xyxy, boxes_xyxy_back)

    def test_clip_boxes(self):
        """Test box clipping to image boundaries."""
        boxes = torch.tensor([
            [-10, -20, 50, 60],      # Out of bounds left/top
            [100, 150, 2000, 1200],  # Out of bounds right/bottom
            [10, 20, 100, 200]       # Valid box
        ], dtype=torch.float32)

        clipped = clip_boxes(boxes, (1080, 1920))

        # First box should be clipped
        assert clipped[0, 0] == 0
        assert clipped[0, 1] == 0

        # Second box should be clipped
        assert clipped[1, 2] == 1920
        assert clipped[1, 3] == 1080

        # Third box should remain unchanged
        assert torch.allclose(clipped[2], boxes[2])

    def test_normalize_boxes(self):
        """Test box normalization."""
        boxes = torch.tensor([
            [0, 0, 1920, 1080],
            [960, 540, 1920, 1080]
        ], dtype=torch.float32)

        normalized = normalize_boxes(boxes, (1080, 1920))

        # First box should span full image (0, 0, 1, 1)
        assert torch.allclose(normalized[0], torch.tensor([0, 0, 1, 1]))

        # Second box
        assert torch.allclose(normalized[1, 0], torch.tensor(0.5))  # x1 = 960/1920
        assert torch.allclose(normalized[1, 1], torch.tensor(0.5))  # y1 = 540/1080


class TestTransforms:
    """Test data augmentation transforms."""

    def test_random_horizontal_flip(self):
        """Test horizontal flip transform."""
        transform = RandomHorizontalFlip(prob=1.0)  # Always flip

        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        boxes = np.array([[100, 200, 300, 400]], dtype=np.float32)
        target = {'boxes': boxes, 'labels': np.array([1])}

        flipped_image, flipped_target = transform(image, target)

        # Boxes should be flipped
        assert flipped_target['boxes'][0, 0] == 1920 - 300  # x1 = W - x2
        assert flipped_target['boxes'][0, 2] == 1920 - 100  # x2 = W - x1

    def test_resize_maintains_16_9(self):
        """Test resize maintains 16:9 aspect ratio."""
        transform = Resize((1920, 1080))

        image = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        target = {'boxes': boxes, 'labels': np.array([1])}

        resized_image, resized_target = transform(image, target)

        # Image should be resized to 1920x1080
        assert resized_image.shape == (1080, 1920, 3)

        # Boxes should be scaled proportionally
        scale_w = 1920 / 1600
        scale_h = 1080 / 900

        assert np.allclose(resized_target['boxes'][0, 0], 100 * scale_w, atol=1.0)
        assert np.allclose(resized_target['boxes'][0, 1], 100 * scale_h, atol=1.0)

    def test_normalize(self):
        """Test normalization transform."""
        transform = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        target = {'boxes': np.array([[100, 100, 200, 200]]), 'labels': np.array([1])}

        normalized_image, normalized_target = transform(image, target)

        # Should be a tensor
        assert isinstance(normalized_image, torch.Tensor)
        assert normalized_image.shape == (3, 1080, 1920)

        # Should be normalized (approximately mean 0, std 1)
        # Note: exact values depend on input, just check type and shape

    def test_random_resize_only_16_9(self):
        """Test random resize only accepts 16:9 sizes."""
        valid_sizes = [(1920, 1080), (1600, 900)]

        # Should not raise
        transform = RandomResize(valid_sizes)

        # Invalid size should raise
        with pytest.raises(ValueError):
            RandomResize([(640, 640)])  # Not 16:9


class TestCollate:
    """Test collate function for batching."""

    def test_collate_fn(self):
        """Test collate function."""
        # Create sample batch
        batch = [
            {
                'image': torch.randn(3, 1080, 1920),
                'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
                'labels': torch.tensor([1, 2]),
                'image_id': 1,
                'orig_size': (1080, 1920)
            },
            {
                'image': torch.randn(3, 1080, 1920),
                'boxes': torch.tensor([[50, 50, 150, 150]]),
                'labels': torch.tensor([3]),
                'image_id': 2,
                'orig_size': (1080, 1920)
            }
        ]

        collated = collate_fn(batch)

        # Images should be stacked
        assert collated['images'].shape == (2, 3, 1080, 1920)

        # Boxes and labels should be lists (variable length)
        assert isinstance(collated['boxes'], list)
        assert isinstance(collated['labels'], list)
        assert len(collated['boxes']) == 2
        assert len(collated['labels']) == 2

        # First sample has 2 boxes, second has 1
        assert collated['boxes'][0].shape == (2, 4)
        assert collated['boxes'][1].shape == (1, 4)


class TestOptimizer:
    """Test optimizer building."""

    def test_build_optimizer(self):
        """Test optimizer construction with parameter grouping."""
        from src.models.rtdetr import build_rtdetr

        model = build_rtdetr(num_classes=80)

        # Create minimal config
        config = ConfigDict({
            'training': {
                'optimizer': {
                    'type': 'AdamW',
                    'lr': 0.0001,
                    'weight_decay': 0.0001,
                    'betas': (0.9, 0.999)
                }
            }
        })

        optimizer = build_optimizer(config, model)

        # Should have 2 parameter groups (with/without decay)
        assert len(optimizer.param_groups) == 2

        # First group should have weight decay
        assert optimizer.param_groups[0]['weight_decay'] == 0.0001

        # Second group should have no weight decay
        assert optimizer.param_groups[1]['weight_decay'] == 0.0

    def test_parameter_grouping(self):
        """Test parameter grouping logic."""
        from src.models.rtdetr import build_rtdetr

        model = build_rtdetr(num_classes=80)

        groups = get_parameter_groups(model)

        # Should have both groups
        assert 'with_decay' in groups
        assert 'without_decay' in groups

        # Should have reasonable split
        assert groups['total_with_decay'] > 0
        assert groups['total_without_decay'] > 0


class TestScheduler:
    """Test learning rate scheduler."""

    def test_warmup_multistep_lr(self):
        """Test warmup multi-step LR scheduler."""
        from src.models.rtdetr import build_rtdetr

        model = build_rtdetr(num_classes=80)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=[40, 55],
            gamma=0.1,
            warmup_epochs=5,
            warmup_factor=0.001
        )

        # Initial LR should be warmup_factor * base_lr
        initial_lr = optimizer.param_groups[0]['lr']

        # After warmup, LR should be base_lr
        for _ in range(5):
            scheduler.step()

        after_warmup_lr = optimizer.param_groups[0]['lr']
        assert after_warmup_lr == pytest.approx(0.0001, abs=1e-6)

        # After milestone, LR should be reduced
        for _ in range(35):  # Move to epoch 40
            scheduler.step()

        after_milestone_lr = optimizer.param_groups[0]['lr']
        assert after_milestone_lr == pytest.approx(0.0001 * 0.1, abs=1e-7)


class TestCheckpoint:
    """Test checkpoint management."""

    def test_checkpoint_save_load(self):
        """Test checkpoint save and load."""
        from src.models.rtdetr import build_rtdetr

        model = build_rtdetr(num_classes=80)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(
                save_dir=tmpdir,
                save_best=True,
                save_last=True
            )

            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                epoch=0,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                metrics={'AP': 0.5}
            )

            # Check files exist
            assert (Path(tmpdir) / 'checkpoint_epoch_0.pth').exists()
            assert (Path(tmpdir) / 'last_checkpoint.pth').exists()
            assert (Path(tmpdir) / 'best_checkpoint.pth').exists()

            # Load checkpoint
            new_model = build_rtdetr(num_classes=80)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.0001)

            checkpoint = checkpoint_manager.load_checkpoint(
                Path(tmpdir) / 'last_checkpoint.pth',
                new_model,
                new_optimizer
            )

            assert checkpoint['epoch'] == 0
            assert checkpoint['metrics']['AP'] == 0.5


class TestMetrics:
    """Test metrics computation."""

    def test_average_meter(self):
        """Test average meter."""
        meter = AverageMeter('loss')

        meter.update(1.0, n=1)
        meter.update(2.0, n=1)
        meter.update(3.0, n=1)

        assert meter.avg == 2.0
        assert meter.count == 3

    def test_timer(self):
        """Test timer."""
        import time

        timer = Timer()
        timer.start()
        time.sleep(0.1)
        elapsed = timer.stop()

        assert elapsed >= 0.1


class TestMisc:
    """Test miscellaneous utilities."""

    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)

        # Generate random numbers
        a = torch.rand(10)
        b = np.random.rand(10)

        # Reset seed
        set_seed(42)

        # Should get same random numbers
        c = torch.rand(10)
        d = np.random.rand(10)

        assert torch.allclose(a, c)
        assert np.allclose(b, d)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
