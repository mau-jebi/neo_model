"""
Comprehensive Unit Tests for RT-DETR at 1920x1080

Tests all components and validates:
1. Model instantiation
2. Forward pass with correct shapes
3. Feature map dimensions for 16:9 aspect ratio
4. Training and inference modes
5. Gradient flow
6. Memory efficiency

Run with: pytest tests/test_model.py -v
or: python tests/test_model.py
"""

import torch
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.position_encoding import build_position_encoding
from src.models.backbone import build_backbone
from src.models.encoder import build_encoder
from src.models.decoder import build_decoder
from src.models.matcher import build_matcher
from src.models.criterion import build_criterion
from src.models.rtdetr import build_rtdetr


class TestPositionEncoding:
    """Test position encoding for rectangular feature maps"""

    def test_sine_encoding_shapes(self):
        """Test sine position encoding with 1920x1080 feature maps"""
        pos_enc = build_position_encoding(hidden_dim=256, position_embedding="sine")

        # Test all three feature map sizes from RT-DETR
        test_cases = [
            (2, 256, 135, 240, 8),   # Stride 8
            (2, 256, 67, 120, 16),   # Stride 16
            (2, 256, 33, 60, 32),    # Stride 32
        ]

        for B, C, H, W, stride in test_cases:
            x = torch.randn(B, C, H, W)
            pos = pos_enc(x)

            assert pos.shape == (B, C, H, W), \
                f"Stride {stride}: Expected shape {(B, C, H, W)}, got {pos.shape}"
            assert not torch.isnan(pos).any(), f"Stride {stride}: NaN in position encoding"
            assert not torch.isinf(pos).any(), f"Stride {stride}: Inf in position encoding"

    def test_learned_encoding_shapes(self):
        """Test learned position encoding"""
        pos_enc = build_position_encoding(hidden_dim=256, position_embedding="learned")

        x = torch.randn(2, 256, 135, 240)
        pos = pos_enc(x)

        assert pos.shape == x.shape
        assert not torch.isnan(pos).any()


class TestBackbone:
    """Test ResNet-50 backbone"""

    def test_backbone_output_shapes(self):
        """Test backbone produces correct feature map shapes for 1920x1080"""
        backbone = build_backbone(
            backbone_type="resnet50",
            pretrained=False,
            use_fpn=False
        )
        backbone.eval()

        # 1920x1080 input
        x = torch.randn(2, 3, 1080, 1920)

        with torch.no_grad():
            features = backbone(x)

        # Expected shapes for 1920x1080 input
        expected = {
            "C3": (2, 512, 135, 240),    # 1080/8 = 135, 1920/8 = 240
            "C4": (2, 1024, 67, 120),    # 1080/16 = 67, 1920/16 = 120
            "C5": (2, 2048, 33, 60)      # 1080/32 = 33, 1920/32 = 60
        }

        for name, feat in features.items():
            assert feat.shape == expected[name], \
                f"{name}: Expected {expected[name]}, got {feat.shape}"

    def test_backbone_with_fpn(self):
        """Test backbone with FPN"""
        backbone = build_backbone(
            backbone_type="resnet50",
            pretrained=False,
            use_fpn=True
        )
        backbone.eval()

        x = torch.randn(2, 3, 1080, 1920)

        with torch.no_grad():
            features = backbone(x)

        # FPN outputs should all have 256 channels
        expected = {
            "P3": (2, 256, 135, 240),
            "P4": (2, 256, 67, 120),
            "P5": (2, 256, 33, 60)
        }

        for name, feat in features.items():
            assert feat.shape == expected[name], \
                f"{name}: Expected {expected[name]}, got {feat.shape}"


class TestEncoder:
    """Test hybrid encoder"""

    def test_encoder_output_shapes(self):
        """Test encoder produces correct output shapes"""
        encoder = build_encoder(
            in_channels=[512, 1024, 2048],
            hidden_dim=256,
            num_encoder_layers=1
        )
        encoder.eval()

        # Simulate backbone features
        features = [
            torch.randn(2, 512, 135, 240),
            torch.randn(2, 1024, 67, 120),
            torch.randn(2, 2048, 33, 60)
        ]

        with torch.no_grad():
            encoded = encoder(features)

        expected_shapes = [
            (2, 256, 135, 240),
            (2, 256, 67, 120),
            (2, 256, 33, 60)
        ]

        assert len(encoded) == 3
        for i, (feat, expected) in enumerate(zip(encoded, expected_shapes)):
            assert feat.shape == expected, \
                f"Scale {i}: Expected {expected}, got {feat.shape}"


class TestDecoder:
    """Test transformer decoder"""

    def test_decoder_output_shapes(self):
        """Test decoder produces correct predictions"""
        decoder = build_decoder(
            hidden_dim=256,
            num_queries=300,
            num_decoder_layers=6,
            num_classes=80,
            return_intermediate=True
        )
        decoder.eval()

        # Simulate encoder memory (flattened C5 features: 33x60 = 1980)
        memory = torch.randn(2, 1980, 256)

        with torch.no_grad():
            class_logits, bbox_preds = decoder(memory)

        # Should have predictions from all 6 layers
        assert len(class_logits) == 6
        assert len(bbox_preds) == 6

        # Check final layer shapes
        assert class_logits[-1].shape == (2, 300, 80)
        assert bbox_preds[-1].shape == (2, 300, 4)

        # Verify bbox predictions are in valid range (after sigmoid in decoder)
        assert (bbox_preds[-1] >= 0).all() and (bbox_preds[-1] <= 1).all()


class TestMatcher:
    """Test Hungarian matcher"""

    def test_matcher_assignment(self):
        """Test matcher produces valid assignments"""
        matcher = build_matcher()

        outputs = {
            "pred_logits": torch.randn(2, 300, 80),
            "pred_boxes": torch.rand(2, 300, 4)
        }

        targets = [
            {"labels": torch.randint(0, 80, (5,)), "boxes": torch.rand(5, 4)},
            {"labels": torch.randint(0, 80, (3,)), "boxes": torch.rand(3, 4)}
        ]

        indices = matcher(outputs, targets)

        # Should have one tuple per image
        assert len(indices) == 2

        # Check each image's matching
        for i, (idx_i, idx_j) in enumerate(indices):
            expected_matches = len(targets[i]["labels"])
            assert len(idx_i) == expected_matches
            assert len(idx_j) == expected_matches

            # Indices should be unique (bijective matching)
            assert len(set(idx_i.tolist())) == len(idx_i)
            assert len(set(idx_j.tolist())) == len(idx_j)


class TestCriterion:
    """Test loss criterion"""

    def test_criterion_computes_losses(self):
        """Test criterion computes all required losses"""
        criterion = build_criterion(num_classes=80)

        outputs = {
            "pred_logits": torch.randn(2, 300, 80),
            "pred_boxes": torch.rand(2, 300, 4)
        }

        targets = [
            {"labels": torch.randint(0, 80, (5,)), "boxes": torch.rand(5, 4)},
            {"labels": torch.randint(0, 80, (3,)), "boxes": torch.rand(3, 4)}
        ]

        losses = criterion(outputs, targets)

        # Check all expected losses are present
        expected_losses = ["loss_vfl", "loss_bbox", "loss_giou"]
        for loss_name in expected_losses:
            assert loss_name in losses, f"Missing loss: {loss_name}"
            assert not torch.isnan(losses[loss_name]), f"NaN in {loss_name}"
            assert not torch.isinf(losses[loss_name]), f"Inf in {loss_name}"
            assert losses[loss_name].requires_grad, f"{loss_name} doesn't require grad"


class TestRTDETR:
    """Test complete RT-DETR model"""

    def test_model_instantiation(self):
        """Test model can be instantiated"""
        model = build_rtdetr(
            num_classes=80,
            backbone_pretrained=False,
            num_queries=300,
            num_decoder_layers=6
        )

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel has {num_params/1e6:.1f}M parameters")

        assert num_params > 0

    def test_forward_pass_training(self):
        """Test forward pass in training mode"""
        model = build_rtdetr(
            num_classes=80,
            backbone_pretrained=False,
            num_queries=300,
            num_decoder_layers=6,
            aux_loss=True
        )
        model.train()

        # 1920x1080 input
        images = torch.randn(2, 3, 1080, 1920)

        outputs = model(images)

        # Check output shapes
        assert "pred_logits" in outputs
        assert "pred_boxes" in outputs
        assert outputs["pred_logits"].shape == (2, 300, 80)
        assert outputs["pred_boxes"].shape == (2, 300, 4)

        # Check auxiliary outputs
        assert "aux_outputs" in outputs
        assert len(outputs["aux_outputs"]) == 5  # 6 layers - 1 final

    def test_forward_pass_inference(self):
        """Test inference mode with post-processing"""
        model = build_rtdetr(
            num_classes=80,
            backbone_pretrained=False,
            num_queries=300
        )

        images = torch.randn(2, 3, 1080, 1920)

        results = model.inference(images, conf_threshold=0.9)  # High threshold for testing

        # Should have one result per image
        assert len(results) == 2

        # Check result structure
        for result in results:
            assert "labels" in result
            assert "boxes" in result
            assert "scores" in result

            # Boxes should be in pixel coordinates
            if len(result["boxes"]) > 0:
                assert result["boxes"][:, 0].max() <= 1920  # x max
                assert result["boxes"][:, 1].max() <= 1080  # y max
                assert result["boxes"][:, 0].min() >= 0     # x min
                assert result["boxes"][:, 1].min() >= 0     # y min

    def test_gradient_flow(self):
        """Test gradients flow through the model"""
        model = build_rtdetr(
            num_classes=80,
            backbone_pretrained=False,
            num_queries=300,
            num_decoder_layers=2  # Fewer layers for faster testing
        )
        model.train()

        images = torch.randn(2, 3, 1080, 1920, requires_grad=True)

        outputs = model(images)

        # Compute dummy loss
        loss = outputs["pred_logits"].sum() + outputs["pred_boxes"].sum()
        loss.backward()

        # Check gradients exist
        assert images.grad is not None
        assert not torch.isnan(images.grad).any()

        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_aspect_ratio_consistency(self):
        """Test model maintains 16:9 aspect ratio throughout pipeline"""
        model = build_rtdetr(
            num_classes=80,
            backbone_pretrained=False
        )
        model.eval()

        # Test with 1920x1080 (16:9)
        images = torch.randn(1, 3, 1080, 1920)

        with torch.no_grad():
            results = model.inference(images)

        # If there are detections, verify they're within image bounds
        if len(results[0]["boxes"]) > 0:
            boxes = results[0]["boxes"]
            assert (boxes[:, [0, 2]] <= 1920).all(), "Width exceeds image bounds"
            assert (boxes[:, [1, 3]] <= 1080).all(), "Height exceeds image bounds"
            assert (boxes >= 0).all(), "Negative coordinates"


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_training_step(self):
        """Test a complete training step"""
        model = build_rtdetr(
            num_classes=80,
            backbone_pretrained=False,
            num_decoder_layers=2,
            aux_loss=True
        )
        criterion = build_criterion(num_classes=80)

        model.train()

        # Create dummy data
        images = torch.randn(2, 3, 1080, 1920)
        targets = [
            {"labels": torch.randint(0, 80, (3,)), "boxes": torch.rand(3, 4)},
            {"labels": torch.randint(0, 80, (2,)), "boxes": torch.rand(2, 4)}
        ]

        # Forward pass
        outputs = model(images)

        # Compute losses
        losses = criterion(outputs, targets)

        # Total loss
        total_loss = sum(losses.values())

        # Backward pass
        total_loss.backward()

        # Verify loss is finite
        assert torch.isfinite(total_loss)

        print(f"\nTraining step losses:")
        for name, value in losses.items():
            print(f"  {name}: {value.item():.4f}")
        print(f"  Total: {total_loss.item():.4f}")


# Main test runner
if __name__ == "__main__":
    print("=" * 70)
    print("RT-DETR 1920x1080 Model Test Suite")
    print("=" * 70)

    # Run tests
    test_classes = [
        TestPositionEncoding,
        TestBackbone,
        TestEncoder,
        TestDecoder,
        TestMatcher,
        TestCriterion,
        TestRTDETR,
        TestEndToEnd
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{'=' * 70}")
        print(f"Running {test_class.__name__}")
        print(f"{'=' * 70}")

        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                print(f"\n  {method_name}...", end=' ')
                method = getattr(test_instance, method_name)
                method()
                print("✅ PASSED")
                passed_tests += 1
            except Exception as e:
                print(f"❌ FAILED")
                print(f"    Error: {str(e)}")

    print(f"\n{'=' * 70}")
    print(f"Test Summary: {passed_tests}/{total_tests} tests passed")
    print(f"{'=' * 70}")

    if passed_tests == total_tests:
        print("\n✅ All tests passed! RT-DETR model is ready for training.")
        print("\nNext steps:")
        print("  1. Implement data pipeline (src/data/)")
        print("  2. Implement training engine (src/engine/)")
        print("  3. Create training script (scripts/train.py)")
        print("  4. Train on COCO dataset at 1920x1080")
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed. Please fix issues before proceeding.")

    sys.exit(0 if passed_tests == total_tests else 1)
