"""ONNX Export Module for RT-DETR Model.

Exports trained PyTorch models to ONNX format for deployment.
Optimized for 1920x1080 input resolution with 16:9 aspect ratio.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class RTDETRExportWrapper(nn.Module):
    """
    Wrapper for ONNX export that handles model output formatting.

    The exported model outputs raw predictions that can be post-processed
    by the deployment target (e.g., TensorRT, ONNX Runtime).
    """

    def __init__(self, model: nn.Module, num_classes: int = 80):
        """
        Args:
            model: Trained RT-DETR model
            num_classes: Number of classes (80 for COCO)
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONNX export.

        Args:
            images: Input tensor [B, 3, 1080, 1920]

        Returns:
            scores: Classification scores [B, num_queries, num_classes]
            boxes: Bounding boxes in cxcywh normalized format [B, num_queries, 4]
        """
        outputs = self.model(images)

        # Get predictions from the last decoder layer
        pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes]
        pred_boxes = outputs['pred_boxes']     # [B, num_queries, 4]

        # Apply sigmoid to get scores
        scores = pred_logits.sigmoid()

        return scores, pred_boxes


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 1080, 1920),
    opset_version: int = 17,
    dynamic_batch: bool = True,
    simplify: bool = True,
    verify: bool = True,
    fp16: bool = False,
    num_classes: int = 80
) -> Dict[str, Any]:
    """
    Export RT-DETR model to ONNX format.

    Args:
        model: Trained RT-DETR model (PyTorch)
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (B, C, H, W)
        opset_version: ONNX opset version (17 recommended for transformer models)
        dynamic_batch: Enable dynamic batch size
        simplify: Apply ONNX simplifier
        verify: Verify exported model against PyTorch output
        fp16: Export with FP16 precision
        num_classes: Number of object classes

    Returns:
        Dictionary with export metadata and verification results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare model for export
    model.eval()
    device = next(model.parameters()).device

    # Wrap model for cleaner ONNX output
    export_model = RTDETRExportWrapper(model, num_classes)
    export_model.eval()
    export_model.to(device)

    # Create dummy input
    batch_size, channels, height, width = input_shape
    dummy_input = torch.randn(input_shape, device=device)

    if fp16:
        dummy_input = dummy_input.half()
        export_model = export_model.half()

    # Define input/output names
    input_names = ['images']
    output_names = ['scores', 'boxes']

    # Define dynamic axes for batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'images': {0: 'batch_size'},
            'scores': {0: 'batch_size'},
            'boxes': {0: 'batch_size'}
        }

    logger.info(f"Exporting model to ONNX: {output_path}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Opset version: {opset_version}")
    logger.info(f"  Dynamic batch: {dynamic_batch}")
    logger.info(f"  FP16: {fp16}")

    # Export to ONNX
    try:
        torch.onnx.export(
            export_model,
            dummy_input,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
        logger.info("ONNX export successful")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise

    # Load and check the model
    onnx_model = onnx.load(str(output_path))

    try:
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")
    except onnx.checker.ValidationError as e:
        logger.error(f"ONNX model validation failed: {e}")
        raise

    # Simplify model if requested
    if simplify:
        try:
            import onnxsim
            logger.info("Simplifying ONNX model...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model, str(output_path))
                logger.info("ONNX simplification successful")
            else:
                logger.warning("ONNX simplification check failed, keeping original")
        except ImportError:
            logger.warning("onnxsim not installed, skipping simplification")
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}, keeping original")

    # Prepare metadata
    metadata = {
        'export_timestamp': datetime.utcnow().isoformat(),
        'input_shape': list(input_shape),
        'input_height': height,
        'input_width': width,
        'num_classes': num_classes,
        'opset_version': opset_version,
        'dynamic_batch': dynamic_batch,
        'fp16': fp16,
        'output_names': output_names,
        'model_size_mb': output_path.stat().st_size / (1024 * 1024)
    }

    # Verify model output if requested
    if verify:
        verification_result = verify_onnx_output(
            pytorch_model=export_model,
            onnx_path=str(output_path),
            input_shape=input_shape,
            device=device,
            fp16=fp16
        )
        metadata['verification'] = verification_result

    # Save metadata alongside ONNX model
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved export metadata to {metadata_path}")

    return metadata


def verify_onnx_output(
    pytorch_model: nn.Module,
    onnx_path: str,
    input_shape: Tuple[int, int, int, int],
    device: torch.device,
    fp16: bool = False,
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Verify ONNX model output matches PyTorch model.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to exported ONNX model
        input_shape: Input tensor shape
        device: PyTorch device
        fp16: Whether model uses FP16
        tolerance: Maximum acceptable difference

    Returns:
        Dictionary with verification results
    """
    logger.info("Verifying ONNX output against PyTorch...")

    # Create test input
    test_input = torch.randn(input_shape, device=device)
    if fp16:
        test_input = test_input.half()
        tolerance = 1e-2  # Relax tolerance for FP16

    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_scores, pytorch_boxes = pytorch_model(test_input)

    pytorch_scores = pytorch_scores.cpu().numpy()
    pytorch_boxes = pytorch_boxes.cpu().numpy()

    # Get ONNX output
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    onnx_input = test_input.cpu().numpy()
    onnx_outputs = session.run(None, {'images': onnx_input})
    onnx_scores, onnx_boxes = onnx_outputs

    # Compare outputs
    scores_diff = np.abs(pytorch_scores - onnx_scores).max()
    boxes_diff = np.abs(pytorch_boxes - onnx_boxes).max()

    scores_match = scores_diff < tolerance
    boxes_match = boxes_diff < tolerance

    result = {
        'passed': scores_match and boxes_match,
        'scores_max_diff': float(scores_diff),
        'boxes_max_diff': float(boxes_diff),
        'tolerance': tolerance,
        'scores_match': scores_match,
        'boxes_match': boxes_match
    }

    if result['passed']:
        logger.info(f"Verification PASSED (scores_diff={scores_diff:.6f}, boxes_diff={boxes_diff:.6f})")
    else:
        logger.warning(f"Verification FAILED (scores_diff={scores_diff:.6f}, boxes_diff={boxes_diff:.6f})")

    return result


def benchmark_onnx_model(
    onnx_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 1080, 1920),
    num_warmup: int = 10,
    num_runs: int = 100,
    use_cuda: bool = True
) -> Dict[str, float]:
    """
    Benchmark ONNX model inference speed.

    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        use_cuda: Use CUDA execution provider if available

    Returns:
        Dictionary with benchmark results (latency in ms)
    """
    import time

    logger.info(f"Benchmarking ONNX model: {onnx_path}")

    # Setup session
    providers = []
    if use_cuda:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    session = ort.InferenceSession(onnx_path, providers=providers)
    active_provider = session.get_providers()[0]
    logger.info(f"Using provider: {active_provider}")

    # Create test input
    test_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        session.run(None, {'images': test_input})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {'images': test_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    results = {
        'provider': active_provider,
        'input_shape': list(input_shape),
        'num_runs': num_runs,
        'mean_latency_ms': float(latencies.mean()),
        'std_latency_ms': float(latencies.std()),
        'min_latency_ms': float(latencies.min()),
        'max_latency_ms': float(latencies.max()),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'throughput_fps': float(1000 / latencies.mean())
    }

    logger.info(f"Benchmark results:")
    logger.info(f"  Mean latency: {results['mean_latency_ms']:.2f} ms")
    logger.info(f"  Throughput: {results['throughput_fps']:.1f} FPS")
    logger.info(f"  P95 latency: {results['p95_latency_ms']:.2f} ms")

    return results


def optimize_onnx_for_inference(
    input_path: str,
    output_path: str,
    optimization_level: int = 99
) -> str:
    """
    Apply ONNX Runtime optimizations to the model.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model
        optimization_level: ORT optimization level (0-99)

    Returns:
        Path to optimized model
    """
    logger.info(f"Optimizing ONNX model with level {optimization_level}")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimized_model_filepath = output_path

    # This will save the optimized model
    _ = ort.InferenceSession(
        input_path,
        session_options,
        providers=['CPUExecutionProvider']
    )

    logger.info(f"Optimized model saved to: {output_path}")
    return output_path


def export_checkpoint_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    input_height: int = 1080,
    input_width: int = 1920,
    num_classes: int = 80,
    num_queries: int = 300,
    **export_kwargs
) -> Dict[str, Any]:
    """
    Export a training checkpoint directly to ONNX.

    This is a convenience function that loads a checkpoint and exports it.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pth)
        output_path: Path to save ONNX model
        config_path: Optional path to training config
        input_height: Input image height (default: 1080)
        input_width: Input image width (default: 1920)
        num_classes: Number of classes
        num_queries: Number of object queries
        **export_kwargs: Additional arguments for export_to_onnx

    Returns:
        Export metadata dictionary
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.models.rtdetr import build_rtdetr

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Build model
    model = build_rtdetr(
        num_classes=num_classes,
        num_queries=num_queries,
        backbone_pretrained=False
    )

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully on {device}")

    # Export to ONNX
    input_shape = (1, 3, input_height, input_width)

    metadata = export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=input_shape,
        num_classes=num_classes,
        **export_kwargs
    )

    # Add checkpoint info to metadata
    metadata['checkpoint_path'] = str(checkpoint_path)
    if 'epoch' in checkpoint:
        metadata['checkpoint_epoch'] = checkpoint['epoch']
    if 'metrics' in checkpoint:
        metadata['checkpoint_metrics'] = checkpoint['metrics']

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export RT-DETR model to ONNX")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save ONNX model')
    parser.add_argument('--height', type=int, default=1080,
                       help='Input image height')
    parser.add_argument('--width', type=int, default=1920,
                       help='Input image width')
    parser.add_argument('--num-classes', type=int, default=80,
                       help='Number of classes')
    parser.add_argument('--opset', type=int, default=17,
                       help='ONNX opset version')
    parser.add_argument('--fp16', action='store_true',
                       help='Export with FP16 precision')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Skip ONNX simplification')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark after export')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Export model
    metadata = export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_height=args.height,
        input_width=args.width,
        num_classes=args.num_classes,
        opset_version=args.opset,
        fp16=args.fp16,
        simplify=not args.no_simplify
    )

    print(f"\nExport completed successfully!")
    print(f"  Output: {args.output}")
    print(f"  Size: {metadata['model_size_mb']:.2f} MB")

    if metadata.get('verification', {}).get('passed'):
        print(f"  Verification: PASSED")
    else:
        print(f"  Verification: FAILED")

    # Run benchmark if requested
    if args.benchmark:
        print("\nRunning benchmark...")
        bench_results = benchmark_onnx_model(
            args.output,
            input_shape=(1, 3, args.height, args.width)
        )
        print(f"  Throughput: {bench_results['throughput_fps']:.1f} FPS")
        print(f"  Mean latency: {bench_results['mean_latency_ms']:.2f} ms")
