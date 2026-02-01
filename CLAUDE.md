# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**neo_model** is an RT-DETR (Real-Time Detection Transformer) implementation adapted for native **1920×1080 (16:9 aspect ratio)** object detection on Jetson Thor GPU. The model is designed for the Jebi AI Engine and targets COCO dataset training with 80 object classes.

**Critical Constraint**: All image processing, transforms, and feature maps MUST maintain 16:9 aspect ratio throughout the entire pipeline. This is the fundamental architectural adaptation from standard RT-DETR implementations.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/verify_installation.py
```

### Dataset Preparation
```bash
# Download COCO 2017 dataset (~21GB total)
python scripts/prepare_data.py --data_dir data/coco --split both

# Verify only (if already downloaded)
python scripts/prepare_data.py --data_dir data/coco --verify_only
```

### Training
```bash
# Overfit test (ALWAYS run this first for sanity check)
python scripts/train.py --config configs/rtdetr_r50_1920x1080.yml --overfit_batches 5
# Expected: Loss should drop to < 1.0 within 50-100 iterations

# Full training
python scripts/train.py --config configs/rtdetr_r50_1920x1080.yml

# Resume from checkpoint
python scripts/train.py --config configs/rtdetr_r50_1920x1080.yml --resume checkpoints/last_checkpoint.pth

# Debug mode
python scripts/train.py --config configs/rtdetr_r50_1920x1080.yml --debug --overfit_batches 5
```

### Evaluation
```bash
# Evaluate checkpoint
python scripts/evaluate.py --config configs/rtdetr_r50_1920x1080.yml --checkpoint checkpoints/best_checkpoint.pth

# With custom thresholds
python scripts/evaluate.py --config configs/rtdetr_r50_1920x1080.yml \
    --checkpoint checkpoints/best_checkpoint.pth \
    --conf_threshold 0.3 --nms_threshold 0.5
```

### Monitoring
```bash
# Launch TensorBoard
tensorboard --logdir outputs/logs
# Access at http://localhost:6006
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run Phase 1 tests (model architecture)
pytest tests/test_model.py -v

# Run Phase 2 tests (training infrastructure)
pytest tests/test_training.py -v

# Run specific test class
pytest tests/test_training.py::TestTransforms -v

# Run single test
pytest tests/test_training.py::TestConfig::test_load_config -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Architecture Overview

### Two-Phase Implementation

**Phase 1: Model Architecture** (Complete)
- ResNet-50 backbone with multi-scale features (C3, C4, C5)
- Hybrid encoder (AIFI + CCFM) for feature enhancement
- 6-layer transformer decoder with denoising training
- VFL + L1 + GIoU loss with Hungarian matching

**Phase 2: Training Infrastructure** (Complete)
- COCO dataset loading with 16:9 transforms
- Mixed precision training (AMP) with gradient accumulation
- AdamW optimizer with parameter grouping
- MultiStepLR scheduler with linear warmup
- Checkpoint management and COCO evaluation

### Data Flow

```
Input Image (1920×1080)
    ↓
Backbone (ResNet-50)
    ↓ [C3: 512×135×240, C4: 1024×67×120, C5: 2048×33×60]
Hybrid Encoder
    ↓ [256×33×60 feature map]
Transformer Decoder (300 queries)
    ↓
Prediction Heads
    ↓
Output: {pred_logits: (B,300,80), pred_boxes: (B,300,4)}
```

### Critical Components

**1. 16:9 Aspect Ratio Preservation** (`src/data/transforms.py`)
- All transforms validate and maintain 16:9 ratio
- Valid multi-scale sizes: (1600,900), (1760,990), (1920,1080), (2080,1170), (2240,1260)
- ANY new transform MUST validate aspect ratio with `compute_aspect_ratio_16_9()`

**2. Positional Encoding** (`src/models/position_encoding.py`)
- Uses 16:9 dimensions (H=1080/32=33.75≈34, W=1920/32=60)
- Sine-cosine encoding adapted for rectangular feature maps
- NOT square like standard DETR implementations

**3. Training Loop** (`src/engine/trainer.py`)
- Mixed precision (FP16) for memory efficiency
- Gradient accumulation (physical batch=4, accumulation=2, effective=8)
- Linear warmup for 5 epochs to prevent divergence
- Gradient clipping (max_norm=0.1) for stability

**4. Loss Computation** (`src/models/criterion.py`)
- Varifocal Loss (VFL) for classification (weight=1.0)
- L1 Loss for box regression (weight=5.0)
- GIoU Loss for box refinement (weight=2.0)
- Hungarian matching for target assignment

### Box Format Conversions

The model uses different box formats at different stages:

- **COCO annotations**: `[x, y, width, height]` (top-left corner)
- **Model input**: `(x1, y1, x2, y2)` in xyxy format, pixel coordinates
- **Model internal**: `(cx, cy, w, h)` in cxcywh format, **normalized [0,1]**
- **Model output**: `(cx, cy, w, h)` in cxcywh format, **normalized [0,1]**
- **COCO evaluation**: `[x, y, width, height]` (top-left corner), pixel coordinates

Use utilities in `src/data/data_utils.py` for conversions:
- `box_xyxy_to_cxcywh()` / `box_cxcywh_to_xyxy()`
- `box_xywh_to_xyxy()` / `box_xyxy_to_xywh()`
- `normalize_boxes()` / `denormalize_boxes()`

## Configuration System

All hyperparameters are in `configs/rtdetr_r50_1920x1080.yml`.

**Key settings**:
- `input.width: 1920` and `input.height: 1080` - MUST be 16:9
- `training.use_amp: true` - Mixed precision (FP16)
- `training.accumulate_grad_batches: 2` - Gradient accumulation
- `training.epochs: 72` - Full training duration
- `training.lr_scheduler.milestones: [40, 55]` - LR decay points
- `data.train.batch_size: 4` - Maximum for 16GB GPU

Config is loaded with validation in `src/utils/config.py`:
```python
from src.utils.config import load_config
config = load_config('configs/rtdetr_r50_1920x1080.yml')
```

## Memory and Performance Constraints

### GPU Memory Management
- **Target**: 16GB VRAM (RTX 4000, Tesla T4, V100)
- **Peak usage**: 14-15GB with batch_size=4 + FP16
- **If OOM**: Reduce `batch_size` to 2, increase `accumulate_grad_batches` to 4

### Training Speed Expectations
- **Throughput**: 100-120 images/sec on RTX 3090
- **Epoch time**: ~1.5-2 hours (118K images)
- **Total training**: ~2-3 days for 72 epochs

### Data Loading Optimization
- Use `cv2.imread()` not `PIL.Image.open()` (3-5× faster for 1920×1080)
- `num_workers=8`, `prefetch_factor=4`, `persistent_workers=True`
- Store data on SSD not HDD for best performance

## When Modifying Code

### Adding New Transforms
1. MUST validate 16:9 aspect ratio in `__init__()`:
   ```python
   if not compute_aspect_ratio_16_9(width, height):
       raise ValueError(f"Size {width}x{height} is not 16:9")
   ```
2. Update boxes proportionally to image resizing
3. Add test in `tests/test_training.py`

### Changing Model Architecture
1. Update feature map sizes in config comments (stride 8/16/32)
2. Verify positional encoding dimensions match
3. Run `pytest tests/test_model.py` to validate forward pass
4. Test with overfit before full training

### Modifying Training Loop
1. Changes to AMP/gradient accumulation affect memory usage
2. Always test with overfit mode first
3. Monitor TensorBoard for unexpected behavior
4. Check that learning rate warmup completes properly

### Adding New Loss Components
1. Add to `src/models/criterion.py`
2. Include in `weight_dict` in config
3. Log to TensorBoard in trainer
4. Verify gradients flow properly (check for NaN)

## Common Debugging Patterns

### Overfit Test (Most Important Sanity Check)
```bash
python scripts/train.py --config configs/rtdetr_r50_1920x1080.yml --overfit_batches 5
```
- Loss should drop below 1.0 within 50-100 iterations
- If not, there's a fundamental issue (data loading, loss computation, optimization)

### Loss becomes NaN
1. Check gradient clipping is enabled (`clip_max_norm: 0.1`)
2. Reduce learning rate (`optimizer.lr: 5e-5`)
3. Increase warmup epochs (`warmup_epochs: 10`)
4. Verify box coordinates are valid (no negative/infinite values)

### Low AP after training
1. Verify pretrained backbone loaded: check logs for "Loading pretrained"
2. Check aspect ratio in transforms: run `pytest tests/test_training.py::TestTransforms`
3. Validate data augmentation not too aggressive
4. Ensure evaluation uses low `conf_threshold=0.01` for COCO

### Slow training
1. Check GPU utilization: `nvidia-smi -l 1` (should be >90%)
2. Increase `num_workers` if GPU util low and CPU has capacity
3. Verify data on SSD not HDD
4. Check `persistent_workers=True` in dataloader

## Integration with Existing Components

### Phase 1 Components (DO NOT MODIFY without careful consideration)
- `src/models/rtdetr.py` - Main model, extensively tested
- `src/models/backbone.py` - ResNet-50 with multi-scale outputs
- `src/models/encoder.py` - Hybrid encoder (AIFI + CCFM)
- `src/models/decoder.py` - Transformer decoder with denoising
- `src/models/criterion.py` - VFL + L1 + GIoU losses
- `src/models/matcher.py` - Hungarian matching algorithm
- `src/models/position_encoding.py` - 16:9 adapted positional encoding

All Phase 1 tests MUST continue passing: `pytest tests/test_model.py -v`

### Phase 2 Integration Points
- Training loop imports model from Phase 1: `from src.models.rtdetr import build_rtdetr`
- Criterion imported by trainer: `from src.models.criterion import build_criterion`
- Data pipeline is independent, only interfaces through `collate_fn`

## File Organization Logic

**Models** (`src/models/`): Pure PyTorch modules, no training logic
**Data** (`src/data/`): Dataset loading, transforms, collation
**Engine** (`src/engine/`): Training loop, evaluation, optimization
**Utils** (`src/utils/`): Configuration, logging, checkpointing, metrics, visualization
**Scripts** (`scripts/`): User-facing entry points
**Tests** (`tests/`): Unit tests organized by phase

## Expected Training Behavior

### Loss Trajectory
- Epoch 1: Loss ~8-10 (initialization)
- Epoch 10: Loss ~3-4
- Epoch 40: Loss ~2-3 (first LR drop)
- Epoch 55: Loss ~2-2.5 (second LR drop)
- Epoch 72: Loss ~1.8-2.5 (final)

### AP Progression
- Epoch 10: AP ~0.30-0.40
- Epoch 40: AP ~0.45-0.50
- Epoch 72: AP >0.52 (target)

If deviating significantly from these, investigate data/model/training issues.

## GPU-Specific Adaptations

### 16GB GPU (RTX 4000, T4, V100)
```yaml
data.train.batch_size: 4
training.accumulate_grad_batches: 2
training.use_amp: true
```

### 24GB GPU (RTX 3090, RTX 4090, A100)
```yaml
data.train.batch_size: 8
training.accumulate_grad_batches: 1
training.use_amp: true
```

### 12GB GPU (Reduce resolution or batch size)
Not recommended for 1920×1080. Consider reducing to 1600×900 temporarily.

## Key Files to Understand

When making changes, these files provide the "big picture":

1. **`src/models/rtdetr.py`** - Complete model pipeline, understand forward pass
2. **`src/engine/trainer.py`** - Training loop with all optimizations
3. **`src/data/transforms.py`** - 16:9 preservation logic
4. **`configs/rtdetr_r50_1920x1080.yml`** - All hyperparameters
5. **`scripts/train.py`** - Entry point, shows component assembly

## Project-Specific Conventions

- **All box coordinates**: Use xyxy format in data pipeline, cxcywh (normalized) in model
- **Image loading**: Use cv2 not PIL for performance
- **Logging**: Use logger from `src.utils.logger`, not print statements
- **Config access**: Use attribute notation (`config.training.epochs`) not dict notation
- **Tests**: Use pytest fixtures, mock COCO data for speed
- **Checkpoints**: Save every 5 epochs, always save best and last
- **Evaluation**: Use low `conf_threshold=0.01` for COCO metrics
- **Documentation**: Update IMPLEMENTATION_SUMMARY.md for major changes
