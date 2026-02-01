"""Verify neo_model installation and setup.

Checks that all components are correctly installed and configured.
Run this before starting training to catch any issues early.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("neo_model Installation Verification")
print("=" * 60)

# Check Python version
print("\n1. Checking Python version...")
py_version = sys.version_info
if py_version.major == 3 and py_version.minor >= 8:
    print(f"   ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
else:
    print(f"   ✗ Python {py_version.major}.{py_version.minor}.{py_version.micro} (requires 3.8+)")
    sys.exit(1)

# Check dependencies
print("\n2. Checking dependencies...")
dependencies = {
    'torch': '2.x',
    'torchvision': '0.x',
    'numpy': '1.x',
    'cv2': 'opencv-python',
    'albumentations': '1.x',
    'pycocotools': '2.x',
    'tensorboard': '2.x',
    'yaml': 'pyyaml',
    'tqdm': '4.x',
    'pytest': '7.x'
}

missing_deps = []
for dep, version in dependencies.items():
    try:
        if dep == 'yaml':
            import yaml
        elif dep == 'cv2':
            import cv2
        else:
            __import__(dep)
        print(f"   ✓ {dep} ({version})")
    except ImportError:
        print(f"   ✗ {dep} NOT FOUND")
        missing_deps.append(f"{dep} ({version})")

if missing_deps:
    print(f"\n   Missing dependencies: {', '.join(missing_deps)}")
    print(f"   Install with: pip install -r requirements.txt")
    sys.exit(1)

# Check PyTorch CUDA
print("\n3. Checking PyTorch CUDA...")
import torch
if torch.cuda.is_available():
    print(f"   ✓ CUDA available (version {torch.version.cuda})")
    print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(f"   ⚠ CUDA not available (will use CPU - very slow)")

# Check project structure
print("\n4. Checking project structure...")
required_dirs = [
    'src/models',
    'src/data',
    'src/engine',
    'src/utils',
    'scripts',
    'tests',
    'configs'
]

for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"   ✓ {dir_path}/")
    else:
        print(f"   ✗ {dir_path}/ NOT FOUND")
        sys.exit(1)

# Check key files
print("\n5. Checking key files...")
key_files = [
    'src/models/rtdetr.py',
    'src/data/coco_dataset.py',
    'src/engine/trainer.py',
    'scripts/train.py',
    'configs/rtdetr_r50_1920x1080.yml'
]

for file_path in key_files:
    if Path(file_path).exists():
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} NOT FOUND")
        sys.exit(1)

# Check config loading
print("\n6. Checking configuration...")
try:
    from src.utils.config import load_config
    config = load_config('configs/rtdetr_r50_1920x1080.yml')
    print(f"   ✓ Config loaded successfully")
    print(f"   ✓ Input: {config.input.width}×{config.input.height}")
    print(f"   ✓ Classes: {config.data.num_classes}")
    print(f"   ✓ Epochs: {config.training.epochs}")

    # Check 16:9 aspect ratio
    ratio = config.input.width / config.input.height
    expected = 16 / 9
    if abs(ratio - expected) < 0.01:
        print(f"   ✓ Aspect ratio: 16:9")
    else:
        print(f"   ✗ Aspect ratio: {ratio:.4f} (expected 16:9)")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Config loading failed: {e}")
    sys.exit(1)

# Check model building
print("\n7. Checking model building...")
try:
    from src.models.rtdetr import build_rtdetr
    model = build_rtdetr(num_classes=80)
    print(f"   ✓ Model built successfully")

    # Count parameters
    from src.utils.misc import count_parameters
    num_params = count_parameters(model)
    print(f"   ✓ Parameters: {num_params:,}")
except Exception as e:
    print(f"   ✗ Model building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check data transforms
print("\n8. Checking data transforms...")
try:
    from src.data.transforms import build_transforms
    from src.data.data_utils import get_valid_16_9_sizes

    valid_sizes = get_valid_16_9_sizes()
    print(f"   ✓ Valid 16:9 sizes: {len(valid_sizes)}")

    transforms = build_transforms(config, is_train=True)
    print(f"   ✓ Training transforms built")

    transforms = build_transforms(config, is_train=False)
    print(f"   ✓ Validation transforms built")
except Exception as e:
    print(f"   ✗ Transform building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check COCO dataset (if available)
print("\n9. Checking COCO dataset...")
data_dir = Path('data/coco')
if data_dir.exists():
    train_images = data_dir / 'train2017'
    val_images = data_dir / 'val2017'
    annotations = data_dir / 'annotations'

    if train_images.exists():
        num_train = len(list(train_images.glob('*.jpg')))
        print(f"   ✓ Training images: {num_train:,}")
    else:
        print(f"   ⚠ Training images not found (run prepare_data.py)")

    if val_images.exists():
        num_val = len(list(val_images.glob('*.jpg')))
        print(f"   ✓ Validation images: {num_val:,}")
    else:
        print(f"   ⚠ Validation images not found (run prepare_data.py)")

    if annotations.exists():
        print(f"   ✓ Annotations directory found")
    else:
        print(f"   ⚠ Annotations not found (run prepare_data.py)")
else:
    print(f"   ⚠ COCO dataset not found at {data_dir}")
    print(f"      Run: python scripts/prepare_data.py --data_dir data/coco")

# Summary
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print("✓ All critical components verified")
print("✓ Ready to start training")
print("")
print("Next steps:")
print("  1. Download COCO dataset (if not done):")
print("     python scripts/prepare_data.py --data_dir data/coco")
print("")
print("  2. Run overfit test (recommended):")
print("     python scripts/train.py --config configs/rtdetr_r50_1920x1080.yml --overfit_batches 5")
print("")
print("  3. Start full training:")
print("     python scripts/train.py --config configs/rtdetr_r50_1920x1080.yml")
print("=" * 60)
