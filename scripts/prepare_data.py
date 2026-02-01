"""Prepare COCO dataset for training.

Downloads COCO 2017 dataset and verifies structure.
"""

import argparse
import sys
from pathlib import Path
import urllib.request
import zipfile
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download file with progress bar.

    Args:
        url: URL to download from
        dest: Destination path
        desc: Description for progress bar
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file with progress bar.

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    print(f"Extracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_to)

    print(f"Extraction complete")


def download_coco(data_dir: Path, split: str = 'train') -> None:
    """Download COCO dataset.

    Args:
        data_dir: Root directory for COCO data
        split: 'train' or 'val'
    """
    # COCO URLs
    base_url = "http://images.cocodataset.org/zips"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    if split == 'train':
        images_url = f"{base_url}/train2017.zip"
        images_zip = data_dir / "train2017.zip"
        images_dir = data_dir / "train2017"
    elif split == 'val':
        images_url = f"{base_url}/val2017.zip"
        images_zip = data_dir / "val2017.zip"
        images_dir = data_dir / "val2017"
    else:
        raise ValueError(f"Unknown split: {split}")

    annotations_zip = data_dir / "annotations_trainval2017.zip"
    annotations_dir = data_dir / "annotations"

    # Download images
    if not images_dir.exists():
        if not images_zip.exists():
            print(f"Downloading COCO {split} images...")
            download_file(images_url, images_zip, desc=f"{split} images")
        else:
            print(f"Found existing {images_zip.name}")

        extract_zip(images_zip, data_dir)

        # Remove zip to save space
        images_zip.unlink()
        print(f"Removed {images_zip.name}")
    else:
        print(f"COCO {split} images already exist at {images_dir}")

    # Download annotations
    if not annotations_dir.exists():
        if not annotations_zip.exists():
            print("Downloading COCO annotations...")
            download_file(annotations_url, annotations_zip, desc="Annotations")
        else:
            print(f"Found existing {annotations_zip.name}")

        extract_zip(annotations_zip, data_dir)

        # Remove zip to save space
        annotations_zip.unlink()
        print(f"Removed {annotations_zip.name}")
    else:
        print(f"COCO annotations already exist at {annotations_dir}")


def verify_coco_dataset(data_dir: Path, split: str = 'train') -> bool:
    """Verify COCO dataset structure and contents.

    Args:
        data_dir: Root directory for COCO data
        split: 'train' or 'val'

    Returns:
        True if valid
    """
    print(f"\nVerifying COCO {split} dataset...")

    # Check directories
    if split == 'train':
        images_dir = data_dir / "train2017"
        annotation_file = data_dir / "annotations" / "instances_train2017.json"
        expected_images = 118287  # Approximate
    else:
        images_dir = data_dir / "val2017"
        annotation_file = data_dir / "annotations" / "instances_val2017.json"
        expected_images = 5000  # Approximate

    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False

    if not annotation_file.exists():
        print(f"❌ Annotation file not found: {annotation_file}")
        return False

    # Count images
    image_files = list(images_dir.glob("*.jpg"))
    num_images = len(image_files)

    print(f"✓ Found {num_images} images (expected ~{expected_images})")

    if num_images < expected_images * 0.95:
        print(f"⚠ Warning: Fewer images than expected")

    # Load and verify annotations
    try:
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        num_annotations = len(coco_data['annotations'])
        num_categories = len(coco_data['categories'])

        print(f"✓ Found {num_annotations} annotations")
        print(f"✓ Found {num_categories} categories")

        if num_categories != 80:
            print(f"⚠ Warning: Expected 80 categories, found {num_categories}")

    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return False

    print(f"✓ COCO {split} dataset verified successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO dataset for training")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/coco',
        help='Root directory for COCO data'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='both',
        choices=['train', 'val', 'both'],
        help='Which split to download'
    )
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Only verify existing dataset, do not download'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"COCO data directory: {data_dir.absolute()}")

    if not args.verify_only:
        # Download datasets
        if args.split in ['train', 'both']:
            download_coco(data_dir, 'train')

        if args.split in ['val', 'both']:
            download_coco(data_dir, 'val')

    # Verify datasets
    success = True

    if args.split in ['train', 'both']:
        success = verify_coco_dataset(data_dir, 'train') and success

    if args.split in ['val', 'both']:
        success = verify_coco_dataset(data_dir, 'val') and success

    if success:
        print("\n✓ Dataset preparation completed successfully!")
        print(f"\nYou can now start training with:")
        print(f"  python scripts/train.py --config configs/rtdetr_r50_1920x1080.yml")
    else:
        print("\n❌ Dataset preparation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
