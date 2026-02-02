"""S3 utilities for automated training pipeline.

Handles downloading training data from S3 and uploading trained models.
Supports both labeled datasets (COCO format) and model artifacts.
"""

import os
import boto3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class S3Client:
    """S3 client wrapper for training pipeline operations."""

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = "us-east-1",
        endpoint_url: Optional[str] = None
    ):
        """
        Initialize S3 client.

        Args:
            aws_access_key_id: AWS access key (or use AWS_ACCESS_KEY_ID env var)
            aws_secret_access_key: AWS secret key (or use AWS_SECRET_ACCESS_KEY env var)
            aws_region: AWS region (default: us-east-1)
            endpoint_url: Custom endpoint URL (for S3-compatible services like MinIO)
        """
        self.aws_region = aws_region

        # Use provided credentials or fall back to environment variables
        session_kwargs = {}
        if aws_access_key_id:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
        if aws_region:
            session_kwargs['region_name'] = aws_region

        self.session = boto3.Session(**session_kwargs)

        client_kwargs = {}
        if endpoint_url:
            client_kwargs['endpoint_url'] = endpoint_url

        self.s3 = self.session.client('s3', **client_kwargs)
        self.s3_resource = self.session.resource('s3', **client_kwargs)

    def validate_credentials(self) -> bool:
        """Validate that S3 credentials are working."""
        try:
            self.s3.list_buckets()
            logger.info("S3 credentials validated successfully")
            return True
        except NoCredentialsError:
            logger.error("No AWS credentials found")
            return False
        except ClientError as e:
            logger.error(f"S3 credential validation failed: {e}")
            return False

    def bucket_exists(self, bucket: str) -> bool:
        """Check if a bucket exists and is accessible."""
        try:
            self.s3.head_bucket(Bucket=bucket)
            return True
        except ClientError:
            return False

    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List objects in an S3 bucket.

        Args:
            bucket: S3 bucket name
            prefix: Object key prefix to filter by
            max_keys: Maximum number of keys to return

        Returns:
            List of object metadata dictionaries
        """
        objects = []
        paginator = self.s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys):
            if 'Contents' in page:
                objects.extend(page['Contents'])

        return objects

    def download_file(
        self,
        bucket: str,
        key: str,
        local_path: str,
        show_progress: bool = True
    ) -> bool:
        """
        Download a single file from S3.

        Args:
            bucket: S3 bucket name
            key: Object key in S3
            local_path: Local path to save file
            show_progress: Show download progress bar

        Returns:
            True if download successful
        """
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if show_progress:
                # Get file size for progress bar
                response = self.s3.head_object(Bucket=bucket, Key=key)
                file_size = response['ContentLength']

                with tqdm(total=file_size, unit='B', unit_scale=True, desc=key.split('/')[-1]) as pbar:
                    self.s3.download_file(
                        bucket, key, str(local_path),
                        Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
                    )
            else:
                self.s3.download_file(bucket, key, str(local_path))

            logger.info(f"Downloaded: s3://{bucket}/{key} -> {local_path}")
            return True

        except ClientError as e:
            logger.error(f"Failed to download s3://{bucket}/{key}: {e}")
            return False

    def download_directory(
        self,
        bucket: str,
        prefix: str,
        local_dir: str,
        max_workers: int = 8,
        show_progress: bool = True
    ) -> bool:
        """
        Download all files from an S3 prefix (directory) to local directory.

        Args:
            bucket: S3 bucket name
            prefix: S3 prefix (directory) to download
            local_dir: Local directory to save files
            max_workers: Number of parallel download threads
            show_progress: Show overall progress bar

        Returns:
            True if all downloads successful
        """
        # Ensure prefix ends with /
        if prefix and not prefix.endswith('/'):
            prefix += '/'

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # List all objects to download
        objects = self.list_objects(bucket, prefix)

        if not objects:
            logger.warning(f"No objects found at s3://{bucket}/{prefix}")
            return False

        logger.info(f"Found {len(objects)} objects to download from s3://{bucket}/{prefix}")

        # Filter out directory markers
        objects = [obj for obj in objects if not obj['Key'].endswith('/')]

        success = True
        failed_files = []

        def download_single(obj):
            key = obj['Key']
            # Calculate relative path
            relative_path = key[len(prefix):] if prefix else key
            local_path = local_dir / relative_path
            return self.download_file(bucket, key, str(local_path), show_progress=False)

        # Download files in parallel with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, obj): obj for obj in objects}

            progress_iter = tqdm(as_completed(futures), total=len(futures),
                                desc="Downloading", disable=not show_progress)

            for future in progress_iter:
                obj = futures[future]
                try:
                    if not future.result():
                        success = False
                        failed_files.append(obj['Key'])
                except Exception as e:
                    logger.error(f"Error downloading {obj['Key']}: {e}")
                    success = False
                    failed_files.append(obj['Key'])

        if failed_files:
            logger.error(f"Failed to download {len(failed_files)} files: {failed_files[:5]}...")

        return success

    def upload_file(
        self,
        local_path: str,
        bucket: str,
        key: str,
        extra_args: Optional[Dict] = None,
        show_progress: bool = True
    ) -> bool:
        """
        Upload a single file to S3.

        Args:
            local_path: Local file path
            bucket: S3 bucket name
            key: Object key in S3
            extra_args: Extra arguments for upload (e.g., ACL, Metadata)
            show_progress: Show upload progress bar

        Returns:
            True if upload successful
        """
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.error(f"Local file does not exist: {local_path}")
                return False

            if show_progress:
                file_size = local_path.stat().st_size
                with tqdm(total=file_size, unit='B', unit_scale=True,
                         desc=local_path.name) as pbar:
                    self.s3.upload_file(
                        str(local_path), bucket, key,
                        ExtraArgs=extra_args,
                        Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
                    )
            else:
                self.s3.upload_file(str(local_path), bucket, key, ExtraArgs=extra_args)

            logger.info(f"Uploaded: {local_path} -> s3://{bucket}/{key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to upload {local_path} to s3://{bucket}/{key}: {e}")
            return False

    def upload_directory(
        self,
        local_dir: str,
        bucket: str,
        prefix: str,
        max_workers: int = 8,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> bool:
        """
        Upload a local directory to S3.

        Args:
            local_dir: Local directory path
            bucket: S3 bucket name
            prefix: S3 prefix (directory) to upload to
            max_workers: Number of parallel upload threads
            include_patterns: File patterns to include (e.g., ['*.pth', '*.onnx'])
            exclude_patterns: File patterns to exclude
            show_progress: Show overall progress bar

        Returns:
            True if all uploads successful
        """
        import fnmatch

        local_dir = Path(local_dir)
        if not local_dir.exists():
            logger.error(f"Local directory does not exist: {local_dir}")
            return False

        # Ensure prefix ends with /
        if prefix and not prefix.endswith('/'):
            prefix += '/'

        # Collect files to upload
        files_to_upload = []
        for local_path in local_dir.rglob('*'):
            if local_path.is_file():
                relative_path = local_path.relative_to(local_dir)

                # Check include patterns
                if include_patterns:
                    if not any(fnmatch.fnmatch(str(relative_path), p) for p in include_patterns):
                        continue

                # Check exclude patterns
                if exclude_patterns:
                    if any(fnmatch.fnmatch(str(relative_path), p) for p in exclude_patterns):
                        continue

                key = prefix + str(relative_path).replace('\\', '/')
                files_to_upload.append((str(local_path), key))

        if not files_to_upload:
            logger.warning(f"No files to upload from {local_dir}")
            return True

        logger.info(f"Uploading {len(files_to_upload)} files to s3://{bucket}/{prefix}")

        success = True

        def upload_single(item):
            local_path, key = item
            return self.upload_file(local_path, bucket, key, show_progress=False)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(upload_single, item): item for item in files_to_upload}

            progress_iter = tqdm(as_completed(futures), total=len(futures),
                                desc="Uploading", disable=not show_progress)

            for future in progress_iter:
                item = futures[future]
                try:
                    if not future.result():
                        success = False
                except Exception as e:
                    logger.error(f"Error uploading {item[0]}: {e}")
                    success = False

        return success


def download_coco_dataset_from_s3(
    s3_client: S3Client,
    bucket: str,
    data_prefix: str,
    local_data_dir: str,
    splits: List[str] = ['train', 'val']
) -> bool:
    """
    Download COCO-format dataset from S3.

    Expected S3 structure:
        s3://{bucket}/{data_prefix}/
            images/
                train2017/
                val2017/
            annotations/
                instances_train2017.json
                instances_val2017.json

    Args:
        s3_client: S3Client instance
        bucket: S3 bucket name
        data_prefix: S3 prefix where data is stored
        local_data_dir: Local directory to save dataset
        splits: Dataset splits to download ('train', 'val', or both)

    Returns:
        True if download successful
    """
    local_data_dir = Path(local_data_dir)

    # Download annotations first (small files)
    logger.info("Downloading annotations...")
    annotations_prefix = f"{data_prefix}/annotations/"
    annotations_dir = local_data_dir / "annotations"

    if not s3_client.download_directory(bucket, annotations_prefix, str(annotations_dir)):
        logger.error("Failed to download annotations")
        return False

    # Download images for each split
    for split in splits:
        split_name = f"{split}2017"
        images_prefix = f"{data_prefix}/images/{split_name}/"
        images_dir = local_data_dir / "images" / split_name

        logger.info(f"Downloading {split} images...")
        if not s3_client.download_directory(bucket, images_prefix, str(images_dir)):
            logger.error(f"Failed to download {split} images")
            return False

    logger.info(f"Dataset downloaded successfully to {local_data_dir}")
    return True


def upload_trained_model_to_s3(
    s3_client: S3Client,
    bucket: str,
    output_prefix: str,
    checkpoint_path: str,
    onnx_path: Optional[str] = None,
    config_path: Optional[str] = None,
    training_logs_dir: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Upload trained model artifacts to S3.

    Args:
        s3_client: S3Client instance
        bucket: S3 bucket name
        output_prefix: S3 prefix for output artifacts
        checkpoint_path: Path to PyTorch checkpoint
        onnx_path: Path to ONNX model (optional)
        config_path: Path to training config (optional)
        training_logs_dir: Path to TensorBoard logs directory (optional)
        metadata: Additional metadata to include

    Returns:
        Dictionary mapping artifact type to S3 URI
    """
    import json
    from datetime import datetime

    # Ensure prefix ends with /
    if output_prefix and not output_prefix.endswith('/'):
        output_prefix += '/'

    uploaded_artifacts = {}

    # Upload PyTorch checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint_key = f"{output_prefix}checkpoints/{Path(checkpoint_path).name}"
        if s3_client.upload_file(checkpoint_path, bucket, checkpoint_key):
            uploaded_artifacts['checkpoint'] = f"s3://{bucket}/{checkpoint_key}"

    # Upload ONNX model
    if onnx_path and Path(onnx_path).exists():
        onnx_key = f"{output_prefix}onnx/{Path(onnx_path).name}"
        if s3_client.upload_file(onnx_path, bucket, onnx_key):
            uploaded_artifacts['onnx'] = f"s3://{bucket}/{onnx_key}"

    # Upload config
    if config_path and Path(config_path).exists():
        config_key = f"{output_prefix}config/{Path(config_path).name}"
        if s3_client.upload_file(config_path, bucket, config_key):
            uploaded_artifacts['config'] = f"s3://{bucket}/{config_key}"

    # Upload training logs (if provided)
    if training_logs_dir and Path(training_logs_dir).exists():
        logs_prefix = f"{output_prefix}logs/"
        if s3_client.upload_directory(training_logs_dir, bucket, logs_prefix):
            uploaded_artifacts['logs'] = f"s3://{bucket}/{logs_prefix}"

    # Create and upload manifest
    manifest = {
        'timestamp': datetime.utcnow().isoformat(),
        'artifacts': uploaded_artifacts,
        'metadata': metadata or {}
    }

    manifest_path = Path('/tmp/manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    manifest_key = f"{output_prefix}manifest.json"
    if s3_client.upload_file(str(manifest_path), bucket, manifest_key, show_progress=False):
        uploaded_artifacts['manifest'] = f"s3://{bucket}/{manifest_key}"

    manifest_path.unlink()  # Clean up temp file

    logger.info(f"Uploaded artifacts: {list(uploaded_artifacts.keys())}")
    return uploaded_artifacts


def verify_s3_dataset_structure(
    s3_client: S3Client,
    bucket: str,
    data_prefix: str
) -> Dict[str, Any]:
    """
    Verify that S3 bucket has expected COCO dataset structure.

    Args:
        s3_client: S3Client instance
        bucket: S3 bucket name
        data_prefix: S3 prefix where data is stored

    Returns:
        Dictionary with verification results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    # Check for annotations
    annotations_prefix = f"{data_prefix}/annotations/"
    annotation_files = s3_client.list_objects(bucket, annotations_prefix)

    required_annotations = ['instances_train2017.json', 'instances_val2017.json']
    found_annotations = [obj['Key'].split('/')[-1] for obj in annotation_files]

    for ann in required_annotations:
        if ann not in found_annotations:
            results['errors'].append(f"Missing annotation file: {ann}")
            results['valid'] = False

    results['stats']['annotation_files'] = len(annotation_files)

    # Check for training images
    train_images_prefix = f"{data_prefix}/images/train2017/"
    train_images = s3_client.list_objects(bucket, train_images_prefix, max_keys=100)
    results['stats']['train_images_sample'] = len(train_images)

    if len(train_images) == 0:
        results['errors'].append("No training images found")
        results['valid'] = False
    elif len(train_images) < 100:
        results['warnings'].append(f"Only {len(train_images)} training images found (expected ~118K)")

    # Check for validation images
    val_images_prefix = f"{data_prefix}/images/val2017/"
    val_images = s3_client.list_objects(bucket, val_images_prefix, max_keys=100)
    results['stats']['val_images_sample'] = len(val_images)

    if len(val_images) == 0:
        results['errors'].append("No validation images found")
        results['valid'] = False
    elif len(val_images) < 100:
        results['warnings'].append(f"Only {len(val_images)} validation images found (expected ~5K)")

    return results
