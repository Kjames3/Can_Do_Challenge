"""
YOLOv8n Fine-tuning Script for Soda Can Detection

This script fine-tunes the YOLOv8n model to detect soda cans using two 
Roboflow datasets. All classes are merged into a single "can" class.

Datasets:
- can1_dataset: 783 train + 88 valid + 44 test images (4 classes -> 1 class)
- can2_dataset: 288 train images (1 class)

Total: ~1,071 training images

Usage:
    python train_yolov8_cans.py [--epochs 100] [--batch 16] [--imgsz 640]
"""

import os
import shutil
import argparse
from pathlib import Path
import yaml


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent


def prepare_combined_dataset(project_root: Path, force_rebuild: bool = False):
    """
    Prepare a combined dataset by merging all can datasets.
    Remaps all class labels to a single 'can' class (index 0).
    
    Supports datasets:
    - can1_dataset: 4 classes (Deformation, Fissure, Open-can, Perfect) -> can
    - can2_dataset: 1 class (soda cans) -> can
    - can3_dataset: 5 classes (0, 1, 2, can, cans) -> can
    - can4_dataset: 3 classes (BIODEGRADABLE, can, distractor) -> can
    
    Args:
        project_root: Path to the project root directory
        force_rebuild: If True, rebuild even if combined dataset exists
        
    Returns:
        Path to the combined dataset directory
    """
    combined_dir = project_root / "datasets" / "combined_cans"
    
    # Check if already prepared
    if combined_dir.exists() and not force_rebuild:
        print(f"✓ Combined dataset already exists at {combined_dir}")
        return combined_dir
    
    # Clean up if rebuilding
    if combined_dir.exists():
        print("Removing existing combined dataset...")
        shutil.rmtree(combined_dir)
    
    # Create directory structure
    for split in ["train", "valid", "test"]:
        (combined_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (combined_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    print("Preparing combined dataset from all available datasets...")
    
    import random
    random.seed(42)  # For reproducibility
    
    stats = {"train": 0, "valid": 0, "test": 0}
    
    # Define all datasets to process
    datasets = [
        {"name": "can1_dataset", "has_splits": True},
        {"name": "can2_dataset", "has_splits": False},  # Only has train
        {"name": "can3_dataset", "has_splits": True},
        {"name": "can4_dataset", "has_splits": True},
    ]
    
    for dataset_info in datasets:
        dataset_name = dataset_info["name"]
        dataset_dir = project_root / "datasets" / dataset_name
        
        if not dataset_dir.exists():
            print(f"  ⚠ {dataset_name} not found, skipping...")
            continue
        
        print(f"\n  Processing {dataset_name}...")
        
        if dataset_info["has_splits"]:
            # Dataset has train/valid/test splits
            for split in ["train", "valid", "test"]:
                count = process_dataset_split(
                    dataset_dir, combined_dir, dataset_name, split
                )
                stats[split] += count
                if count > 0:
                    print(f"    ✓ {split}: {count} images")
        else:
            # Dataset only has train - we'll split it 80/10/10
            src_images = dataset_dir / "train" / "images"
            src_labels = dataset_dir / "train" / "labels"
            
            if not src_images.exists():
                print(f"    ⚠ No train/images found, skipping...")
                continue
            
            image_files = list(src_images.glob("*"))
            image_files = [f for f in image_files 
                          if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
            
            random.shuffle(image_files)
            
            n_files = len(image_files)
            n_train = int(n_files * 0.8)
            n_valid = int(n_files * 0.1)
            
            splits = {
                "train": image_files[:n_train],
                "valid": image_files[n_train:n_train + n_valid],
                "test": image_files[n_train + n_valid:]
            }
            
            for split, files in splits.items():
                dst_images = combined_dir / split / "images"
                dst_labels = combined_dir / split / "labels"
                
                for img_file in files:
                    new_name = f"{dataset_name}_{img_file.name}"
                    shutil.copy2(img_file, dst_images / new_name)
                    
                    label_file = src_labels / f"{img_file.stem}.txt"
                    if label_file.exists():
                        remap_label_file(label_file, dst_labels / f"{dataset_name}_{img_file.stem}.txt")
                    else:
                        (dst_labels / f"{dataset_name}_{img_file.stem}.txt").touch()
                    
                    stats[split] += 1
                
                print(f"    ✓ {split}: {len(files)} images (auto-split)")
    
    # Create data.yaml configuration
    data_yaml = {
        "path": str(combined_dir.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["can"]
    }
    
    yaml_path = combined_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n{'='*50}")
    print(f"✓ Combined dataset created at {combined_dir}")
    print(f"  - Training images: {stats['train']}")
    print(f"  - Validation images: {stats['valid']}")
    print(f"  - Test images: {stats['test']}")
    print(f"  - Total: {sum(stats.values())}")
    print(f"  - Config: {yaml_path}")
    print(f"{'='*50}")
    
    return combined_dir


def process_dataset_split(dataset_dir: Path, combined_dir: Path, 
                          dataset_name: str, split: str) -> int:
    """
    Process a single split (train/valid/test) from a dataset.
    
    Returns:
        Number of images processed
    """
    src_images = dataset_dir / split / "images"
    src_labels = dataset_dir / split / "labels"
    
    if not src_images.exists():
        return 0
    
    dst_images = combined_dir / split / "images"
    dst_labels = combined_dir / split / "labels"
    
    count = 0
    for img_file in src_images.glob("*"):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            # Copy image with prefix to avoid name conflicts
            new_name = f"{dataset_name}_{img_file.name}"
            shutil.copy2(img_file, dst_images / new_name)
            
            # Copy and remap label file
            label_file = src_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                remap_label_file(label_file, dst_labels / f"{dataset_name}_{img_file.stem}.txt")
            else:
                # Create empty label file if no annotations
                (dst_labels / f"{dataset_name}_{img_file.stem}.txt").touch()
            
            count += 1
    
    return count


def remap_label_file(src_path: Path, dst_path: Path):
    """
    Remap all class indices in a YOLO label file to class 0.
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    """
    with open(src_path, "r") as f:
        lines = f.readlines()
    
    remapped_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # Replace class index with 0, keep bounding box coordinates
            parts[0] = "0"
            remapped_lines.append(" ".join(parts) + "\n")
    
    with open(dst_path, "w") as f:
        f.writelines(remapped_lines)


def train_model(
    data_yaml: Path,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    pretrained_weights: str = "yolov8n.pt",
    project_name: str = "can_detection",
    run_name: str = "yolov8n_cans",
    resume: bool = False,
    device: str = None
):
    """
    Train YOLOv8n model for can detection.
    
    Args:
        data_yaml: Path to dataset configuration file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        pretrained_weights: Path to pretrained weights or model name
        project_name: Project name for organizing runs
        run_name: Name for this training run
        resume: Whether to resume from last checkpoint
        device: Device to use (None for auto-detect, '0' for GPU 0, 'cpu' for CPU)
    """
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("Error: ultralytics package not found!")
        print("Install it with: pip install ultralytics")
        return None
    
    # Auto-detect CUDA and set device
    if device is None:
        if torch.cuda.is_available():
            device = "0"  # Use first GPU
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 CUDA GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            device = "cpu"
            print("⚠️  No CUDA GPU detected, using CPU (training will be slower)")
    
    print(f"\n{'='*60}")
    print("YOLOv8n Can Detection Training")
    print(f"{'='*60}")
    print(f"Dataset config: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    print(f"Pretrained weights: {pretrained_weights}")
    print(f"{'='*60}\n")
    
    # Load pretrained model
    project_root = get_project_root()
    weights_path = project_root / pretrained_weights
    
    if weights_path.exists():
        print(f"Loading pretrained weights from {weights_path}")
        model = YOLO(str(weights_path))
    else:
        print(f"Downloading {pretrained_weights} from Ultralytics hub...")
        model = YOLO(pretrained_weights)
    
    # Training arguments
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "project": str(project_root / "runs" / project_name),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "rect": False,
        "cos_lr": True,
        "close_mosaic": 10,
        "resume": resume,
        "amp": True,  # Automatic mixed precision
        "patience": 50,  # Early stopping patience
        "save": True,
        "save_period": -1,  # Save checkpoint every epoch
        "cache": True,  # Cache images in RAM for faster training
        "workers": 0,  # Use 0 on Windows to avoid multiprocessing issues
        "plots": True,  # Generate training plots
        "device": device,  # Explicitly set device
    }

    
    # Data augmentation settings (good defaults for object detection)
    train_args.update({
        "hsv_h": 0.015,  # Hue augmentation
        "hsv_s": 0.7,    # Saturation augmentation
        "hsv_v": 0.4,    # Value augmentation
        "degrees": 0.0,  # Rotation
        "translate": 0.1,  # Translation
        "scale": 0.5,    # Scale
        "shear": 0.0,    # Shear
        "perspective": 0.0,  # Perspective
        "flipud": 0.0,   # Vertical flip
        "fliplr": 0.5,   # Horizontal flip
        "mosaic": 1.0,   # Mosaic augmentation
        "mixup": 0.0,    # Mixup augmentation
        "copy_paste": 0.0,  # Copy-paste augmentation
    })
    
    # Start training
    print("Starting training...")
    results = model.train(**train_args)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    print(f"Last model saved to: {results.save_dir}/weights/last.pt")
    
    return model, results


def evaluate_model(model_path: Path, data_yaml: Path):
    """
    Evaluate the trained model on the test set.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found!")
        return None
    
    print(f"\n{'='*60}")
    print("Model Evaluation")
    print(f"{'='*60}")
    
    model = YOLO(str(model_path))
    
    # Validate on test set
    results = model.val(
        data=str(data_yaml),
        split="test",
        verbose=True,
        plots=True
    )
    
    print(f"\nTest Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results


def export_model(model_path: Path, formats: list = None):
    """
    Export the trained model to various formats.
    
    Args:
        model_path: Path to the trained .pt model
        formats: List of export formats (default: ['onnx'])
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found!")
        return None
    
    if formats is None:
        formats = ["onnx"]
    
    print(f"\n{'='*60}")
    print("Model Export")
    print(f"{'='*60}")
    
    model = YOLO(str(model_path))
    
    exported_paths = []
    for fmt in formats:
        print(f"\nExporting to {fmt.upper()}...")
        try:
            path = model.export(format=fmt)
            exported_paths.append(path)
            print(f"  ✓ Exported to: {path}")
        except Exception as e:
            print(f"  ✗ Failed to export to {fmt}: {e}")
    
    return exported_paths


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n for soda can detection"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: '0' for GPU, 'cpu' for CPU (default: auto)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--rebuild-dataset", action="store_true",
        help="Force rebuild of combined dataset"
    )
    parser.add_argument(
        "--eval-only", type=str, default=None,
        help="Only evaluate a trained model (provide path to .pt file)"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export model after training"
    )
    parser.add_argument(
        "--export-formats", type=str, nargs="+", default=["onnx"],
        help="Export formats (default: onnx)"
    )
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    
    # Prepare combined dataset
    combined_dir = prepare_combined_dataset(
        project_root, 
        force_rebuild=args.rebuild_dataset
    )
    data_yaml = combined_dir / "data.yaml"
    
    # Evaluation only mode
    if args.eval_only:
        model_path = Path(args.eval_only)
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            return
        evaluate_model(model_path, data_yaml)
        return
    
    # Train the model
    model, results = train_model(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        resume=args.resume
    )
    
    if model is None:
        return
    
    # Get best model path
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    
    # Evaluate on test set
    evaluate_model(best_model_path, data_yaml)
    
    # Export if requested
    if args.export:
        export_model(best_model_path, args.export_formats)
    
    print(f"\n{'='*60}")
    print("All Done!")
    print(f"{'='*60}")
    print(f"\nTo use your trained model:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{best_model_path}')")
    print(f"  results = model.predict('image.jpg')")


if __name__ == "__main__":
    main()
