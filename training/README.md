# YOLO Model Training

This folder contains scripts for training custom object detection models.

## Files

| File | Description |
|------|-------------|
| `train_yolov8_cans.py` | YOLO11n fine-tuning script for soda can detection |

---

## Training Pipeline

### Overview

The training script fine-tunes YOLO11n to detect soda cans by merging classes from multiple datasets into a unified "can" class.

### Datasets

Download these Roboflow datasets in **YOLOv8 format**:

| Dataset | Images | Link |
|---------|--------|------|
| Cans Dataset | 783 | [Roboflow - Cans](https://universe.roboflow.com/heho/cans-p8c8x/dataset/4/download) |
| Soda Can Dataset | 288 | [Roboflow - Soda Cans](https://universe.roboflow.com/soda-can-dataset/my-first-project-qqbah) |

Extract to:
```
datasets/
├── can1_dataset/
└── can2_dataset/
```

### Training

```bash
# Basic training (100 epochs)
python train_yolov8_cans.py --epochs 100

# Full options
python train_yolov8_cans.py --epochs 100 --batch 16 --imgsz 640

# Resume interrupted training
python train_yolov8_cans.py --resume
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch` | 16 | Batch size |
| `--imgsz` | 640 | Training image size |
| `--resume` | False | Resume from last checkpoint |
| `--device` | auto | CUDA device (0, 1, cpu) |

### Output

Trained models are saved to:
```
runs/can_detection/yolo11n_cans/weights/
├── best.pt     # Best validation model
└── last.pt     # Last epoch model
```

### Deployment

Copy the trained model to the project root:

```bash
copy runs\can_detection\yolo11n_cans\weights\best.pt ..\yolo11n_cans.pt
```

---

## Model Export

The training script can also export to ONNX format for deployment on edge devices:

```bash
# Automatic export after training
python train_yolov8_cans.py --epochs 100 --export onnx
```

---

## See Also

- [Main README](../README.md) - Project overview
- [datasets/](../datasets/) - Training data location
- [runs/](../runs/) - Training run history
