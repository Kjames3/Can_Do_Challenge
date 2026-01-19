from ultralytics import YOLO

# Load the YOLOv11 model
model_path = 'yolo11n_cans.pt'
print(f"Loading model: {model_path}...")
model = YOLO(model_path)

# Export the model to NCNN format
print("Exporting to NCNN format...")
model.export(format='ncnn')

print("Export complete! Look for 'yolo11n_cans_ncnn_model' folder.")
