from ultralytics import YOLO
import sys

def download_model(model_name: str):
    """
    Downloads an Ultralytics YOLO model by initializing it.
    If the model is an official Ultralytics model (like 'yolo11n.pt', 'yolov8n.pt'),
    it will be downloaded automatically to the current directory if it doesn't exist.
    """
    print(f"Attempting to download/load model: {model_name}")
    try:
        # Initializing the YOLO class with a model name triggers the download 
        # from ultralytics release assets if it is an official model.
        model = YOLO(model_name)
        print(f"Successfully loaded/downloaded {model_name}")
        
        # Print model info
        model.info()
        
    except Exception as e:
        print(f"Error downloading or loading {model_name}: {e}\n")
        print("Note: If 'yolov26n.pt' is a custom model, you may need to provide the direct URL or path.")
        print("If you meant an official model, double check the version (e.g., 'yolo11n.pt', 'yolov8n.pt').")

if __name__ == "__main__":
    # You can pass the model name as a command line argument, or it defaults to 'yolov26n.pt'
    # E.g., python download_model.py yolo11n.pt
    model_to_download = sys.argv[1] if len(sys.argv) > 1 else 'yolov26n.pt'
    download_model(model_to_download)
