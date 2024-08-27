from ultralytics import YOLO

# Path to the dataset's YAML configuration file
DATA_YAML_PATH = 'path/to/dataset/data.yaml'

# Load a pretrained YOLOv8 nano model
# This model is lightweight and optimized for real-time detection on resource-constrained devices
model = YOLO('yolov8n.pt')

# Train the model on the custom dataset
model.train(
    data=DATA_YAML_PATH,    # Path to the dataset configuration file
    epochs=100,             # Number of training epochs
    imgsz=640,              # Image size used for training
    batch=16,               # Batch size    # Adjust for CPU/GPU
    name='cone_detector'    # Name of the training run; results will be saved under this directory
)

# Note: 
# - Using a GPU is highly recommended for faster training. 
# - Training on a CPU will be significantly slower, especially with larger datasets or higher epoch counts.