# Cone Detection and Localization Pipeline

## Step 0: Installation, requirements, and environment setup

(Installation on Raspberry Pi 4/5)

Requirements: 
- Depthai (`pip install depthai` on Linux)
- OpenCV (cv2) (`pip install opencv-python`)

## Step 1: Capture Images with `camera_images.py`

`camera_images.py` captures images from a DepthAI device, with an optional live stream display using OpenCV. Images are saved automatically in a specified directory.

### Features

- **Image Capture**: Captures images and saves them with a timestamped filename.
- **Optional Live Stream**: Display the video stream with `--display`.
- **Automatic Directory Creation**: Ensures the output directory exists.

### Usage

#### Basic Capture

```bash
python camera_images.py
```

#### Capture with Live Stream

```bash
python camera_images.py --display
```

#### Stopping the Script

- **Without Display**: Press `Ctrl+C`.
- **With Display**: Press 'q' in the window or `Ctrl+C`.


## Step 2: Manually Filter Images

Manually filter the images that might contain poor/unwanted samples. Create a new directory with the filtered images. 

## Step 3: Label Dataset using `label.py`

This script annotates image datasets using the GroundedSAM model.

### Requirements
- Python 3.x
- GPU and substantial memory for large datasets
- Libraries: `autodistill`

### Setup
1. Update paths in the script:
   - `DATASET_DIR_PATH`: path to save annotated dataset
   - `IMAGE_DIR_PATH`: path to images

2. Install necessary libraries:
   ```bash
   pip install autodistill
   ```

### Running the Script
Navigate to the script's directory and execute:
```bash
python label.py
```
Output is saved to the directory specified in `DATASET_DIR_PATH`.

> **Ensure the provided paths are correct and the system has adequate resources.**

## Step 4: Update the Dataset using `update_dataset.py`

### Purpose

The `update_dataset.py` script processes an image dataset by:

1. **Resizing Images:** Standardizes images to 640x640 pixels.
2. **Color Classification:** Classifies cones based on their color (yellow or blue) using hue analysis.
3. **Dataset Structuring:** Organizes processed images and labels in a format compatible with machine learning frameworks.
4. **Metadata Update:** Generates a `data.yaml` file with class information and paths for seamless integration.

### Usage

Update the `source_dir` and `target_dir` paths as needed, then run:

```bash
python update_dataset.py
```

### Requirements

Install the necessary packages:

```bash
pip install numpy pillow tqdm matplotlib
```

These cover both the script and `cone_utils.py` dependencies.

Here's a short markdown section for your `train_yolo.py` script:

---

## Step 5: Cone Detector Training (`train_yolo.py`)

The purpose of this script is to train a YOLOv8n (nano) model to detect cones using a custom dataset. This script leverages the YOLOv8n architecture, which is optimized for real-time object detection on resource-constrained devices. 

### Dependencies
Before running the script, ensure you have the necessary dependencies installed. You can install them using pip:

```bash
pip install ultralytics
```

### Usage
To train the cone detector, run the `train_yolo.py` script with the following command:

```bash
python train_yolo.py
```

The script will train the model for 100 epochs with an image size of 640 and a batch size of 16. The training results will be saved under a directory named `cone_detector`.

#### Note
> For faster training, it is highly recommended to use a GPU. Training on a CPU can be significantly slower, especially with larger datasets and higher epoch counts.

After running the training script, the model file should be available inside `runs>detect>MODEL_NAME>weights>best.pt`
Obtain the `best.pt` weights file for the next step.

## Step 6: Converting Blob

Take the weight file and upload it to this [blob conversion tool](https://tools.luxonis.com/)
Select the following settings: 

- YOLO Version: 8 (Detection)
- RVC2
- Input Image shape: '640 640'
- Shaves: 5
- Use OpenVINO 2021.4 ✅

Download the `results.zip` file and unpack it. The directory unzipped contents should contain a `blob` file which we can use to deploy the model to the OAKD Pro.

## Step 7: `spatial_yolo.py`: Spatial Cone Detection Pipeline

This script detects and localizes cones (e.g., traffic cones) in 3D space using an OAK-D device and a YOLO-based neural network.

### Configuration Options

- **`nn_blob_path`:** Path to the neural network blob file (e.g., `models/yolov8n_cones_3510_yb_st_100_5s.blob`).
- **`camera_height`:** Height of the camera from the ground in millimeters (e.g., `290 mm`).
- **`camera_alpha`:** Angle of the camera's tilt relative to the ground in degrees (e.g., `20°`).
- **`labelMap`:** Label map for detected objects (e.g., `["Yellow", "Blue"]`).
- **`syncNN`:** Synchronize the neural network processing with the camera feed (e.g., `True`).
- **`show_depth:`** Visualize depth
- **`dot_projector:`** Use IR Laser Dot Projections for Active Stereo Depth Estimation

### Usage

1. Ensure your OAK-D device is connected and configured.
2. Update the `detector_configs` with your settings.
3. Run the script to start real-time cone detection.

A minimal version of thie script can be found here: `spatial_yolo_simple.py` which is later converted into a ROS node for integration within the pipeline