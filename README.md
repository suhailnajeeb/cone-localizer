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