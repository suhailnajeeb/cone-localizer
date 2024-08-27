import depthai as dai
import cv2
import os
import glob

def ensure_dir(directory):
    """
    Ensure that the specified directory exists. If it does not exist, create it.
    
    Parameters:
        directory (str): The path to the directory to ensure.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

def initialize_pipeline():
    """
    Initializes and returns a DepthAI pipeline with a color camera and XLink output,
    including a vertical flip of the camera output.

    Returns:
    - pipeline: The configured DepthAI pipeline.
    """
    pipeline = dai.Pipeline()

    # Create a color camera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setImageOrientation(dai.CameraImageOrientation.VERTICAL_FLIP)  # Set vertical flip

    # Create XLinkOut node for the color camera
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    return pipeline

def capture_and_return_frame(q_rgb, vflip=False):
    """
    Captures a single frame from the specified output queue and optionally flips it vertically.

    Args:
    - q_rgb: Output queue from which to get the rgb frames.
    - vflip: Flag to indicate if the image should be vertically flipped.

    Returns:
    - frame: The captured and possibly flipped frame, or None if no frame was captured.
    """
    in_rgb = q_rgb.tryGet()  # Try to get a frame from the queue

    if in_rgb is not None:
        frame = in_rgb.getCvFrame()
        if vflip:
            frame = cv2.flip(frame, 0)  # Flip the frame vertically
        return frame
    return None

def save_frame(frame, file_name_prefix="output_image", image_counter=0):
    """
    Saves the given frame to a file with a unique number appended to the file name.

    Args:
    - frame: The frame to be saved.
    - file_name_prefix: Prefix for the file name where the image will be saved.
    - image_counter: Counter used to append a unique number to the file name.
    """
    file_name = f"{file_name_prefix}_{image_counter}.jpg"
    cv2.imwrite(file_name, frame)
    print(f"Image has been saved as {file_name}")

def get_image_count(dir_path):
    """
    Count the number of jpg images in the specified directory.

    Parameters:
        dir_path (str): The path to the directory containing the images.

    Returns:
        int: The number of jpg images found in the directory.
             Returns "Invalid directory path" if the specified path is not a valid directory.
    """
    # Check if the directory exists
    if not os.path.isdir(dir_path):
        return "Invalid directory path"

    # Count the number of jpg images in the directory
    jpg_files = glob.glob(os.path.join(dir_path, '*.jpg'))
    count = len(jpg_files)

    return count