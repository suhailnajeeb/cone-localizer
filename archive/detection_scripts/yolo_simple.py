import cv2
import numpy as np
import depthai as dai
import time
#from YOLOSeg import YOLOSeg

# Path to the YOLOv8 segment model blob file
pathYoloBlob = "best.blob"

# Create OAK-D pipeline
pipeline = dai.Pipeline()

# Set up the RGB camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 640)  # Set preview size to 640x640
cam_rgb.setInterleaved(False)  # Disable interleaved mode

# Set up the neural network node with the YOLOv8 model
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(pathYoloBlob)  # Load the YOLOv8 model blob

# Link the camera preview output to the neural network input
cam_rgb.preview.link(nn.input)

# Set up XLink output nodes for RGB frames and neural network results
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn_yolo = pipeline.createXLinkOut()
xout_nn_yolo.setStreamName("nn_yolo")
nn.out.link(xout_nn_yolo.input)

# Start the application
with dai.Device(pipeline) as device:
    # Get the output queues for RGB frames and neural network results
    q_rgb = device.getOutputQueue("rgb")
    q_nn_yolo = device.getOutputQueue("nn_yolo")

    print("Device initialized, waiting for 2 seconds")

    # Wait for 2 seconds to let the camera initialize
    time.sleep(2)

    # Capture one frame
    in_rgb = q_rgb.get()
    frame = in_rgb.getCvFrame()

    print("Captured frame and segmentation.")

    # Run the YOLO model on the captured frame
    in_nn_yolo = q_nn_yolo.get()
   
    output0 = np.reshape(in_nn_yolo.getLayerFp16("output0"), newshape=([1, 38, 8400]))
    output1 = np.reshape(in_nn_yolo.getLayerFp16("output1"), newshape=([1, 32, 160, 160]))

    # Print the YOLO model output to the terminal
    print("YOLO Model Output:")
    print("Output0:", output0)
    print("Output1:", output1)

    # Display the captured image
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(0)  # Wait for a key press to close the image display window
    cv2.destroyAllWindows()