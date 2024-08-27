"""
This Code is taken from this StackOverFlow Thread:
https://stackoverflow.com/questions/78153689/numpy-array-slow-with-large-list

I have found this out from this discussion: 
https://discuss.luxonis.com/d/3646-how-do-i-deploy-a-yolov8-segment-model-on-an-oak-d-pro-camera/2
"""
import cv2
import numpy as np
import depthai as dai
import time
from YOLOSeg import YOLOSeg

# Path to the YOLOv8 segment model blob file
pathYoloBlob = "best.blob"

# Create OAK-D pipeline
pipeline = dai.Pipeline()

# Set up the RGB camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 640)  # Set preview size to 640x640
#cam_rgb.setPreviewSize(1280, 720)
cam_rgb.setInterleaved(False)  # Disable interleaved mode

# Additional manips for resizing the frame to 640x640
#cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
#manip = pipeline.create(dai.node.ImageManip)
#manip.initialConfig.setResize(640, 640)
#manip.initialConfig.setKeepAspectRatio(False)
#manip.setMaxOutputFrameSize(1228800)
#cam_rgb.preview.link(manip.inputImage)

# Set up the neural network node with the YOLOv8 model
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(pathYoloBlob)  # Load the YOLOv8 model blob

# Link the camera preview output to the neural network input
cam_rgb.preview.link(nn.input)

# Link the resized image with the neural network input
#manip.out.link(nn.input)

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
    q_rgb = device.getOutputQueue("rgb", maxSize = 4, blocking = False)
    q_nn_yolo = device.getOutputQueue("nn_yolo", maxSize = 4, blocking = False)

    frame = None

    # Function to normalize bounding box coordinates
    # Since the detections returned by nn have values from <0..1> range, 
    # they need to be multiplied by frame width/height to receive the actual position of the bounding box on the image
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    # Main host-side application loop
    while True:
        in_rgb = q_rgb.tryGet()  # Try to get the next RGB frame
        in_nn_yolo = q_nn_yolo.tryGet()  # Try to get the next neural network result

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()  # Get the RGB frame

            if in_nn_yolo is not None:
                # Reshape the neural network output layers to the expected shapes
                output0 = np.reshape(in_nn_yolo.getLayerFp16("output0"), newshape=([1, 38, 8400]))
                output1 = np.reshape(in_nn_yolo.getLayerFp16("output1"), newshape=([1, 32, 160, 160]))

                # If both outputs are available, we can calculate the final mask
                if len(output0) > 0 and len(output1) > 0:
                    # Post-process, this is fast, no problems here
                    yoloseg = YOLOSeg("", conf_thres=0.3, iou_thres=0.5)  # Initialize YOLOSeg object
                    yoloseg.prepare_input_for_oakd(frame.shape[:2])  # Prepare input
                    yoloseg.segment_objects_from_oakd(output0, output1)  # Segment objects
                    combined_img = yoloseg.draw_masks(frame.copy())  # Draw masks on the frame
                    cv2.imshow("Output", combined_img)  # Display the combined image with masks

            else:
                print("in_nn_yolo EMPTY")  # Print message if neural network output is empty

        else:
            #pass
            print("in_rgb EMPTY")  # Print message if RGB frame is empty
        
        # At any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break  # Break the loop and exit if 'q' is pressed