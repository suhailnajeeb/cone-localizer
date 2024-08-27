import cv2
import numpy as np
import depthai as dai
import time
from YOLOSeg import YOLOSeg

# Path to the YOLOv8 segment model blob file
pathYoloBlob = "yolov8n_det_3510_yb.blob"

# Create OAK-D pipeline
pipeline = dai.Pipeline()

# Set up the RGB camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setImageOrientation(dai.CameraImageOrientation.VERTICAL_FLIP)
cam_rgb.setPreviewSize(640, 640)
cam_rgb.setPreviewKeepAspectRatio(False)
cam_rgb.setInterleaved(False)

# Set up the neural network node with the YOLOv8 model
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(pathYoloBlob)  # Load the YOLOv8 model blob
nn.input.setBlocking(False)
nn.input.setQueueSize(1)

# Set up XLink output nodes for RGB frames and neural network results
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
xout_nn_yolo = pipeline.createXLinkOut()
xout_nn_yolo.setStreamName("nn_yolo")

# cam_rgb.preview -> xout_rgb
cam_rgb.preview.link(xout_rgb.input)
# cam_rgb.preview -> nn
cam_rgb.preview.link(nn.input)
# nn.out -> xout_nn
nn.out.link(xout_nn_yolo.input)

# Start the application
with dai.Device(pipeline) as device:
    # Get the output queues for RGB frames and neural network results
    q_rgb = device.getOutputQueue("rgb", maxSize = 4, blocking = False)
    #q_rgb_manip = device.getOutputQueue("rgb_manip", maxSize = 4, blocking = False)
    q_nn_yolo = device.getOutputQueue("nn_yolo", maxSize = 4, blocking = False)

    print("Device initialized, waiting for 2 seconds")

    # Wait for 2 seconds to let the camera initialize
    time.sleep(2)

    #counter = 0

    while True:
        #timestamp = time.time()
        # Try to get frame and NN output
        in_rgb = q_rgb.get()
        #in_rgb_manip = q_rgb_manip.get()

        #in_rgb_data = in_rgb.getData()
        #in_rgb_manip_data = in_rgb_manip.getData()

        #breakpoint()
        in_nn_yolo = q_nn_yolo.get()
        #in_nn_yolo = None

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            #print(frame.shape)

            if in_nn_yolo is not None:
                # Reshape the NN outputs to get expected shapes
                output0 = np.reshape(in_nn_yolo.getLayerFp16("output0"), newshape=([1, 6, 8400]))

                # If output is available, we can display the detection
                if len(output0) > 0:
                    # Post-process the detection
                    yoloseg = YOLOSeg("", conf_thres = 0.3, iou_thres = 0.5, num_masks = 0) # Initialize YOLOSeg object
                    yoloseg.prepare_input_for_oakd(frame.shape[:2]) # Prepare input
                    boxes, scores, class_ids = yoloseg.detect_objects_from_oakd(output0) # Detect objects
                    #counter += 1
                    #print(f"Objects detected, frames processed: {counter}")
                    #fps = 1 / (time.time() - timestamp)
                    #print(f"FPS: {fps}")

                detections_img = yoloseg.draw_detections(frame.copy())
                cv2.imshow("Output", detections_img)

            #Break the loop when 'q' key is pressed
            if cv2.waitKey(1) == ord('q'):
                break