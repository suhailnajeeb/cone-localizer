
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
from types import SimpleNamespace

from transforms import CameraToWorld

class SpatialConeDetector:
    def __init__(self, configs):
        self.nnBlobPath = configs.nn_blob_path
        self.show_depth = configs.show_depth
        self.labelMap = configs.labelMap
        self.dot_projector = configs.dot_projector
        self.cam2world = CameraToWorld(configs.camera_height, configs.camera_alpha)
        self.pipeline = self.setup_spatial_detection_pipeline(configs)
    
    @staticmethod
    def setup_camrgb(camRgb):
        # Camera properties
        camRgb.setPreviewSize(640, 640)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setImageOrientation(dai.CameraImageOrientation.VERTICAL_FLIP)    # Flip the image vertically due to reverse camera mount
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setPreviewKeepAspectRatio(False)
        camRgb.setFps(10)  # Set Camera FPS
    
    @staticmethod
    def setup_stereo(monoLeft, monoRight, stereo):
        # Stereo Camera properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setCamera("left")
        monoLeft.setImageOrientation(dai.CameraImageOrientation.VERTICAL_FLIP)  # Flip the image vertically

        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setCamera("right")
        monoRight.setImageOrientation(dai.CameraImageOrientation.VERTICAL_FLIP)  # Flip the image vertically

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
        stereo.setSubpixel(True)
    
    def setup_sdn(self, spatialDetectionNetwork):
        # Spatial Detection Network Configs
        spatialDetectionNetwork.setBlobPath(self.nnBlobPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        #spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])     # Not needed for YOLOv8
        #spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })       # Not needed for YOLOv8
        spatialDetectionNetwork.setIouThreshold(0.5)

        # Additional Settings
        spatialDetectionNetwork.setNumInferenceThreads(2)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.input.setQueueSize(1)       # Make sure frames are real-time

    def setup_spatial_detection_pipeline(self, configs):
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        # Output Nodes
        nnNetworkOut = pipeline.create(dai.node.XLinkOut)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutNN = pipeline.create(dai.node.XLinkOut)

        # Set output streams
        xoutRgb.setStreamName("rgb")
        xoutNN.setStreamName("detections")        
        nnNetworkOut.setStreamName("nnNetwork")

        # Setup Nodes
        self.setup_camrgb(camRgb)
        self.setup_stereo(monoLeft, monoRight, stereo)
        self.setup_sdn(spatialDetectionNetwork)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)
        if configs.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        else:
            camRgb.preview.link(xoutRgb.input)

        spatialDetectionNetwork.out.link(xoutNN.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

        # Additional nodes for depth visualization
        if self.show_depth:
            xoutDepth = pipeline.create(dai.node.XLinkOut)
            xoutDepth.setStreamName("depth")
            spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

        return pipeline
    
    def start_pipeline(self):
        # Connect to OAKD device and start pipeline
        with dai.Device(self.pipeline) as device:
            # Set IR Laser Dot Projector
            if self.dot_projector: device.setIrLaserDotProjectorIntensity(1.0)    # Intensity can be between 0 ~ 1

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            if self.show_depth: depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            printOutputLayersOnce = True

            while True:
                frameStart = time.time()
                inPreview = previewQueue.get()
                inDet = detectionNNQueue.get()
                inNN = networkQueue.get()

                # TODO: Get rid of this print statement
                if printOutputLayersOnce:
                    toPrint = 'Output layer names:'
                    for ten in inNN.getAllLayerNames():
                        toPrint = f'{toPrint} {ten},'
                    print(toPrint)
                    printOutputLayersOnce = False
                
                frame = inPreview.getCvFrame()
                
                # Code for showing the depth frame
                if self.show_depth: 
                    depth = depthQueue.get()
                    depthFrame = depth.getFrame() # depthFrame values are in millimeters

                    depth_downscaled = depthFrame[::4]
                    if np.all(depth_downscaled == 0):
                        min_depth = 0  # Set a default minimum depth value when all elements are zero
                    else:
                        min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
                    max_depth = np.percentile(depth_downscaled, 99)
                    depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
                    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                detections = inDet.detections

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width  = frame.shape[1]

                for detection in detections:
                    if self.show_depth:
                        roiData = detection.boundingBoxMapping
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)
                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)

                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    # Process Spatial Co-ordinates
                    x_c = detection.spatialCoordinates.x
                    y_c = detection.spatialCoordinates.y
                    z_c = detection.spatialCoordinates.z

                    # Transform coordinates to body frame
                    x_w, y_w, z_w = self.cam2world.transform_to_body_frame(x_c, y_c, z_c)

                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"X: {x_w/10:.2f} cm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"Y: {y_w/10:.2f} cm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"Z: {z_w/10:.2f} cm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                frameEnd = time.time()  # Timestamp after processing the frame
                latency = (frameEnd - frameStart) * 1000  # Convert to milliseconds

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
                cv2.putText(frame, f"Latency: {latency:.2f} ms", (2, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                if self.show_depth: cv2.imshow("depth", depthFrameColor)
                cv2.imshow("rgb", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

detector_configs = SimpleNamespace(
    nn_blob_path = 'models/yolov8n_cones_3510_yb_st_100_5s.blob',
    camera_height = 290,    # mm
    camera_alpha = 20,      # degrees
    labelMap = ["Yellow", "Blue"],   # label map for detected objects
    syncNN = True,
    dot_projector = True,
    show_depth = False,
)

cone_detector = SpatialConeDetector(detector_configs)
cone_detector.start_pipeline()