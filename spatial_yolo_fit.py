
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
from types import SimpleNamespace

from transforms import CameraToWorld

from sklearn.linear_model import LinearRegression

# Fit lines using linear regression
def fit_line(x, y):
    model = LinearRegression()
    model.fit(np.array(x).reshape(-1, 1), np.array(y))
    return model.coef_[0], model.intercept_

class SpatialConeDetector:
    def __init__(self, configs):
        self.nnBlobPath = configs.nn_blob_path
        self.show_depth = configs.show_depth
        self.show_preview = configs.show_preview
        self.labelMap = configs.labelMap
        self.dot_projector = configs.dot_projector
        self.scatter_plot = configs.scatter_plot
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
    
    # @staticmethod
    def setup_stereo(self, monoLeft, monoRight, stereo):
        # Stereo Camera properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoLeft.setCamera("left")
        #monoLeft.setImageOrientation(dai.CameraImageOrientation.VERTICAL_FLIP)  # Flip the image vertically

        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoRight.setCamera("right")
        #monoRight.setImageOrientation(dai.CameraImageOrientation.VERTICAL_FLIP)  # Flip the image vertically

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        # stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
        stereo.setSubpixel(True)
        self.maxDisparity = stereo.initialConfig.getMaxDisparity()
    
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
        spatialDetectionNetwork.setNumInferenceThreads(1)
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
        # nnNetworkOut = pipeline.create(dai.node.XLinkOut)
        xoutNN = pipeline.create(dai.node.XLinkOut)

        # Set output cams
        xoutNN.setStreamName("detections")        
        # nnNetworkOut.setStreamName("nnNetwork")

        # Setup Nodes
        self.setup_camrgb(camRgb)
        self.setup_stereo(monoLeft, monoRight, stereo)
        self.setup_sdn(spatialDetectionNetwork)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)
        spatialDetectionNetwork.out.link(xoutNN.input)

        # Vertical Flip the Depth Image
        # Create ImageManip node
        manipDepth = pipeline.create(dai.node.ImageManip)
        manipDepth.initialConfig.setVerticalFlip(True)
        manipDepth.initialConfig.setFrameType(dai.ImgFrame.Type.RAW16)

        # Set the maximum output frame size to handle the larger frame
        # You need to set this to at least the size of your frame in bytes (e.g., 1843200B)
        manipDepth.setMaxOutputFrameSize(1843200)

        # Link stereo depth output to ImageManip input
        stereo.depth.link(manipDepth.inputImage)

        # Link ImageManip output to the spatial detection network depth input
        manipDepth.out.link(spatialDetectionNetwork.inputDepth)

        #stereo.depth.link(spatialDetectionNetwork.inputDepth)
        # spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

        # Additional nodes for preview visualization
        if self.show_preview:
            xoutRgb = pipeline.create(dai.node.XLinkOut)
            xoutRgb.setStreamName("rgb")
            if configs.syncNN:
                spatialDetectionNetwork.passthrough.link(xoutRgb.input)
            else:
                camRgb.preview.link(xoutRgb.input)

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
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            if self.show_preview: previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            if self.show_depth: depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            
            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            printOutputLayersOnce = True

            scale_factor = 10  # Adjust based on your space scale to fit the plot in the window

            height = 640
            width = 640

            while True:
                # Initialization before the loop
                scatter_plot = np.zeros((500, 500, 3), dtype=np.uint8)  # Change size as needed

                frameStart = time.time()
                inDet = detectionNNQueue.get()

                if self.show_preview:
                    inPreview = previewQueue.get()          
                    frame = inPreview.getCvFrame()
                    # If the frame is available, draw bounding boxes on it and show the frame
                    # height = frame.shape[0]
                    # width  = frame.shape[1]

                # Code for showing the depth frame
                if self.show_depth: 
                    depth = depthQueue.get()
                    depthFrame = depth.getFrame() # depthFrame values are in millimeters                    
                    # Optional, extend range 0..95 -> 0..255, for a better visualisation
                    if 1: depthFrame = (depthFrame * 255. / self.maxDisparity).astype(np.uint8)
                    # Optional, apply false colorization
                    if 1: depthFrame = cv2.applyColorMap(depthFrame, cv2.COLORMAP_HOT)
                    depthFrameColor = np.ascontiguousarray(depthFrame)
                    #cv2.imshow(depthWindowName, depthFrame)

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                detections = inDet.detections

                x_list = []
                y_list = []
                c_list = []

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

                    # Process Spatial Co-ordinates
                    x_c = detection.spatialCoordinates.x
                    y_c = detection.spatialCoordinates.y
                    z_c = detection.spatialCoordinates.z

                    # Transform coordinates to body frame
                    x_w, y_w, z_w = self.cam2world.transform_to_body_frame(x_c, y_c, z_c)
                    label = detection.label

                    x_list.append(x_w)
                    y_list.append(y_w)
                    c_list.append(label)

                    if self.show_preview:
                        label = self.labelMap[detection.label]
                        # Denormalize bounding box
                        x1 = int(detection.xmin * width)
                        x2 = int(detection.xmax * width)
                        y1 = int(detection.ymin * height)
                        y2 = int(detection.ymax * height)
                        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"X: {x_w/10:.2f} cm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"Y: {y_w/10:.2f} cm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"Z: {z_w/10:.2f} cm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                    # if self.scatter_plot:
                    #     # Map to scatter plot dimensions (you might need to adjust the transformation)
                    #     plot_x = int(x_w / scale_factor)# + scatter_plot.shape[1] / 2)
                    #     plot_y = int(y_w / scale_factor + scatter_plot.shape[0] / 2)

                    #     if label == "Yellow":
                    #         label_color = (0, 255, 255)  # Yellow
                    #     elif label == "Blue":
                    #         label_color = (255, 0, 0)  # Blue
                    #     else:
                    #         label_color = (255, 255, 255)  # Default to white if label is unknown

                    #     # Draw on the scatter plot image
                    #     cv2.circle(scatter_plot, (plot_x, plot_y), 5, label_color, -1)  # Green circle
                
                if self.scatter_plot:
                    if len(x_list) > 0:
                        # print(c_list)
                        x_np = np.array(x_list) / scale_factor
                        y_np = np.array(y_list) / scale_factor + 250
                        c_np = np.array(c_list)

                        # Create boolean masks based on c_list for yellow (0) and blue (1)
                        yellow_mask = c_np == 0
                        blue_mask = c_np == 1

                        bx = x_np[blue_mask]
                        by = y_np[blue_mask]
                        yx = x_np[yellow_mask]
                        yy = y_np[yellow_mask]
                        
                        # Draw the scatter plots for blue points
                        for i in range(len(bx)):
                            cv2.circle(scatter_plot, (int(bx[i]), int(by[i])), 3, (255, 0, 0), -1)  # Blue points (BGR)

                        # Draw the scatter plots for yellow points
                        for i in range(len(yx)):
                            cv2.circle(scatter_plot, (int(yx[i]), int(yy[i])), 3, (0, 255, 255), -1)  # Yellow points (BGR)
                        
                        # Fit a line to yellow points if there are enough points
                        if len(yx) > 1:
                            try:
                                yellow_fit = np.polyfit(yx, yy, 1)  # Linear fit (y = mx + b)
                                y_start_yellow = int(yellow_fit[0] * 0 + yellow_fit[1])  # Line starting at x = 0
                                y_end_yellow = int(yellow_fit[0] * 500 + yellow_fit[1])  # Line ending at x = 500
                                cv2.line(scatter_plot, (0, y_start_yellow), (500, y_end_yellow), (0, 255, 255), 2)  # Yellow line
                            except:
                                print("Could not fit yellow line")

                        # # Fit a line to blue points if there are enough points
                        if len(bx) > 1:
                            try:
                                blue_fit = np.polyfit(bx, by, 1)  # Linear fit (y = mx + b)
                                y_start_blue = int(blue_fit[0] * 0 + blue_fit[1])  # Line starting at x = 0
                                y_end_blue = int(blue_fit[0] * 500 + blue_fit[1])  # Line ending at x = 500
                                cv2.line(scatter_plot, (0, y_start_blue), (500, y_end_blue), (255, 0, 0), 2)  # Blue line
                            except:
                                print("Could not fit blue line")
                        

                    cv2.imshow("Scatter Plot", scatter_plot)

                if self.show_preview:
                    frameEnd = time.time()  # Timestamp after processing the frame
                    latency = (frameEnd - frameStart) * 1000  # Convert to milliseconds
                    cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
                    cv2.putText(frame, f"Latency: {latency:.2f} ms", (2, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.imshow("rgb", frame)

                if self.show_depth: cv2.imshow("depth", depthFrameColor)

                if cv2.waitKey(1) == ord('q'):
                    break

detector_configs = SimpleNamespace(
    nn_blob_path = 'models/yolov8n_cones_3510_yb_st_100_5s.blob',
    # nn_blob_path = 'models/yolov8n_det_3510_yb_6shave.blob',
    camera_height = 290,    # mm
    camera_alpha = 20,      # degrees
    labelMap = ["Yellow", "Blue"],   # label map for detected objects
    syncNN = True,
    dot_projector = True,
    show_depth = False,
    show_preview = False,
    scatter_plot = True,
)

cone_detector = SpatialConeDetector(detector_configs)
cone_detector.start_pipeline()