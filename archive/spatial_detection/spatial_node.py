#!/usr/bin/env python3

import cv2
import depthai as dai
from calc import HostSpatialsCalc
from utility import *
import numpy as np
import math

from transform import *

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

# Outputs
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

# Set Stream Names
xoutDepth.setStreamName("depth")
#xoutDepth.setStreamName("disp") # This looks redundant
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")


# Camera Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setImageOrientation(dai.CameraImageOrientation.VERTICAL_FLIP)  # Flip the image vertically

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setImageOrientation(dai.CameraImageOrientation.VERTICAL_FLIP)  # Flip the image vertically

# SpatialLocationCalculator configuration
topLeft = dai.Point2f(0.48, 0.47)
bottomRight = dai.Point2f(0.52, 0.53)
stepSize = 0.05
newConfig = False

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Stereo Properties
stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

#stereo.depth.link(xoutDepth.input)
#stereo.disparity.link(xoutDepth.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    #dispQ = device.getOutputQueue(name="disp", maxSize=4, blocking=False)
    spatialDataQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    configQueue = device.getInputQueue("spatialCalcConfig")

    text = TextHelper()
    #hostSpatials = HostSpatialsCalc(device)
    #y = 200
    #x = 300
    #step = 3
    #delta = 5
    #hostSpatials.setDeltaRoi(delta)

    print("Use WASD keys to move ROI.\nUse 'r' and 'f' to change ROI size.")

    while True:
        inDepth = depthQueue.get()
        # Calculate spatial coordiantes from depth frame
        #spatials, centroid = hostSpatials.calc_spatials(depthData, (x,y)) # centroid == x/y in our case
        spatialData = spatialDataQueue.get().getSpatialLocations()
        depthFrame = inDepth.getFrame()

        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)

        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        for data in spatialData:
            roi = data.config.roi.denormalize(
                width = depthFrameColor.shape[1],
                height = depthFrameColor.shape[0])
            xmin, ymin = int(roi.topLeft().x), int(roi.topLeft().y)
            xmax, ymax = int(roi.bottomRight().x), int(roi.bottomRight().y)
            # # Get disparity frame for nicer depth visualization
            # disp = dispQ.get().getFrame()
            # disp = (disp * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
            # disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

            x_c = data.spatialCoordinates.x
            y_c = data.spatialCoordinates.y
            z_c = data.spatialCoordinates.z

            #x_c, y_c, z_c = spatials['x'], spatials['y'], spatials['z']
            x_w, y_w, z_w = transform_to_body_frame(x_c, y_c, z_c)

            #text.rectangle(disp, (x-delta, y-delta), (x+delta, y+delta))
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            text.putText(depthFrameColor, "X: " + ("{:.2f} m".format(x_w/1000)), (xmin + 10, ymin + 20))
            text.putText(depthFrameColor, "Y: " + ("{:.2f} m".format(y_w/1000)), (xmin + 10, ymin + 35))
            text.putText(depthFrameColor, "Z: " + ("{:.2f} m".format(z_w/1000)), (xmin + 10, ymin + 50))

        # Show the frame
        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                newConfig = True
        elif key == ord('a'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                newConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True
        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            config.calculationAlgorithm = calculationAlgorithm
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            configQueue.send(cfg)
            newConfig = False
        # elif key == ord('r'): # Increase Delta
        #     if delta < 50:
        #         delta += 1
        #         hostSpatials.setDeltaRoi(delta)
        # elif key == ord('f'): # Decrease Delta
        #     if 3 < delta:
        #         delta -= 1
        #         hostSpatials.setDeltaRoi(delta)
        
        #print(f"Centroid: {x}, {y}")