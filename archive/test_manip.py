import cv2
import depthai as dai
import time

pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_rgb.setPreviewSize(640, 640)
cam_rgb.setPreviewKeepAspectRatio(False)

# Manipulation for letterboxing
#manip = pipeline.createImageManip()
#manip.initialConfig.setResizeThumbnail(640, 640, 0, 0, 0)
#manip.initialConfig.setKeepAspectRatio(True) # Preserve Aspect Ratio
#manip.setMaxOutputFrameSize(640 * 640 *3)
# cam_rgb.video -> manip.input
#cam_rgb.preview.link(manip.inputImage)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
# manip.out -> xout_rgb.input
#manip.out.link(xout_rgb.input)
cam_rgb.preview.link(xout_rgb.input)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize = 4, blocking = False)
    # Wait for 2 seconds to let the camera initialise
    time.sleep(2)

    while True:
        in_rgb = q_rgb.get()

        if in_rgb is not None:
            in_rgb_data = in_rgb.getData()
            frame = in_rgb.getCvFrame()
            print(f"Data Length: {len(in_rgb_data)}")
            print(frame.shape)
            cv2.imshow("Output", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break