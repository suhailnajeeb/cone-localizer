import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define source
monoLeft = pipeline.create(dai.node.MonoCamera)

# Set properties for mono camera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setImageOrientation(dai.CameraImageOrientation.VERTICAL_FLIP)  # Flip the image vertically

# Create XLinkOut for streaming the mono camera output
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("monoLeft")
monoLeft.out.link(xout.input)

# Connect to device and start the pipeline
with dai.Device(pipeline) as device:
    # Turn on IR Dot Projection
    device.setIrLaserDotProjectorIntensity(1.0) # in %, from 0 to 1
    # Output queue will be used to get the frames from the camera
    monoQueue = device.getOutputQueue(name="monoLeft", maxSize=2, blocking=False)

    while True:
        frame = monoQueue.get()  # Get frame from camera
        img = frame.getCvFrame()  # Convert to OpenCV format

        cv2.imshow("Mono Left Camera", img)  # Display the image

        if cv2.waitKey(1) == ord('q'):  # Exit loop if 'q' is pressed
            break

cv2.destroyAllWindows()

