import argparse  # Import argparse for command-line argument parsing
from capture_utils import *
import time
import cv2  # Import OpenCV

out_path = 'test'
ensure_dir(out_path)

wait = 0.5

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Capture images with or without live stream display.")
parser.add_argument('--display', action='store_true', help="Enable live stream display using OpenCV.")
args = parser.parse_args()

# Main script
if __name__ == "__main__":
    pipeline = initialize_pipeline()

    # Connect to the device and start the pipeline
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        print(f"Press Enter to start capturing images every {wait} seconds, type 'q' and press Enter to quit.")
        image_counter = get_image_count(out_path)
        user_input = input()
        if user_input.strip().lower() == 'q':
            pass
        else:
            try:
                if args.display:
                    cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)  # Create a window for display if --display is set.
                
                while True:
                    frame = capture_and_return_frame(q_rgb, vflip=False)
                    if frame is not None:
                        if args.display:
                            cv2.imshow("Live Stream", frame)  # Display the frame if --display is set
                            key = cv2.waitKey(1)  # Wait for 1 ms
                            if key == ord('q'):  # Quit if 'q' is pressed
                                break
                        
                        save_frame(
                            frame,
                            file_name_prefix=f'{out_path}/cone',
                            image_counter=image_counter)
                        image_counter += 1

                    time.sleep(wait)  # Wait before capturing the next frame
            except KeyboardInterrupt:
                print("Stopped by user.")
            finally:
                if args.display:
                    cv2.destroyAllWindows()  # Clean up the OpenCV windows if they were created