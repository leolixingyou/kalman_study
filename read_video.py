import cv2
import numpy as np

class VIDEO_READER:
    def __init__(self, file_path):
        # Path to the MP4 video file
        self.video_file_path = file_path
        
        # Create a VideoCapture object
        self.cap = cv2.VideoCapture(self.video_file_path)

        # Check if the video opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video FPS: {self.fps}, Width: {self.width}, Height: {self.height}")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Process the frame (e.g., convert to grayscale)
        return frame

    def release(self):
        # Release the VideoCapture object and close windows
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ =='__main__':
    file_path = '/workspace/src_code/tl_detector/kalman_study/singleball.mp4'
    video_reader = VIDEO_READER(file_path)

    while video_reader.cap.isOpened():
        frame = video_reader.read_frame()
        if frame is None:
            break

        # Display the frame
        cv2.imshow('Frame', frame)

        # Press 'q' to exit the video early
        if cv2.waitKey(int(1000 / video_reader.fps)) & 0xFF == ord('q'):
            break

    video_reader.release()

