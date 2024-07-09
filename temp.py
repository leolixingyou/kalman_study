import cv2
import copy
import numpy as np

class KalmanFilterForTracking:
    def __init__(self):
        self.video_path = '/workspace/src_code/tl_detector/kalman_study/singleball.mp4'
        # self.video_path = '/home/cvlab-swlee/Desktop/log/postgraduates/AVAS_autonomous_vehicle_auto_shipment/ros_refact/ros_vision_refact/src_code/tl_detector/kalman_study/singleball.mp4' 
        self.video = cv2.VideoCapture(self.video_path)
        self.fgbg = None
        # self.fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=100)
        self.track_initialized = False
        self.kf = None

    def initialize_kalman_filter(self, initial_location):
        kf = cv2.KalmanFilter(6, 2)

        kf.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0],
                                        [0, 1, 0, 1, 0, 0.5],
                                        [0, 0, 1, 0, 1, 0],
                                        [0, 0, 0, 1, 0, 1],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]], np.float32)   # F. input

        kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0]], np.float32) # H. input


        kf.transitionMatrix = np.array([[1, 0, 1, 0, 0,     0],
                                        [0, 1, 0, 1, 0,     0],
                                        [0, 0, 1, 0, 1,     0],
                                        [0, 0, 0, 1, 0,     1],
                                        [0, 0, 0, 0, 1,     0],
                                        [0, 0, 0, 0, 0,     1]], np.float32)   # F. input

        kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0]], np.float32) # H. input

        kf.processNoiseCov = 1e-1 * np.array([[1, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]], np.float32) # Q. input

        kf.measurementNoiseCov = 1e-2 * np.array([[1, 0],
                                           [0, 1]], np.float32) # R. input

        kf.errorCovPost = np.eye(6, dtype=np.float32) * 1e2 # P._k|k  KF state var
        kf.statePost = np.array([initial_location[0], initial_location[1], 0, 0, 0, 0], np.float32) # x^_k|k  KF state var

        return kf

    def detect_object(self, frame):
        frame_dev = cv2.absdiff(frame, self.fgbg)
        _, self.frame_detect = cv2.threshold(frame_dev, 100, 255, cv2.THRESH_BINARY)

        if np.count_nonzero(self.frame_detect) > 0 :
            contours, _ = cv2.findContours(cv2.cvtColor(self.frame_detect, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = np.array([x, y]) 
                
                return center, True
        return np.array([0, 0]), False

    def annotate_tracked_object(self, detected_location, is_object_detected, tracked_location, label):

        if is_object_detected is not None:
            cv2.circle(self.frame_detect, (int(detected_location[0]), int(detected_location[1])), 5, (0, 0, 255), -1)
            cv2.circle(self.frame_combined, (int(detected_location[0]), int(detected_location[1])), 5, (255, 100, 0), -1)

        if tracked_location is not None:
            cv2.circle(self.frame_track, (int(tracked_location[0]), int(tracked_location[1])), 5, (0, 0, 255), -1)
            cv2.putText(self.frame_track, label, (int(tracked_location[0]), int(tracked_location[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.circle(self.frame_combined, (int(tracked_location[0]), int(tracked_location[1])), 5, (0, 0, 255), -1)
            cv2.putText(self.frame_combined, label, (int(tracked_location[0]), int(tracked_location[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    def track_single_object(self):
        count = 0
        ret, self.fgbg = self.video.read()
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break

            self.frame_detect = copy.copy(frame)
            self.frame_track = copy.copy(frame)
            self.frame_combined = copy.copy(frame)

            detected_location, is_object_detected = self.detect_object(frame)

            if not self.track_initialized:
                if is_object_detected:
                    self.kf = self.initialize_kalman_filter(detected_location)
                    self.track_initialized = True
                    tracked_location = self.kf.statePost[:2]
                    label = 'Initial'
                else:
                    tracked_location = None
                    label = ''
            else:
                if is_object_detected:
                    self.kf.correct(detected_location.astype(np.float32))
                    tracked_location = self.kf.statePost[:2]
                    label = 'Corrected' 
                else:
                    tracked_location = self.kf.predict()[:2]
                    label = 'Predicted'

            self.annotate_tracked_object(detected_location, is_object_detected, tracked_location, label)
            cv2.imshow('Raw image', frame)
            cv2.imshow('Detection', self.frame_detect)
            cv2.imshow('Kalman Filter Tracking', self.frame_track)
            cv2.imshow('Result', self.frame_combined)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = KalmanFilterForTracking()
    tracker.track_single_object()
