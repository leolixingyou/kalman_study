import cv2 
import numpy as np

#### change detection and give properties for class such as age counts and histories
class Opencv_Tracker:
    def __init__(self, detected_location) -> None:
        self.kalman = cv2.KalmanFilter(6, 2)
        self.param_kalman(detected_location)

    def param_kalman(self, initial_location):
        # ### scratch try to not constant velocity
        self.kalman.transitionMatrix = np.array([ [ 1, 0, 0, 0, 1, 0],
                                        [ 0, 1, 0, 0, 0, 0],
                                        [ 0, 0, 1, 0, 0, 1],
                                        [ 0, 0, 0, 1, 0, 0], 
                                        [ 0, 0, 0, 0, 1, 0],
                                        [ 0, 0, 0, 0, 0, 1]], np.float32)   # F. input

        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 0]], np.float32) # H. input

        self.kalman.processNoiseCov =   np.array([[25, 0, 0, 0, 0, 0],
                                            [0, 10, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 25, 0, 0],
                                            [0, 0, 0, 0, 10, 0],
                                            [0, 0, 0, 0, 0, 1],], np.float32) # Q. input


        self.kalman.measurementNoiseCov = 25 * np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) # R. input , 25

        self.kalman.errorCovPost = np.eye(6, dtype=np.float32) * 1e6 # P._k|k  KF state var
        self.kalman.statePost = np.array([initial_location[0], initial_location[1], initial_location[2], initial_location[3], 0, 0], np.float32) # x^_k|k  KF state var
        

### work for single axis motion
class Single_Tracker_Kalman:
    def __init__(self) -> None:
        self.kalman_init = False 
        self.is_object_detected = False 
    
        self.prediction6x1 = None
        self.correction6x1 = None

    def tracking_update(self, box_det):
        ## Tracking
        if not self.kalman_init:
            if self.is_object_detected:
                self.kalman_track = Opencv_Tracker(box_det)
                self.kalman_init = True
        else:
            self.prediction6x1 = self.kalman_track.kalman.predict()
            if self.is_object_detected:
                self.kalman_track.kalman.correct(box_det.astype(np.float32))
                self.correction6x1 = self.kalman_track.kalman.statePost.copy()

        return self.prediction6x1, self.correction6x1

### work for single axis motion
class Tracker_Kalman_for_MOT:
    count = 0
    def __init__(self, box_det) -> None:
        self.kalman_track = Opencv_Tracker(box_det)
    
        self.prediction6x1 = None
        self.correction6x1 = None
        self.id = Tracker_Kalman_for_MOT.count
        Tracker_Kalman_for_MOT.count += 1

        ## 'Predict' and 'Correction'
        self.state_tracker = 'Predict'


    def tracking_update(self, box_det):
        self.state_tracker = 'Predict'
        self.prediction6x1 = self.kalman_track.kalman.predict()
        if box_det is not []:
            self.state_tracker = 'Correction'
            self.kalman_track.kalman.correct(box_det.astype(np.float32))
            self.correction6x1 = self.kalman_track.kalman.statePost.copy()