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
        
