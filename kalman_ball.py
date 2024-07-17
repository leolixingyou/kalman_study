import cv2
import copy
import numpy as np


def detect_object(frame, firts_frame):
    frame_dev = cv2.absdiff(frame, firts_frame)
    _, frame_detect = cv2.threshold(frame_dev, 100, 255, cv2.THRESH_BINARY)

    if np.count_nonzero(frame_detect) > 0 :
        contours, _ = cv2.findContours(cv2.cvtColor(frame_detect, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = np.array([x, y]) 
            
            return center, True, largest_contour
    return np.array([]), False, None


def annotate_tracked_object(frame, detected_location, is_object_detected, tracked_location, label):
    frame_detect = copy.copy(frame)
    frame_combined = copy.copy(frame)
    frame_track = copy.copy(frame)

    if is_object_detected:
        cv2.circle(frame_detect, (int(detected_location[0]), int(detected_location[1])), 5, (0, 0, 255), -1)
        cv2.circle(frame_combined, (int(detected_location[0]), int(detected_location[1])), 5, (255, 100, 0), -1)

    if tracked_location is not None:
        cv2.circle(frame_track, (int(tracked_location[0]), int(tracked_location[1])), 5, (0, 0, 255), -1)
        cv2.putText(frame_track, label, (int(tracked_location[0]), int(tracked_location[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame_track, str([tracked_location[0], tracked_location[1]]), (int(tracked_location[0]), int(tracked_location[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.circle(frame_combined, (int(tracked_location[0]), int(tracked_location[1])), 5, (0, 0, 255), -1)
        cv2.putText(frame_combined, label, (int(tracked_location[0]), int(tracked_location[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame_combined, str([tracked_location[0], tracked_location[1]]), (int(tracked_location[0]), int(tracked_location[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame_detect, frame_track, frame_combined 


class Kalman_Single_Ball:
    def __init__(self,):
        self.track_initialized = False
        pass

    def tracking_state_judgement(self, detected_location, is_object_detected):
        if not self.track_initialized:
            if is_object_detected:
                self.kf = self.initialize_kalman_filter(detected_location)
                self.track_initialized = True
                # ### opencv
                # tracked_location = self.kf.statePost[:2]
                # ### scratch
                # tracked_location = self.kf.x[:2]

                tracked_location_l = self.kf.kalman_correct(detected_location)
                tracked_location = [tracked_location_l[0][0], tracked_location_l[0][1]]
                
                label = 'Initial'
            else:
                tracked_location = None
                label = ''
        else:

            if is_object_detected:
                # ### opencv
                # self.kf.correct(detected_location.astype(np.float32))
                # tracked_location = self.kf.statePost[:2]
                # ### scratch
                self.kf.kalman_predict()
                detection_l = detected_location.astype(np.float32)
                tracked_location_l = self.kf.kalman_correct(detection_l)
                tracked_location = [tracked_location_l[0][0], tracked_location_l[0][1]]

                label = 'Corrected' 
            else:
                # ### opencv
                # tracked_location = self.kf.predict()[:2]
                # ### scratch
                tracked_location_l = self.kf.kalman_predict()
                tracked_location = [tracked_location_l[0][0], tracked_location_l[0][1]]
                label = 'Predicted'

        return tracked_location, label

    ### 엄청 important 데스
    def initialize_kalman_filter(self, initial_location):
        flag_list = ['Constant_Velocity', 'Constant_Accelerate']
        flag = flag_list[0]

        F_transitionMatrix, H_measurementMatrix, Q_processNoiseCov, R_measurementNoiseCov, P_StateCov_errorCovPost, x_State_statePost = self.param_kalman(initial_location, flag)

        kf = KalmanFilter(initial_location, F=F_transitionMatrix, H=H_measurementMatrix, Q=Q_processNoiseCov, R=R_measurementNoiseCov, P=P_StateCov_errorCovPost, x0=x_State_statePost)

        return kf
    
    def param_kalman(self, initial_location, flag):

        if flag == 'Constant_Accelerate':
            # ### scratch try to not constant velocity 
            F_transitionMatrix = np.array([ [ 1, 1, 0.5, 0, 0, 0],
                                            [ 0, 1, 1, 0, 0, 0],
                                            [ 0, 0, 1, 0, 0, 0],
                                            [ 0, 0, 0, 1, 1, 0.5],
                                            [ 0, 0, 0, 0, 1, 1],
                                            [ 0, 0, 0, 0, 0, 1]], np.float32)   # F. input

            H_measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 0]], np.float32) # H. input

            Q_processNoiseCov =   np.array([[25, 0, 0, 0, 0, 0],
                                            [0, 10, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 25, 0, 0],
                                            [0, 0, 0, 0, 10, 0],
                                            [0, 0, 0, 0, 0, 1],], np.float32) # Q. input

            R_measurementNoiseCov = 25 * np.array([[1, 0],
                                                [0, 1]], np.float32) # R. input , 25
            
            P_StateCov_errorCovPost = np.eye(6, dtype=np.float32) * 1e6 # P._k|k  KF state var
            x_State_statePost = np.array([initial_location[0], 0, 0, initial_location[1], 0, 0], np.float32) # x^_k|k  KF state var
            # x_State_statePost = np.array([0, 0, 0, 0, 0, 0], np.float32) # x^_k|k  KF state var
            
        if flag == 'Constant_Velocity':
            # ### scratch try to constant velocity 
            F_transitionMatrix = np.array([[ 1, 1, 0, 0],
                                        [ 0, 1, 0, 0],
                                        [ 0, 0, 1, 1],
                                        [ 0, 0, 0, 1]], np.float32)   # F. input

            H_measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 0, 1, 0]], np.float32) # H. input

            Q_processNoiseCov =   np.array([[25, 0, 0, 0],
                                            [0, 10, 0, 0],
                                            [0, 0, 25, 0],
                                            [0, 0, 0, 10],], np.float32) # Q. input

            R_measurementNoiseCov = np.array([[25, 0],
                                                [0, 25]], np.float32) # R. input
            
            P_StateCov_errorCovPost = np.eye(4, dtype=np.float32) * 1e6 # P._k|k  KF state var
            x_State_statePost = np.array([initial_location[0], 0, initial_location[1], 0], np.float32) # x^_k|k  KF state var

        return F_transitionMatrix, H_measurementMatrix, Q_processNoiseCov, R_measurementNoiseCov, P_StateCov_errorCovPost, x_State_statePost

class KalmanFilter(object):
    def __init__(self, initial_location, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None ):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) * 1e6 if P is None else P
        # self.x = np.zeros((self.n, 1)) if x0 is None else x0
        # self.x = np.array([initial_location[0], initial_location[1], 0, 0, 0, 0], np.float32) # x^_k|k  KF state var
        self.x = np.array([initial_location[0], 0, initial_location[1], 0], np.float32) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def correction(self,):
        ### 1. Compute the Kalman G gain 
        ### 1. P(bar)_k * H(T) * [H * P(bar)_k * H(T) + R]^(-1)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        ### 2. Update The Estimate via Z => z is measurement
        ### 2. x(head)_k = x(head,bar)_k + K_k * [z_k - H * x(head,bar)_k]
        z = [self.state[0], self.state[2]]
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        ### 3. Update The Error Covariance 
        ### 3. P_k = (I - K_k * H) * P(bar)_k
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        # self.P = self.P - K * self.H * self.P

    def kalman_predict(self):
        """ 
        StateTransitionModel => F
        pProcessNoise => Q
        pStateCovariance => P(-)_k
        MeasurementModel => H
        pMeasurementNoise => R
        pState => x(-)_k
        """
        StateTransitionModel = self.F
        pProcessNoise = self.Q
        MeasurementModel = self.H
        pState = np.reshape(self.x,(self.x.size,1))
        pStateCovariance = self.P

        x = np.dot(StateTransitionModel, pState)

        # Predict the state covariance
        P_pred = np.dot(np.dot(StateTransitionModel, pStateCovariance), StateTransitionModel.T) + pProcessNoise

        # State is a column vector internally; but it is a row vector for output
        z_pred = np.dot(MeasurementModel, x).T


        # Update internal state and covariance
        self.x = x
        self.P = P_pred


        return z_pred

    def kalman_correct(self, z):

        """ 
        pStateCovariance => P(-)_k
        MeasurementModel => H
        pMeasurementNoise => R
        pState => x(-)_k
        """
        pStateCovariance = self.F
        MeasurementModel = self.H
        pMeasurementNoise = self.R
        # pState = self.x
        pState = np.reshape(self.x,(self.x.size,1))
        pStateCovariance = self.P

        # Set the measurement
        pMeasurement = z.reshape(-1, 1)

        # Compute the Kalman Gain
        gain_numerator = np.dot(pStateCovariance, MeasurementModel.T)
        residualCovariance = np.dot(np.dot(MeasurementModel, pStateCovariance), MeasurementModel.T) + pMeasurementNoise
        gain = np.dot(gain_numerator, np.linalg.inv(residualCovariance))

        # Update the state estimate
        innovation = pMeasurement - np.dot(MeasurementModel, pState)
        x = pState + np.dot(gain, innovation)

        # Update the state covariance estimate
        P_corr = pStateCovariance - np.dot(np.dot(gain, MeasurementModel), pStateCovariance)

        # Corrected state and measurement
        z_corr = np.dot(MeasurementModel, x).T

        # Update internal state and covariance
        self.x = x
        self.P = P_corr

        return z_corr