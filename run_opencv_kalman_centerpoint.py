import cv2 
import copy 
import numpy as np

from read_video import VIDEO_READER
from kalman_ball import detect_object
from tools import plot_detection_and_tracking_2


def param_kalman(kalman, initial_location):
    # ### scratch try to not constant velocity 
    kalman.transitionMatrix = np.array([ [ 1, 1, 0.5, 0, 0, 0],
                                    [ 0, 1, 1, 0, 0, 0],
                                    [ 0, 0, 1, 0, 0, 0],
                                    [ 0, 0, 0, 1, 1, 0.5],
                                    [ 0, 0, 0, 0, 1, 1],
                                    [ 0, 0, 0, 0, 0, 1]], np.float32)   # F. input

    kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0]], np.float32) # H. input

    kalman.processNoiseCov =   np.array([[25, 0, 0, 0, 0, 0],
                                    [0, 10, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 25, 0, 0],
                                    [0, 0, 0, 0, 10, 0],
                                    [0, 0, 0, 0, 0, 1],], np.float32) # Q. input

    kalman.measurementNoiseCov = 25 * np.array([[1, 0],
                                        [0, 1]], np.float32) # R. input , 25
    
    kalman.errorCovPost = np.eye(6, dtype=np.float32) * 1e6 # P._k|k  KF state var
    kalman.statePost = np.array([initial_location[0], 0, 0, initial_location[1], 0, 0], np.float32) # x^_k|k  KF state var
       
    return kalman

def annotate_tracked_object_kalman(frame, detected_location, is_object_detected, prediction6x1, correction6x1):
    frame_detect = copy.copy(frame)
    frame_predict = copy.copy(frame)
    frame_correct = copy.copy(frame)
    frame_combined = copy.copy(frame)

    if prediction6x1 is not None:
        pt = list(map(int,prediction6x1[::3]))

        cv2.circle(frame_predict, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame_predict, str(pt), pt - np.array([0,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.circle(frame_combined, pt, 5, (255, 0, 255), -1)
        cv2.putText(frame_combined, str(pt), pt - np.array([20,40]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(frame_combined, 'Prediction', pt - np.array([20,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if is_object_detected:
        pt = list(map(int,detected_location))
        cv2.circle(frame_detect, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame_detect, str(pt), pt - np.array([0,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.circle(frame_combined, pt, 5, (255, 100, 0), -1)
        cv2.putText(frame_combined, str(pt), pt - np.array([-20,40]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        cv2.putText(frame_combined, 'Detection', pt - np.array([-20,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    if correction6x1 is not None:
        pt = list(map(int,correction6x1[::3]))
        cv2.circle(frame_correct, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame_correct, str(pt), pt - np.array([0,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.circle(frame_combined, pt, 5, (0, 255, 255), -1)
        cv2.putText(frame_combined, str(pt), pt + np.array([20,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame_combined, 'Correction', pt + np.array([20,40]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame_detect, frame_predict, frame_correct, frame_combined 



file_path = '/workspace/singleball.mp4'
video_reader = VIDEO_READER(file_path)

ret,first_frame = video_reader.cap.read()
kalman = cv2.KalmanFilter(6, 2)
kalman_init = False
detection_list = []
prediction_list = []
correction_list = []
prediction6x1 = None
correction6x1 = None

while video_reader.cap.isOpened():
    frame = video_reader.read_frame()
    if frame is None:
        break

    ## detection
    detected_location, is_object_detected, _ = detect_object(frame, first_frame)

    if not kalman_init:
        if is_object_detected:
            kalman = param_kalman(kalman, detected_location)
            kalman_init = True
    else:
        prediction6x1 = kalman.predict()
        if is_object_detected:
            kalman.correct(detected_location.astype(np.float32))
            correction6x1 = kalman.statePost.copy()
    
    frame_detect, frame_predict, frame_correct, frame_combined = annotate_tracked_object_kalman(frame, detected_location, is_object_detected, prediction6x1, correction6x1)

    detection_list.append(detected_location if detected_location!=[] else None)
    correction_list.append(correction6x1[::3] if correction6x1 is not None else None)
    prediction_list.append(prediction6x1[::3] if prediction6x1 is not None else None)

    # Display the frame
    cv2.imshow('Raw Image', frame)
    cv2.imshow('Detection Image', frame_detect)
    cv2.imshow('Predict Image', frame_predict)
    cv2.imshow('Correct Image', frame_correct)
    cv2.imshow('Commbined Image', frame_combined)
    # ###Press 'q' to exit the video early
    # if cv2.waitKey(int(1000 / video_reader.fps)) & 0xFF == ord('q'):
    # ###frame by frame
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

video_reader.release()

### plotting the detection location on first frame and tracking location

# 绘制检测和跟踪的位置轨迹
plot_detection_and_tracking_2(first_frame, detection_list, prediction_list, correction_list)