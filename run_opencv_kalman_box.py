import cv2 
import copy 
import numpy as np

from read_video import VIDEO_READER
from kalman_ball import detect_object
from plot_kalman_figures import plot_detection_and_tracking_2


def fan_fu_transform(A):
    m, n = A.shape
    B = np.zeros((m*n, m*n), dtype=int)
    
    for i in range(m):
        for j in range(n):
            B[i*n+j, i*n+j] = A[i, j]
    
    return B

def param_kalman(kalman, initial_location):
    # ### scratch try to not constant velocity
    kalman.transitionMatrix = np.array([ [ 1, 0, 0, 0, 1, 0],
                                    [ 0, 1, 0, 0, 0, 0],
                                    [ 0, 0, 1, 0, 0, 1],
                                    [ 0, 0, 0, 1, 0, 0], 
                                    [ 0, 0, 0, 0, 1, 0],
                                    [ 0, 0, 0, 0, 0, 1]], np.float32)   # F. input

    kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0]], np.float32) # H. input

    kalman.processNoiseCov =   np.array([[25, 0, 0, 0, 0, 0],
                                        [0, 10, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 25, 0, 0],
                                        [0, 0, 0, 0, 10, 0],
                                        [0, 0, 0, 0, 0, 1],], np.float32) # Q. input


    kalman.measurementNoiseCov = 25 * np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) # R. input , 25

    kalman.errorCovPost = np.eye(6, dtype=np.float32) * 1e6 # P._k|k  KF state var
    kalman.statePost = np.array([initial_location[0], initial_location[1], initial_location[2], initial_location[3], 0, 0], np.float32) # x^_k|k  KF state var
       
    return kalman

def boundary_limits(pts):
    if pts[0] < 0:
        pts[0] = 0
    if pts[0] > img_w:
        pts[0] = img_w

    if pts[1] < 0:
        pts[1] = 0
    if pts[1] > img_h:
        pts[1] = img_h

    pts = [int(x) for x in pts]

    return pts

def annotate_box_tracked_object_kalman(frame, detected_location, is_object_detected, prediction6x1, correction6x1, img_h, img_w):
    frame_detect = copy.copy(frame)
    frame_predict = copy.copy(frame)
    frame_correct = copy.copy(frame)
    frame_combined = copy.copy(frame)

    if prediction6x1 is not None:
        pt = np.reshape([prediction6x1[0],prediction6x1[2]],-1).astype(np.int32)
        pt_top = np.array([prediction6x1[0], prediction6x1[2]]) - (1/2) * np.array([prediction6x1[1], prediction6x1[3]])
        pt_bottom = np.array([prediction6x1[0], prediction6x1[2]]) + (1/2) * np.array([prediction6x1[1], prediction6x1[3]])

        pt_top = boundary_limits(pt_top)
        pt_bottom = boundary_limits(pt_bottom)

        cv2.rectangle(frame_predict, pt_top, pt_bottom, (0, 0, 255), 2)
        cv2.putText(frame_predict, str(pt), pt - np.array([0,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.rectangle(frame_combined, pt_top, pt_bottom, (255, 0, 255), 2)
        cv2.putText(frame_combined, str(pt), pt - np.array([20,40]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(frame_combined, 'Prediction', pt - np.array([20,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if is_object_detected:
        pt = np.reshape([detected_location[0],detected_location[2]],-1).astype(np.int32)
        pt_top = np.array([detected_location[0], detected_location[2]]) - (1/2) * np.array([detected_location[1], detected_location[3]])
        pt_bottom = np.array([detected_location[0], detected_location[2]]) + (1/2) * np.array([detected_location[1], detected_location[3]])
        
        pt_top = boundary_limits(pt_top)
        pt_bottom = boundary_limits(pt_bottom)

        cv2.rectangle(frame_detect, pt_top, pt_bottom, (0, 0, 255), 2)
        cv2.putText(frame_detect, str(pt), pt - np.array([0,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.rectangle(frame_combined, pt_top, pt_bottom, (255, 100, 0), 2)
        cv2.putText(frame_combined, str(pt), pt - np.array([-20,40]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        cv2.putText(frame_combined, 'Detection', pt - np.array([-20,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    if correction6x1 is not None:
        pt = np.reshape([correction6x1[0],correction6x1[2]],-1).astype(np.int32)
        pt_top = np.array([correction6x1[0], correction6x1[2]]) - (1/2) * np.array([correction6x1[1], correction6x1[3]])
        pt_bottom = np.array([correction6x1[0], correction6x1[2]]) + (1/2) * np.array([correction6x1[1], correction6x1[3]])

        pt_top = boundary_limits(pt_top)
        pt_bottom = boundary_limits(pt_bottom)

        cv2.rectangle(frame_correct, pt_top, pt_bottom, (0, 0, 255), 2)
        cv2.putText(frame_correct, str(pt), pt - np.array([0,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.rectangle(frame_combined, pt_top, pt_bottom, (0, 255, 255), 2)
        cv2.putText(frame_combined, str(pt), pt + np.array([20,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame_combined, 'Correction', pt + np.array([20,40]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame_detect, frame_predict, frame_correct, frame_combined 



file_path = '/workspace/singleball.mp4'
video_reader = VIDEO_READER(file_path)

ret,first_frame = video_reader.cap.read()
img_h, img_w, _ = first_frame.shape
kalman = cv2.KalmanFilter(6, 2)
kalman_init = False
detection_list = []
prediction_list = []
correction_list = []
box = np.array([])
prediction6x1 = None
correction6x1 = None

while video_reader.cap.isOpened():
    frame = video_reader.read_frame()
    if frame is None:
        break

    ## detection
    detected_location, is_object_detected, contours = detect_object(frame, first_frame)

    if is_object_detected:
        box_2d = np.reshape([np.min(contours, axis=0), np.max(contours, axis=0)],(2,2))
        box_xmid, box_ymid = np.mean(box_2d,axis=0)
        box_width, box_height = box_2d[1][0]-box_2d[0][0], box_2d[1][1] - box_2d[0][1]
        box = np.array([box_xmid, box_width, box_ymid, box_height]).astype(np.int32)
    else:
        box = np.array([])

    if not kalman_init:
        if is_object_detected:
            kalman = param_kalman(kalman, box)
            kalman_init = True
    else:
        prediction6x1 = kalman.predict()
        if is_object_detected:
            kalman.correct(box.astype(np.float32))
            correction6x1 = kalman.statePost.copy() 
    
    frame_detect, frame_predict, frame_correct, frame_combined  = annotate_box_tracked_object_kalman(frame, box, is_object_detected, prediction6x1, correction6x1, img_h, img_w)

    # detection_list.append([box[0],box[2]] if box != [] else None)
    # correction_list.append(np.reshape([correction6x1[0],correction6x1[2]],-1).astype(np.int32) if correction6x1 is not None else None)
    # prediction_list.append(np.reshape([prediction6x1[0],prediction6x1[2]],-1).astype(np.int32) if prediction6x1 is not None else None)

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

# ### plotting the detection location on first frame and tracking location

# # 绘制检测和跟踪的位置轨迹
# plot_detection_and_tracking_2(first_frame, detection_list, prediction_list, correction_list)