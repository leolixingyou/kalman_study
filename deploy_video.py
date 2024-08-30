import os
import cv2
import copy
import numpy as np

from read_video import VIDEO_READER
from opencv_tracker import Opencv_Tracker
from tools import opencv_box_detection, annotate_box_tracked_object_kalman, \
            plot_detection_and_tracking_2, xyxy_to_xywh
from detection_trt.detector_yolov7 import Detecotr_YoloV7

    
def read_video_first_video(file_path):
    video_reader = VIDEO_READER(file_path)

    ret,first_frame = video_reader.cap.read()
    img_h, img_w, _ = first_frame.shape
    return video_reader, first_frame, [img_h, img_w]


def run_single_tracker(detector_yolov, file_path):
    ## single tracker
    kalman_init = False

    video_reader, first_frame, [img_h, img_w] = read_video_first_video(file_path)

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

        ## Detection
        # box, is_object_detected = opencv_box_detection(frame, first_frame)
        filter_img, tl_boxes = detector_yolov.image_process(frame, '1')
        if tl_boxes != []:
            tl_boxes = xyxy_to_xywh(tl_boxes[-1])

        is_object_detected = True if np.any(tl_boxes) else False
        ## Tracking
        if not kalman_init:
            if is_object_detected:
                kalman_track = Opencv_Tracker(tl_boxes)
                kalman_init = True
        else:
            prediction6x1 = kalman_track.kalman.predict()
            if is_object_detected:
                kalman_track.kalman.correct(tl_boxes.astype(np.float32))
                correction6x1 = kalman_track.kalman.statePost.copy()

        ## Visualization
        frame_detect, frame_predict, frame_correct, frame_combined  = annotate_box_tracked_object_kalman(frame, tl_boxes, is_object_detected, prediction6x1, correction6x1, img_h, img_w)

        ## Process for Visualization 
        detection_list.append([tl_boxes[0],tl_boxes[2]] if tl_boxes != [] else None)
        correction_list.append(np.reshape([correction6x1[0],correction6x1[2]],-1).astype(np.int32) if correction6x1 is not None else None)
        prediction_list.append(np.reshape([prediction6x1[0],prediction6x1[2]],-1).astype(np.int32) if prediction6x1 is not None else None)

        # Display the frame
        # cv2.imshow('Raw Image', frame)
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

if __name__ == "__main__":
    file_path = '/workspace/time_2024_06_19_14_48_f60.mp4'
    detector_yolov = Detecotr_YoloV7()
    run_single_tracker(detector_yolov, file_path)