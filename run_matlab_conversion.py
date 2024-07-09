import cv2

from src_code.tl_detector.kalman_study.read_video import VIDEO_READER
from src_code.tl_detector.kalman_study.kalman_ball import detect_object, annotate_tracked_object, Kalman_Single_Ball
from plot_kalman_figures import plot_detection_and_tracking

file_path = '/workspace/src_code/tl_detector/kalman_study/singleball.mp4'
video_reader = VIDEO_READER(file_path)

ret,first_frame = video_reader.cap.read()
kmsb = Kalman_Single_Ball()
count = 0
detection_list = []
tracking_list = []

while video_reader.cap.isOpened():
    frame = video_reader.read_frame()
    if frame is None:
        break

    ## process6
    detected_location, is_object_detected = detect_object(frame, first_frame)
    # if count % 3 == 0:
    #     is_object_detected = True
    # else:
    #     is_object_detected = False

    tracked_location, label = kmsb.tracking_state_judgement(detected_location, is_object_detected)
    frame_detect, frame_track, frame_combined = annotate_tracked_object(frame, detected_location, is_object_detected, tracked_location, label)

    detection_list.append(detected_location if detected_location!=[] else None)
    tracking_list.append(tracked_location if tracked_location!=[] else None)

    # Display the frame
    cv2.imshow('Raw Image', frame) 
    cv2.imshow('Detection Image', frame_detect)
    cv2.imshow('Tracking Image', frame_track)
    cv2.imshow('Commbined Image', frame_combined)
    count +=1 
    # ###Press 'q' to exit the video early
    # if cv2.waitKey(int(1000 / video_reader.fps)) & 0xFF == ord('q'):
    # ###frame by frame
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

video_reader.release()

### plotting the detection location on first frame and tracking location

# 绘制检测和跟踪的位置轨迹
plot_detection_and_tracking(first_frame, detection_list, tracking_list)