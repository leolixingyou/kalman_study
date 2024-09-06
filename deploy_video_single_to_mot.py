import cv2
import numpy as np

from read_video import VIDEO_READER
from multi_tracker import Multi_Object_Tracking
from tools import opencv_box_detection, annotate_box_tracked_object_kalman, \
            plot_detection_and_tracking_2, xyxy_to_xywh, iou_batch
from detection_trt.detector_yolov7 import Detecotr_YoloV7


def linear_assignment(cost_matrix):
    try:
        import lap #linear assignment problem solver
        _, x, y = lap.lapjv(cost_matrix, extend_cost = True)
        return np.array([[y[i],i] for i in x if i>=0])
    except ImportError: 
        from scipy.optimize import linear_sum_assignment
        x,y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x,y)))
    
def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.0):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of 
    1. matches,
    2. unmatched_detections
    3. unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    print(f'iou_matrix is {iou_matrix}')
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() ==1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    
    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def read_video_first_video(file_path):
    video_reader = VIDEO_READER(file_path)

    ret,first_frame = video_reader.cap.read()
    img_h, img_w, _ = first_frame.shape
    return video_reader, first_frame, [img_h, img_w]


def run_single_tracker(detector_yolov, file_path):
    ## single tracker
    kalman_init = False

    video_reader, first_frame, [img_h, img_w] = read_video_first_video(file_path)

    mot_tarcker = Multi_Object_Tracking(max_age = 1, min_hits = 3 , iou_threshold = 0.3)

    detection_list = []
    prediction_list = []
    correction_list = []

    trts_list = []

    det_hist = []
    while video_reader.cap.isOpened():
        frame = video_reader.read_frame()
        if frame is None:
            break

        ## Detection
        # box, is_object_detected = opencv_box_detection(frame, first_frame)
        filter_img, tl_boxes = detector_yolov.image_process(frame, '1')
        
        ## Take xyxy from tl_boxes
        if tl_boxes != []:
            # ## Make tl_box from xyxy to xywh 
            # tl_boxes_det = [xyxy_to_xywh(x[-1]) for x in tl_boxes]
            ## Keep xyxy
            tl_boxes_det = [x[-1] for x in tl_boxes]
        else:
            tl_boxes_det = []

        if tl_boxes_det != [] and det_hist[-1] != []:
            matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(tl_boxes_det, det_hist[-1])
            print(matches, unmatched_detections, unmatched_trackers)

        det_hist.append(tl_boxes_det)
        # ### Tracking
        # single_tarcker.tracking_update(tl_boxes_det)



        # ## Visualization
        # frame_detect, frame_predict, frame_correct, frame_combined  = annotate_box_tracked_object_kalman(frame, tl_boxes_det, is_object_detected, prediction6x1, correction6x1, img_h, img_w)

        # ## Process for Visualization 
        # detection_list.append([tl_boxes_det[0],tl_boxes_det[2]] if tl_boxes_det != [] else None)
        # correction_list.append(np.reshape([correction6x1[0],correction6x1[2]],-1).astype(np.int32) if correction6x1 is not None else None)
        # prediction_list.append(np.reshape([prediction6x1[0],prediction6x1[2]],-1).astype(np.int32) if prediction6x1 is not None else None)

        # # Display the frame
        # # cv2.imshow('Raw Image', frame)
        # cv2.imshow('Detection Image', frame_detect)
        # cv2.imshow('Predict Image', frame_predict)
        # cv2.imshow('Correct Image', frame_correct)
        # cv2.imshow('Commbined Image', frame_combined)
        # # ###Press 'q' to exit the video early
        # # if cv2.waitKey(int(1000 / video_reader.fps)) & 0xFF == ord('q'):
        # # ###frame by frame
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

    video_reader.release()

    ### plotting the detection location on first frame and tracking location
    # 绘制检测和跟踪的位置轨迹
    # plot_detection_and_tracking_2(first_frame, detection_list, prediction_list, correction_list)

if __name__ == "__main__":
    file_path = '/workspace/time_2024_06_19_14_48_f60.mp4'## inverse
    # file_path = '/workspace/time_2024_09_05_02_28_f60.mp4'
    detector_yolov = Detecotr_YoloV7()
    run_single_tracker(detector_yolov, file_path)