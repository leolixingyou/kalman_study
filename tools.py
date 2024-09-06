import cv2 
import copy 
import numpy as np
import matplotlib.pyplot as plt


# 绘制检测位置和跟踪位置的轨迹
def plot_detection_and_tracking(first_frame, detection_list, tracking_list):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

    detection_x = [loc[0] for loc in detection_list if loc is not None]
    detection_y = [loc[1] for loc in detection_list if loc is not None]
    tracking_x = [loc[0] for loc in tracking_list if loc is not None]
    tracking_y = [loc[1] for loc in tracking_list if loc is not None]

    ax.plot(detection_x, detection_y, 'rx', label='Detections', markersize = 10)
    ax.plot(tracking_x, tracking_y, 'yo', label='Tracking', markersize = 5)

    # 标注检测点的索引
    for i, (x, y) in enumerate(zip(detection_x, detection_y)):
        ax.text(x, y-2, str(i), color='red', fontsize=8)

    # 标注跟踪点的索引
    for i, (x, y) in enumerate(zip(tracking_x, tracking_y)):
        ax.text(x, y-4, str(i), color='yellow', fontsize=8)

    ax.legend()
    plt.show()


# 绘制检测位置和跟踪位置的轨迹
def plot_detection_and_tracking_2(first_frame, detection_list, prediction_list, correction_list):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

    detection_x = [loc[0] for loc in detection_list if loc is not None]
    detection_y = [loc[1] for loc in detection_list if loc is not None]
    prediction_list_x = [loc[0] for loc in prediction_list if loc is not None]
    prediction_list_y = [loc[1] for loc in prediction_list if loc is not None]
    correction_list_x = [loc[0] for loc in correction_list if loc is not None]
    correction_list_y = [loc[1] for loc in correction_list if loc is not None]


    ax.plot(detection_x, detection_y, 'rx', label='Detections', markersize = 10)
    ax.plot(prediction_list_x, prediction_list_y, 'ro', label='prediction_list_x', markersize = 5, alpha = 0.5)
    ax.plot(correction_list_x, correction_list_y, 'bo', label='correction_list_x', markersize = 5, alpha = 0.1)

    # 标注检测点的索引
    for i, (x, y) in enumerate(zip(detection_x, detection_y)):
        ax.text(x, y-2, str(i), color='red', fontsize=8)

    # 标注跟踪点的索引
    for i, (x, y) in enumerate(zip(prediction_list_x, prediction_list_y)):
        ax.text(x, y-4, str(i), color='yellow', fontsize=8)

    # 标注跟踪点的索引
    for i, (x, y) in enumerate(zip(correction_list_x, correction_list_y)):
        ax.text(x, y-4, str(i), color='blue', fontsize=8)

    ax.legend()
    plt.show()
        
def opencv_box_detection(frame, first_frame):
    detected_location, is_object_detected, contours = detect_object(frame, first_frame)
    
    if is_object_detected:
        box_2d = np.reshape([np.min(contours, axis=0), np.max(contours, axis=0)],(2,2))
        box_xmid, box_ymid = np.mean(box_2d,axis=0)
        box_width, box_height = box_2d[1][0]-box_2d[0][0], box_2d[1][1] - box_2d[0][1]
        box = np.array([box_xmid, box_width, box_ymid, box_height]).astype(np.int32)
    else:
        box = np.array([])

    return box, is_object_detected

def boundary_limits(pts, img_h, img_w):
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

        pt_top = boundary_limits(pt_top, img_h, img_w)
        pt_bottom = boundary_limits(pt_bottom, img_h, img_w)

        cv2.rectangle(frame_predict, pt_top, pt_bottom, (0, 0, 255), 2)
        cv2.putText(frame_predict, str(pt), pt - np.array([0,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.rectangle(frame_combined, pt_top, pt_bottom, (255, 0, 255), 2)
        cv2.putText(frame_combined, str(pt), pt - np.array([20,40]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(frame_combined, 'Prediction', pt - np.array([20,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if detected_location != []:
        pt = np.reshape([detected_location[0],detected_location[2]],-1).astype(np.int32)
        pt_top = np.array([detected_location[0], detected_location[2]]) - (1/2) * np.array([detected_location[1], detected_location[3]])
        pt_bottom = np.array([detected_location[0], detected_location[2]]) + (1/2) * np.array([detected_location[1], detected_location[3]])
        
        pt_top = boundary_limits(pt_top, img_h, img_w)
        pt_bottom = boundary_limits(pt_bottom, img_h, img_w)

        cv2.rectangle(frame_detect, pt_top, pt_bottom, (0, 0, 255), 2)
        cv2.putText(frame_detect, str(pt), pt - np.array([0,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.rectangle(frame_combined, pt_top, pt_bottom, (255, 100, 0), 2)
        cv2.putText(frame_combined, str(pt), pt - np.array([-20,40]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        cv2.putText(frame_combined, 'Detection', pt - np.array([-20,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    if correction6x1 is not None:
        pt = np.reshape([correction6x1[0],correction6x1[2]],-1).astype(np.int32)
        pt_top = np.array([correction6x1[0], correction6x1[2]]) - (1/2) * np.array([correction6x1[1], correction6x1[3]])
        pt_bottom = np.array([correction6x1[0], correction6x1[2]]) + (1/2) * np.array([correction6x1[1], correction6x1[3]])

        pt_top = boundary_limits(pt_top, img_h, img_w)
        pt_bottom = boundary_limits(pt_bottom, img_h, img_w)

        cv2.rectangle(frame_correct, pt_top, pt_bottom, (0, 0, 255), 2)
        cv2.putText(frame_correct, str(pt), pt - np.array([0,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.rectangle(frame_combined, pt_top, pt_bottom, (0, 255, 255), 2)
        cv2.putText(frame_combined, str(pt), pt + np.array([20,20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame_combined, 'Correction', pt + np.array([20,40]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame_detect, frame_predict, frame_correct, frame_combined 


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

def xyxy_to_xywh(box):
    x_top, y_top, x_bottom, y_bottom = box
    box_xmid = (x_top+x_bottom)/2
    box_ymid = (y_top+y_bottom)/2
    box_width = x_bottom - x_top
    box_height = y_bottom - y_top

    box = np.array([box_xmid, box_width, box_ymid, box_height]).astype(np.int32)
    return box

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)

    return(o)

def iou_batch_scrached(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    
    return(o)