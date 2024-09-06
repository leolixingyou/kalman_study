import numpy as np

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

    return o

def judge_class_score(box_info):
    box_class = box_info[..., 0]
    box_score = box_info[..., 2]
    box_area = max(box_info[..., 1])
    box_location = max(box_info[..., 3])

    unique_class, counts_class = np.unique(box_class, return_counts=True)
    unique_score, counts_score = np.unique(box_score, return_counts=True)

    if not np.unique(counts_class): 
        new_box_class = unique_class[np.where(counts_class == max(counts_class))]
    else:
        new_box_class = max(box_class)

    if not np.unique(unique_score):
        new_box_score = unique_score[np.where(counts_score == max(counts_score))]
    else:
        new_box_score = max(box_score)

    return np.resize([new_box_class, box_area, new_box_score, box_location],(1,4))


def merge_boxes(boxes, iou_threshold=0.5):
    boxes = np.array(boxes)
    
    box_info = np.array([np.array(x[-1]) for x in boxes])
    box_result_iou = iou_batch(box_info, box_info)
    overlabpped_box_ind = [i for i, x in enumerate(box_result_iou) for j, y in enumerate(x) if i!=j and y > iou_threshold]
    merged_box = judge_class_score(boxes[overlabpped_box_ind])## one list [class, area, score, box]
    ### Remove overlapped box and add merged box
    mask = np.array([not any(np.array_equal(row, x) for x in boxes[overlabpped_box_ind]) for row in boxes])
    ## Remove overlapped box by using mask
    removed_box = boxes[mask]
    ## Merge boxes into one
    merged_boxes = np.vstack([removed_box, merged_box])

    return merged_boxes

# 输入数组
input_array = np.array([
    [11, 14184, 0.8160549998283386, [1205, 60, 1302, 132]],
    [11, 14016, 0.47701379656791687, [1105, 58, 1297, 131]],
    [9, 14016, 0.47701379656791687, [1105, 58, 1297, 131]]
])

# 合并框
result = merge_boxes(input_array)

print(result)