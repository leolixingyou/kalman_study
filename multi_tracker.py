### MOT 

import numpy as np
from opencv_tracker import Tracker_Kalman_for_MOT

def linear_assignment(cost_matrix):
    try:
        import lap #linear assignment problem solver
        _, x, y = lap.lapjv(cost_matrix, extend_cost = True)
        return np.array([[y[i],i] for i in x if i>=0])
    except ImportError: 
        from scipy.optimize import linear_sum_assignment
        x,y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x,y)))

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[...,0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)

def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
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
    # print(f'iou_matrix is {iou_matrix}')
    
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

class Multi_Object_Tracking:
    def __init__(self, max_age = 1, min_hits = 3 , iou_threshold = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracker_list = []
        self.frame_count = 0

    def get_tracker(self):
        return self.tracker_list
    
    def preprocess_box(self, detection):
        """
        box = x1,y1,x2,y2 or xmid,ymid,w,h or xmid, ymid, _, _ ;
        detection is the numpy array as [[box1], [box2], [box3],......]
        The previous work get it from [[class1, area1, score1, [box1]], [class2, area2, score2, [box2]],......]
        Recommand use np.array(boxresult)[:3]
        """
        dets_to_sort = np.empty((0,6)) ### based on the system model
        for i, box in enumerate(detection):
            x0, y0, x1, y1 = box
            dets_to_sort = np.vstack((dets_to_sort, 
                        np.array([x0, y0, x1, y1, 0, 0])))
        return dets_to_sort
            
    ### detection shape based on system model
    def update(self, dets=np.empty((0,6))):

        self.frame_count += 1

        ### Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)): ### Any num in pos is True
                to_del.append(t)
        
        ### Mask out nan values
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:

            trk = KalmanBoxTracker(np.hstack((dets[i,:], np.array([0]))))
            
            #trk = KalmanBoxTracker(np.hstack(dets[i,:])
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) #+1'd because MOT benchmark requires positive value
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update >self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0,6))

class Multi_Object_Tracking:
    def __init__(self, max_age = 1, min_hits = 3 , iou_threshold = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracker_list = []
        self.frame_count = 0

    def preprocess_box(self, detection):
        """
        box = x1,y1,x2,y2 or xmid,ymid,w,h or xmid, ymid, _, _ ;
        detection is the numpy array as [[box1], [box2], [box3],......]
        The previous work get it from [[class1, area1, score1, [box1]], [class2, area2, score2, [box2]],......]
        Recommand use np.array(boxresult)[:3]
        """
        dets_to_sort = np.empty((0,4)) ### based on the system model
        for i, box in enumerate(detection):
            x0, y0, x1, y1 = box
            dets_to_sort = np.vstack((dets_to_sort, 
                        np.array([x0, y0, x1, y1])))
        return dets_to_sort
    
    ### detection shape based on system model
    def update(self, detections = np.empty((0,4))):

        trackers = np.zeros_like(self.tracker_list)
        for i, tracker in enumerate(self.tracker_list):
            tracker[i].tracking_update(detections)
            tracker[:] = tracker[i].prediction6x1
            if tracker[i].state_tracker == 'Corrction':
                tracker[:] = tracker[i].correction6x1

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers, self.iou_threshold)
        
        for un_det in unmatched_dets:
            trakcker_mot = Tracker_Kalman_for_MOT(un_det)
        
        self.tracker_list



def main(multi_object_tracking):
    multi_object_tracking.preprocess_box()
    multi_object_tracking.update()

if __name__ == "__main__":
    multi_object_tracking = Multi_Object_Tracking()
    main(multi_object_tracking)

