import os
import cv2
import copy
import time
import numpy as np
from utils import BaseEngine
import pycuda.driver as cuda

def get_file_list(path, ftype):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in ftype:
                image_names.append(apath)
    return image_names

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

RGB_DAY_LIST = [(148, 108, 138), (207, 134, 240), (138, 88, 55), (26, 99, 86), (38, 125, 80), (72, 80, 57), # 6 
                (118, 23, 200), (117, 189, 183), (0, 128, 128), (60, 185, 90), (20, 20, 150), (13, 131, 204), 
                (30, 200, 200), (43, 38, 105), (104, 235, 178), (135, 68, 28), (140, 202, 15), (67, 115, 220),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30)]

def draw_img_filter(orig_img, boxes):
    height, weight, _ = orig_img.shape
    tl = 3 or round(0.002 * (height + weight) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    cur_img = copy.copy(orig_img)

    if len(boxes) > 0:
        for box_info in boxes:
            box = box_info[3]
            cls_id = box_info[0]
            score = box_info[2]

            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            x0 = 0 if x0 < 0 else x0
            y0 = 0 if y0 < 0 else y0

            _COLORS = RGB_DAY_LIST

            c1, c2 = (x0,y0), (x1,y1)
            cv2.rectangle(cur_img, c1, c2, _COLORS[6], thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            text = '{}'.format('traffic light')
            t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(cur_img, c1, c2, _COLORS[6], -1, cv2.LINE_AA)  # filled
            cv2.putText(cur_img, text, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

        img = cur_img
        return img, None


class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 18  # your model classes
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'green3_h', 'bus',
                            'red3_h', 'truck', 'yellow3_h', 'green4_h', 'red4_h', 'yellow4_h',
                            'redgreen4_h', 'redyellow4_h', 'greenarrow4_h', 'red_v', 'yellow_v', 'green_v']

if __name__ == '__main__':
    pred = Predictor(engine_path='/workspace/weights/integrate_each_class_fp16_0314.trt')
    class_name = pred.class_names

    orig_img = cv2.imread('/workspace/tl_026.png')
    box_result_ori = pred.steam_inference(orig_img, conf=0.1, end2end=True, day_night=1)

    draw_orig_img, _ = draw_img_filter(orig_img, box_result_ori)

    cv2.imshow('draw_orig_img', draw_orig_img)

    cv2.waitKey(0)