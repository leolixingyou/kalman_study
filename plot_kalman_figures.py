import cv2
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
    ax.plot(prediction_list_x, prediction_list_y, 'yo', label='prediction_list_x', markersize = 5)
    ax.plot(correction_list_x, correction_list_y, 'bo', label='correction_list_x', markersize = 5)

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