import os
import cv2
import time

def get_bag_list(path,file_type):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        print(maindir)
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in file_type:
                image_names.append(apath)
    return image_names

def create_video_from_images(images, video_name, fps=30.0, fourcc='XVID'):
    # Read the first image to get the size (width, height)
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    video = cv2.VideoWriter(video_name, fourcc_code, fps, (width, height))

    for image_file in sorted(images):
        video.write(cv2.imread(image_file))

    # Release the video writer object
    video.release()
    print(f"Video saved as {video_name}")

def make_video(ingput_list, save_video_dir):
    fps=30.0
    fourcc='XVID'
    his_fname = None
    flag = False
    for file_path in ingput_list:
        if not flag:
            video_name = f"{save_video_dir}time_{time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))}_{file_path.split(os.sep)[-2]}.mp4"
            if his_fname != video_name:
                his_fname = video_name
                try:
                    frame = cv2.imread(file_path)
                    height, width, layers = frame.shape
                    # Define the codec and create VideoWriter object
                    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
                    video = cv2.VideoWriter(video_name, fourcc_code, fps, (width, height))
                except:
                    flag = True
            else:
                video.write(cv2.imread(file_path))
        else:
            video.release()
if __name__ == '__main__':
    root_path = '/workspace/demo/'
    img_path = f'{root_path}runs/img_tl_dis_1/f60/'
    save_video_dir = f'{root_path}'
    input_path = f'{img_path}'
    ingput_list = sorted(get_bag_list(input_path,'.png'))
    make_video(ingput_list, save_video_dir)