import tensorflow as tf
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from imgaug import augmenters as ia
from sklearn import model_selection
from tensorflow.image import resize
from pprint import pprint


# from pytictoc import TicToc


def config_args(vids_path, factor):
    args = {'videos': []}
    for root, dirs, files in os.walk(vids_path):
        for file in files:
            args['videos'].append(os.path.join(root, file))
    h, w = 9 * factor, 16 * factor
    return args, h, w


def config_gpus(memory_limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit * 1024)]
            )
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPU(s),", len(logical_gpus), "Logical GPU(s)")
        except RuntimeError as e:
            # must enable memory growth when starting program
            print(e)


def split_trts(video_volume, ts_size):
    vol_tr, vol_ts = model_selection.train_test_split(video_volume, test_size=ts_size)
    vol_tr, vol_ts = np.asarray(vol_tr), np.asarray(vol_ts)
    vol_tr /= 255.
    vol_ts /= 255.
    return vol_tr, vol_ts


def prepare_dataset(video_path: str, vid_no: str, w: int, h: int, frame_interval: int, ts_size: float,
                    augmenter: ia.Augmenter, augmenter_name: str) -> np.array:
    clean_vp = VideoProcessor(video_path=video_path)
    clean_vp.rw_volume(frame_interval=frame_interval, max_frame_count=100000)  # Takes most time
    aug_video_path = os.path.join('augmented_videos', f'vid{vid_no}_{augmenter_name}.avi')
    if not os.path.exists(aug_video_path):
        print(f'{os.path.split(aug_video_path)[-1]}: new augmentation!')
        clean_vp.augment_vid(aug_video_path, augmenter)
    noisy_vp = VideoProcessor(video_path=aug_video_path)
    noisy_vp.rw_volume(frame_interval=1)
    if clean_vp.vid_vol.shape != noisy_vp.vid_vol.shape:
        print(f'{os.path.split(aug_video_path)[-1]}: new shape!')
        clean_vp.augment_vid(aug_video_path, augmenter)
        noisy_vp = VideoProcessor(video_path=aug_video_path)
        noisy_vp.rw_volume(frame_interval=1)
    clean_vp.vid_vol = resize(clean_vp.vid_vol, (h, w))
    noisy_vp.vid_vol = resize(noisy_vp.vid_vol, (h, w))
    zipped_vol = np.array([*zip(clean_vp.vid_vol, noisy_vp.vid_vol)])
    noisy = noisy_vp.vid_vol / 255.
    tr, ts = split_trts(zipped_vol, ts_size)
    print('Train volume shape:')
    print('  ', tr.shape)
    print('Test volume shape: ')
    print('  ', ts.shape)
    print()
    return noisy, tr, ts


def print_history(history):
    print(history.keys())
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


class VideoProcessor:
    def __init__(self, video_path: str):
        if type(video_path) is str:
            self.cap = cv2.VideoCapture(video_path)
            print(f'Capturing {os.path.split(video_path)[-1]} ...')
            self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_ch = int(self.cap.get(cv2.CAP_PROP_CHANNEL))
        self.vid_vol = None

    def rw_volume(self, frame_interval: int, max_frame_count=100000) -> None:
        ret, frame = self.cap.read()
        fc, volume = 0, []
        while ret and fc < max_frame_count:
            fc += 1
            if fc % frame_interval == 0:
                volume.append(frame)
            ret, frame = self.cap.read()
        self.cap.release()
        volume = np.stack(volume, axis=0)
        print(f"Volume shape: {volume.shape}")
        self.vid_vol = volume

    def augment_vid(self, file_name: str, augmenter: ia.Augmenter) -> None:
        if self.vid_vol is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            volume = self.vid_vol
            volume = augmenter.augment_images(volume)
            noisy_out = cv2.VideoWriter(file_name, fourcc, 5, (self.frame_w, self.frame_h))
            for frame in volume:
                noisy_out.write(frame)
            noisy_out.release()
            cv2.destroyAllWindows()
        else:
            print('Run self.rw_volume first!')

    @staticmethod
    def draw_bounding_box(img, classes, colors, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
