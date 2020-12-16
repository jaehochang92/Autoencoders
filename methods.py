import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imgaug import augmenters as ia
from sklearn import model_selection
# from pytictoc import TicToc
from tensorflow.image import resize


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
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU(s),", len(logical_gpus), "Logical GPU(s)")
        except RuntimeError as e:
            # must enable memory growth when starting program
            print(e)


def prepare_dataset(original_video_path: str, vid_no: int, w: int, h: int, frame_interval: int, ts_size: float,
                    augmenter: ia.Augmenter, augmenter_name: str) -> tuple:
    vid_path = os.path.join('augmented_videos', f'vid{vid_no}.avi')
    if not os.path.exists(vid_path):
        my_vp = VideoProcessor(original_video_path)
        clean_vol = my_vp.read_video(h, w, frame_interval, 10 ** 5)
        write_video(clean_vol, vid_path)
    else:
        my_vp = VideoProcessor(vid_path)
        clean_vol = my_vp.read_video(my_vp.org_h, my_vp.org_w, frame_interval, 10 ** 5)
    avid_path = os.path.join('augmented_videos', f'vid{vid_no}({augmenter_name}).avi')
    if not os.path.exists(avid_path):
        noisy_vol = augmenter.augment_images(clean_vol)
        write_video(noisy_vol, avid_path)
    else:
        my_vp = VideoProcessor(avid_path)
        noisy_vol = my_vp.read_video(h, w, frame_interval, 10 ** 5)
    zipped_vol = np.array([*zip(clean_vol, noisy_vol)])
    del clean_vol, noisy_vol

    # Split videos
    train, test = model_selection.train_test_split(zipped_vol, test_size=ts_size)
    print(f'Before  splitting:')
    print('  ', zipped_vol.shape)
    print('Train volume shape:')
    print('  ', train.shape)
    print('Test volume shape: ')
    print('  ', test.shape)
    print()

    return avid_path, train, test


def print_history(history):
    # Plot train-validation trace
    print(history.keys())
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


class VideoProcessor:
    def __init__(self, r_path=''):
        if r_path:
            self.cap = cv2.VideoCapture(r_path)
            print(f'Capturing {r_path} ...')
            self.org_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.org_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.org_ch = int(self.cap.get(cv2.CAP_PROP_CHANNEL))

    def read_video(self, h: int, w: int, frame_interval: int, max_frame_count: int) -> np.ndarray:
        is_new_shape = self.org_w != w and self.org_h != h
        if is_new_shape:
            print(f'New shape detected: {(h, w)}')
            resizer = ia.Resize({'height': h, 'width': w})
        vol, ret, fc = [], 1, 0
        while ret and fc < max_frame_count:
            if fc > 0 and fc % frame_interval == 0:
                if is_new_shape:
                    frame = resizer.augment_image(frame)
                vol.append(frame)
            ret, frame = self.cap.read()
            fc += 1
        self.cap.release()
        return np.asarray(vol)


def write_video(volume: np.ndarray, w_path: str) -> None:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vshape = (volume.shape[2:0:-1])
    print(f'Writing {vshape} to {w_path}...')
    out = cv2.VideoWriter(w_path, fourcc, 5, vshape)
    for frame in volume:
        out.write(frame)
    out.release()