import tensorflow as tf
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from imgaug import augmenters as ia
from sklearn import model_selection
# from pytictoc import TicToc

def config_args(vids_path, factor):
    args = {'VIDEOS': []}
    for root, dirs, files in os.walk(vids_path):
        for file in files:
            args['VIDEOS'].append(os.path.join(root, file))
    h, w = 9 * 4 * factor, 16 * 4 * factor
    return args, h, w

class VideoProcessor:
    """
    args, _, _ = config_args('D:/20.share/jaehochang/SP2Robotics/videos', 1)
    vp = VideoProcessor('D:/20.share/jaehochang/SP2Robotics/videos/vid3.mkv',
                        ia.AdditiveGaussianNoise(loc=0, scale=(.0, 1 / 4 * 255), per_channel=.2))
    vid_arr = vp.read_frames(5)
    """
    def __init__(self, video_path, augmenter):
        self.cap = cv2.VideoCapture(video_path)
        self.augmenter = augmenter
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_ch = int(self.cap.get(cv2.CAP_PROP_CHANNEL))

    def read_frames(self, frame_interval, max_frame_count=10000):
        ret, frame = self.cap.read()
        fc, volume = 0, []
        while ret and fc < max_frame_count:
            fc += 1
            if fc % frame_interval == 0:
                volume.append(frame)
            ret, frame = self.cap.read()
        print(f"{len(volume)} frames' been read!")
        self.vid_vol = np.stack(volume, axis=0)

    def split_trts(self, frames, aimgs, ts_size):
        fa = np.asarray([*zip(frames, aimgs)])
        print()
        print('Dataset shape:')
        print(fa.shape)
        print()
        X_train, X_test = model_selection.train_test_split(fa, test_size=ts_size)
        X_train, X_test = np.asarray(X_train), np.asarray(X_test)
        X_train = X_train.astype("float32") / 255.
        X_test = X_test.astype("float32") / 255.
        print('Dataset shape after split (train, test):')
        print(X_train.shape, X_test.shape)
        print()
        return X_train, X_test

    def wr_imgs(self, c, augmenter_name, augmenter, resizer, w, h, max_frame, frame_interval):

        frames, aimgs = [], []
        # gray_scaler = ia.Grayscale()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        nname = f"{c}_{augmenter_name}_{os.path.split(VIDEO)[-1].split('.')[0]}.avi"
        noisy_out = cv2.VideoWriter(nname, fourcc, 5, (w, h))

        k = int(max_frame / frame_interval)

        cap.release()
        noisy_out.release()
        cv2.destroyAllWindows()
        return np.array([*zip(frames, aimgs)])

    def iaug_setup(vids_path, vid, augmenters_c, factor, max_frame, frame_interval):
        resizer, args, h, w = config_args(vids_path, factor)
        print(f'resized frames: {w} x {h}')
        for c, augmenters in augmenters_c.items():
            for augmenter_name, augmenter in augmenters.items():
                for VIDEO in args['VIDEOS'][vid - 1:vid]:
                    yield wr_imgs(VIDEO, c, augmenter_name, augmenter, resizer, w, h, max_frame, frame_interval)

    def prepare_dataset(vids_path, augmenters_c, max_frame, vid, frame_interval, whfactor, ts_size):
        fname = f'vid{vid}-{frame_interval}fi-{whfactor}fctr.npz'
        if not os.path.exists(fname):  # When you build initial dataset
            dataset = iaug_setup(vids_path, vid, augmenters_c, whfactor, max_frame, frame_interval)
            dataset = np.array([*dataset])[vid - 1]  # (videos, frames, clean / noisy, height, width, RGB)
            X_train, X_test = split_trts(dataset[:, 0], dataset[:, 1], ts_size)
            np.savez(fname, train=X_train, test=X_test)
        else:  # Load past dataset
            dataset = np.load(fname)
            X_train, X_test = dataset['train'], dataset['test']
        print(X_train.shape, X_test.shape)
        return fname, X_train, X_test

    def draw_bounding_box(self, img, classes, COLORS, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def print_history(history):
    print(history.keys())
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def config_gpus(memory_lim_GB):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate ?GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_lim_GB * 1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(gpus)
            print(logical_gpus)
            print(len(gpus), "Physical GPU(s),", len(logical_gpus),
                  f"Logical GPU(s) with memory limit of {memory_lim_GB}GB")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)