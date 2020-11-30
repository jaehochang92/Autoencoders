import tensorflow as tf
import pandas as pd
import pickle

from methods import *
from sklearn import model_selection
from tensorflow import keras
# from tensorflow.keras import regularizers
# from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, add
from tensorflow.keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
# from tensorflow.python.client import device_lib

augmenters_c = {
    'Env': {
        # 'Snowy': ia.FastSnowyLandscape(lightness_threshold=150, lightness_multiplier=2.5),
        # 'Clouds': ia.Clouds(),
        # 'Fog': ia.Fog(),
        # 'Snowflakes': ia.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
        # 'Rain': ia.Rain(drop_size=(0.10, 0.20), speed=(0.1, 0.3)),
        # 'Darken': ia.Multiply(.3, per_channel=.5)
    },
    'Trs': {
        'GN': ia.AdditiveGaussianNoise(loc=0, scale=(0.0, 1 / 10 * 255), per_channel=0.2),
        # 'VertFlip': ia.Flipud(),
        # 'HorizFlip': ia.HorizontalFlip(),
        # 'Rotate': ia.Rotate(rotate=(-45, 45)),
        # 'Zoom': ia.Affine(scale={"x": (1, 1.5), "y": (1, 1.5)})
    }
}

def check_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            print('Available GPUs:')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(gpus)
        except RuntimeError as e:
            print(e)

def prepare_frames(augmented_imgs_path, augmenters_c, frames, factor = 10):
    args = {'IMAGES': []}
    for root, dirs, files in os.walk(augmented_imgs_path):
        for file in files:
            args['IMAGES'].append(os.path.join(root, file))
    h, w = 9 * 4 * factor, 16 * 4 * factor
    frames, aimgs = prprdata(w, h, frames, args['IMAGES'][:1], augmenters_c)
    return frames, aimgs

def split_trts(frames, aimgs, ts_size):
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

def show_imgs(idx, X_train, X_test):
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 2), plt.imshow(X_train[idx][0])
    plt.subplot(2, 2, 1), plt.imshow(X_train[idx][1])
    plt.subplot(2, 2, 4), plt.imshow(X_test[idx][0])
    plt.subplot(2, 2, 3), plt.imshow(X_test[idx][1])
    plt.show()

def build_model(input_shape, depths, gpu='/GPU:0'):
    with tf.device(gpu):
        x = keras.layers.Input(input_shape)
        model = keras.Sequential()
        model.add(x)
        # Encoder
        for depth in depths:
            model.add(Conv2D(depth, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2), padding='same'))
            model.add(BatchNormalization())
        # Decoder
        for depth in depths[::-1]:
            model.add(Conv2D(depth, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(UpSampling2D((2, 2)))
            model.add(BatchNormalization())
        model.add(Conv2D(3, (3, 3), activation='linear', padding='same'))
        optmz = keras.optimizers.SGD(.05)
        loss = keras.losses.MeanSquaredError()
        model.compile(optmz, loss)

        return model

def print_history(history):
    print(history.keys())
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def inspect_model(X_test, ae):
    h = ae.layers[1].output
    activation_model = Model(inputs=ae.input, outputs=h)
    activations = activation_model.predict(np.expand_dims(X_test, axis=0))
    print(activations[0].shape)
    plt.imshow(activations[0][:, :, 1:2])
    plt.show()