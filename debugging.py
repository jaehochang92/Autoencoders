# This is a standalone code for reproducing bugs or issues.
import tensorflow as tf  # tf == 2.3.1
import numpy as np
import nvgpu

from pprint import pprint
from sklearn import model_selection
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense
tf.keras.backend.clear_session()

print('NVIDIA GPU info.:')
pprint(nvgpu.gpu_info())
print()


def split_trts(video_volume, ts_size):
    vol_tr, vol_ts = model_selection.train_test_split(video_volume, test_size=ts_size)
    vol_tr, vol_ts = np.asarray(vol_tr), np.asarray(vol_ts)
    vol_tr = vol_tr.astype("float32") / 255.
    vol_ts = vol_ts.astype("float32") / 255.
    return vol_tr, vol_ts


def prepare_dataset(volume: np.array, ts_size: float) -> np.array:
    zipped_vol = np.array([*zip(volume[:, 0], volume[:, 1])])
    tr, ts = split_trts(zipped_vol, ts_size)
    print('Train volume shape:')
    print('  ', tr.shape)
    print('Test volume shape: ')
    print('  ', ts.shape)
    print()
    return tr, ts


def config_gpus(memory_limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit * 1024)]
            )
        except RuntimeError as e:
            print(e)


def build_model(input_shape, cnn_filters):
    model = tf.keras.Sequential()
    model.add(Input(input_shape))
    # Encoder
    for depth in cnn_filters:
        model.add(Conv2D(depth, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(BatchNormalization())
    # Decoder
    for depth in cnn_filters[::-1]:
        model.add(Conv2D(depth, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(BatchNormalization())
    for depth in cnn_filters:
        model.add(Conv2D(depth, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
    model.add(Dense(3))
    # model.add(Conv2D(3, (3, 3), activation='linear', padding='same'))
    optmz = tf.keras.optimizers.SGD(momentum=.05)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optmz, loss)
    return model


config_gpus(5)
tf.debugging.set_log_device_placement(True)
foo_volume = tf.random.uniform(
    (1000, 2, 128, 128, 3), minval=0, maxval=255, dtype=tf.dtypes.int32, seed=None, name=None
)
train, test = prepare_dataset(foo_volume, ts_size=0.4)
my_model = build_model(train.shape[2:], [64, 64])
print('Your model:'), print(my_model.summary())
if input("Proceed? */n: ") != 'n':
    history = my_model.fit(train[:, 1], train[:, 0],  # noisy train, clean train
                           batch_size=4, epochs=2000, verbose=True,
                           validation_data=(test[:, 1], test[:, 0])).history
