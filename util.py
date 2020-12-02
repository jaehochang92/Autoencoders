import tensorflow as tf
import cv2, os, time
import numpy as np
import matplotlib.pyplot as plt

from imgaug import augmenters as ia
from sklearn import model_selection
# from tensorflow.keras import regularizers
# from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model

def draw_bounding_box(img, classes, COLORS, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

def print_history(history):
    print(history.keys())
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def inspect_model(li, X_test, ae):
    h = ae.layers[li].output
    activation_model = Model(inputs=ae.input, outputs=h)
    activations = activation_model.predict(np.expand_dims(X_test, axis=0))
    print('Inspecting...:')
    print(activations[0].shape)
    plt.figure(figsize=(45, 15))
    plt.subplot(3, 1, 1), plt.imshow(activations[0][:, :, 0])
    plt.subplot(3, 1, 2), plt.imshow(activations[0][:, :, 1])
    plt.subplot(3, 1, 3), plt.imshow(activations[0][:, :, 2])
    plt.show()

def check_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            print('Available GPUs:')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(gpus)
        except RuntimeError as e:
            print(e)

def config_args(augmented_vids_path, factor):
    args = {'VIDEOS': []}
    for root, dirs, files in os.walk(augmented_vids_path):
        for file in files:
            args['VIDEOS'].append(os.path.join(root, file))
    h, w = 9 * 4 * factor, 16 * 4 * factor
    resizer = ia.size.Resize({"height": h, "width": w})
    return resizer, args, h, w