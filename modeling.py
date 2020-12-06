from util import *
from tensorflow import keras
# from tensorflow.keras.layers import Input, add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler

# from tensorflow.python.client import device_lib


def lr_schedule(epoch):
    lrate = 0.01
    if epoch > 75:
        lrate = 0.005
    elif epoch > 100:
        lrate = 0.001
    return lrate


def build_model(input_shape, cnn_filters):
    model = keras.Sequential()
    model.add(keras.layers.Input(input_shape))
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
    optmz = keras.optimizers.SGD(momentum=.05)
    loss = keras.losses.MeanSquaredError()
    model.compile(optmz, loss)
    return model


def comparison(X_test, denoised_imgs):
    yn, i = 'n', 0
    while yn == 'n':
        i += 1
        plt.figure(figsize=(30, 15))
        # display original testset
        plt.subplot(1, 2, 1), plt.imshow(X_test[:, 1][i])
        # display reconstructed figure
        plt.subplot(1, 2, 2), plt.imshow(denoised_imgs[i])
        plt.show()
        yn = input('Stop? y/n: ')


def inspect_model(li, X_test, ae):
    h = ae.layers[li].output
    activation_model = Model(inputs=ae.input, outputs=h)
    activations = activation_model.predict(np.expand_dims(X_test, axis=0))
    print('Inspecting...:')
    print(activations[0].shape)
    plt.figure(figsize=(45, 15))
    plt.subplot(3, 1, 1), plt.imshow(activations[0][:, :, 0])  # first channel
    plt.subplot(3, 1, 2), plt.imshow(activations[0][:, :, 1])  # second channel
    plt.subplot(3, 1, 3), plt.imshow(activations[0][:, :, 2])  # third channel
    plt.show()