from util import *
# from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, InputLayer, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler



# from tensorflow.python.client import device_lib


def lr_schedule(epoch):
    lrate = 0.01
    if epoch > 100:
        lrate = 0.005
    elif epoch > 200:
        lrate = 0.001
    return lrate


def build_cnn_ae(input_shape, cnn_filters):
    model = Sequential()
    model.add(InputLayer(input_shape))
    for depth in cnn_filters:
        model.add(Conv2D(depth, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
    for depth in cnn_filters[::-1]:
        model.add(Conv2D(depth, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(UpSampling2D())
        model.add(BatchNormalization())
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # model.add(Dense(3))
    optmz = tf.keras.optimizers.SGD(momentum=.05)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optmz, loss)
    return model


def build_ae(input_shape, code_size):
    # The encoder
    ae = Sequential()
    ae.add(InputLayer(input_shape))
    ae.add(Flatten(data_format='channels_last'))
    ae.add(Dense(code_size))
    # The decoder
    ae.add(
        Dense(np.prod(input_shape)))  # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    ae.add(Reshape(input_shape))
    # The reconstruction
    optmz = tf.keras.optimizers.SGD(momentum=.05)
    loss = tf.keras.losses.MeanSquaredError()
    ae.compile(optmz, loss)
    return ae


def do_denoising(model, model_name, test, batch_size):
    denoised_video = model.predict(test, batch_size=batch_size)
    denoised_vp = VideoProcessor(video_path=0)
    denoised_vp.vid_vol = (denoised_video * 255).astype(np.uint8)
    denoised_vp.frame_w = denoised_video.shape[2]
    denoised_vp.frame_h = denoised_video.shape[1]
    denoised_vp.augment_vid(os.path.join('denoised_videos', model_name + '.avi'),
                            ia.Identity())


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
