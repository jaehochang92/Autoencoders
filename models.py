import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, regularizers, losses
# from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


# from tensorflow.keras.utils import plot_model
# from tensorflow.python.client import device_lib


def lr_schedule(epoch: int) -> float:
    lrate = 0.0002 * 5
    if epoch > 50:
        lrate /= 2
    elif epoch > 100:
        lrate /= 4
    return lrate


def build_cnn_ae(input_shape: tuple) -> Model:
    cnn_filters = [16]
    input_tensor = layers.Input(shape=input_shape)
    dorate = .0
    # encoding
    for i, filter in enumerate(cnn_filters):
        x = layers.Conv2D(filter, (3, 3), kernel_regularizer=regularizers.l1(1e-4),
                          padding='same', activation='relu')(input_tensor if i == 0 else x)
        x = layers.Dropout(dorate)(x)
        x = layers.BatchNormalization()(x)
        # x = layers.UpSampling2D()(x)
        x = layers.MaxPooling2D((2, 2), padding='same',
                                name='encoded' if i == len(cnn_filters) - 1 else f'maxpool{i + 1}')(x)
    # decoding
    for i, filter in enumerate(cnn_filters[::-1]):
        x = layers.Conv2D(filter, (3, 3), kernel_regularizer=regularizers.l1(1e-4),
                          padding='same', activation='relu')(x)
        x = layers.Dropout(dorate)(x)
        x = layers.BatchNormalization()(x)
        # x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.UpSampling2D(name='decoded' if i == len(cnn_filters) - 1 else f'upsample{i + 1}')(x)
    decoded = layers.BatchNormalization()(
        layers.Conv2D(3, (9, 16), padding='same', activation='relu')(x)
    )
    optmz = Adam()
    loss = losses.MeanSquaredError()
    model = Model(input_tensor, decoded)
    model.compile(optimizer=optmz, loss=loss)
    return model


def build_ae(input_shape: tuple) -> Sequential:
    # The encoder
    ae = Sequential()
    ae.add(layers.InputLayer(input_shape))
    ae.add(layers.Flatten(data_format='channels_last'))
    ae.add(layers.Dense(512))
    # The decoder
    ae.add(layers.Dense(np.prod(input_shape)))
    ae.add(layers.Reshape(input_shape))
    # The reconstruction
    optmz = SGD(momentum=.01)
    loss = MeanSquaredError()
    ae.compile(optmz, loss)
    return ae


def inspect_model(layer_names: list, test: np.ndarray, model: Model) -> None:
    for layer_name in layer_names:
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(np.expand_dims(test, axis=0))
        w = int(np.ceil(np.sqrt(intermediate_output.shape[-1])))
        fig, axes = plt.subplots(w, w)
        i, j = np.meshgrid(range(w), range(w))
        xys = np.column_stack((i.flatten(), j.flatten()))
        for i, xy in enumerate(xys):
            try:
                axes[xy[0], xy[1]].imshow(intermediate_output[0, :, :, i])
            except:
                pass
        plt.show()