from tensorflow import keras
# from tensorflow.keras.layers import Input, add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from util import *

# from pytictoc import TicToc

# from tensorflow.python.client import device_lib

augmenters_c = {
    'Env': {
        # 'Snowy': ia.FastSnowyLandscape(lightness_threshold=150, lightness_multiplier=2.5),
        # 'Clouds': ia.Clouds(),
        # 'Fog': ia.Fog(),
        'Snowflakes': ia.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
        # 'Rain': ia.Rain(drop_size=(0.10, 0.20), speed=(0.1, 0.3)),
        # 'Darken': ia.Multiply(.3, per_channel=.5)
    },
    'Trs': {
        'GN': ia.AdditiveGaussianNoise(loc=0, scale=(0.0, 1 / 4 * 255), per_channel=0.2),
        # 'VertFlip': ia.Flipud(),
        # 'HorizFlip': ia.HorizontalFlip(),
        # 'Rotate': ia.Rotate(rotate=(-45, 45)),
        # 'Zoom': ia.Affine(scale={"x": (1, 1.5), "y": (1, 1.5)})
    }
}


def write_imgs(VIDEO, c, augmenter_name, augmenter, resizer, w, h, max_frame, frame_interval):
    index = 1
    frames, aimgs = [], []
    # gray_scaler = ia.Grayscale()
    cap = cv2.VideoCapture(VIDEO)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    nname = f"{c}_{augmenter_name}_{os.path.split(VIDEO)[-1].split('.')[0]}.avi"
    noisy_out = cv2.VideoWriter(nname, fourcc, 5, (w, h))
    ret, frame = cap.read()
    k = int(max_frame / frame_interval)
    while ret and index < frame_interval * k:
        if index % frame_interval == 0:
            frame = resizer.augment_image(frame)
            # frame = gray_scaler.augment_image(frame)
            frames.append(frame)
            aimg = augmenter(image=frame)
            aimgs.append(aimg)
            noisy_out.write(aimg)
        index += 1
        ret, frame = cap.read()
    cap.release()
    noisy_out.release()
    cv2.destroyAllWindows()
    return np.array([*zip(frames, aimgs)])


def iaug_setup(augmented_vids_path, vid, augmenters_c, factor, max_frame, frame_interval):
    resizer, args, h, w = config_args(augmented_vids_path, factor)
    print(f'resized frames: {w} x {h}')
    for c, augmenters in augmenters_c.items():
        for augmenter_name, augmenter in augmenters.items():
            for VIDEO in args['VIDEOS'][vid - 1:vid]:
                yield write_imgs(VIDEO, c, augmenter_name, augmenter, resizer, w, h, max_frame, frame_interval)


def build_model(input_shape, depths, gpu='/GPU:0'):
    with tf.device(gpu):
        model = keras.Sequential()
        model.add(keras.layers.Input(input_shape))
        # Encoder
        for depth in depths:
            model.add(Conv2D(depth, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(UpSampling2D((2, 2)))
            model.add(BatchNormalization())
        # Decoder
        for depth in depths[::-1]:
            model.add(Conv2D(depth, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2), padding='same'))
            model.add(BatchNormalization())
        model.add(Conv2D(3, (3, 3), activation='linear', padding='same'))
        optmz = keras.optimizers.SGD(.01, .1)
        loss = keras.losses.MeanSquaredError()
        model.compile(optmz, loss)
        return model
