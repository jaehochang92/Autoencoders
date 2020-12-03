from modeling import *
from pprint import pprint
import nvgpu


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
        'GN': ia.AdditiveGaussianNoise(loc=0, scale=(0.0, 1 / 4 * 255), per_channel=0.2),
        # 'VertFlip': ia.Flipud(),
        # 'HorizFlip': ia.HorizontalFlip(),
        # 'Rotate': ia.Rotate(rotate=(-45, 45)),
        # 'Zoom': ia.Affine(scale={"x": (1, 1.5), "y": (1, 1.5)})
    }
}

vids_path = 'D:/20.share/jaehochang/SP2Robotics/videos'
pprint(nvgpu.gpu_info())

# Input set
ts_size = .4
batch_size = 2
epochs = 500

# Prepare dataset
vid = 1
frame_interval = 20
max_frame = 100000
whfactor = 13

# Configure model
CNN_filters = [48]
max_gpu_mem_GB = 6.9

fname, X_train, X_test = prepare_dataset(vids_path, augmenters_c, max_frame, vid, frame_interval, whfactor, ts_size)

if __name__ == '__main__':
    config_gpus(max_gpu_mem_GB)
    # Train your model
    mname = f'vid{vid}-{X_train.shape[2:4]}-{epochs}e-{batch_size}b'
    print()
    print("===== You're trying ...")
    print(f'fname:      {fname}')
    print(f'mname:      {mname}')
    print("=====")
    print()
    while input('Proceed? y/n: ') == 'y':
        if not os.path.exists(mname):
            my_model = build_model(X_train[0, 0].shape, CNN_filters)
            print('Your model:'), print(my_model.summary())
            if input('Fit? y/n: ') == 'y':
                history = my_model.fit(X_train[:, 1], X_train[:, 0],  # noisy train, clean train
                                       batch_size=batch_size, epochs=epochs, verbose=True,
                                       validation_data=(X_test[:, 1], X_test[:, 0])).history
                my_model.save(mname)
                pd.to_pickle(history, f'{mname}/history')
            else:
                break
        else:  # get pretrained one
            my_model = tf.keras.models.load_model(mname)
            print('Your model:'), print(my_model.summary())
            history = pd.read_pickle(os.path.join(mname, 'history'))

        # Denoise image
        denoised_imgs = my_model.predict(X_test[:, 1], batch_size=batch_size)
        denoised_imgs.save(os.path.join(mname, 'decoded_imgs.npy'))
        pd.to_pickle(denoised_imgs, )
        comparison(X_test, denoised_imgs)

        print_history(history)
        inspect_model(2, X_test[:, 1][-1], my_model)  # noisy test
