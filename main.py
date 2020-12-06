from modeling import *
import nvgpu
tf.keras.backend.clear_session()

print('NVIDIA GPU info.:')
pprint(nvgpu.gpu_info())
print()

augmenters_dict = {
    # 'Snowy': ia.FastSnowyLandscape(lightness_threshold=150, lightness_multiplier=2.5),
    # 'Clouds': ia.Clouds(),
    # 'Fog': ia.Fog(),
    # 'Snowflakes': ia.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
    # 'Rain': ia.Rain(drop_size=(0.10, 0.20), speed=(0.1, 0.3)),
    'AGN': ia.AdditiveGaussianNoise(loc=0, scale=(0.0, 1 / 4 * 255), per_channel=0.2),
}

# Prepare dataset
raw_videos_path = 'D:/20.share/jaehochang/SP2Robotics/videos'
vid_no = 3
frame_interval = 3

# Configure model
ts_size = .4
batch_size = 2  # Start with batch size = 1
epochs = 200
size_factor = 10
CNN_filters = [
    [8],
    [64, 64, 64]
][0]
max_gpu_mem_GB = 5

if __name__ == '__main__':
    args, h, w = config_args(raw_videos_path, size_factor)
    pprint(args)
    if input('Videos found correctly? */n: ') != 'n':
        for augmenter_name, augmenter in augmenters_dict.items():
            noisy, train, test = prepare_dataset(args['videos'][vid_no - 1], vid_no, w, h, frame_interval,
                                                ts_size, augmenter, augmenter_name)
            # Train your model
            mname = f'vid{vid_no}-{augmenter_name}-{epochs}epc-{batch_size}btc'
            print("\n=== You're trying ...")
            print(f'mname:      {mname}')
            print("===\n")
        # Modeling
        config_gpus(max_gpu_mem_GB)
        tf.debugging.set_log_device_placement(True)
        if not os.path.exists(mname):
            my_model = build_model(train.shape[2:], CNN_filters)
            print('Your model:'), print(my_model.summary())
            if input("You're fitting a new model. Proceed? */n: ") != 'n':
                history = my_model.fit(train[:, 1], train[:, 0],  # noisy train, clean train
                                       batch_size=batch_size, epochs=epochs, verbose=True,
                                       validation_data=(test[:, 1], test[:, 0]),
                                       callbacks=[LearningRateScheduler(lr_schedule)]).history
                my_model.save(mname)
                pd.to_pickle(history, f'{mname}/history')
        else:  # get pretrained one
            my_model = tf.keras.models.load_model(mname)
            print('Your model:'), print(my_model.summary())
            history = pd.read_pickle(os.path.join(mname, 'history'))
        # Denoising video
        do_denoising(my_model, mname, noisy, batch_size)
        # comparison(test, denoised_video)
        print_history(history)
        inspect_model(2, noisy[-1], my_model)  # noisy test
    del noisy
