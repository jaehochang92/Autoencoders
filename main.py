from pprint import pprint

import pandas as pd
from methods import *
from models import *
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping


# import nvgpu


def main(raw_videos_path, size_factor, augmenters_dict, vid_no, frame_interval, ts_size, batch_size, epochs,
         model_builder, layeroi, interactive=True):
    args, h, w = config_args(raw_videos_path, size_factor)
    h, w = int(h), int(w)
    pprint((args, f'w: {w}, h: {h}'))
    ask = input('Videos found correctly? */n: ') if interactive else 'y'
    if ask != 'n':
        for augmenter_name, augmenter in augmenters_dict.items():
            avid_path, train, test = prepare_dataset(args['videos'][vid_no - 1], vid_no, w, h, frame_interval,
                                                     ts_size, augmenter, augmenter_name)
            train, test = train / 255., test / 255.
            # Train your model
            mname = os.path.join(
                'models',
                f'vid{vid_no}-{augmenter_name}-{size_factor}szf-{epochs}epc-{batch_size}btc'
            )
            print("\n=== You're trying ...")
            print(f'mname:      {mname}')
            print("===\n")
            # Modeling
            if not os.path.exists(mname):
                my_model = model_builder(train.shape[2:])
                print('Your model:'), print(my_model.summary())
                ask = input("You're fitting a new model. Proceed? */n: ") if interactive else 'y'
                if ask != 'n':
                    history = my_model.fit(train[:, 1], train[:, 0],  # noisy train, clean train
                                           batch_size=batch_size, epochs=epochs, verbose=True,
                                           validation_data=(test[:, 1], test[:, 0]),
                                           callbacks=[LearningRateScheduler(lr_schedule),
                                                      EarlyStopping(min_delta=0.0005, patience=20)],
                                           ).history
                    my_model.save(mname)
                    pd.to_pickle(history, f'{mname}/history')
                else:
                    return
            else:  # get pretrained one
                my_model = tf.keras.models.load_model(mname)
                print('Your model:'), print(my_model.summary())
                history = pd.read_pickle(os.path.join(mname, 'history'))
            del train
            # Denoising video
            denoised_video = my_model.predict(test[:, 1], batch_size=batch_size)
            for i, f in enumerate(denoised_video):
                if input('See next denoised frame?: */n') != 'n':
                    fig, ax = plt.subplots(1, 2)
                    ax[0].imshow(test[i, 1]), ax[1].imshow(denoised_video[i])
                    plt.show()
                else:
                    break
            print_history(history)
            inspect_model(layeroi, test[-1, 1], my_model)  # third layer


if __name__ == '__main__':
    config_gpus(memory_limit=6)
    # print('NVIDIA GPU info.:')
    # pprint(nvgpu.gpu_info())
    # print()
    augmenters_dict = {
        # 'Snowy': ia.FastSnowyLandscape(lightness_threshold=150, lightness_multiplier=2.5),
        # 'Clouds': ia.Clouds(),
        # 'Fog': ia.Fog(),
        # 'Snowflakes': ia.Snowflakes(flake_size=(0.4, 0.6), speed=(0.03, 0.05)),
        # 'SnowyFlakes': ia.Sequential([
        #     ia.FastSnowyLandscape(lightness_threshold=150, lightness_multiplier=2.5),
        #     ia.Snowflakes(flake_size=(0.6, 0.8), speed=(0.03, 0.05))
        # ])
        # 'Rain': ia.Rain(drop_size=(0.10, 0.20), speed=(0.1, 0.3)),
        # 'AGN': ia.AdditiveGaussianNoise(loc=0, scale=(25, 27), per_channel=.5),
        'APN': ia.AdditivePoissonNoise(lam=20, per_channel=.5),
    }
    main(
        # data loading variables
        raw_videos_path='D:/20.share/jaehochang/SP2Robotics/videos',
        vid_no=3,
        frame_interval=1,
        size_factor=2 ** 5,
        augmenters_dict=augmenters_dict,
        # train setup variables
        ts_size=.4,
        batch_size=24,  # I have 48 cores
        epochs=200,
        model_builder=build_cnn_ae,
        layeroi=['encoded', 'decoded'],
        interactive=False
    )
