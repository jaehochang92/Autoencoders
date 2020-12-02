from DenAE import *
import tensorflow as tf
import pandas as pd

augmented_vids_path = 'D:/20.share/jaehochang/SP2Robotics/videos'

# Input set that invokes memory exhaustion
# whfactor = 8
# batch_size = 32

# Input set
ts_size = .4
batch_size = 2
epochs = 200

# Prepare dataset
vid = 2
frame_interval = 5
max_frame = 10000
whfactor = 12

def

fnames = (f'vid{vid}-tr-{frame_interval}fi-{whfactor}fctr.pkl', f'vid{vid}-ts-{frame_interval}fi-{whfactor}fctr.pkl')
if not os.path.exists(fnames[0]):  # When you build initial dataset
    dataset = iaug_setup(augmented_vids_path, vid, augmenters_c, whfactor, max_frame, frame_interval)
    dataset = np.array([*dataset])[vid - 1]  # (videos, frames, clean / noisy, height, width, RGB)
    X_train, X_test = split_trts(dataset[:, 0], dataset[:, 1], ts_size)
    pd.to_pickle(X_train, fnames[0]), pd.to_pickle(X_test, fnames[1])
else:  # Load past dataset
    X_train, X_test = pd.read_pickle(fnames[0]), pd.read_pickle(fnames[1])
print(X_train.shape, X_test.shape)

if __name__ == '__main__':
    check_gpus()
    # Train your model
    mname = f'{X_train.shape[2:4]}-{epochs}e-{batch_size}b'

    print("=== You're trying... ===")
    print(f'fnames:     {fnames}')
    print(f'mname:      {mname}')
    print("========================")

    while input('Proceed? y/n: ') == 'y':
        if not os.path.exists(mname):
            # Build your model
            filters = [64]
            autoencoder = build_model(X_train[0, 0].shape, filters)
            print('Your model:')
            print(autoencoder.summary())
            if input('Fit? y/n: ') == 'y':
                history = autoencoder.fit(X_train[:, 1], X_train[:, 0],  # noisy train, clean train
                                          batch_size=batch_size, epochs=epochs, verbose=True,
                                          validation_data=(X_test[:, 1], X_test[:, 0])).history
                autoencoder.save(mname)
                pd.to_pickle(history, f'{mname}/history')
            else:
                break
        else:  # get pretrained one
            autoencoder = tf.keras.models.load_model(mname)
            print(autoencoder.summary())
            history = pd.read_pickle(f'{mname}/history')

        decoded_imgs = autoencoder.predict(X_test[:, 1], batch_size=batch_size)
        pd.to_pickle(decoded_imgs, 'decoded_imgs.pkl')

        yn, i = 'n', 0
        while yn == 'n':
            i += 1
            plt.figure(figsize=(30, 15))
            # display original test set
            plt.subplot(1, 2, 1), plt.imshow(X_test[:, 1][i])
            # display reconstructed figure
            plt.subplot(1, 2, 2), plt.imshow(decoded_imgs[i])
            plt.show()
            yn = input('Stop? y/n: ')

        print_history(history)
        inspect_model(2, X_test[:, 1][-1], autoencoder)  # noisy test
