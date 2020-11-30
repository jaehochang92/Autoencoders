from DenAE import *

augmented_imgs_path = 'D:/20.share/jaehochang/SP2Robotics/videos/'

# Input set that invokes exhaustive modeling
# frame_interval = 80
# whfactor = 8
# batch_size = 32

# Input set
ts_size = .4
batch_size = 4
epochs = 15
idx = 11

# Prepare dataset
frame_interval = 25
whfactor = 30
fnames = (f'tr-{frame_interval}fi-{whfactor}fctr.pkl', f'ts-{frame_interval}fi-{whfactor}fctr.pkl')
if not os.path.exists(fnames[0]):  # When you build initial dataset
    frames, aimgs = prepare_frames(augmented_imgs_path, augmenters_c, frame_interval, whfactor)
    X_train, X_test = split_trts(frames, aimgs, ts_size)
    pd.to_pickle(X_train, fnames[0]), pd.to_pickle(X_test, fnames[1])
else:  # Load past dataset
    X_train, X_test = pd.read_pickle(fnames[0]), pd.read_pickle(fnames[1])
    print(X_train.shape, X_test.shape)

# Build your model
filters = [16, 32]
check_gpus()
autoencoder = build_model(X_train[0, 0].shape, filters)
print('Your model:')
print(autoencoder.summary())

# Train your model
mname = f'{X_train.shape[2:4]}-{epochs}e-{batch_size}b'
if not os.path.exists(mname):
    history = autoencoder.fit(X_train[:, 1], X_train[:, 0],  # noisy train, train
                              batch_size=batch_size, epochs=epochs, verbose=True,
                              validation_data=(X_test[:, 1], X_test[:, 0])).history
    autoencoder.save(mname)
    pd.to_pickle(history, f'{mname}/history')
else:
    autoencoder = tf.keras.models.load_model(mname)
    history = pd.read_pickle(f'{mname}/history')
decoded_imgs = autoencoder.predict(X_test[:, 1], batch_size=batch_size)

# Write decoded video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
decoded = cv2.VideoWriter(f"{X_train.shape[2:4]}-{epochs}e-{batch_size}b.avi", fourcc, 5, X_train.shape[2:4])
for di in decoded_imgs:
    decoded.write(di)
decoded.release()
cv2.destroyAllWindows()

plt.figure(figsize=(20, 10))
for i, j in enumerate([idx]):
    # display original
    ax = plt.subplot(1, 2, 1)
    plt.imshow(X_test[:, 0][j])

    # display reconstruction
    ax = plt.subplot(1, 2, 2)
    plt.imshow(decoded_imgs[j])
plt.show()

print_history(history)
inspect_model(X_test[:, 1][idx], autoencoder)