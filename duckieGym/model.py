from keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from os import listdir

import numpy as np
from tensorflow.python.keras.layers import Dropout

from LoggerCallback import LoggerCallback


def scheduler(epoch):
    if epoch < 10:
        return 0.01
    if epoch < 15:
        return 0.005
    if epoch < 20:
        return 0.002
    if epoch < 25:
        return 0.001
    if epoch < 30:
        return 0.0005
    if epoch < 35:
        return 0.0002
    if epoch < 40:
        return 0.0001
    if epoch < 50:
        return 0.00005
    else:
        return 0.00002


def scale(X, Y):
    X = X / 255.0

    velocity_steer_scaler = StandardScaler()
    Y = velocity_steer_scaler.fit_transform(Y)

    return (X, Y), velocity_steer_scaler


def read_image(filename):
    image = tf.keras.preprocessing.image.load_img(filename)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)

    return input_arr


def split_data(X, Y, ratio=(0.7, 0.15, 0.15)):
    train_split = ratio[0]
    valid_split = ratio[1]
    test_split = ratio[2]
    nb_samples = X.shape[0]

    X_train = X[0:int(nb_samples * train_split)]
    Y_train = Y[0:int(nb_samples * train_split)]
    X_valid = X[int(nb_samples * train_split):int(nb_samples * (1 - test_split))]
    Y_valid = Y[int(nb_samples * train_split):int(nb_samples * (1 - test_split))]
    X_test = X[int(nb_samples * (1 - test_split)):]
    Y_test = Y[int(nb_samples * (1 - test_split)):]

    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)


def read_data(images_dir_name="preprocessedImages", label_file="my_app.txt"):
    img_list = np.zeros((len(listdir(images_dir_name)), 48, 85, 3))
    for i, f in enumerate(listdir(images_dir_name)):
        img_list[i] = (read_image(images_dir_name + "/" + f))

    X = np.asarray(img_list, dtype='float32')

    with open(label_file, "r") as my_app:
        lines = my_app.readlines()
        Y = np.zeros((len(lines), 2))
        for i, line in enumerate(lines):
            y = list(map(float, line.split(" ")[1:]))
            y = np.asarray(y, dtype="float32")
            Y[i] = y

    return X, Y


def create_model(input_shape):
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(5, 5), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(4, 4)))  # TODO kernel size
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.15))

    model.add(Dense(12000, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="linear"))

    return model

early_stopping = EarlyStopping(patience=10, verbose=1, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, min_lr=10e-5)
checkpoint = ModelCheckpoint(filepath='duckie.hdf5', verbose=1, save_best_only=True)
change_lr = LearningRateScheduler(scheduler)

def train_model(model, X_train, Y_train, X_valid, Y_valid):

    model.compile(loss='mse',
                  optimizer=Adam(),
                  metrics=["mse"])

    print(model.summary())

    model.fit(X_train, Y_train, batch_size=32, epochs=10000, validation_data=(X_valid, Y_valid),
              callbacks=[early_stopping, reduce_lr, checkpoint, change_lr, LoggerCallback("stat.csv")], verbose=1,
              shuffle=True)



"""
X, Y = read_data()

#TODO swap scale and split_data order
(X_scaled, Y_scaled), velocity_steering_scaler = scale(X, Y)

(X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = split_data(X_scaled, Y_scaled)

model = create_model(X_train[0].shape)

train_model(model, X_train, Y_train, X_valid, Y_valid)

print("eval score: ", model.evaluate(X_test, Y_test, batch_size=32))

y_test_pred = model.predict(X_test)

for idx, pred in enumerate(y_test_pred): #TODO maybe pyplot history after training
    print(pred, Y_test[idx])
"""
