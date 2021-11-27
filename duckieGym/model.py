from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from os import listdir

import numpy as np


def scheduler(epoch):
    if epoch < 5:
        return 0.001
    if epoch < 10:
        return 0.0005
    if epoch < 20:
        return 0.0002
    if epoch < 40:
        return 0.0001
    else:
        return 0.00005


def read_image(filename):
    image = tf.keras.preprocessing.image.load_img(filename)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    for i in range(3):
        scaler = MinMaxScaler().fit(input_arr[:, :, i])
        input_arr[:, :, i] = scaler.transform(input_arr[:, :, i])

    return input_arr


def read_data(images_dir_name= "preprocessedImages", label_file="my_app.txt"):
    img_list = np.zeros((len(listdir(images_dir_name)), 48, 85, 3))
    for i, f in enumerate(listdir(images_dir_name)):
        img_list[i] = (read_image(images_dir_name + f))

    X = np.asarray(img_list, dtype='float32')

    with open(label_file, "r") as my_app:
        lines = my_app.readlines()
        Y = np.zeros((len(lines), 2))
        for i, line in enumerate(lines):
            y = list(map(float, line.split(" ")[1:]))
            y = np.asarray(y, dtype="float32")
            Y[i] = y


    train_split=0.7
    valid_split=0.15
    test_split=0.15
    nb_samples=X.shape[0]

    X_train = X[0:int(nb_samples*(train_split))]
    Y_train = Y[0:int(nb_samples*(train_split))]
    X_valid = X[int(nb_samples*(train_split)):int(nb_samples*(1-test_split))]
    Y_valid = Y[int(nb_samples*(train_split)):int(nb_samples*(1-test_split))]
    X_test = X[int(nb_samples*(1-test_split)):]
    Y_test = Y[int(nb_samples*(1-test_split)):]

    return (X_train,Y_train,X_valid,Y_valid,X_test,Y_test)


def create_model(input_shape):
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(5, 5), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(5, 5)))  # TODO kernel size
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(5, 5)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(12800, activation="relu"))
    model.add(Dense(2, activation="linear"))

    return model


def run_model(model,X_train,Y_train,X_valid,Y_valid):


    early_stopping = EarlyStopping(patience=20, verbose=1, monitor='loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=20, min_lr=10e-5)
    checkpoint = ModelCheckpoint(filepath='duckie.hdf5', verbose=1, save_best_only=True)
    change_lr = LearningRateScheduler(scheduler)


    model.compile(loss='mse',
                  optimizer=Adam(),
                  metrics=['mse'])

    print(model.summary())

    model.fit(X_train, Y_train, batch_size=32, epochs=10000, validation_split=0.15,
              callbacks=[early_stopping, reduce_lr, checkpoint, change_lr], verbose=1)


X_train,Y_train,X_valid,Y_valid,X_test,Y_test = read_data() 
model = create_model(X_train[0].shape)
run_model(model,X_train,Y_train,None,None)
