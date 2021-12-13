from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from os import listdir
import numpy as np


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
