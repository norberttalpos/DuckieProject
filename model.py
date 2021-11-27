from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization,MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback,TensorBoard, EarlyStopping,ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization,MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback,TensorBoard, EarlyStopping,ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau,TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from os import listdir
from os.path import isfile, join
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
    scaler=MinMaxScaler().fit(input_arr[:,:,i])
    input_arr[:,:,i] = scaler.transform(input_arr[:,:,i])


  return input_arr


img_list=np.zeros( (len(listdir("preprocessedImages")),48,85,3))
for i,f in enumerate(listdir("preprocessedImages")):
  img_list[i]=(read_image("preprocessedImages/"+f))

X_train=np.asarray(img_list,dtype='float32')


with open("my_app.txt","r") as my_app:
  lines=my_app.readlines()
  Y_train=np.zeros((len(lines),2))
  for i,line in enumerate(lines):
    y=list(map(float,line.split(" ")[1:]))
    y=np.asarray(y,dtype="float32")
    Y_train[i]=y

tb = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=1)
early_stopping = EarlyStopping(patience=20, verbose=1, monitor='loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=20, min_lr=10e-5)
checkpoint = ModelCheckpoint(filepath='duckie.hdf5', verbose=1)
change_lr = LearningRateScheduler(scheduler)
tensorboard_callback = TensorBoard(log_dir="./logs", write_graph=True, histogram_freq=1)

model = Sequential()

input_shape = X_train[0].shape
model.add(Conv2D(filters=16, kernel_size=(5, 5), input_shape=input_shape))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(5, 5)))
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

model.compile(loss='mse',
              optimizer=Adam(),
              metrics=['mse'])

print(model.summary())

model.fit(X_train, Y_train, batch_size=1, epochs=100,validation_split = 0.2,callbacks=[early_stopping,reduce_lr, checkpoint, tb, change_lr, tensorboard_callback], verbose = 1)