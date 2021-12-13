import argparse
import os

import numpy as np
from gym_duckietown.envs import DuckietownEnv
from keras.models import load_model
from tensorflow import keras

from dagger_learner import DaggerLearner
from dagger_teacher import DaggerTeacher
from IIL import InteractiveImitationLearning
from dagger_sandbox import MyInteractiveImitationLearning
from detector import preprocess_image
from data_reader import read_data, scale
from tensorflow.keras.callbacks import EarlyStopping


img_dim = [48,85,3]
action_dim = 2
steps = 1000
batch_size = 32
nb_epoch = 100

def get_teacher_action(ob):
    #steer = ob.angle*10/np.pi
    #steer -= ob.trackPos*0.10
    action = np.array([0.1, 0.0])
    return action

def img_reshape(input_img):
    #_img = np.transpose(input_img, (1, 2, 0))
    #_img = np.flipud(_img)
    _img = np.reshape(preprocess_image(input_img), (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img

images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]))
actions_all = np.zeros((0,action_dim))
rewards_all = np.zeros((0,))


img_list = []
action_list = []
reward_list = []

env = env = DuckietownEnv(
        map_name="zigzag_dists",
        max_steps=1000,
        draw_curve=False,
        draw_bbox=False,
        domain_rand=False,
        distortion=True,
        accept_start_angle_deg=4,
        full_transparency=True,
    )
ob = env.reset()



#(obs, reward, done, info) = self.env.step(action)

teacher=DaggerTeacher(env)

print('Collecting data...')
for i in range(steps):
    #if i == 0:
    act = np.array([0.0, 0.0])
    #else:
        act = teacher.predict(env,ob)#act = get_teacher_action(ob)
    print(act)
    if i%100 == 0:
        print(i)
    (ob, reward, done, info) = env.step(act)
    #ob, reward, done, _ = env.step(vel,angle)
    #print(ob)
    img_list.append(ob)
    action_list.append(act)
    reward_list.append(np.array([reward]))

env.close()

print('Packing data into arrays...')
for img, act, rew in zip(img_list, action_list, reward_list):
    images_all = np.concatenate([images_all, img_reshape(img)], axis=0)
    actions_all = np.concatenate([actions_all, np.reshape(act, [1,action_dim])], axis=0)
    rewards_all = np.concatenate([rewards_all, rew], axis=0)


from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Convolution2D(32, 3, 3, padding='same',
                        input_shape=img_dim))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 1, 1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(action_dim))
model.add(Activation('tanh'))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-4),
              metrics=['mean_squared_error'])
              
model = load_model("/tmp/dagger2")

model.fit(images_all, actions_all,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True)

output_file = open('results.txt', 'w')

#aggregate and retrain
dagger_itr = 5
for itr in range(dagger_itr):
    ob_list = []

    env = env = DuckietownEnv(
        map_name="zigzag_dists",
        max_steps=1000,
        draw_curve=False,
        draw_bbox=False,
        domain_rand=False,
        distortion=True,
        accept_start_angle_deg=4,
        full_transparency=True,
    )
    ob = env.reset()
    (ob, reward, done, info) = env.step([0.0,0.0])
    reward_sum = 0.0
    
    #teacher=DaggerTeacher(env)
    learner=DaggerLearner(model)

    for i in range(steps):
        #act = model.predict(img_reshape(ob))
        act=learner.predict(env,ob)
        print(act)
        (ob, reward, done, info) = env.step(act)
        #ob, reward, done, _ = env.step(act)
        if done is True:
            break
        else:
            ob_list.append(ob)
        reward_sum += reward
        print(i, reward, reward_sum, done, act)
    print('Episode done ', itr, i, reward_sum)
    output_file.write('Number of Steps: %02d\t Reward: %0.04f\n'%(i, reward_sum))
    env.close()

    if i==(steps-1):
        break

    for ob in ob_list:
        images_all = np.concatenate([images_all, img_reshape(ob)], axis=0)
        actions_all = np.concatenate([actions_all, np.reshape(get_teacher_action(ob), [1,action_dim])], axis=0)

    model.fit(images_all, actions_all,
                  batch_size=batch_size,
                  epochs=nb_epoch,
                  shuffle=True)
keras.models.save_model(model,"/tmp/dagger2")

