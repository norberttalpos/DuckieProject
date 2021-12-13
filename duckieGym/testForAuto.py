import os
import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from keras.models import load_model
from tensorflow import keras
from PIL import Image

from dagger_learner import DaggerLearner
from dagger_teacher import DaggerTeacher
from IIL import InteractiveImitationLearning
from dagger_sandbox import MyInteractiveImitationLearning
from detector import preprocess_image
from data_reader import *
from tensorflow.keras.callbacks import EarlyStopping



from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def img_reshape(input_img):
    _img = np.reshape(preprocess_image(input_img), (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img


if __name__ == "__main__":
    # ! Parser sector:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-name", default="34")
    
    args = parser.parse_args()
    print(args.map_name)
    



    img_dim = [48,85,3]
    action_dim = 2
    steps = 300
    batch_size = 32
    nb_epoch = 100



    images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]))
    actions_all = np.zeros((0,action_dim))
    rewards_all = np.zeros((0,))

    img_list = []
    action_list = []
    reward_list = []

    env = env = DuckietownEnv(
        map_name=args.map_name,
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
        #act = np.array([0.0, 0.0])
        #else:
        act = teacher.predict(env,ob)
        act[1]*=10.0
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

#i=0
#for ob in img_list:
    #obs = Image.fromarray(ob, 'RGB')
    #obs.save(os.path.join(os.getcwd(),"alma", str(i) + ".png"))
    #i+=1

    print('Packing data into arrays...')
    for img, act, rew in zip(img_list, action_list, reward_list):
        images_all = np.concatenate([images_all, img_reshape(img)], axis=0)
        actions_all = np.concatenate([actions_all, np.reshape(act, [1,action_dim])], axis=0)
        rewards_all = np.concatenate([rewards_all, rew], axis=0)


model = load_model("/tmp/dagger2")

model.fit(images_all, actions_all,
              batch_size=batch_size,
              validation_split=0.15,
              epochs=nb_epoch,
              shuffle=True)

output_file = open('results.txt', 'w')

#aggregate and retrain
dagger_itr = 5
for itr in range(dagger_itr):
    ob_list = []

    env = env = DuckietownEnv(
        map_name=args.map_name,
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
    teacher=DaggerTeacher(env)
    learner=DaggerLearner(model)

    for i in range(steps):
        act = model.predict(img_reshape(ob))
        act = np.array([act[0][0],act[0][1]])
        print(act)
        #act=learner.predict(env,ob)
        
        #print(act)
        #act[1]*=2.0
        
        #ob, reward, done, _ = env.step(act)
        if done is True:
            break
        else:
            ob_list.append(ob)
        (ob, reward, done, info) = env.step(act)
        actions_all = np.concatenate([actions_all, np.reshape(teacher.predict(env,ob), [1,action_dim])], axis=0)
        reward_sum += reward
        print("Teacher: ",np.reshape(teacher.predict(env,ob), [1,action_dim]))
        
        print(i, reward, reward_sum, done, act)
    print('Episode done ', itr, i, reward_sum)
    output_file.write('Number of Steps: %02d\t Reward: %0.04f\n'%(i, reward_sum))
    env.close()
    
    i=0
    for ob in ob_list:
        images_all = np.concatenate([images_all, img_reshape(ob)], axis=0)
        
        #obs = Image.fromarray(ob, 'RGB')
        #obs.save(os.path.join(os.getcwd(),"barack", str(i) + ".png"))
        #i+=1

    model.fit(images_all, actions_all,
                  batch_size=batch_size,
                  validation_split=0.15,
                  epochs=nb_epoch,
                  shuffle=True)
                  
    if i==(steps-1):
        break
        
        
print ("alma")
keras.models.save_model(model,"/tmp/dagger2")
print(model.summary())