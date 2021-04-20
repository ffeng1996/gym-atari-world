"""
saves ~ 200 episodes generated from a random policy
"""
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import random
import sys
# import tensorflow as tf
import os
import gym
# import keras
import gym_cartpole_world
from dqn.ddqn import DoubleDQNAgent

from xvfbwrapper import Xvfb
import cv2
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
# gpu_config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=False, log_device_placement=False)
# gpu_config.gpu_options.allow_growth = True
#
# # K.tensorflow_backend._get_available_gpus()
# sess = tf.Session(config=gpu_config)
# keras.backend.set_session(sess)


SCREEN_X = 64
SCREEN_Y = 64

vdisplay = Xvfb(width=SCREEN_X, height=SCREEN_Y)
vdisplay.start()

MAX_FRAMES = 40
MAX_TRIALS = 10000
RANDOM_TRIALS = 10000

cartpole_version = sys.argv[1]
# cartpole_version = 'v20'
env = gym.make('CartPoleWorld-{}'.format(cartpole_version))
env.initialize()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

theta_threshold = env.theta_threshold
x_threshold = env.x_threshold

dataset_name = 'dataset_len{}_xthreshold{}_thetathreshold{}_trial{}_randomtrial{}_newgravity'.format(
    MAX_FRAMES, x_threshold, theta_threshold, MAX_TRIALS, RANDOM_TRIALS)

if not os.path.exists(dataset_name):
    os.makedirs(dataset_name)

if not os.path.exists(os.path.join(dataset_name, cartpole_version)):
    os.makedirs(os.path.join(dataset_name, cartpole_version))

agent = DoubleDQNAgent(state_size, action_size)
# agent.model.load_weights("./save_model/cartpoleworld_theta_{}_x_{}_{}.h5".format(theta_threshold,
#                                                                                  x_threshold,
#                                                                                  cartpole_version))


env = gym.wrappers.Monitor(env, "save_demo", video_callable=False, force=True, write_upon_reset=False)

trial = 0
while trial < MAX_TRIALS:  # 200 trials per worker

    recording_state = []
    recording_obs = []
    recording_action = []
    recording_reward = []

    success = 0

    state = env.reset()

    for frame in range(MAX_FRAMES):
        # add state
        recording_state.append(state)

        # add obs
        obs = env.render(mode='rgb_array')
        # obs = obs[72:328, 172:428, :]
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (128, 128), interpolation=cv2.INTER_CUBIC)
        obs = cv2.normalize(obs, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        recording_obs.append(obs)

        # add action
        state = np.reshape(state, [1, state_size])
        if trial < RANDOM_TRIALS:
            action = 1 if np.random.rand() > 0.5 else 0
        else:
            action = agent.get_action(state, is_training=False)
        recording_action.append(action)

        # add reward
        next_state, reward, done, info = env.step(action)
        recording_reward.append(reward)

        state = next_state

        if done:
            if frame == (MAX_FRAMES - 1):
                success = 1
                print("Done. Episode {} finished after {} timesteps".format(trial, frame + 1))
            else:
                trial = trial - 1

            break
        elif frame == (MAX_FRAMES - 1):
            success = 1
            print("Episode {} finished after {} timesteps".format(trial, frame + 1))
    else:

        env.stats_recorder.save_complete()
        env.stats_recorder.done = True

    if success == 1:
        recording_obs = np.array(recording_obs, dtype=np.float16)
        recording_state = np.array(recording_state, dtype=np.float16)
        recording_action = np.array(recording_action, dtype=np.float16)
        recording_reward = np.array(recording_reward, dtype=np.float16)

        filename = '{}/{}/trail_{}.npz'.format(dataset_name, cartpole_version, trial)
        np.savez_compressed(filename,
                            obs=recording_obs,
                            state=recording_state,
                            action=recording_action,
                            reward=recording_reward)

    trial += 1

env.close()
env.env.close()
vdisplay.stop()