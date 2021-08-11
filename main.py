
from utils.tensorboard_video import VideoRecorderCallback

import gym
import  os
from stable_baselines3 import A2C, PPO, DDPG, TD3, SAC

# tensorboard_log
tensorboard_log = 'logs/'

# 環境リスト 
env_names = ['Pendulum-v0']

# シード数
random_seeds = 1
render_freq = 2500
total_timesteps = 10000
for env_name in env_names:
    for _ in  range(random_seeds):
        env = gym.make(env_name)
        video_recorder = VideoRecorderCallback(env, render_freq=render_freq)
        model = DDPG("MlpPolicy", env_name, tensorboard_log=tensorboard_log, verbose=1)
        model.learn(total_timesteps=total_timesteps, callback=video_recorder)
        env.close()
