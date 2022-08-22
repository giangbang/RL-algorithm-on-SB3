import gym
import numpy as np  

from sac_discrete import DiscreteSAC

env = gym.make("FrozenLake-v1")

model = DiscreteSAC("MlpPolicy", env, verbose=1, learning_rate=1e-2)
model.learn(total_timesteps=1000, log_interval=20)