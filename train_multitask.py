import gym
import numpy as np  

from algs.distral import DiscreteDistral
from parse_args import parse_args
from stable_baselines3.common.logger import configure
from get_multitask import envs_name
from stable_baselines3.common.utils import DummyVecEnv
from algs.distral.callbacks import LogRewardSeperateEnvs

args = parse_args()
# test envs

envs = envs_name['grid-tworoom']
envs = DummyVecEnv(envs)

model = DiscreteDistral("MlpPolicy",
    envs, verbose=1, learning_rate=args.learning_rate,
    buffer_size=args.buffer_size,
    learning_starts=args.learning_starts,
    batch_size=args.batch_size,
    tau = args.tau,
    gamma=args.gamma,
    train_freq=args.train_freq,
    gradient_steps=args.gradient_steps, 
    alpha=0.5,
    beta=5)
new_logger = configure('./', ["stdout", "csv"])
model.set_logger(new_logger)

callback = LogRewardSeperateEnvs()

model.learn(total_timesteps=args.total_timesteps, 
    log_interval=10, callback=callback)