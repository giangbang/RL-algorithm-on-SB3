import gym
import numpy as np

from algs.distral import DiscreteDistral
from parse_args import parse_args
from stable_baselines3.common.logger import configure
from get_multi_envs import envs_name
from stable_baselines3.common.vec_env import DummyVecEnv
from common.callbacks import LogReturnSeparateEnvs
import os

from stable_baselines3 import HerReplayBuffer, DQN
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

args = parse_args()
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
# If True the HER transitions will get sampled online
online_sampling = True

env_name = 'HER-' + args.env_name

log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

envs = envs_name[env_name]
envs = envs[0]()
# envs = DummyVecEnv(envs)
# envs = Monitor(envs, log_dir)
print(envs.reset())

model = DQN(
    "MultiInputPolicy",
    envs,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=100
    ),
    verbose=1,
)

# new_logger = configure('./', ["stdout", "csv"])
# model.set_logger(new_logger)

# callback_return = LogReturnSeparateEnvs()

model.learn(total_timesteps=args.total_timesteps, log_interval=1000)
