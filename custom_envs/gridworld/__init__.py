from .gridworld_env import GridworldEnv, MultiGoalGridWorld
from gym.envs.registration import register

register(
    id="Gridworld-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
)

register(
    id="HER:Gridworld-v0",
    entry_point=__name__ + ":MultiGoalGridWorld",
    max_episode_steps=100,
)