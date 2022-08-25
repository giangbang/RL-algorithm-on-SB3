import gym
import custom_envs

envs_name = {
    'grid-tworoom': [lambda: gym.make('Gridworld-v0', plan=4),
                     lambda: gym.make('Gridworld-v0', plan=5),
                     lambda: gym.make('Gridworld-v0', plan=6),],

}
