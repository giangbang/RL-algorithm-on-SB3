import gym
import custom_envs



envs_name = {
    'grid-tworoom': [lambda: gym.make('Gridworld-v0', plan=4),
                     lambda: gym.make('Gridworld-v0', plan=5),
                     lambda: gym.make('Gridworld-v0', plan=6),],
    'grid-alley': [lambda: gym.make('Gridworld-v0', plan=20),
                   lambda: gym.make('Gridworld-v0', plan=21),],
    'grid-maze': [lambda: gym.make('Gridworld-v0', plan=30), 
                  lambda: gym.make('Gridworld-v0', plan=31),
                  lambda: gym.make('Gridworld-v0', plan=32),],
    'grid-seperate-corridor': [lambda: gym.make('Gridworld-v0', plan=50),
                lambda: gym.make('Gridworld-v0', plan=51),],
    'HER:grid-maze': [
        lambda: gym.make('HER:Gridworld-v0', plan=30), 
        lambda: gym.make('HER:Gridworld-v0', plan=31),
        lambda: gym.make('HER:Gridworld-v0', plan=32),
    ],
    'HER:grid-alley': [lambda: gym.make('HER:Gridworld-v0', plan=20),
                   lambda: gym.make('HER:Gridworld-v0', plan=21),],
    'HER:grid-seperate-corridor': [lambda: gym.make('HER:Gridworld-v0', plan=50),
                lambda: gym.make('HER:Gridworld-v0', plan=51),],
    
}
