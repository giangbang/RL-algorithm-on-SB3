from custom_envs.gridworld import GridworldEnv

envs_name = {
    'grid-tworoom': [lambda: GridworldEnv(4), lambda: GridworldEnv(5),
                     lambda: GridworldEnv(6)],

}
