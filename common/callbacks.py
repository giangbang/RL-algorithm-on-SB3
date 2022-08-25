from stable_baselines3.common.callbacks import BaseCallback
from typing import List
import numpy as np
import copy


class LogReturnSeparateEnvs(BaseCallback):
    """
    Logging returns of each environment in VecEnv separately.
    This callback helps when working with multitask problem,
    when tasks' individual performance is important.
    """

    def __init__(self, verbose=0, envs_names: List[str] = None):
        super().__init__(verbose)
        self.total_rewards = None
        self.cnt_episode = None
        self.envs_names = envs_names

    def _on_step(self):
        assert ('rewards' in self.locals and 'dones' in self.locals)
        if self.total_rewards is None:
            self.total_rewards = copy.deepcopy(self.locals['rewards'])
            self.cnt_episode = np.zeros_like(self.total_rewards)
        else:
            self.total_rewards += self.locals['rewards']

        dones = self.locals['dones']
        for idx, done in enumerate(dones):
            if done:
                self.cnt_episode[idx] += 1
                env_name = str(idx) if self.envs_names is None else self.envs_names[idx]
                self.logger.record(f'train/task {env_name} returns', self.total_rewards[idx])
                self.total_rewards[idx] = 0
        return True
