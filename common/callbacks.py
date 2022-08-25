from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import copy

class LogReturnSeperateEnvs(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_rewards = None
        self.cnt_episode = None
        
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
                self.logger.record(f'train/task return {idx}', self.total_rewards[idx])
                self.total_rewards[idx] = 0
        return True
