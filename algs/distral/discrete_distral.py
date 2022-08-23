from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from .policies import DiscreteDistral
from .MultitaskReplayBuffer import MultitaskReplayBuffer


class BaseDistral(OffPolicyAlgorithm):
    """
    Base class for Distral 
    Paper: https://arxiv.org/abs/1707.04175
    """
    
    def __init__(
        self,
        policy: Union[str, Type[DiscreteSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float = 0.5, # Hyper params of Distral
        beta: float  = 0.5, # Hyper params of Distral
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = MultitaskReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 1,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        supported_action_spaces: gym.spaces: None,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            supported_action_spaces=supported_action_spaces,
            support_multi_env=True,
        )
        
        self.alpha = alpha
        self.beta = beta
        
        if _init_setup_model:
            self._setup_model()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actors
        self.critic = self.policy.critics
        
class DiscreteDistral(BaseDistral):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": DiscreteDistral,
    }

    def __init__(self, *args, **kwargs):
        kwargs.update(supported_action_spaces=(gym.spaces.Discrete))
        super().__init__(self, *args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)

        # Update optimizers learning rate
        optimizers = self.policy.actors_optims + self.policy.critics_optims

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # Update pi0

            _, log_pi0 = self.policy.pi0.




        