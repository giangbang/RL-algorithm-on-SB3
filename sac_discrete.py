from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


class DiscreteSAC(OffPolicyAlgorithm):
  """
  Discrete Soft-Actor Critic Algorithm
  """
  policy_aliases: Dict[str, Type[BasePolicy]] = {
    "MlpPolicy": MlpPolicy,
  }
  
  def __init__(
    self,
    policy: Union[str, Type[SACPolicy]],
    env: Union[GymEnv, str],
    learning_rate: Union[float, Schedule] = 3e-4,
    buffer_size: int = 1_000_000,  # 1e6
    learning_starts: int = 100,
    batch_size: int = 256,
    tau: float = 0.005,
    gamma: float = 0.99,
    train_freq: Union[int, Tuple[int, str]] = 1,
    gradient_steps: int = 1,
    action_noise: Optional[ActionNoise] = None,
    replay_buffer_class: Optional[ReplayBuffer] = None,
    replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
    optimize_memory_usage: bool = False,
    ent_coef: Union[str, float] = "auto",
    target_update_interval: int = 1,
    target_entropy: Union[str, float] = "auto",
    tensorboard_log: Optional[str] = None,
    create_eval_env: bool = False,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
    seed: Optional[int] = None,
    device: Union[th.device, str] = "auto",
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
      action_noise,
      replay_buffer_class=replay_buffer_class,
      replay_buffer_kwargs=replay_buffer_kwargs,
      policy_kwargs=policy_kwargs,
      tensorboard_log=tensorboard_log,
      verbose=verbose,
      device=device,
      create_eval_env=create_eval_env,
      seed=seed,
      sde_support=False,
      supported_action_spaces=(gym.spaces.Discrete),
      support_multi_env=True,
    )
    
    self.target_entropy = target_entropy
    self.log_ent_coef = None  # type: Optional[th.Tensor]
    # Entropy coefficient / Entropy temperature
    # Inverse of the reward scale
    self.ent_coef = ent_coef
    self.target_update_interval = target_update_interval
    self.ent_coef_optimizer = None

    if _init_setup_model:
      self._setup_model()
  
  def _setup_model(self) -> None:
    pass
    
  def train(self, gradient_steps: int, batch_size: int = 64) -> None:
    self.policy.set_training_mode(True)
    
    # Update optimizers learning rate
    optimizers = [self.actor.optimizer, self.critic.optimizer]
    if self.ent_coef_optimizer is not None:
      optimizers += [self.ent_coef_optimizer]
      
    # Update learning rate according to lr schedule
    self._update_learning_rate(optimizers)
    
    ent_coef_losses, ent_coefs = [], []
    actor_losses, critic_losses = [], []

    for gradient_step in range(gradient_steps):
      # Sample replay buffer
      replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
      actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
    
      ent_coef_loss = None
      if self.ent_coef_optimizer is not None:
        ent_coef = th.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        ent_coef_losses.append(ent_coef_loss.item())
      else:
        ent_coef = self.ent_coef_tensor
        
      # Optimize entropy coefficient, also called
      # entropy temperature or alpha in the paper
      if ent_coef_loss is not None:
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()
        
      with th.no_grad():
        # Select action according to policy
        next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
        # Compute the next Q values: min over all critics targets
        next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
        next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
        # add entropy term
        next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
        # td error + entropy term
        target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values