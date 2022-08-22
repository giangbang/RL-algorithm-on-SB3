from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from policy import MlpPolicy, DiscreteSACPolicy
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

class DiscreteSAC(OffPolicyAlgorithm):
  """
  Discrete Soft-Actor Critic Algorithm
  """
  policy_aliases: Dict[str, Type[BasePolicy]] = {
    "MlpPolicy": MlpPolicy,
  }
  
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
    train_freq: Union[int, Tuple[int, str]] = 1,
    gradient_steps: int = 1,
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
    super()._setup_model()
    self._create_aliases()
    # Target entropy is used when learning the entropy coefficient
    if self.target_entropy == "auto":
      # automatically set target entropy if needed
      self.target_entropy = np.log(self.env.action_space.n)/5
    else:
      # Force conversion
      # this will also throw an error for unexpected string
      self.target_entropy = float(self.target_entropy)
      
    # The entropy coefficient or entropy can be learned automatically
    # see Automating Entropy Adjustment for Maximum Entropy RL section
    # of https://arxiv.org/abs/1812.05905
    if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
      # Default initial value of ent_coef when learned
      init_value = 1.0
      if "_" in self.ent_coef:
        init_value = float(self.ent_coef.split("_")[1])
        assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

      # Note: we optimize the log of the entropy coeff which is slightly different from the paper
      # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
      self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
      self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
    else:
      # Force conversion to float
      # this will throw an error if a malformed string (different from 'auto')
      # is passed
      self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)
    
  def _create_aliases(self) -> None:
    self.actor = self.policy.actor
    self.critic = self.policy.critic
    
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
    
      ent_coef_loss = None
      if self.ent_coef_optimizer is not None:
        ent_coef = th.exp(self.log_ent_coef.detach())
        with th.no_grad():
          entropy = self.actor.get_distribution(replay_data.observations).entropy()
        
        ent_coef_loss = -(self.log_ent_coef * (-entropy + self.target_entropy).detach()).mean()
        ent_coef_losses.append(ent_coef_loss.item())
      else:
        ent_coef = self.ent_coef_tensor

      ent_coefs.append(ent_coef.item())

      # Optimize entropy coefficient, also called
      # entropy temperature or alpha in the paper
      if ent_coef_loss is not None:
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

      with th.no_grad():
        # Select action according to policy
        next_action_probs, next_entropy = self.actor.action_log_prob(replay_data.next_observations)
        # Compute the next Q values: min over all critics targets
        next_q_values = self.critic.critic_target(replay_data.next_observations)
        next_q_values = th.minimum(*next_q_values)
        next_q_values = (next_q_values * next_action_probs).sum(dim=1, keepdims=True)
        # add entropy term
        next_q_values = next_q_values + ent_coef * next_entropy.reshape(-1, 1)
        # td error + entropy term
        target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
        
      # Get current Q-values estimates for each critic network
      # using action from the replay buffer
      current_q_values = self.critic.critic_online(replay_data.observations)
      current_q_values = [current_q.gather(1, replay_data.actions) for current_q in current_q_values]

      # Compute critic loss
      critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
      critic_losses.append(critic_loss.item())

      # Optimize the critic
      self.critic.optimizer.zero_grad()
      critic_loss.backward()
      self.critic.optimizer.step()
      
      # Compute actor loss
      probs, entropy = self.actor.action_log_prob(replay_data.observations)
      
      q_values_pi = self.critic.critic_online(replay_data.observations)
      min_q_values = th.minimum(*q_values_pi)
      actor_loss = (probs * min_q_values).sum(dim=1, keepdims=True) + ent_coef * entropy.reshape(-1, 1)
      actor_loss = -actor_loss.mean()
      
      # Optimize the actor
      self.actor.optimizer.zero_grad()
      actor_loss.backward()
      self.actor.optimizer.step()
      actor_losses.append(actor_loss.item())

      # Update target networks
      if gradient_step % self.target_update_interval == 0:
        polyak_update(self.critic.critic_online_parameters(), self.critic.critic_target_parameters(), self.tau)

    self._n_updates += gradient_steps

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/ent_coef", np.mean(ent_coefs))
    self.logger.record("train/actor_loss", np.mean(actor_losses))
    self.logger.record("train/critic_loss", np.mean(critic_losses))
    if len(ent_coef_losses) > 0:
      self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))