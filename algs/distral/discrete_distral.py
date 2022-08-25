from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from .policies import DiscreteDistralPolicy, DistralBasePolicies
from common.buffers import MultitaskReplayBuffer
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.policies import BasePolicy


class BaseDistral(OffPolicyAlgorithm):
    """
    Base class for Distral 
    Paper: https://arxiv.org/abs/1707.04175
    This implementation is different from what presented in the paper
    we use an actor critic architecture to avoid calculating the softmax 
    of the Q function, which can be unstable when Q values are large
    """

    def __init__(
            self,
            policy: Union[str, Type[DistralBasePolicies]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: float = 0.5,  # Hyper params of Distral
            beta: float = 5,  # Hyper params of Distral
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            replay_buffer_class: Optional[MultitaskReplayBuffer] = MultitaskReplayBuffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            target_update_interval: int = 1,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            supported_action_spaces: gym.spaces = None,
            _init_setup_model: bool = True,
    ):
        distral_args = {"alpha": alpha, "beta": beta}
        if policy_kwargs is None:
            policy_kwargs = distral_args
        else:
            assert isinstance(policy_kwargs, Dict)
            policy_kwargs.update(distral_args)

        self.n_envs = env.num_envs
        policy_kwargs.update(n_envs=self.n_envs)
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
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
            sde_support=False,
        )

        self.alpha = alpha
        self.beta = beta
        self.target_update_interval = target_update_interval

        if _init_setup_model:
            self._setup_model()

    def _create_aliases(self) -> None:
        self.actor = self.policy
        self.critic = self.policy


class DiscreteDistral(BaseDistral):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": DiscreteDistralPolicy,
    }

    def __init__(self, *args, **kwargs):
        kwargs.update(supported_action_spaces=(gym.spaces.Discrete))
        super().__init__(*args, **kwargs)
        self.epsilon_in_log = 1e-6

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)

        # Update optimizers learning rate
        optimizers = self.policy.actors_optims + self.policy.critics_optims

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        critic_losses, actor_losses = [], []

        gradient_steps *= self.n_envs

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # Update pi0
            probs_pi0, _ = self.policy.pi0.action_log_prob(replay_data.observations, False)
            with th.no_grad():
                probs_pi, _ = self.actor.action_log_prob(replay_data.env_indices,
                                                         replay_data.observations, False)

            log_probs_pi0 = th.log(probs_pi0 + self.epsilon_in_log)
            cross_entropy = (log_probs_pi0 * probs_pi).sum(dim=1, keepdims=True)
            cross_entropy = self.alpha / self.beta * cross_entropy
            pi0_loss = -cross_entropy.mean()

            self.policy.pi0_optim.zero_grad()
            pi0_loss.backward()
            self.policy.pi0_optim.step()

            # Update value function of tasks
            with th.no_grad():
                # Compute the next Q values: min over all critics targets
                next_q_values = self.critic.critic_target_task(replay_data.env_indices, replay_data.next_observations)
                next_q_values = th.minimum(*next_q_values)

                next_probs_pi, next_entropy = self.actor.action_log_prob(
                    replay_data.env_indices, replay_data.next_observations
                )

                next_prob_pi0, _ = self.policy.pi0.action_log_prob(replay_data.next_observations, False)
                next_log_prob_pi0 = th.log(next_prob_pi0 + self.epsilon_in_log)
                next_cross_ent = (next_log_prob_pi0 * next_probs_pi).sum(dim=1, keepdims=True)
                next_q_values = (
                        (next_q_values * next_probs_pi).sum(dim=1, keepdims=True) +
                        self.alpha / self.beta * next_cross_ent +
                        1 / self.beta * next_entropy.reshape(-1, 1)
                )

                target_q_values = (
                        replay_data.rewards
                        + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            current_q_values = self.critic.critic_task(replay_data.env_indices, replay_data.observations)
            current_q_values = [
                current_q.gather(1, replay_data.actions)
                for current_q in current_q_values
            ]

            # Compute critic loss
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.critics_optims[replay_data.env_indices].zero_grad()
            critic_loss.backward()
            self.critic.critics_optims[replay_data.env_indices].step()

            # Compute actor loss
            probs, entropy = self.actor.action_log_prob(replay_data.env_indices, replay_data.observations)

            q_values_pi = self.critic.critic_task(replay_data.env_indices, replay_data.observations)
            min_q_values = th.minimum(*q_values_pi)

            # entropy coefficient is 1, since we only want to project the softmax of Q to another actor
            actor_loss = (probs * min_q_values).sum(
                dim=1, keepdims=True
            ) + 1 * entropy.reshape(-1, 1)
            actor_loss = -actor_loss.mean()

            # Optimize the actor
            self.actor.actors_optims[replay_data.env_indices].zero_grad()
            actor_loss.backward()
            self.actor.actors_optims[replay_data.env_indices].step()
            actor_losses.append(actor_loss.item())

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.critics[replay_data.env_indices].critic_parameters(),
                    self.critic.critics[replay_data.env_indices].critic_target_parameters(),
                    self.tau,
                )
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
