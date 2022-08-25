"""
For simplicity in the implementation, we do not allow (or take into consideration) the sharing 
of feature extractors between policies and Q networks, algorithms that do require the 
sharing extractors (for example, CARE) should maintain its own extractor in its respective algorithm class
"""
from typing import Any, Dict, List, Optional, Type, Union, Tuple

import gym
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.distributions import (
    Distribution,
    CategoricalDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, BaseModel
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)


class DiscretePolicy(BasePolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        assert isinstance(net_arch, list)

        self.net_arch = net_arch
        self.features_extractor = features_extractor
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.action_dim = self.action_space.n  # number of actions
        self.action_dist = make_proba_distribution(action_space)
        self.pi, self.action_net = None, None

        self._build()

    def _build_pi(self) -> None:
        pi = create_mlp(
            self.features_dim, self.net_arch[-1], self.net_arch[:-1], self.activation_fn
        )
        self.pi = nn.Sequential(*pi)

    def _build(self) -> None:
        self._build_pi()

        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.net_arch[-1]
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        latent_pi = self.pi(features)

        logits = self.action_net(latent_pi)
        return logits

    def logits(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        latent_pi = self.pi(features)

        logits = self.action_net(latent_pi)
        return logits

    def _get_distribution_from_logit(self, logits: th.Tensor) -> Distribution:
        return self.action_dist.proba_distribution(action_logits=logits)

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        logits = self(obs)
        return self._get_distribution_from_logit(logits)

    def _predict(
            self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def action_log_prob(self, obs: th.Tensor, return_ent=True) -> th.Tensor:
        logits = self(obs)
        probs = F.softmax(logits, dim=1)
        if not return_ent:
            ent = None
        else:
            ent = self._get_distribution_from_logit(logits).entropy()

        return probs, ent


class DiscreteActor(BasePolicy):
    """
    Wrapper class for discrete actor
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            features_extractor_class: nn.Module,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
        )
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.policy = None
        self._build()

    def _build(self):
        policy_args = self._update_features_extractor(self.net_args)
        self.policy = DiscretePolicy(**policy_args)

    def _predict(
            self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        return self.policy._predict(observation, deterministic)

    def forward(self, observation: th.Tensor):
        return self._predict(observation)

    def logits(self, obs: th.Tensor) -> th.Tensor:
        logits = self.policy.logits(obs)
        return logits

    def action_log_prob(self, obs: th.Tensor, return_ent=True) -> th.Tensor:
        return self.policy.action_log_prob(obs, return_ent)

    def _get_distribution_from_logit(self, logits: th.Tensor) -> Distribution:
        return self.policy._get_distribution_from_logit(logits)


class DiscreteTwinDelayedDoubleQNetworks(BasePolicy):
    """
    Double Q Network with delayed target networks
    with total of four Q Networks at its core
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            features_extractor_class: nn.Module,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            normalize_images=normalize_images,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
        )

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": features_extractor_kwargs,
            "lr_schedule": get_schedule_fn(1), # dummy learning schedule, do not matter
        }

        self.q1_net, self.q2_net = None, None
        self._build()

    def _build(self) -> None:
        self.q1_net = self.make_dqn()
        self.q2_net = self.make_dqn()

    def make_dqn(self) -> DQNPolicy:
        dqn_net = DQNPolicy(**self.net_args)
        # discard optimizer of DQN, as we will have the optimizer in other class
        dqn_net.optimizer = None
        return dqn_net

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.q1_net.q_net(obs), self.q2_net.q_net(obs)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        return th.minimum(*self.forward(obs))

    def critic_target(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.q1_net.q_net_target(obs), self.q2_net.q_net_target(obs)

    def critic(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.q1_net.q_net(obs), self.q2_net.q_net(obs)

    def critic_target_parameters(self):
        """
        Return list of parameters of the target networks
        This will be helpful when making optimizers or polyak update
        """
        return list(self.q1_net.q_net_target.parameters()) + list(
            self.q2_net.q_net_target.parameters()
        )

    def critic_parameters(self):
        """
        Return list of parameters of the target networks
        This will be helpful when making optimizers or polyak update
        """
        return list(self.q1_net.q_net.parameters()) + list(
            self.q2_net.q_net.parameters()
        )


class ContinuousTwinDelayedDoubleQNetworks(BasePolicy):
    """
    Twin delayed Double Q networks for continuous action space
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            features_extractor_class: nn.Module,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            normalize_images=normalize_images,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
        )

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": features_extractor_kwargs,
        }

        self.critic, self.target_critic = None, None

        self._build()

    def _build(self) -> None:
        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.net_args, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def critic_target_parameters(self):
        return self.critic_target.parameters()

    def critic_parameters(self):
        return self.critic.parameters()

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.critic(obs)
