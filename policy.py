from typing import Any, Dict, List, Optional, Type, Union

import gym
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.distributions import (
  Distribution,
  CategoricalDistribution,
  make_proba_distribution,
)
from stable_baselines3.common.torch_layers import (
  BaseFeaturesExtractor,
  FlattenExtractor,
  create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule


class TwinDelayedQNetworks(BasePolicy):
  def __init__(
    self,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    lr_schedule: Schedule,
    net_arch: Optional[List[int]] = None,
    activation_fn: Type[nn.Module] = nn.ReLU,
    normalize_images: bool = True
  ):
    super().__init__(
      observation_space,
      action_space,
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
      "lr_schedule": lr_schedule,
    }

    self.q1_net, self.q2_net = None, None
    self._build()

  def _build(self) -> None:
    self.q1_net = self.make_dqn_net()
    self.q2_net = self.make_dqn_net()

  def make_dqn_net(self) -> DQNPolicy:
    dqn_net = DQNPolicy(**self.net_args)
    dqn_net.optimizer = None
    return dqn_net

  def forward(self, obs: th.Tensor) -> th.Tensor:
    return self.q1_net.q_net(obs), self.q2_net.q_net(obs)

  def _predict(self, obs: th.Tensor) -> th.Tensor:
    return th.minimum(*self.forward(obs))
        
  def critic_target(self, obs: th.Tensor) -> th.Tensor:
    return self.q1_net.q_net_target(obs), self.q2_net.q_net_target(obs)
    
  def critic_online(self, obs: th.Tensor) -> th.Tensor:
    return self.q1_net.q_net(obs), self.q2_net.q_net(obs)
    
  def critic_target_parameters(self):
    return list(self.q1_net.q_net_target.parameters()) + \
           list(self.q2_net.q_net_target.parameters())
           
  def critic_online_parameters(self):
    return list(self.q1_net.q_net.parameters()) + \
           list(self.q2_net.q_net.parameters())
    

class DiscreteActor(BasePolicy):
  def __init__(
    self,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    net_arch: List[int],
    features_extractor: nn.Module,
    features_dim: int, 
    lr_schedule: Schedule,
    activation_fn: Type[nn.Module] = nn.ReLU,
    normalize_images: bool = True,
  ):
    super().__init__(
      observation_space,
      action_space,
      features_extractor=features_extractor,
      normalize_images=normalize_images,
    )
    assert isinstance(net_arch, List)

    self.net_arch = net_arch
    self.features_extractor = features_extractor
    self.activation_fn = activation_fn
    self.features_dim = features_dim
    self.action_dim = self.action_space.n  # number of actions
    self.action_dist = make_proba_distribution(action_space)
    self.pi, self.action_net = None, None

    self._build()

  def _build_pi(self) -> None:
    pi = create_mlp(self.features_dim, self.net_arch[-1], self.net_arch[:-1], self.activation_fn)
    self.pi = nn.Sequential(*pi)

  def _build(self) -> None:
    self._build_pi()

    self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.net_arch[-1])

  def forward(self, obs: th.Tensor) -> th.Tensor:
    features = self.extract_features(obs)
    latent_pi = self.pi(features)

    logits = self.action_net(latent_pi)
    return logits
      
  def _get_distribution_from_logit(self, logits: th.Tensor) -> Distribution:
    return self.action_dist.proba_distribution(action_logits=logits)

  def get_distribution(self, obs: th.Tensor) -> Distribution:
    logits = self(obs)

    return self._get_distribution_from_logit(logits)

  def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
    return self.get_distribution(observation).get_actions(deterministic=deterministic)

  def action_log_prob(self, obs: th.Tensor) -> th.Tensor:
    logits = self(obs)
    probs = F.softmax(logits, dim=1)

    return probs, self._get_distribution_from_logit(logits).entropy()

class DiscreteSACPolicy(BasePolicy):
  def __init__(
    self,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    lr_schedule: Schedule,
    net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    activation_fn: Type[nn.Module] = nn.ReLU,
    features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
    features_extractor_kwargs: Optional[Dict[str, Any]] = None,
    normalize_images: bool = True,
    optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
  ):

    super().__init__(
      observation_space,
      action_space,
      features_extractor_class,
      features_extractor_kwargs,
      optimizer_class=optimizer_class,
      optimizer_kwargs=optimizer_kwargs,
    )

    if net_arch is None:
      net_arch = [256, 256]

    self.net_arch = net_arch
    self.activation_fn = activation_fn
    self.net_args = {
      "observation_space": self.observation_space,
      "action_space": self.action_space,
      "net_arch": net_arch,
      "activation_fn": self.activation_fn,
      "normalize_images": normalize_images,
      "lr_schedule": lr_schedule,
    }

    self.actor, self.critic = None, None

    self._build(lr_schedule)

  def _build(self, lr_schedule: Schedule) -> None:
    self.actor = self.make_actor()
    self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    self.critic = self.make_critic()
    critic_parameters = self.critic.critic_online_parameters()
    self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

  def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DiscreteActor:
    actor_kwargs = self._update_features_extractor(self.net_args, features_extractor)
    print(actor_kwargs)
    return DiscreteActor(**actor_kwargs).to(self.device)

  def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> TwinDelayedQNetworks:
    # critic_kwargs = self._update_features_extractor(self.net_args, features_extractor)
    return TwinDelayedQNetworks(**self.net_args).to(self.device)

  def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
    return self.actor._predict(observation, deterministic)
    
  def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
    return self._predict(obs, deterministic=deterministic)

  def set_training_mode(self, mode: bool) -> None:
    """
    Put the policy in either training or evaluation mode.

    This affects certain modules, such as batch normalisation and dropout.

    :param mode: if true, set to training mode, else set to evaluation mode
    """
    self.actor.set_training_mode(mode)
    self.critic.set_training_mode(mode)
    self.training = mode

MlpPolicy = DiscreteSACPolicy