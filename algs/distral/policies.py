from typing import Any, Dict, List, Optional, Type, Union

import gym
import torch as th
from torch import nn
from torch.nn import functional as F
from algs.sac_discrete.policies import DiscreteActor, TwinDelayedQNetworks
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, BaseModel


class TwinDelayedContinuousQNetworks(BaseModel):
    """
    Twin delayed continuous Q networks for continuous action space
    """
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
        
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.critic, self.critic_target = None, None
        
    def _build(self) -> None:
        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    
    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.net_args, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.critic(self.extract_features(obs))
        
        
class DistralBasePolicies(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        n_envs: int, 
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
        self.n_envs = n_envs
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actors, self.critics = None, None
        
        self._build(lr_schedule)
        
    def _build(self, lr_schedule: Schedule):
        self._build_actor_critic()
        self._build_optims()
        
    def _build_list_module(self, module_cls: BasePolicy, 
            args: Optional[Dict[str, Any]], n_module: int,
            features_extractor: Optional[BaseFeaturesExtractor] = None):
        args = self._update_features_extractor(args, features_extractor)
        return [module_cls(**args) for _ in range(n_module)]
        
    def _build_list_optimizer(self, optimizer_class: Type[th.optim.Optimizer],
            optimizer_kwargs: Optional[Dict[str, Any]], 
            parameters,
            lr_schedule
    ):
        optims = []
        for params in parameters:
            optims.append(optimizer_class(params, lr=lr_schedule(1), **optimizer_kwargs))
        return optims
        
class DiscreteDistral(DistralBasePolicies):
    def _build(self, lr_schedule: Schedule):
        self.actors = self._build_list_module(DiscreteActor, self.net_args, self.n_envs)
        self.critics = self._build_list_module(TwinDelayedQNetworks, self.net_args, self.n_envs)
        
        actor_params = [list(actor.parameters()) for actor in self.actors]
        critic_params = [list(critic.critic.parameters()) for critic in self.critics]
        
        self.actors_optims = self._build_list_optimizer(
            self.optimizer_class, self.optimizer_kwargs, actor_params, lr_schedule
        )
        
        self.critics_optims = self._build_list_optimizer(
            self.optimizer_class, self.optimizer_kwargs, critic_params, lr_schedule
        )