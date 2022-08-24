from typing import Any, Dict, List, Optional, Type, Union

import gym
import torch as th
from torch import nn
from algs.sac_discrete.policies import DiscreteActor, TwinDelayedQNetworks
from stable_baselines3.sac.policies import Actor as ContinuousActor
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, BaseModel
from stable_baselines3.common.type_aliases import Schedule

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)

class TwinDelayedContinuousQNetworks(BasePolicy):
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
            "net_arch": net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.critic, self.critic_target = None, None
    
    def critic_target_parameters(self):
        return self.critic_target.parameters()
        
    def critic_online_parameters(self):
        return self.critic.parameters()
    
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
        return self.critic(obs)

import inspect
def filter_dict(dict_to_filter, thing_with_kwargs):
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {filter_key:dict_to_filter[filter_key] for filter_key in filter_keys}
    return filtered_dict

class DistralBasePolicies(BasePolicy):
    """
    This implementation uses a actor critic architecture for learning Distral
    Which is mainly based on the Discrete SAC
    Actor Critic approach has an advantage is that it avoids calculating the 
    softmax distribution over the Q function, which can cause instability 
    when the values of Q function become too large 
    """
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
        alpha: float = 0.5,
        beta: float = 0.5,
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
        self.alpha = alpha
        self.beta = beta
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actors, self.critics = None, None
        self.pi0, self.pi0_optim = None, None
        self.actors_optims, self.critics_optims = None, None
        
        self._build(lr_schedule)
        
    def _build(self, lr_schedule: Schedule):
        self._build_actor_critic()
        self._build_optims(lr_schedule)

    def _build_actor_critic(self):
        pass 

    def _build_optims(self, lr_schedule: Schedule):
        actor_params = [list(actor.parameters()) for actor in self.actors]
        critic_params = [list(critic.critic.parameters()) for critic in self.critics]
        
        self.actors_optims = self._build_list_optimizer(
            self.optimizer_class, self.optimizer_kwargs, actor_params, lr_schedule
        )
        
        self.critics_optims = self._build_list_optimizer(
            self.optimizer_class, self.optimizer_kwargs, critic_params, lr_schedule
        )

        self.pi0_optim = self.optimizer_class(
            self.pi0.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        
        self.h0_optim = self.optimizer_class(
            self.h0.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        
    def _build_list_module(self, module_cls: BasePolicy, 
            args: Optional[Dict[str, Any]], n_module: int,
            features_extractor: Optional[BaseFeaturesExtractor] = None):
        args = self._update_features_extractor(args, features_extractor)
        args = filter_dict(args, module_cls)
        return [module_cls(**args) for _ in range(n_module)]
        
    def _build_list_optimizer(self, optimizer_class: Type[th.optim.Optimizer],
            optimizer_kwargs: Optional[Dict[str, Any]], 
            parameters, lr_schedule
    ):
        optims = []
        for params in parameters:
            optims.append(optimizer_class(params, lr=lr_schedule(1), **optimizer_kwargs))
        return optims

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic)

    def action_log_prob(self, 
            i_task: int, 
            observation: th.Tensor,
            return_ent: bool = True) -> th.Tensor:
        return self.actors[i_task].action_log_prob(observation, return_ent)

    def critic_target_task(self, i_task: int, observation: th.Tensor) -> th.Tensor:
        return self.critics[i_task].critic_target(observation)
        
    def critic_task(self, i_task: int, observation: th.Tensor) -> th.Tensor:
        return self.critics[i_task].critic(observation)

class DiscreteDistralPolicy(DistralBasePolicies):
    def _build_actor_critic(self):
        self.actors = self._build_list_module(DiscreteActor, self.net_args, self.n_envs)
        self.critics = self._build_list_module(TwinDelayedQNetworks, self.net_args, self.n_envs)
        self.pi0 = DiscreteActor(**self.net_args)
        self.h0  = TwinDelayedQNetworks(**self.net_args)
        
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        actions = []
        for obs, actor in zip(observation, self.actors):
            obs = obs.reshape(1, -1)
            action_logits_pi = actor._logits(obs)
            action_logits_pi0 = self.pi0._logits(obs)
            action_logits = self.beta*action_logits_pi + self.alpha*action_logits_pi0
            action = actor._get_distribution_from_logit(action_logits).get_actions(
                deterministic=deterministic
            )
            actions.append(action)
        actions = th.cat(actions, dim=0)
        return actions

class ContinuousDistral(DistralBasePolicies):
    def _build_actor_critic(self):
        self.actors = self._build_list_module(ContinuousActor, self.net_args, self.n_envs)
        self.critics = self._build_list_module(TwinDelayedContinuousQNetworks, self.net_args, self.n_envs)
        self.pi0 = ContinuousActor(**self.net_args)