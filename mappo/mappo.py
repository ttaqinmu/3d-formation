#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.envs import Compose, EnvBase, Transform
from torchrl.modules import (
    IndependentNormal,
    MaskedCategorical,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators
from torchrl.data import (
    Composite,
    Categorical,
    LazyMemmapStorage,
    LazyTensorStorage,
    OneHot,
    ReplayBuffer,
    Unbounded,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import (
    PrioritizedSampler,
    RandomSampler,
    SamplerWithoutReplacement,
)
from torchrl.objectives.utils import HardUpdate, SoftUpdate, TargetNetUpdater

from .model import ModelConfig
from .utils import _read_yaml_config, DEVICE_TYPING


class Algorithm(ABC):
    def __init__(self, experiment):
        self.experiment = experiment

        self.device: DEVICE_TYPING = experiment.config.train_device
        self.buffer_device: DEVICE_TYPING = experiment.config.buffer_device
        self.experiment_config = experiment.config
        self.model_config = experiment.model_config
        self.critic_model_config = experiment.critic_model_config
        self.on_policy = experiment.on_policy
        self.group_map = experiment.group_map
        self.observation_spec = experiment.observation_spec
        self.action_spec = experiment.action_spec
        self.state_spec = experiment.state_spec
        self.action_mask_spec = experiment.action_mask_spec
        self.has_independent_critic = (
            experiment.algorithm_config.has_independent_critic()
        )
        self.has_centralized_critic = (
            experiment.algorithm_config.has_centralized_critic()
        )
        self.has_critic = experiment.algorithm_config.has_critic
        self.has_rnn = self.model_config.is_rnn or (
            self.critic_model_config.is_rnn and self.has_critic
        )

        # Cached values that will be instantiated only once and then remain fixed
        self._losses_and_updaters = {}
        self._policies_for_loss = {}
        self._policies_for_collection = {}

        self._check_specs()

    def _check_specs(self):
        if self.state_spec is not None:
            if len(self.state_spec.keys(True, True)) != 1:
                raise ValueError(
                    "State spec must contain one entry per group"
                    " to follow the library conventions, "
                    "you can apply a transform to your environment to satisfy this criteria."
                )
        for group in self.group_map.keys():
            if (
                len(self.action_spec[group].keys(True, True)) != 1
                or list(self.action_spec[group].keys())[0] != "action"
            ):
                raise ValueError(
                    "Action spec must contain one entry per group named 'action'"
                    " to follow the library conventions, "
                    "you can apply a transform to your environment to satisfy this criteria."
                )
            if (
                self.action_mask_spec is not None
                and group in self.action_mask_spec.keys()
                and (
                    len(self.action_mask_spec[group].keys(True, True)) != 1
                    or list(self.action_mask_spec[group].keys())[0] != "action_mask"
                )
            ):
                raise ValueError(
                    "Action mask spec must contain one entry per group named 'action_mask'"
                    " to follow the library conventions, "
                    "you can apply a transform to your environment to satisfy this criteria."
                )

    def get_loss_and_updater(self, group: str) -> Tuple[LossModule, TargetNetUpdater]:
        if group not in self._losses_and_updaters.keys():
            action_space = self.action_spec[group, "action"]
            continuous = not isinstance(action_space, (Categorical, OneHot))
            loss, use_target = self._get_loss(
                group=group,
                policy_for_loss=self.get_policy_for_loss(group),
                continuous=continuous,
            )
            if use_target:
                if self.experiment_config.soft_target_update:
                    target_net_updater = SoftUpdate(
                        loss, tau=self.experiment_config.polyak_tau
                    )
                else:
                    target_net_updater = HardUpdate(
                        loss,
                        value_network_update_interval=self.experiment_config.hard_target_update_frequency,
                    )
            else:
                target_net_updater = None
            self._losses_and_updaters.update({group: (loss, target_net_updater)})
        return self._losses_and_updaters[group]

    def get_replay_buffer(
        self, group: str, transforms: List[Transform] = None
    ) -> ReplayBuffer:
        memory_size = self.experiment_config.replay_buffer_memory_size(self.on_policy)
        sampling_size = self.experiment_config.train_minibatch_size(self.on_policy)
        if self.has_rnn:
            sequence_length = -(
                -self.experiment_config.collected_frames_per_batch(self.on_policy)
                // self.experiment_config.n_envs_per_worker(self.on_policy)
            )
            memory_size = -(-memory_size // sequence_length)
            sampling_size = -(-sampling_size // sequence_length)

        # Sampler
        if self.on_policy:
            sampler = SamplerWithoutReplacement()
        elif self.experiment_config.off_policy_use_prioritized_replay_buffer:
            sampler = PrioritizedSampler(
                memory_size,
                self.experiment_config.off_policy_prb_alpha,
                self.experiment_config.off_policy_prb_beta,
            )
        else:
            sampler = RandomSampler()

        # Storage
        if self.buffer_device == "disk" and not self.on_policy:
            storage = LazyMemmapStorage(
                memory_size,
                device=self.device,
                scratch_dir=self.experiment.folder_name / f"buffer_{group}",
            )
        else:
            storage = LazyTensorStorage(
                memory_size,
                device=self.device if self.on_policy else self.buffer_device,
            )

        return TensorDictReplayBuffer(
            storage=storage,
            sampler=sampler,
            batch_size=sampling_size,
            priority_key=(group, "td_error"),
            transform=Compose(*transforms) if transforms is not None else None,
        )

    def get_policy_for_loss(self, group: str) -> TensorDictModule:
        """
        Get the non-explorative policy for a specific group loss.
        This function calls the abstract :class:`~benchmarl.algorithms.Algorithm._get_policy_for_loss()` which needs to be implemented.
        The function will cache the output at the first call and return the cached values in future calls.

        Args:
            group (str): agent group of the policy

        Returns: TensorDictModule representing the policy
        """
        if group not in self._policies_for_loss.keys():
            action_space = self.action_spec[group, "action"]
            continuous = not isinstance(action_space, (Categorical, OneHot))
            self._policies_for_loss.update(
                {
                    group: self._get_policy_for_loss(
                        group=group,
                        continuous=continuous,
                        model_config=self.model_config,
                    )
                }
            )
        return self._policies_for_loss[group]

    def get_policy_for_collection(self) -> TensorDictSequential:
        """
        Get the explorative policy for all groups together.
        This function calls the abstract :class:`~benchmarl.algorithms.Algorithm._get_policy_for_collection()` which needs to be implemented.
        The function will cache the output at the first call and return the cached values in future calls.

        Returns: TensorDictSequential representing all explorative policies
        """
        policies = []
        for group in self.group_map.keys():
            if group not in self._policies_for_collection.keys():
                policy_for_loss = self.get_policy_for_loss(group)
                action_space = self.action_spec[group, "action"]
                continuous = not isinstance(action_space, (Categorical, OneHot))
                policy_for_collection = self._get_policy_for_collection(
                    policy_for_loss,
                    group,
                    continuous,
                )
                self._policies_for_collection.update({group: policy_for_collection})
            policies.append(self._policies_for_collection[group])
        return TensorDictSequential(*policies)

    def get_parameters(self, group: str) -> Dict[str, Iterable]:
        """
        Get the dictionary mapping loss names to the relative parameters to optimize for a given group.
        This function calls the abstract :class:`~benchmarl.algorithms.Algorithm._get_parameters()` which needs to be implemented.

        Returns: a dictionary mapping loss names to a parameters' list
        """
        return self._get_parameters(
            group=group,
            loss=self.get_loss_and_updater(group)[0],
        )

    def process_env_fun(
        self,
        env_fun: Callable[[], EnvBase],
    ) -> Callable[[], EnvBase]:
        """
        This function can be used to wrap env_fun

        Args:
            env_fun (callable): a function that takes no args and creates an enviornment

        Returns: a function that takes no args and creates an enviornment

        """

        return env_fun

    ###############################
    # Abstract methods to implement
    ###############################

    @abstractmethod
    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        """
        Implement this function to return the LossModule for a specific group.

        Args:
            group (str): agent group of the loss
            policy_for_loss (TensorDictModule): the policy to use in the loss
            continuous (bool): whether to return a loss for continuous or discrete actions

        Returns: LossModule and a bool representing if the loss should have target parameters
        """
        raise NotImplementedError

    @abstractmethod
    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        """
        Get the dictionary mapping loss names to the relative parameters to optimize for a given group loss.

        Returns: a dictionary mapping loss names to a parameters' list
        """
        raise NotImplementedError

    @abstractmethod
    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        """
        Get the non-explorative policy for a specific group.

        Args:
            group (str): agent group of the policy
            model_config (ModelConfig): model config class
            continuous (bool): whether the policy should be continuous or discrete

        Returns: TensorDictModule representing the policy
        """
        raise NotImplementedError

    @abstractmethod
    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        """
        Implement this function to add an explorative layer to the policy used in the loss.

        Args:
            policy_for_loss (TensorDictModule): the group policy used in the loss
            group (str): agent group
            continuous (bool): whether the policy is continuous or discrete

        Returns: TensorDictModule representing the explorative policy
        """
        raise NotImplementedError

    @abstractmethod
    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        """
        This function can be used to reshape data coming from collection before it is passed to the policy.

        Args:
            group (str): agent group
            batch (TensorDictBase): the batch of data coming from the collector

        Returns: the processed batch

        """
        raise NotImplementedError

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        """
        Here you can modify the loss_vals tensordict containing entries loss_name->loss_value
        For example, you can sum two entries in a new entry, to optimize them together.

        Args:
            group (str): agent group
            loss_vals (TensorDictBase): the tensordict returned by the loss forward method

        Returns: the processed loss_vals
        """
        return loss_vals


@dataclass
class AlgorithmConfig:
    """
    Dataclass representing an algorithm configuration.
    This should be overridden by implemented algorithms.
    Implementors should:

        1. add configuration parameters for their algorithm
        2. implement all abstract methods

    """

    def get_algorithm(self, experiment) -> Algorithm:
        """
        Main function to turn the config into the associated algorithm

        Args:
            experiment (Experiment): the experiment class

        Returns: the Algorithm

        """
        return self.associated_class()(
            **self.__dict__,  # Passes all the custom config parameters
            experiment=experiment,
        )

    @staticmethod
    def _load_from_yaml(name: str) -> Dict[str, Any]:
        yaml_path = (
            pathlib.Path(__file__).parent.parent
            / "conf"
            / "algorithm"
            / f"{name.lower()}.yaml"
        )
        return _read_yaml_config(str(yaml_path.resolve()))

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        """
        Load the algorithm configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from.
                If None, it will default to
                ``benchmarl/conf/algorithm/self.associated_class().__name__``

        Returns: the loaded AlgorithmConfig
        """

        if path is None:
            config = AlgorithmConfig._load_from_yaml(
                name=cls.associated_class().__name__
            )

        else:
            config = _read_yaml_config(path)
        return cls(**config)

    @staticmethod
    @abstractmethod
    def associated_class() -> Type[Algorithm]:
        """
        The algorithm class associated to the config
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def on_policy() -> bool:
        """
        If the algorithm has to be run on policy or off policy
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_continuous_actions() -> bool:
        """
        If the algorithm supports continuous actions
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_discrete_actions() -> bool:
        """
        If the algorithm supports discrete actions
        """
        raise NotImplementedError

    @staticmethod
    def has_independent_critic() -> bool:
        """
        If the algorithm uses an independent critic
        """
        return False

    @staticmethod
    def has_centralized_critic() -> bool:
        """
        If the algorithm uses a centralized critic
        """
        return False

    def has_critic(self) -> bool:
        """
        If the algorithm uses a critic
        """
        if self.has_centralized_critic() and self.has_independent_critic():
            raise ValueError(
                "Algorithm can either have a centralized critic or an indpendent one"
            )
        return self.has_centralized_critic() or self.has_independent_critic()

class Mappo(Algorithm):
    def __init__(
        self,
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: bool,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        scale_mapping: str,
        use_tanh_normal: bool,
        minibatch_advantage: bool,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.share_param_critic = share_param_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.loss_critic_type = loss_critic_type
        self.lmbda = lmbda
        self.scale_mapping = scale_mapping
        self.use_tanh_normal = use_tanh_normal
        self.minibatch_advantage = minibatch_advantage

    #############################
    # Overridden abstract methods
    #############################

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        # Loss
        loss_module = ClipPPOLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        return {
            "loss_objective": list(loss.actor_network_params.flatten_keys().values()),
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )

        actor_output_spec = Composite(
            {
                group: Composite(
                    {"logits": Unbounded(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else TanhNormal
                ),
                distribution_kwargs=(
                    {
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    }
                    if self.use_tanh_normal
                    else {}
                ),
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )

        else:
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys={
                        "logits": (group, "logits"),
                        "mask": (group, "action_mask"),
                    },
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )

        return policy

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        # MAPPO uses the same stochastic actor for collection
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )

        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        loss = self.get_loss_and_updater(group)[0]
        if self.minibatch_advantage:
            increment = -(
                -self.experiment.config.train_minibatch_size(self.on_policy)
                // batch.shape[1]
            )
        else:
            increment = batch.batch_size[0] + 1
        last_start_index = 0
        start_index = increment
        minibatches = []
        while last_start_index < batch.shape[0]:
            minimbatch = batch[last_start_index:start_index]
            minibatches.append(minimbatch)
            with torch.no_grad():
                loss.value_estimator(
                    minimbatch,
                    params=loss.critic_network_params,
                    target_params=loss.target_critic_network_params,
                )
            last_start_index = start_index
            start_index += increment

        batch = torch.cat(minibatches, dim=0)
        return batch

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        loss_vals.set(
            "loss_objective", loss_vals["loss_objective"] + loss_vals["loss_entropy"]
        )
        del loss_vals["loss_entropy"]
        return loss_vals

    #####################
    # Custom new methods
    #####################

    def get_critic(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if self.share_param_critic:
            critic_output_spec = Composite({"state_value": Unbounded(shape=(1,))})
        else:
            critic_output_spec = Composite(
                {
                    group: Composite(
                        {"state_value": Unbounded(shape=(n_agents, 1))},
                        shape=(n_agents,),
                    )
                }
            )

        if self.state_spec is not None:
            input_has_agent_dim = False
            critic_input_spec = self.state_spec

        else:
            input_has_agent_dim = True
            critic_input_spec = Composite(
                {group: self.observation_spec[group].clone().to(self.device)}
            )

        value_module = self.critic_model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=True,
            input_has_agent_dim=input_has_agent_dim,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
        )
        if self.share_param_critic:
            expand_module = TensorDictModule(
                lambda value: value.unsqueeze(-2).expand(
                    *value.shape[:-1], n_agents, 1
                ),
                in_keys=["state_value"],
                out_keys=[(group, "state_value")],
            )
            value_module = TensorDictSequential(value_module, expand_module)

        return value_module


@dataclass
class MappoConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Mappo`."""

    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Mappo

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
