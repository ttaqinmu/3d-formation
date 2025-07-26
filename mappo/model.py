#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pathlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, MISSING
from typing import Any, Dict, List, Optional, Sequence, Type

import torch
import torch.nn.functional as F
from torch import nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from tensordict.utils import NestedKey, expand_as_right, unravel_key_list
from torchrl.data import Composite, TensorSpec, Unbounded
from torchrl.modules import MLP, MultiAgentMLP, LSTMCell

from .utils import _class_from_name, _read_yaml_config, DEVICE_TYPING


def _check_spec(tensordict, spec):
    if not spec.is_in(tensordict):
        raise ValueError(f"TensorDict {tensordict} not in spec {spec}")


def parse_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    del cfg["name"]
    kwargs = {}
    for key, value in cfg.items():
        if key.endswith("class") and value is not None:
            value = _class_from_name(cfg[key])
        kwargs.update({key: value})
    return kwargs


def output_has_agent_dim(share_params: bool, centralised: bool) -> bool:
    if share_params and centralised:
        return False
    else:
        return True


class Model(TensorDictModuleBase, ABC):
    def __init__(
        self,
        input_spec: Composite,
        output_spec: Composite,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        action_spec: Composite,
        model_index: int,
        is_critic: bool,
    ):
        TensorDictModuleBase.__init__(self)

        self.input_spec = input_spec
        self.output_spec = output_spec
        self.agent_group = agent_group
        self.input_has_agent_dim = input_has_agent_dim
        self.centralised = centralised
        self.share_params = share_params
        self.device = device
        self.n_agents = n_agents
        self.action_spec = action_spec
        self.model_index = model_index
        self.is_critic = is_critic

        self.in_keys = list(self.input_spec.keys(True, True))
        self.out_keys = list(self.output_spec.keys(True, True))

        self.out_key = self.out_keys[0]
        self.output_leaf_spec = self.output_spec[self.out_key]

        self._perform_checks()

    @property
    def output_has_agent_dim(self) -> bool:
        return output_has_agent_dim(self.share_params, self.centralised)

    @property
    def in_key(self) -> NestedKey:
        if len(self.in_keys) > 1:
            raise ValueError("Model has more than one input key")
        return self.in_keys[0]

    @property
    def input_leaf_spec(self) -> TensorSpec:
        return self.input_spec[self.in_key]

    def _perform_checks(self):
        if not self.input_has_agent_dim and not self.centralised:
            raise ValueError(
                "If input does not have an agent dimension the model should be marked as centralised"
            )

        if len(self.out_keys) > 1:
            raise ValueError("Currently models support just one output key")

        if self.agent_group in self.input_spec.keys() and self.input_spec[
            self.agent_group
        ].shape != (self.n_agents,):
            raise ValueError(
                "If the agent group is in the input specs, its shape should be the number of agents"
            )
        if self.agent_group in self.output_spec.keys() and self.output_spec[
            self.agent_group
        ].shape != (self.n_agents,):
            raise ValueError(
                "If the agent group is in the output specs, its shape should be the number of agents"
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = self._forward(tensordict)
        return tensordict

    def share_params_with(self, other_model):
        if (
            self.share_params != other_model.share_params
            or self.centralised != other_model.centralised
            or self.input_has_agent_dim != other_model.input_has_agent_dim
            or self.input_spec != other_model.input_spec
            or self.output_spec != other_model.output_spec
        ):
            warnings.warn(
                "Sharing parameters with models that are not identical. "
                "This might result in unintended behavior or error."
            )
        for param, other_param in zip(self.parameters(), other_model.parameters()):
            other_param.data[:] = param.data

    ###############################
    # Abstract methods to implement
    ###############################

    @abstractmethod
    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError


class SequenceModel(Model):
    """A sequence of :class:`~benchmarl.models.Model`

    Args:
       models (list of Model): the models in the sequence
    """

    def __init__(
        self,
        models: List[Model],
    ):
        super().__init__(
            n_agents=models[0].n_agents,
            input_spec=models[0].input_spec,
            output_spec=models[-1].output_spec,
            centralised=models[0].centralised,
            share_params=models[0].share_params,
            device=models[0].device,
            agent_group=models[0].agent_group,
            input_has_agent_dim=models[0].input_has_agent_dim,
            action_spec=models[0].action_spec,
            model_index=models[0].model_index,
            is_critic=models[0].is_critic,
        )
        self.models = TensorDictSequential(*models)
        self.in_keys = self.models.in_keys
        self.out_keys = self.models.out_keys

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.models(tensordict)


@dataclass
class ModelConfig(ABC):
    def get_model(
        self,
        input_spec: Composite,
        output_spec: Composite,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        action_spec: Composite,
        model_index: int = 0,
    ) -> Model:
        return self.associated_class()(
            **asdict(self),
            input_spec=input_spec,
            output_spec=output_spec,
            agent_group=agent_group,
            input_has_agent_dim=input_has_agent_dim,
            n_agents=n_agents,
            centralised=centralised,
            share_params=share_params,
            device=device,
            action_spec=action_spec,
            model_index=model_index,
            is_critic=self.is_critic,
        )

    @staticmethod
    @abstractmethod
    def associated_class():
        """
        The associated Model class
        """
        raise NotImplementedError

    @property
    def is_rnn(self) -> bool:
        """
        Whether the model is an RNN
        """
        return False

    @property
    def is_critic(self):
        """
        Whether the model is a critic
        """
        if not hasattr(self, "_is_critic"):
            self._is_critic = False
        return self._is_critic

    @is_critic.setter
    def is_critic(self, value):
        """
        Set whether the model is a critic
        """
        self._is_critic = value

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        return Composite()

    def _get_model_state_spec_inner(
        self, model_index: int = 0, group: str = None
    ) -> Composite:
        return self.get_model_state_spec(model_index)

    @staticmethod
    def _load_from_yaml(name: str) -> Dict[str, Any]:
        yaml_path = (
            pathlib.Path(__file__).parent.parent
            / "conf"
            / "model"
            / "layers"
            / f"{name.lower()}.yaml"
        )
        return _read_yaml_config(str(yaml_path.resolve()))

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        if path is None:
            config = ModelConfig._load_from_yaml(name=cls.associated_class().__name__)
        else:
            config = _read_yaml_config(path)
        config = parse_model_config(config)
        return cls(**config)


@dataclass
class SequenceModelConfig(ModelConfig):
    model_configs: Sequence[ModelConfig]
    intermediate_sizes: Sequence[int]

    def __post_init__(self):
        for model_config in self.model_configs:
            if isinstance(model_config, EnsembleModelConfig):
                raise TypeError(
                    "SequenceModelConfig cannot contain EnsembleModelConfig layers, but the opposite can be done."
                )

    def get_model(
        self,
        input_spec: Composite,
        output_spec: Composite,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        action_spec: Composite,
        model_index: int = 0,
    ) -> Model:
        n_models = len(self.model_configs)
        if not n_models > 0:
            raise ValueError(
                f"SequenceModelConfig expects n_models > 0, got {n_models}"
            )
        if len(self.intermediate_sizes) != n_models - 1:
            raise ValueError(
                f"SequenceModelConfig intermediate_sizes len should be {n_models - 1}, got {len(self.intermediate_sizes)}"
            )

        out_has_agent_dim = output_has_agent_dim(share_params, centralised)
        next_centralised = not out_has_agent_dim
        intermediate_specs = [
            Composite(
                {
                    f"_{agent_group}{'_critic' if self.is_critic else ''}_intermediate_{i}": Unbounded(
                        shape=(n_agents, size) if out_has_agent_dim else (size,)
                    )
                }
            )
            for i, size in enumerate(self.intermediate_sizes)
        ] + [output_spec]

        models = [
            self.model_configs[0].get_model(
                input_spec=input_spec,
                output_spec=intermediate_specs[0],
                agent_group=agent_group,
                input_has_agent_dim=input_has_agent_dim,
                n_agents=n_agents,
                centralised=centralised,
                share_params=share_params,
                device=device,
                action_spec=action_spec,
                model_index=0,
            )
        ]

        next_models = [
            self.model_configs[i].get_model(
                input_spec=intermediate_specs[i - 1],
                output_spec=intermediate_specs[i],
                agent_group=agent_group,
                input_has_agent_dim=out_has_agent_dim,
                n_agents=n_agents,
                centralised=next_centralised,
                share_params=share_params,
                device=device,
                action_spec=action_spec,
                model_index=i,
            )
            for i in range(1, n_models)
        ]
        models += next_models
        return SequenceModel(models)

    @staticmethod
    def associated_class():
        return SequenceModel

    @property
    def is_critic(self):
        if not hasattr(self, "_is_critic"):
            self._is_critic = False
        return self._is_critic

    @is_critic.setter
    def is_critic(self, value):
        self._is_critic = value
        for model_config in self.model_configs:
            model_config.is_critic = value

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        spec = Composite()
        for i, model_config in enumerate(self.model_configs):
            spec.update(model_config.get_model_state_spec(model_index=i))
        return spec

    @property
    def is_rnn(self) -> bool:
        is_rnn = False
        for model_config in self.model_configs:
            is_rnn += model_config.is_rnn
        return is_rnn

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        raise NotImplementedError


@dataclass
class EnsembleModelConfig(ModelConfig):

    model_configs_map: Dict[str, ModelConfig]

    def get_model(self, agent_group: str, **kwargs) -> Model:
        if agent_group not in self.model_configs_map.keys():
            raise ValueError(
                f"Environment contains agent group '{agent_group}' not present in the EnsembleModelConfig configuration."
            )
        return self.model_configs_map[agent_group].get_model(
            **kwargs, agent_group=agent_group
        )

    @staticmethod
    def associated_class():
        class EnsembleModel(Model):
            pass

        return EnsembleModel

    @property
    def is_critic(self):
        if not hasattr(self, "_is_critic"):
            self._is_critic = False
        return self._is_critic

    @is_critic.setter
    def is_critic(self, value):
        self._is_critic = value
        for model_config in self.model_configs_map.values():
            model_config.is_critic = value

    def _get_model_state_spec_inner(
        self, model_index: int = 0, group: str = None
    ) -> Composite:
        return self.model_configs_map[group].get_model_state_spec(
            model_index=model_index
        )

    @property
    def is_rnn(self) -> bool:
        is_rnn = False
        for model_config in self.model_configs_map.values():
            is_rnn += model_config.is_rnn
        return is_rnn

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        raise NotImplementedError


class Mlp(Model):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        )
        self.output_features = self.output_leaf_spec.shape[-1]

        if self.input_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs=self.input_features,
                n_agent_outputs=self.output_features,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **kwargs,
            )
        else:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_features=self.input_features,
                        out_features=self.output_features,
                        device=self.device,
                        **kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

    def _perform_checks(self):
        super()._perform_checks()

        input_shape = None
        for input_key, input_spec in self.input_spec.items(True, True):
            if (self.input_has_agent_dim and len(input_spec.shape) == 2) or (
                not self.input_has_agent_dim and len(input_spec.shape) == 1
            ):
                if input_shape is None:
                    input_shape = input_spec.shape[:-1]
                else:
                    if input_spec.shape[:-1] != input_shape:
                        raise ValueError(
                            f"MLP inputs should all have the same shape up to the last dimension, got {self.input_spec}"
                        )
            else:
                raise ValueError(
                    f"MLP input value {input_key} from {self.input_spec} has an invalid shape, maybe you need a CNN?"
                )
        if self.input_has_agent_dim:
            if input_shape[-1] != self.n_agents:
                raise ValueError(
                    "If the MLP input has the agent dimension,"
                    f" the second to last spec dimension should be the number of agents, got {self.input_spec}"
                )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the MLP output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = torch.cat([tensordict.get(in_key) for in_key in self.in_keys], dim=-1)

        # Has multi-agent input dimension
        if self.input_has_agent_dim:
            res = self.mlp.forward(input)
            if not self.output_has_agent_dim:
                # If we are here the module is centralised and parameter shared.
                # Thus the multi-agent dimension has been expanded,
                # We remove it without loss of data
                res = res[..., 0, :]

        # Does not have multi-agent input dimension
        else:
            if not self.share_params:
                res = torch.stack(
                    [net(input) for net in self.mlp],
                    dim=-2,
                )
            else:
                res = self.mlp[0](input)

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class MlpConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Mlp`."""

    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    activation_class: Type[nn.Module] = MISSING
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Mlp


class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        device: DEVICE_TYPING,
        n_layers: int,
        dropout: float,
        bias: bool,
        time_dim: int = -2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.time_dim = time_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bias = bias

        self.lstms = torch.nn.ModuleList(
            [
                LSTMCell(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    device=self.device,
                    bias=self.bias,
                )
                for i in range(self.n_layers)
            ]
        )

    def forward(self, input, is_init, h, c):
        hs = []

        h = list(h.unbind(dim=-2))
        c = list(c.unbind(dim=-2))

        for in_t, init_t in zip(
            input.unbind(self.time_dim), is_init.unbind(self.time_dim)
        ):
            for layer in range(self.n_layers):
                h[layer] = torch.where(init_t, 0, h[layer])
                c[layer] = torch.where(init_t, 0, c[layer])

                h[layer], c[layer] = self.lstms[layer](in_t, (h[layer], c[layer]))

                if layer < self.n_layers - 1 and self.dropout:
                    in_t = F.dropout(h[layer], p=self.dropout, training=self.training)
                else:
                    in_t = h[layer]

            hs.append(in_t)
        h_n = torch.stack(h, dim=-2)
        c_n = torch.stack(c, dim=-2)
        output = torch.stack(hs, self.time_dim)

        return output, h_n, c_n


def get_net(input_size, hidden_size, n_layers, bias, device, dropout, compile):
    lstm = LSTM(
        input_size,
        hidden_size,
        n_layers=n_layers,
        bias=bias,
        device=device,
        dropout=dropout,
    )
    if compile:
        lstm = torch.compile(lstm, mode="reduce-overhead")
    return lstm


class MultiAgentLSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_agents: int,
        device: DEVICE_TYPING,
        centralised: bool,
        share_params: bool,
        n_layers: int,
        dropout: float,
        bias: bool,
        compile: bool,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_agents = n_agents
        self.hidden_size = hidden_size
        self.device = device
        self.centralised = centralised
        self.share_params = share_params
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.compile = compile

        if self.centralised:
            input_size = input_size * self.n_agents

        agent_networks = [
            get_net(
                input_size=input_size,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                bias=self.bias,
                device=self.device,
                dropout=self.dropout,
                compile=self.compile,
            )
            for _ in range(self.n_agents if not self.share_params else 1)
        ]
        self._make_params(agent_networks)

        with torch.device("meta"):
            self._empty_lstm = get_net(
                input_size=input_size,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                bias=self.bias,
                device="meta",
                dropout=self.dropout,
                compile=self.compile,
            )
            # Remove all parameters
            TensorDict.from_module(self._empty_lstm).data.to("meta").to_module(
                self._empty_lstm
            )

    def forward(
        self,
        input,
        is_init,
        h_0=None,
        c_0=None,
    ):
        # Input and output always have the multiagent dimension
        # Hidden states always have it apart from when it is centralized and share params
        # is_init never has it

        assert is_init is not None, "We need to pass is_init"
        training = h_0 is None

        missing_batch = False
        if (
            not training and len(input.shape) < 3
        ):  # In evaluation the batch might be missing
            missing_batch = True
            input = input.unsqueeze(0)
            h_0 = h_0.unsqueeze(0)
            c_0 = c_0.unsqueeze(0)
            is_init = is_init.unsqueeze(0)

        if (
            not training
        ):  # In collection we emulate the sequence dimension and we have the hidden state
            input = input.unsqueeze(1)

        # Check input
        batch = input.shape[0]
        seq = input.shape[1]
        assert input.shape == (batch, seq, self.n_agents, self.input_size)

        if not training:  # Collection
            # Set hidden to 0 when is_init
            h_0 = torch.where(expand_as_right(is_init, h_0), 0, h_0)
            c_0 = torch.where(expand_as_right(is_init, c_0), 0, c_0)
            is_init = is_init.unsqueeze(
                1
            )  # If in collection emulate the sequence dimension

        assert is_init.shape == (batch, seq, 1)
        is_init = is_init.unsqueeze(-2).expand(batch, seq, self.n_agents, 1)

        if training:
            if self.centralised and self.share_params:
                shape = (
                    batch,
                    self.n_layers,
                    self.hidden_size,
                )
            else:
                shape = (
                    batch,
                    self.n_agents,
                    self.n_layers,
                    self.hidden_size,
                )
            h_0 = torch.zeros(
                shape,
                device=self.device,
                dtype=torch.float,
            )
            c_0 = h_0.clone()
        if self.centralised:
            input = input.view(batch, seq, self.n_agents * self.input_size)
            is_init = is_init[..., 0, :]

        output, h_n, c_n = self.run_net(input, is_init, h_0, c_0)

        if self.centralised and self.share_params:
            output = output.unsqueeze(-2).expand(
                batch, seq, self.n_agents, self.hidden_size
            )

        if not training:
            output = output.squeeze(1)
        if missing_batch:
            output = output.squeeze(0)
            h_n = h_n.squeeze(0)
            c_n = c_n.squeeze(0)
        return output, h_n, c_n

    def run_net(self, input, is_init, h_0, c_0):
        if not self.share_params:
            if self.centralised:
                output, h_n, c_n = self.vmap_func_module(
                    self._empty_lstm,
                    (0, None, None, -3, -3),
                    (-2, -3, -3),
                )(self.params, input, is_init, h_0, c_0)
            else:
                output, h_n, c_n = self.vmap_func_module(
                    self._empty_lstm,
                    (0, -2, -2, -3, -3),
                    (-2, -3, -3),
                )(self.params, input, is_init, h_0, c_0)
        else:
            with self.params.to_module(self._empty_lstm):
                if self.centralised:
                    output, h_n, c_n = self._empty_lstm(input, is_init, h_0, c_0)
                else:
                    output, h_n, c_n = torch.vmap(
                        self._empty_lstm,
                        in_dims=(-2, -2, -3, -3),
                        out_dims=(-2, -3, -3),
                    )(input, is_init, h_0, c_0)

        return output, h_n, c_n

    def vmap_func_module(self, module, *args, **kwargs):
        def exec_module(params, *input):
            with params.to_module(module):
                return module(*input)

        return torch.vmap(exec_module, *args, **kwargs)

    def _make_params(self, agent_networks):
        if self.share_params:
            self.params = TensorDict.from_module(agent_networks[0], as_module=True)
        else:
            self.params = TensorDict.from_modules(*agent_networks, as_module=True)


class Lstm(Model):
    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        bias: bool,
        dropout: float,
        compile: bool,
        **kwargs,
    ):

        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.hidden_state_name_h = (
            self.agent_group,
            f"_hidden_lstm_h_{self.model_index}",
        )
        self.hidden_state_name_c = (
            self.agent_group,
            f"_hidden_lstm_c_{self.model_index}",
        )

        self.rnn_keys = unravel_key_list(
            ["is_init", self.hidden_state_name_c, self.hidden_state_name_h]
        )
        self.in_keys += self.rnn_keys

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.compile = compile

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        )
        self.output_features = self.output_leaf_spec.shape[-1]

        if self.input_has_agent_dim:
            self.lstm = MultiAgentLSTM(
                self.input_features,
                self.hidden_size,
                self.n_agents,
                self.device,
                bias=self.bias,
                n_layers=self.n_layers,
                centralised=self.centralised,
                share_params=self.share_params,
                dropout=self.dropout,
                compile=self.compile,
            )
        else:
            self.lstm = nn.ModuleList(
                [
                    get_net(
                        input_size=self.input_features,
                        hidden_size=self.hidden_size,
                        n_layers=self.n_layers,
                        bias=self.bias,
                        device=self.device,
                        dropout=self.dropout,
                        compile=self.compile,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

        mlp_net_kwargs = {
            "_".join(k.split("_")[1:]): v
            for k, v in kwargs.items()
            if k.startswith("mlp_")
        }
        if self.output_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs=self.hidden_size,
                n_agent_outputs=self.output_features,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **mlp_net_kwargs,
            )
        else:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_features=self.hidden_size,
                        out_features=self.output_features,
                        device=self.device,
                        **mlp_net_kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

    def _perform_checks(self):
        super()._perform_checks()

        input_shape = None
        for input_key, input_spec in self.input_spec.items(True, True):
            if (self.input_has_agent_dim and len(input_spec.shape) == 2) or (
                not self.input_has_agent_dim and len(input_spec.shape) == 1
            ):
                if input_shape is None:
                    input_shape = input_spec.shape[:-1]
                else:
                    if input_spec.shape[:-1] != input_shape:
                        raise ValueError(
                            f"LSTM inputs should all have the same shape up to the last dimension, got {self.input_spec}"
                        )
            else:
                raise ValueError(
                    f"LSTM input value {input_key} from {self.input_spec} has an invalid shape, maybe you need a CNN?"
                )
        if self.input_has_agent_dim:
            if input_shape[-1] != self.n_agents:
                raise ValueError(
                    "If the LSTM input has the agent dimension,"
                    f" the second to last spec dimension should be the number of agents, got {self.input_spec}"
                )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the LSTM output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = torch.cat(
            [
                tensordict.get(in_key)
                for in_key in self.in_keys
                if in_key not in self.rnn_keys
            ],
            dim=-1,
        )
        h_0 = tensordict.get(self.hidden_state_name_h, None)
        c_0 = tensordict.get(self.hidden_state_name_c, None)
        is_init = tensordict.get("is_init")

        training = h_0 is None

        # Has multi-agent input dimension
        if self.input_has_agent_dim:
            output, h_n, c_n = self.lstm(input, is_init, h_0, c_0)
            if not self.output_has_agent_dim:
                output = output[..., 0, :]
        else:  # Is a global input, this is a critic
            # Check input
            batch = input.shape[0]
            seq = input.shape[1]
            assert input.shape == (batch, seq, self.input_features)
            assert is_init.shape == (batch, seq, 1)

            h_0 = torch.zeros(
                (batch, self.n_layers, self.hidden_size),
                device=self.device,
                dtype=torch.float,
            )
            c_0 = h_0.clone()
            if self.share_params:
                output, _, _ = self.lstm[0](input, is_init, h_0, c_0)
            else:
                outputs = []
                for net in self.lstm:
                    output, _, _ = net(input, is_init, h_0, c_0)
                    outputs.append(output)
                output = torch.stack(outputs, dim=-2)

        # Mlp
        if self.output_has_agent_dim:
            output = self.mlp.forward(output)
        else:
            if not self.share_params:
                output = torch.stack(
                    [net(output) for net in self.mlp],
                    dim=-2,
                )
            else:
                output = self.mlp[0](output)

        tensordict.set(self.out_key, output)
        if not training:
            tensordict.set(("next", *self.hidden_state_name_h), h_n)
            tensordict.set(("next", *self.hidden_state_name_c), c_n)
        return tensordict


@dataclass
class LstmConfig(ModelConfig):
    hidden_size: int = MISSING
    n_layers: int = MISSING
    bias: bool = MISSING
    dropout: float = MISSING
    compile: bool = MISSING

    mlp_num_cells: Sequence[int] = MISSING
    mlp_layer_class: Type[nn.Module] = MISSING
    mlp_activation_class: Type[nn.Module] = MISSING

    mlp_activation_kwargs: Optional[dict] = None
    mlp_norm_class: Type[nn.Module] = None
    mlp_norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Lstm

    @property
    def is_rnn(self) -> bool:
        return True

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        spec = Composite(
            {
                f"_hidden_lstm_c_{model_index}": Unbounded(
                    shape=(self.n_layers, self.hidden_size)
                ),
                f"_hidden_lstm_h_{model_index}": Unbounded(
                    shape=(self.n_layers, self.hidden_size)
                ),
            }
        )
        return spec
