import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooEnv

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from environment.multi_quadcopter_formation import MultiQuadcopterFormation


class MultiquadcopterFormationClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        return lambda: MultiQuadcopterFormation.from_json(
            **config
        )

    def supports_continuous_actions(self) -> bool:
        return True

    
    def supports_discrete_actions(self) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return env.max_duration_seconds * 60

    def has_render(self, env: EnvBase) -> bool:
        return True

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return {"agents": [agent.name for agent in env.agents]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        if "state" in env.observation_spec:
            return Composite({"state": env.observation_spec["state"].clone()})
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "action_mask":
                    del group_obs_spec[key]
            if group_obs_spec.is_empty():
                del observation_spec[group]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        if observation_spec.is_empty():
            return None
        return observation_spec

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return "multiquadcopter_formation"


class MultiquadcopterFormationTask(Task):
    """Enum for PettingZoo tasks."""

    AUTO_SWITCH = None

    @staticmethod
    def associated_class():
        return MultiquadcopterFormationClass

