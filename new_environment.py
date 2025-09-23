from time import process_time
import numpy as np
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field
from gymnasium.spaces import Box
from PyFlyt.core.aviary import Aviary


@dataclass
class Info:
    reward: float


@dataclass
class RewardConfig:
    delta_distance: bool = True
    exceede_boundary: float = 10.0
    collision: Optional[float] = None
    crowding: Optional[float] = None
    approach_same_target: Optional[float] = None

    @classmethod
    def default(cls) -> "RewardConfig":
        return cls(
            delta_distance=True
        )


class ObservationEnum(Enum):
    ang_vel_local = ["vp", "vq", "vr"]
    ang_pos = ["p", "q", "r"]
    lin_vel_local = ["u", "v", "w"]
    lin_pos_global = ["x", "y", "z"]
    pos_agent_relative = ["ca_x", "ca_y", "ca_z"]
    pos_target_relative = ["ct_x", "ct_y", "ct_z"]


@dataclass
class ObservationConfig:
    components: List[ObservationEnum]
    max_agent_neighbors: int = 3
    max_target_neighbors: int = 3

    @classmethod
    def default(cls) -> "ObservationConfig":
        return cls(
            components=[
                ObservationEnum.ang_vel_local,
                ObservationEnum.ang_pos,
                ObservationEnum.lin_vel_local,
                ObservationEnum.lin_pos_global,
                ObservationEnum.pos_agent_relative,
                ObservationEnum.pos_target_relative,
            ]
        )

    @property
    def dim_size(self) -> int:
        size = 0
        for component in self.components:
            match component:
                case ObservationEnum.ang_vel_local:
                    size += 3
                case ObservationEnum.ang_pos:
                    size += 3
                case ObservationEnum.lin_vel_local:
                    size += 3
                case ObservationEnum.lin_pos_global:
                    size += 3
                case ObservationEnum.pos_agent_relative:
                    size += 3 * self.max_agent_neighbors
                case ObservationEnum.pos_target_relative:
                    size += 3 * self.max_target_neighbors
        return size


@dataclass
class ShareObservationConfig:
    agent_ang_vel: bool = True
    agent_ang_pos: bool = True
    agent_lin_vel: bool = True
    agent_lin_pos: bool = True
    target_pos: bool = True
    target_reach: bool = True
    current_action: bool = True


class ActionEnum(Enum):
    PWM = -1
    ANG_VEL_THRUST = 0
    ANG_POS = 3
    LOCAL_LIN_VEL_ATT = 4
    LOCAL_LIN_VEL = 5
    GLOBAL_POS = 7


@dataclass
class ExtraConfig:
    reward: RewardConfig = field(default_factory=RewardConfig.default)
    observation: ObservationConfig = field(default_factory=ObservationConfig.default)
    action: ActionEnum = ActionEnum.GLOBAL_POS
    shared_observation: ShareObservationConfig = field(
        default_factory=ShareObservationConfig
    )


class MultiQuadFormation:
    def __init__(
        self,
        start_positions: np.ndarray,
        target_positions: np.ndarray,
        render: Optional[str] = None,
        np_random: Optional[np.random.Generator] = None,
        extra_config: Optional[RewardConfig] = None,
    ):
        assert start_positions.shape == target_positions.shape

        if not np_random:
            np_random = np.random.default_rng()

        if not extra_config:
            extra_config = ExtraConfig()    # type: ignore

        self.start_positions = start_positions
        self.start_orn = np.zeros((self.start_positions.shape[0], 3))
        self.target_positions = target_positions
        
        self.np_random = np_random
        self.extra_config: ExtraConfig = extra_config   # type: ignore

        self.num_agents = start_positions.shape[0]
        self.render = render

        self.step_count: int = 0

        self.aviary = Aviary(
            start_pos=self.start_positions,
            start_orn=self.start_orn,
            drone_type="quadx",
            np_random=self.np_random,
            render=True if self.render else False,
        )
 

    @property
    def action_space(self) -> Box:
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
        )


    @property
    def observation_space(self) -> Box:
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.extra_config.observation.dim_size,),
        )


    @property
    def shared_observation_space(self) -> Box:
        dim_size = 0
        config = self.extra_config.shared_observation

        if config.agent_ang_vel:
            dim_size += 3 * self.num_agents

        if config.agent_ang_pos:
            dim_size += 3 * self.num_agents

        if config.agent_lin_vel:
            dim_size += 3 * self.num_agents

        if config.agent_lin_pos:
            dim_size += 3 * self.num_agents

        if config.target_pos:
            dim_size += 3 * self.target_positions.shape[0]

        if config.target_reach:
            dim_size += self.target_positions.shape[0]

        if config.current_action:
            dim_size += 4 * self.num_agents

        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
        )


    def update_info(self):
        pass


    def get_observation(self, agent_index: int) -> np.ndarray:
        config = self.extra_config.observation
        raw = self.aviary.state(agent_index)
        
        ang_vel = raw[0]
        ang_pos = raw[1]
        lin_vel = raw[2]
        lin_pos = raw[3]

        obs = []

        for component in config.components:
            match component:
                case ObservationEnum.ang_vel_local:
                    obs.extend(ang_vel)
                
                case ObservationEnum.ang_pos:
                    obs.extend(ang_pos)
                
                case ObservationEnum.lin_vel_local:
                    obs.extend(lin_vel)
                
                case ObservationEnum.lin_pos_global:
                    obs.extend(lin_pos)

                case ObservationEnum.pos_agent_relative:
                    rel_positions = []
                    for i in range(self.num_agents):
                        if i != agent_index:
                            rel_positions.append(self.aviary.state(i)[3] - lin_pos)
                    rel_positions = np.array(rel_positions)
                    if rel_positions.shape[0] > config.max_agent_neighbors:
                        dists = np.linalg.norm(rel_positions, axis=1)
                        nearest_indices = np.argsort(dists)[: config.max_agent_neighbors]
                        rel_positions = rel_positions[nearest_indices]
                    elif rel_positions.shape[0] < config.max_agent_neighbors:
                        padding = np.zeros((config.max_agent_neighbors - rel_positions.shape[0], 3))
                        rel_positions = np.vstack((rel_positions, padding))
                    obs.extend(rel_positions.flatten())

                case ObservationEnum.pos_target_relative:
                    rel_targets = self.target_positions - lin_pos
                    if rel_targets.shape[0] > config.max_target_neighbors:
                        dists = np.linalg.norm(rel_targets, axis=1)
                        nearest_indices = np.argsort(dists)[: config.max_target_neighbors]
                        rel_targets = rel_targets[nearest_indices]
                    elif rel_targets.shape[0] < config.max_target_neighbors:
                        padding = np.zeros((config.max_target_neighbors - rel_targets.shape[0], 3))
                        rel_targets = np.vstack((rel_targets, padding))
                    obs.extend(rel_targets.flatten())

        return np.array(obs)


    def close(self) -> None:
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

    def reset(self):
        self.step_count = 0
        self.aviary.reset()

        for _ in range(10):
            self.aviary.step()


    def step(self, actions: np.ndarray):
        assert actions.shape == (self.num_agents, 4)

        self.aviary.set_all_setpoints(actions)
        
        for _ in range(4):
            self.aviary.step()

        self.update_info()

        self.step_count += 1
