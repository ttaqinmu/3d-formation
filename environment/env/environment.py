import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Space
from models.aviary import Aviary


@dataclass
class Config:
    init_agent_pos: np.ndarray
    init_target_pos: np.ndarray
    max_duration_seconds: int
    render: bool

    @classmethod
    def base_parsing_pos(cls, data: List[List[float]]) -> np.ndarray:
        return np.array([
            [d[0], d[1], d[2]]
            if isinstance(d, list)
            else [
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10),
                np.random.uniform(0, 10)
            ]
            for d in data if isinstance(d, list)
        ])

    @classmethod
    def from_json(cls, json_str: str) -> "Config":
        data = json.loads(json_str)
        
        assert len(data["agents"]) == len(data["targets"])

        return cls(
            init_agent_pos=cls.base_parsing_pos(data["agents"]),
            init_target_pos=cls.base_parsing_pos(data["targets"]),
            max_duration_seconds=data.get("max_duration_seconds", 60),
            render=data.get("render", False)
        )

'''
State:
aviary->state[agent_idx]
- linear position (global) [3]
- angular velocity [0]
- linear velocity [2]
- relative to target
- relative to other agent

Action:
- pwm motor 1
- pwm motor 2
- pwm motor 3
- pwm motor 4
'''


class MultiQuadcopterFormation(ParallelEnv):
    metadata = {
        "name": "multi_quadcopter_formation_v0",
    }

    def __init__(self, config: Config):
        self.config: Config = config

        self.num_agents: int = len(config.init_agent_pos)
        self.obs_size: int = 9 + (self.num_agents * 3) + ((self.num_agents-1) * 3)

        self._init_action_space()


    def _init_action_space(self):
        angular_rate_limit = np.pi
        thrust_limit = 0.8
        high = np.array(
            [
                angular_rate_limit,
                angular_rate_limit,
                angular_rate_limit,
                thrust_limit,
            ]
        )
        low = np.array(
            [
                -angular_rate_limit,
                -angular_rate_limit,
                -angular_rate_limit,
                0.0,
            ]
        )
        self._action_space = Box(low=low, high=high, dtype=np.float64)

    def _init_observation_space(self):
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float64)

    def reset(self, seed=None, options=None):
        if hasattr(self, "aviary"):
            if isinstance(self.aviary, Aviary): # type: ignore
                self.aviary.disconnect() # type: ignore

        self.step_count = 0
        self.aviary = Aviary(
            start_pos=self.config.init_agent_pos,
            start_orn=np.zeros((self.config.init_agent_pos.shape[0], 3)),
        )

    def step(self, actions: Dict[str, np.ndarray]):
        pass

    def render(self):
        pass

    def observation_space(self, agent_index: int) -> Space:
        return self.observation_spaces[agent_index]

    def action_space(self, agent_index: int) -> Box:    # type: ignore
        return self.action_spaces[agent_index]
