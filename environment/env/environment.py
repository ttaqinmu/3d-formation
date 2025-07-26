import numpy as np
from typing import Dict
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box


class Agent:
    pass


class Target:
    pass


class MultiQuadcopterFormation(ParallelEnv):
    metadata = {
        "name": "multi_quadcopter_formation_v0",
    }

    def __init__(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions: Dict[str, np.ndarray]):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent) -> Box:
        return self.action_spaces[agent]
