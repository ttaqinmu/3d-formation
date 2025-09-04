import numpy as np
import json
from typing import Dict, List, Any, Sequence
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Space
from environment.env.models.aviary import Aviary
import environment.env.config

"""
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
"""


class MultiQuadcopterFormation(ParallelEnv):
    metadata = {
        "name": "multi_quadcopter_formation_v0",
    }

    @classmethod
    @classmethod
    def from_json(cls, json_path: str) -> "MultiQuadcopterFormation":
        def base_parsing_pos(datas: List[List[float]]) -> np.ndarray:
            return np.array(
                [
                    (
                        [d[0], d[1], d[2]]
                        if isinstance(d, list)
                        else [
                            np.random.uniform(-10, 10),
                            np.random.uniform(-10, 10),
                            np.random.uniform(0, 10),
                        ]
                    )
                    for d in datas
                    if isinstance(d, list)
                ]
            )

        data = json.load(open(json_path))
        assert len(data["agents"]) == len(data["targets"])
        return cls(
            start_pos=base_parsing_pos(data["agents"]),
            target_pos=base_parsing_pos(data["targets"]),
            max_duration_seconds=data.get("max_duration_seconds", 60),
            render=data.get("render", False),
        )

    def __init__(
        self,
        start_pos,
        target_pos,
        max_duration_seconds: int = 60,
        render: bool = False,
    ):
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.max_duration_seconds = max_duration_seconds
        self.is_render = render

        self.num_agents: int = len(start_pos)
        self.obs_size: int = 9 + (self.num_agents * 3) + ((self.num_agents - 1) * 3)

        self._init_action_space()
        self._init_observation_space()

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
        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float64
        )

    def reset(self, **kwargs):
        if hasattr(self, "aviary"):
            if isinstance(self.aviary, Aviary):  # type: ignore
                self.aviary.disconnect()  # type: ignore

        self.step_count = 0
        self.aviary = Aviary(
            start_pos=self.start_pos,
            start_orn=np.zeros((self.start_pos.shape[0], 3)),
            render=self.is_render,
        )

    def end_reset(
        self, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> None:
        self.aviary.register_all_new_bodies()

        self.aviary.set_mode(0)

        for _ in range(10):
            self.aviary.step()
        self.update_states()

    def update_states(self) -> None:
        pass

    def compute_attitude_by_id(
        self, agent_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """state.

        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - quaternion (vector of 4 values)
        """
        raw_state = self.aviary.state(agent_id)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quaternion angles
        quaternion = p.getQuaternionFromEuler(ang_pos)

        return ang_vel, ang_pos, lin_vel, lin_pos, quaternion

    def compute_observation_by_id(self, agent_id: int) -> Any:
        raise NotImplementedError

    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """compute_term_trunc_reward_info_by_id.

        Args:
            agent_id (int): agent_id

        Returns:
            Tuple[bool, bool, float, dict[str, Any]]:

        """
        raise NotImplementedError

    def step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """step.

        Args:
            actions (dict[str, np.ndarray]): actions

        Returns:
            tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:

        """
        # copy over the past actions
        self.past_actions = self.current_actions.copy()

        # set the new actions and send to aviary
        self.current_actions *= 0.0
        for k, v in actions.items():
            self.current_actions[self.agent_name_mapping[k]] = v
        self.aviary.set_all_setpoints(self.current_actions)

        # observation and rewards dictionary
        observations = dict()
        terminations = {k: False for k in self.agents}
        truncations = {k: False for k in self.agents}
        rewards = {k: 0.0 for k in self.agents}
        infos = {k: dict() for k in self.agents}

        # step enough times for one RL step
        for _ in range(self.env_step_ratio):
            self.aviary.step()
            self.update_states()

            # update reward, term, trunc, for each agent
            # TODO: make it so this doesn't have to be computed every aviary step
            for ag in self.agents:
                ag_id = self.agent_name_mapping[ag]

                # compute term trunc reward
                term, trunc, rew, info = self.compute_term_trunc_reward_info_by_id(
                    ag_id
                )
                terminations[ag] |= term
                truncations[ag] |= trunc
                rewards[ag] += rew
                infos[ag].update(info)

                # compute observations
                observations[ag] = self.compute_observation_by_id(ag_id)

        # increment step count and cull dead agents for the next round
        self.step_count += 1
        self.agents = [
            agent
            for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]

        return observations, rewards, terminations, truncations, infos

    def render(self) -> bool:
        return self.is_render

    def observation_space(self, agent_index: int) -> Space:
        return self.observation_spaces[agent_index]

    def action_space(self, agent_index: int) -> Box:  # type: ignore
        return self.action_spaces[agent_index]
