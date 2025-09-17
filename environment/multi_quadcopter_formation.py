import numpy as np
import json
import pybullet as p
import os
from PyFlyt.pz_envs.quadx_envs.ma_quadx_base_env import MAQuadXBaseEnv
from PyFlyt.core.aviary import Aviary
from typing import List, Any, Optional
from gymnasium.spaces import Box, Space


class MultiQuadcopterFormation(MAQuadXBaseEnv):
    metadata = {
        "name": "multi_quadcopter_formation_v0",
    }

    @classmethod
    def random_start_pos(cls, num_agents: int) -> np.ndarray:
        print(num_agents)
        pos = []
        for _ in range(num_agents):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            z = np.random.uniform(0, 0.1)
            pos.append([x, y, z])
        return np.array(pos)

    @classmethod
    def random_target_pos(cls, num_targets: int) -> np.ndarray:
        pos = []
        for _ in range(num_targets):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            z = np.random.uniform(2, 3)
            pos.append([x, y, z])
        return np.array(pos)

    @classmethod
    def from_json(cls, json_path: str) -> "MultiQuadcopterFormation":
        def base_parsing_pos(
            datas: List[List[float]],
        ) -> np.ndarray:
            return np.array(
                [([d[0], d[1], d[2]]) for d in datas if isinstance(d, list)]
            )

        data = json.load(open(json_path))
        return cls(
            start_pos=(
                base_parsing_pos(data["agents"]) if data.get("agents") else None
            ),
            target_pos=(
                base_parsing_pos(data["targets"]) if data.get("targets") else None
            ),
            max_duration_seconds=data.get("max_duration_seconds", 60),
            render=data.get("render", False),
        )

    def __init__(
        self,
        num_targets: int = 10,
        target_pos: Optional[np.ndarray] = None,
        start_pos: Optional[np.ndarray] = None,
        max_duration_seconds: int = 60,
        render: bool = False,
    ):
        if start_pos is None:
            start_pos = self.random_start_pos(num_targets)

        if target_pos is None:
            target_pos = self.random_target_pos(num_targets)

        assert start_pos.shape == target_pos.shape

        super().__init__(
            start_pos=start_pos,
            start_orn=np.zeros((start_pos.shape[0], 3)),
            flight_mode=7,
            flight_dome_size=30,
            max_duration_seconds=max_duration_seconds,
            angle_representation="euler",
            agent_hz=40,
            render_mode="human" if render else None,
        )

        self.sparse_reward = False

        self.target_pos = target_pos
        self.target_info = [
            dict(
                id=i,
                reached=False,
                distance_to_agent=np.full(len(self.start_pos), np.inf),
                nearest_agent_id=None,
            )
            for i in range(len(target_pos))
        ]

        num_agents = start_pos.shape[0]
        self.agent_info = [
            dict(
                id=i,
                reached_target=False,
                distance_to_target=np.full(num_agents, np.inf),
                nearest_target_id=None,
            )
            for i in range(num_agents)
        ]

        self._observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(9 + (num_agents * 3) + ((num_agents - 1) * 3),),
            dtype=np.float64,
        )

        self.file_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_gui(self):
        if self.render_mode is None:
            return

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        floor_col = p.createCollisionShape(p.GEOM_PLANE)
        floor_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[50, 50, 0.01],
            rgbaColor=[0.60, 0.69, 0.84, 1.0],
            specularColor=[0, 0, 0],
        )
        floor = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_col,
            baseVisualShapeIndex=floor_vis,
            basePosition=[0, 0, 0],
        )

        targ_obj_dir = os.path.join(self.file_dir, "data/target.urdf")
        self.target_visual = []
        for target in self.target_pos:
            self.target_visual.append(
                p.loadURDF(
                    targ_obj_dir,
                    basePosition=target,
                    useFixedBase=True,
                    globalScaling=0.05,
                )
            )

        for i, visual in enumerate(self.target_visual):
            p.changeVisualShape(
                visual,
                linkIndex=-1,
                rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
            )

    def observation_space(self, agent: Any = None) -> Space:
        return self._observation_space

    def update_states(self) -> None:
        # TODO: update aux_state
        pass

    def reset(
        self, seed=None, options=dict()
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

        self.step_count = 0
        self.agents = self.possible_agents[:]

        # rebuild the environment
        self.aviary = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="quadx",
            render=bool(self.render_mode),
            drone_options=options,
            seed=seed,
        )

        self.setup_gui()

        self.aviary.register_all_new_bodies()

        self.aviary.set_mode(7)
        # self.aviary.drones[1].set_mode(7)
        # self.aviary.drones[0].set_mode(7)
        # setpoint = np.array([0.0, 0.0, 0.0, 1.5])
        # self.aviary.set_setpoint(0, self.target_pos[0])

        for _ in range(10):
            self.aviary.step()

        self.update_states()

        observations = {
            ag: self.compute_observation_by_id(self.agent_name_mapping[ag])
            for ag in self.agents
        }
        infos = {ag: dict() for ag in self.agents}
        return observations, infos

    def compute_observation_by_id(self, agent_id: int) -> np.ndarray:
        """
        State:
        aviary->state[agent_idx]
        - linear position (global) [3]
        - angular velocity [0]
        - linear velocity [2]
        - relative to target
        - relative to other agent
        """

        raw_state = self.aviary.state(agent_id)
        # aux_state = self.aviary.aux_state(agent_id)

        # Basic state
        lin_pos = raw_state[3]
        ang_vel = raw_state[0]
        lin_vel = raw_state[2]

        # Relative to target
        rel_pos_target = (self.target_pos - lin_pos).flatten()

        # Relative to other agent
        rel_pos_agent = []
        for i in range(len(self.agents)):
            if i != agent_id:
                rel_pos_agent.append(self.aviary.state(i)[3] - lin_pos)

        rel_pos_agent = np.array(rel_pos_agent).flatten()

        return np.concatenate(
            [
                lin_pos,  # 3
                ang_vel,  # 3
                lin_vel,  # 3
                rel_pos_target,  # 3 * num_agents
                rel_pos_agent,  # 3 * num_agents - 1
                # self.past_actions[agent_id],
                # self.start_pos[agent_id],
            ],
            axis=-1,
        )

    def update_agent_info(self, agent_id: int) -> None:
        agent_pos = self.aviary.state(agent_id)[3]
        distance_to_target = []

        min_dist = np.inf

        for target_index, target in enumerate(self.target_pos):
            dist = np.linalg.norm(agent_pos - target)

            if dist <= 0.01:
                self.agent_info[agent_id]["reached"] = True
                self.target_info[target_index]["reached"] = True

            distance_to_target.append(dist)
            self.target_info[target_index]["distance_to_agent"][agent_id] = dist  # type: ignore

            if dist < min_dist:
                min_dist = dist
                self.agent_info[agent_id]["nearest_target_id"] = target_index

            if dist <= self.target_info[target_index]["distance_to_agent"].min():  # type: ignore
                self.target_info[target_index]["nearest_agent_id"] = agent_id

        self.agent_info[agent_id]["distance_to_target"] = np.array(distance_to_target)  # type: ignore

    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Computes the termination, truncation, and reward of the current timestep.
        1. delta position to target
        2. obstacle avoidance
        3. resource consumption
        """
        reward = 0.0
        term = False
        trunc = self.step_count > self.max_steps
        info = dict()

        self.update_agent_info(agent_id)

        # All targets reached
        if all(t["reached"] for t in self.target_info):
            reward += 100.0
            term |= True
            info["all_targets_reached"] = True

        # Delta position
        reward += 10.0 * (1.0 - np.tanh(self.agent_info[agent_id]["distance_to_target"].min()))  # type: ignore

        # Approach same target as another agent
        nearest_target_id = self.agent_info[agent_id]["nearest_target_id"]
        if nearest_target_id is not None:
            if self.target_info[nearest_target_id]["nearest_agent_id"] != agent_id:
                reward -= 5.0
                info["crowding"] = True

        # Collision
        if np.any(self.aviary.contact_array[self.aviary.drones[agent_id].Id]):
            reward -= 10.0
            info["collision"] = True

        # Exceed flight dome
        if np.linalg.norm(self.aviary.state(agent_id)[-1]) > self.flight_dome_size:
            reward -= 20.0
            info["out_of_bounds"] = True
            term |= True

        return term, trunc, reward, info
