import numpy as np
import json
import pybullet as p
import os
from PyFlyt.pz_envs.quadx_envs.ma_quadx_base_env import MAQuadXBaseEnv
from PyFlyt.core.aviary import Aviary
from typing import List, Any, Optional
from gymnasium.spaces import Box, Space
from random import randint
from .utils import minmax_scale, array_zfill


class MultiQuadcopterFormation(MAQuadXBaseEnv):
    metadata = {
        "name": "multi_quadcopter_formation_v0",
    }

    @classmethod
    def random_start_pos(cls, num_agents: int) -> np.ndarray:
        pos = []
        for _ in range(num_agents):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            z = 0.1
            pos.append([x, y, z])
        return np.array(pos)

    @classmethod
    def random_target_pos(cls, num_targets: int) -> np.ndarray:
        pos = []
        for _ in range(num_targets):
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            z = np.random.uniform(1, 2)
            pos.append([x, y, z])
        return np.array(pos)

    @classmethod
    def from_json(cls, filename: str, control_mode: int=-1,default_render: Optional[str] = None) -> "MultiQuadcopterFormation":
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"formations/{filename}")

        def base_parsing_pos(
            datas: List[List[float]],
        ) -> np.ndarray:
            return np.array(
                [([d[0], d[1], d[2]]) for d in datas if isinstance(d, list)]
            )

        data = json.load(open(file_path))
        return cls(
            control_mode=control_mode,
            starts_pos=np.array([(
                base_parsing_pos(data["agents"]) if data.get("agents") else cls.random_start_pos(10)
            )]),
            targets_pos=np.array([(
                base_parsing_pos(data["targets"]) if data.get("targets") else cls.random_target_pos(10)
            )]),
            max_duration_seconds=data.get("max_duration_seconds", 60),
            render=data.get("render") if data.get("render") else default_render,
        )

    def get_random_pos(self):
        if len(self.targets_pos) == 1:
            return self.targets_pos[0], self.starts_pos[0]

        return (
            self.targets_pos[randint(0, len(self.targets_pos)-1)],
            self.starts_pos[randint(0, len(self.starts_pos)-1)],
        )

    def __init__(
        self,
        num_targets: int = 10,
        control_mode: int = -1,
        targets_pos: Optional[np.ndarray] = None,
        starts_pos: Optional[np.ndarray] = None,
        max_target_neighbor: int = 3,
        max_agent_neighbor: int = 3,
        max_duration_seconds: int = 60,
        max_doom_size: float = 30.0,
        render: Optional[str] = None,
        random_when_reset: bool = False, 
    ):
        self.control_mode = control_mode
        self.max_doom_size = max_doom_size
        self.max_target_neighbor = max_target_neighbor
        self.max_agent_neighbor = max_agent_neighbor

        if starts_pos is None:
            self.starts_pos = np.array([self.random_start_pos(num_targets)])
        else:
            self.starts_pos = starts_pos

        if targets_pos is None:
            self.targets_pos = np.array([self.random_target_pos(num_targets)])
        else:
            self.targets_pos = targets_pos

        self.target_pos, self.start_pos = self.get_random_pos()

        num_agents = self.start_pos.shape[0]

        self.random_when_reset = random_when_reset

        # assert self.target_pos.shape == self.start_pos.shape

        super().__init__(
            start_pos=self.start_pos,
            start_orn=np.zeros((self.start_pos.shape[0], 3)),
            flight_mode=control_mode,
            flight_dome_size=max_doom_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation="euler",
            agent_hz=40,
            render_mode=render,
        )

        self.sparse_reward = False

        self.target_info = [
            dict(
                id=i,
                reached=False,
                distance_to_agent=np.full(num_agents, 1),
                nearest_agent_id=None,
            )
            for i in range(len(self.target_pos))
        ]

        self.n = num_agents
        num_obs = 9 + (max_target_neighbor * 3) + (max_agent_neighbor * 3)

        self.agent_info = [
            dict(
                id=i,
                reached_target=False,
                distance_to_target=np.full(num_targets, 1),
                nearest_target_id=None,
            )
            for i in range(num_agents)
        ]

        self._observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_obs,),
            dtype=np.float64,
        )

        self.share_observation_space = []

        share_obs_dim = (
            num_agents * 3 +    # agent pos 30
            num_agents * 3 +    # agent vel 30
            num_agents * 3 +    # agent ang vel 30
            num_targets * 3 +   # target pos 30
            num_targets         # target reached 10
        )
        self.share_observation_space = Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)

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

    def reset(  # type: ignore
        self, seed=None, options=dict()
    ):
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

        self.step_count = 0
        self.agents = [
            "uav_" + str(r) for r in range(self.n)
        ]
        
        if self.random_when_reset:
            self.target_pos = self.random_target_pos(self.num_agents)
            self.start_pos = self.random_start_pos(self.num_agents)
        else:
            self.target_pos, self.start_pos = self.get_random_pos()

        # rebuild the environment
        self.aviary = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="quadx",
            render=bool(self.render_mode),
            drone_options=options,
            seed=seed,
        )

        self.agent_info = [
            dict(
                id=i,
                reached_target=False,
                distance_to_target=np.full(len(self.target_pos), np.inf),
                nearest_target_id=None,
            )
            for i in range(self.n)
        ]

        self.target_info = [
            dict(
                id=i,
                reached=False,
                distance_to_agent=np.full(self.n, np.inf),
                nearest_agent_id=None,
            )
            for i in range(len(self.target_pos))
        ]

        self.setup_gui()

        self.aviary.register_all_new_bodies()
        self.aviary.set_mode(self.control_mode)

        for _ in range(10):
            self.aviary.step()

        self.update_states()

        observations = [
            self.compute_observation_by_id(self.agent_name_mapping[ag])
            for ag in self.agents
        ]
        
        share_obs = self.compute_share_observation()

        return observations, share_obs, []

    def compute_observation_by_id(self, agent_id: int) -> np.ndarray:
        """
        State:
        aviary->state[agent_idx]
        - linear position (global) 3
        - angular velocity 3
        - linear velocity 3
        - relative to closest target 3 * neighbors
        - relative to closest agent 3 * neighbors
        """
        self.update_agent_info(agent_id)

        raw_state = self.aviary.state(agent_id)
        # aux_state = self.aviary.aux_state(agent_id)

        # Basic state
        lin_pos = raw_state[3]
        ang_vel = raw_state[0]
        lin_vel = raw_state[2]

        # Relative to target
        rel_pos_target = []
        target_indexes = self.agent_info[agent_id]["distance_to_target"].argsort()[:self.max_target_neighbor]
        for index in target_indexes:
            rel_pos_target.append(self.target_pos[index] - lin_pos)

        rel_pos_target = array_zfill(
            np.array(rel_pos_target).flatten(),
            self.max_target_neighbor * 3,
            self.max_doom_size,
        )

        # Relative to other agent
        rel_pos_agent = []
        for index in range(len(self.agents)):
            if index != agent_id:
                pos = self.aviary.state(index)
                dist = np.linalg.norm(pos[3] - raw_state[3])
                rel_pos_agent.append(dist)
            else:
                rel_pos_agent.append(np.inf)

        agent_indexes = np.array(rel_pos_agent).argsort()[:self.max_agent_neighbor]
        rel_pos_agent = []
        for index in agent_indexes:
            if index != agent_id:
                rel_pos_agent.append(self.aviary.state(index)[3] - raw_state[3])
            else:
                rel_pos_agent.append([self.max_doom_size, self.max_doom_size, self.max_doom_size])

        rel_pos_agent = array_zfill(
            np.array(rel_pos_agent).flatten(),
            self.max_target_neighbor * 3,
            self.max_doom_size,
        )

        return np.concatenate(
            [
                lin_pos,  # 3
                ang_vel,  # 3
                lin_vel,  # 3
                rel_pos_target,  # 3 * target
                rel_pos_agent,  # 3 * agents
                # self.past_actions[agent_id],
                # self.start_pos[agent_id],
            ],
            axis=-1,
        )

    def compute_share_observation(self) -> np.ndarray:
        """
        State:
        aviary->state[agent_idx]
        - linear position (global) 3 * num agent 30
        - angular velocity 3 * num agent 30
        - linear velocity 3 * num agent 30
        - target pos 3 * num target 30
        - target reach 1 * num target 10
        """
        obs = []
        for ag_id in range(self.n):
            raw_state = self.aviary.state(ag_id)
            lin_pos = raw_state[3]
            ang_vel = raw_state[0]
            lin_vel = raw_state[2]
            obs.append(lin_pos)
            obs.append(ang_vel)
            obs.append(lin_vel)

        obs.extend(self.target_pos)
        target_reached = [1.0 if t["reached"] else 0.0 for t in self.target_info]
        obs.append(np.array(target_reached))
        return np.concatenate(obs, axis=-1)

    def update_agent_info(self, agent_id: int) -> None:
        agent_pos = self.aviary.state(agent_id)[3]
        distance_to_target = []

        min_dist = np.inf

        is_reach_target = False

        for target_index, target in enumerate(self.target_pos):
            self.target_info[target_index]["reached"] = False
            dist = np.linalg.norm(agent_pos - target)

            distance_to_target.append(dist)
            self.target_info[target_index]["distance_to_agent"][agent_id] = dist  # type: ignore

            if dist < min_dist:
                min_dist = dist
                self.agent_info[agent_id]["nearest_target_id"] = target_index

            if dist <= self.target_info[target_index]["distance_to_agent"].min():  # type: ignore
                self.target_info[target_index]["nearest_agent_id"] = agent_id

            if (
                dist <= 0.1
                and self.target_info[target_index]["nearest_agent_id"] == agent_id
            ):
                is_reach_target = True
                self.target_info[target_index]["reached"] = True

        self.agent_info[agent_id]["reached_target"] = is_reach_target
        self.agent_info[agent_id]["distance_to_target"] = np.array(distance_to_target)  # type: ignore

    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Computes the termination, truncation, and reward of the current timestep.
        1. delta position to target
        2. obstacle avoidance
        3. resource consumption
        """
        # initialize
        reward = 0.0
        term = False
        trunc = self.step_count > self.max_steps
        info = {
            "all_targets_reached": False,
            "target_reached": False,
            "closest_distance_to_target": None,
            "crowding": False,
            "collision": False,
            "out_of_bounds": False,
        }

        # All targets reached
        if all(t["reached"] for t in self.target_info):
            reward += 100.0
            term |= True
            info["all_targets_reached"] = True

        # Target reached
        if self.agent_info[agent_id]["reached_target"]:
            reward += 30
            info["target_reached"] = True

        # Delta position
        d = self.agent_info[agent_id]["distance_to_target"].min()   # type: ignore
        info["closest_distance_to_target"] = d
        delta_position = 10.0 * np.exp(-d)
        reward += delta_position

        # Approach same target as another agent
        nearest_target_id = self.agent_info[agent_id]["nearest_target_id"]
        if nearest_target_id is not None:
            if self.target_info[nearest_target_id]["nearest_agent_id"] != agent_id:
                reward -= 5.0
                info["crowding"] = True

        # Collision
        if np.any(self.aviary.contact_array[self.aviary.drones[agent_id].Id]):
            reward -= 20.0
            info["collision"] = True

        # Exceed flight dome
        if np.linalg.norm(self.aviary.state(agent_id)[-1]) > self.flight_dome_size:
            reward -= 30.0
            info["out_of_bounds"] = True

        # Shared team reward
        team_progress = np.mean([t["distance_to_target"] for t in self.agent_info]) # type: ignore
        reward += 2.0 * (1.0 - np.tanh(team_progress))

        # Time penalty
        # reward -= 0.01 * self.step_count

        return term, trunc, reward, info

    def clip_control(self, action):
        if self.control_mode == 7:
            return np.clip(action, -30, 30)
        elif self.control_mode == -1:
            return np.clip(action, -1, 1)
        else:
            return action

    def step(self, actions: Any):   # type: ignore
        if not isinstance(actions, dict):
            if len(actions) == 1:
                actions = np.array(actions[0])
            actions = {ag: self.clip_control(actions[i]) for i, ag in enumerate(self.agents)}

        observations, share_obs, rewards, terminations, infos = self._step(actions)

        return list(observations.values()), share_obs, list(rewards.values()), list(terminations.values()), infos, []

    def _step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, Any],
        np.ndarray,
        dict[str, float],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
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
        share_obs = []

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

            share_obs = self.compute_share_observation()

        self.step_count += 1

        infos["global"] = {
            "agent": self.agent_info,
            "target": self.target_info,
        }

        return observations, share_obs, rewards, terminations, infos

    def render(self, conf):
        pass

