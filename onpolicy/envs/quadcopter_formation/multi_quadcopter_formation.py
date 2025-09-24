import numpy as np
import json
import pybullet as p
import os
from PyFlyt.pz_envs.quadx_envs.ma_quadx_base_env import MAQuadXBaseEnv
from PyFlyt.core.aviary import Aviary
from typing import List, Any, Optional
from gymnasium.spaces import Box, Space, Discrete
from random import randint, uniform
from .utils import minmax_scale, array_zfill


class MultiQuadcopterFormation(MAQuadXBaseEnv):
    metadata = {
        "name": "multi_quadcopter_formation_v0",
    }

    @classmethod
    def random_start_pos(cls, num_agents: int) -> np.ndarray:
        pos = []
        for _ in range(num_agents):
            x = uniform(-1, 1)
            y = uniform(-1, 1)
            pos.append([x, y, 0.1])
        return np.array(pos)

    @classmethod
    def random_target_pos(cls, num_targets: int) -> np.ndarray:
        pos = []
        for _ in range(num_targets):
            x = uniform(-1.5,1.5)
            y = uniform(-1.5,1.5)
            z = uniform(1,2)
            pos.append([x, y, z])
        return np.array(pos)

    @classmethod
    def from_json(
        cls,
        filename: str,
        control_mode: int=-1,
        default_render: Optional[str] = None
    ) -> "MultiQuadcopterFormation":
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"formations/{filename}")

        def base_parsing_pos(
            datas: List[List[float]],
        ) -> np.ndarray:
            return np.array(
                [([d[0], d[1], d[2]]) for d in datas if isinstance(d, list)]
            )

        data = json.load(open(file_path))
        return cls(
            num_targets=len(data.get("targets")),
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

    def masking_control(self):
        if self.control_mode in [8, 9, 10]:
            return 7
        return self.control_mode

    def __init__(
        self,
        num_targets: int = 10,
        control_mode: int = -1,
        targets_pos: Optional[np.ndarray] = None,
        starts_pos: Optional[np.ndarray] = None,
        max_target_neighbor: int = 4,
        max_agent_neighbor: int = 3,
        max_duration_seconds: int = 60,
        max_doom_size: float = 10.0,
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

        super().__init__(
            start_pos=self.start_pos,
            start_orn=np.zeros((self.start_pos.shape[0], 3)),
            flight_mode=self.masking_control(),
            flight_dome_size=max_doom_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation="euler",
            agent_hz=40,
            render_mode=render,
        )

        if control_mode == 9:
            self._action_space = Discrete(max_target_neighbor)
        elif control_mode == 8:
            self._action_space = Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float64)
        elif control_mode == 10:
            self._action_space = Discrete(4)

        self.max_distance = np.linalg.norm(
            np.full(3, self.max_doom_size) - np.full(3, -self.max_doom_size)
        )

        self.target_info = [
            dict(
                id=i,
                reached=False,
                distance_to_agent=np.full(num_agents, self.max_distance),
                nearest_agent_id=None,
            )
            for i in range(len(self.target_pos))
        ]

        self.n = num_agents
        num_obs = (
            3 * 4 +     # 12
            (max_target_neighbor * 3) +  # 12
            (max_agent_neighbor * 3) +  # 9
            max_target_neighbor # target flag   # 4
        )

        self.agent_info = [
            dict(
                id=i,
                reached_target=False,
                distance_to_agent=np.full(num_agents, self.max_distance),
                distance_to_target=np.full(num_targets, self.max_distance),
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
            num_agents * 3 +    # agent ang 30
            num_agents * 3 +    # agent ang vel 30
            num_agents * 4 +    # agent action
            num_targets * 3 +   # target pos 30
            num_targets         # target reached 10
        )
        self.share_observation_space = Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)

        self.file_dir = os.path.dirname(os.path.realpath(__file__))

        print(num_obs)

    def setup_gui(self):
        if self.render_mode is None:
            return

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
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
    
    def observation_space(self, agent: Any = None) -> Space:
        return self._observation_space

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

        self.aviary = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="quadx",
            render=bool(self.render_mode),
        )

        self.agent_info = [
            dict(
                id=i,
                reached_target=False,
                distance_to_target=np.full(len(self.target_pos), self.max_distance),
                distance_to_agent=np.full(self.num_agents, self.max_distance),
                nearest_target_id=None,
            )
            for i in range(self.n)
        ]

        self.target_info = [
            dict(
                id=i,
                reached=False,
                distance_to_agent=np.full(self.n, self.max_distance),
                nearest_agent_id=None,
            )
            for i in range(len(self.target_pos))
        ]

        self.setup_gui()

        self.aviary.register_all_new_bodies()
        self.aviary.set_mode(self.masking_control())

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
        - angular velocity 3 -> vp, vq, vr
        - angular pos 3 -> p, q, r
        - linear velocity (local) 3 -> u, v, w
        - linear position (global) 3 -> x,y,z
        - relative to closest target 3 * neighbors
        - relative to closest agent 3 * neighbors
        """

        raw_state = self.aviary.state(agent_id)
        # aux_state = self.aviary.aux_state(agent_id)

        # Basic state
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # Relative to target
        rel_pos_target = []
        acq_idx = []
        target_indexes = self.agent_info[agent_id]["distance_to_target"].argsort()
        for index in target_indexes:
            if len(rel_pos_target) >= self.max_target_neighbor:
                continue

            if self.target_info[index]["reached"]:
                acq_idx.append(1.0)
            else:
                acq_idx.append(0.0)

            rel_pos_target.append(self.target_pos[index] - lin_pos)

        rel_pos_target = array_zfill(
            np.array(rel_pos_target).flatten(),
            self.max_target_neighbor * 3,
            self.max_doom_size / 2,
        )

        # Relative to other agent
        rel_pos_agent = []
        agent_indexes = self.agent_info[agent_id]["distance_to_agent"].argsort()
        rel_pos_agent = []
        for index in agent_indexes:
            if (
                index != agent_id
                and len(rel_pos_agent) < self.max_agent_neighbor
            ):
                rel_pos_agent.append(self.aviary.state(index)[3] - lin_pos)

        rel_pos_agent = array_zfill(
            np.array(rel_pos_agent).flatten(),
            self.max_agent_neighbor * 3,
            self.max_doom_size / 2,
        )

        return np.concatenate(
            [
                ang_vel,
                ang_pos,
                lin_vel,
                lin_pos,
                rel_pos_target,  # 3 * max target neighbors
                rel_pos_agent,  # 3 * max agents neighbors
                acq_idx,
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
            ang_vel = raw_state[0]
            ang_pos = raw_state[1]
            lin_vel = raw_state[2]
            lin_pos = raw_state[3]
            obs.append(ang_vel)
            obs.append(ang_pos)
            obs.append(lin_vel)
            obs.append(lin_pos)

        obs.extend(self.target_pos)
        target_reached = [1.0 if t["reached"] else 0.0 for t in self.target_info]
        obs.append(np.array(target_reached))
        obs.append(np.concatenate(self.current_actions))

        result = np.concatenate(obs, axis=-1)
        return result

    def update_info(self) -> None:
        reached_target_id = []

        for agent_id in range(self.n):
            agent_pos = self.aviary.state(agent_id)[3]
            distance_to_target = []
            distance_to_agent = []

            min_dist = np.inf

            is_reach_target = False

            for target_index, target in enumerate(self.target_pos):
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
                    reached_target_id.append(target_index)

            for index in range(self.n):
                if index == agent_id:
                    distance_to_agent.append(self.max_distance)
                    continue

                other_pos = self.aviary.state(index)[3]
                dist = np.linalg.norm(agent_pos - other_pos)
                distance_to_agent.append(dist)

            self.agent_info[agent_id]["reached_target"] = is_reach_target
            self.agent_info[agent_id]["distance_to_agent"] = np.array(distance_to_agent)  # type: ignore
            self.agent_info[agent_id]["distance_to_target"] = np.array(distance_to_target)  # type: ignore

        for target_index in range(len(self.target_pos)):
            if target_index in reached_target_id:
                self.target_info[target_index]["reached"] = True
            #     p.changeVisualShape(
            #         self.target_visual[target_index],
            #         linkIndex=-1,
            #         rgbaColor=(0.5, 0.9, 0.3, 0.6),
            #     )
            # else:
            #     self.target_info[target_index]["reached"] = False
            #     p.changeVisualShape(
            #         self.target_visual[target_index],
            #         linkIndex=-1,
            #         rgbaColor=(1, 1, 1, 1),
            #     )

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
            "approach_same_target": False
        }

        # All targets reached
        if all(t["reached"] for t in self.target_info):
            reward += 10
            term |= True
            info["all_targets_reached"] = True

        # Target reached
        if self.agent_info[agent_id]["reached_target"]:
            reward += 2
            info["target_reached"] = True

        # Shared team reward
        # team_progress = np.mean([t["distance_to_target"] for t in self.agent_info]) # type: ignore
        # reward += 2.0 * (1.0 - np.tanh(team_progress))

        # Delta position
        d = self.agent_info[agent_id]["distance_to_target"].min()   # type: ignore
        info["closest_distance_to_target"] = d
        reward += 0.1 / d

        # Crowding
        distance_to_agent: list[int] = self.agent_info[agent_id]["distance_to_agent"]   # type: ignore
        count_crowd = sum([1 for d in distance_to_agent if d <= 0.5])
        if count_crowd > 0:
            info["crowding"] = True
            # reward -= count_crowd * 0.05

        # Approach same target as another agent
        nearest_target_id = self.agent_info[agent_id]["nearest_target_id"]
        if nearest_target_id is not None:
            if self.target_info[nearest_target_id]["nearest_agent_id"] != agent_id:
                # reward -= 0.2
                info["approach_same_target"] = True

        # Collision
        if np.any(self.aviary.contact_array[self.aviary.drones[agent_id].Id]):
            reward -= 0.2
            info["collision"] = True
            term |= True

        # Exceed flight dome
        if np.linalg.norm(self.aviary.state(agent_id)[-1]) > self.flight_dome_size:
            reward -= 0.5
            info["out_of_bounds"] = True
            term |= True

        # Time penalty
        # reward -= 0.01 * self.step_count

        return term, trunc, reward, info

    def masking_action(self, action: np.ndarray, agent_id) -> np.ndarray:
        if self.control_mode == 7:
            return np.clip(action, -self.max_doom_size, self.max_doom_size)
        elif self.control_mode == 9:
            index = action.argmax()
            return np.insert(self.target_pos[index], 2, 0)
        elif self.control_mode == 8:
            current_pos = self.aviary.state(agent_id)[3]
            current_pos += action
            return np.insert(current_pos, 2, 0)
        elif self.control_mode == 10:
            index = action.argmax()
            if index == 3:
                current_pos = self.aviary.state(agent_id)[3]
                return np.insert(current_pos, 2, 0)
            target_indexes = self.agent_info[agent_id]["distance_to_target"].argsort()[:self.max_target_neighbor]
            return np.insert(self.target_pos[target_indexes[index]], 2, 0) 
        else:
            return action

    def step(self, actions: Any):   # type: ignore
        if not isinstance(actions, dict):
            if len(actions) == 1:
                actions = np.array(actions[0])
            actions = {ag: self.masking_action(actions[i], i) for i, ag in enumerate(self.agents)}
        else:
            actions = {ag: self.masking_action(actions[ag], i) for i, ag in enumerate(self.agents)}

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

            self.update_info()

            share_obs = self.compute_share_observation()

        self.step_count += 1

        infos["global"] = {
            "agent": self.agent_info,
            "target": self.target_info,
        }

        # if self.step_count % 100 == 0:
        #     print(rewards)
        #     print(observations)
        #     print(self.agent_info)
        #     print(self.target_info)
        #
        #     print("----")

        # return observations, share_obs, rewards, terminations, infos

        return list(observations.values()), share_obs, list(rewards.values()), list(terminations.values()), infos, []

