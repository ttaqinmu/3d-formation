import time
from warnings import warn

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
from typing import Union

from .quadcopter import Quadcopter


class Aviary(bullet_client.BulletClient):

    def __init__(
        self,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        render: bool = False,
    ):
        super().__init__(p.GUI if render else p.DIRECT)

        assert (
            len(start_pos.shape) == 2
        ), f"start_pos must be shape (n, 3), currently {start_pos.shape}."
        assert (
            start_pos.shape[-1] == 3
        ), f"start_pos must be shape (n, 3), currently {start_pos.shape}."
        assert (
            start_orn.shape == start_pos.shape
        ), f"start_orn must be same shape as start_pos, currently {start_orn.shape}."

        world_scale = 1.0
        
        physics_hz = 240
        
        # constants
        self.num_drones = start_pos.shape[0]
        self.start_pos = start_pos
        self.start_orn = start_orn

        self.physics_hz = physics_hz
        self.physics_period = 1.0 / physics_hz

        # set the world scale and directories
        self.world_scale = world_scale
        self.setAdditionalSearchPath(pybullet_data.getDataPath())

        # render
        self.render = render
        self.rtf_debug_line = self.addUserDebugText(
            text="RTF here", textPosition=[0, 0, 0], textColorRGB=[1, 0, 0]
        )


    def reset(self, seed: None | int = None):
        self.resetSimulation()
        self.setGravity(0, 0, -9.81)
        self.physics_steps: int = 0
        self.aviary_steps: int = 0
        self.elapsed_time: float = 0

        # reset the camera position to a sane place
        self.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1],
        )

        # define new RNG
        self.np_random = np.random.RandomState(seed=seed)

        # construct the world
        self.planeId = self.loadURDF(
            "plane.urdf", useFixedBase=True, globalScaling=self.world_scale
        )

        # spawn drones
        self.drones: list[Quadcopter] = []
        for start_pos, start_orn in zip(
            self.start_pos,
            self.start_orn,
        ):
            self.drones.append(
                Quadcopter(
                    self,
                    start_pos=start_pos,
                    start_orn=start_orn,
                    physics_hz=self.physics_hz,
                    np_random=self.np_random,
                )
            )

        all_control_hz = [int(1.0 / drone.control_period) for drone in self.drones]
        self.updates_per_step = int(self.physics_hz / np.min(all_control_hz))
        self.update_period = 1.0 / np.min(all_control_hz)

        if len(all_control_hz) > 0:
            all_control_hz.sort()
            all_ratios = np.array(all_control_hz)[1:] / np.array(all_control_hz)[:-1]
            assert all(
                r % 1.0 == 0.0 for r in all_ratios
            ), "Looprates must form common multiples of each other."

        self.now = time.time()
        self._frame_elapsed = 0.0
        self._sim_elapsed = 0.0

        self.register_all_new_bodies()
        self.set_armed(True)

        [drone.reset() for drone in self.drones]
        [drone.update_state() for drone in self.drones]
        [drone.update_last() for drone in self.drones]

    def register_all_new_bodies(self):
        # the collision array is a scipy sparse, upper triangle array
        num_bodies = (
            np.max([self.getBodyUniqueId(i) for i in range(self.getNumBodies())]) + 1
        )
        self.contact_array = np.zeros((num_bodies, num_bodies), dtype=bool)

    def state(self, index: int) -> np.ndarray:
        """Returns the state for the indexed drone.

        This is a (4, 3) array, where:
            - `state[0, :]` represents body frame angular velocity
            - `state[1, :]` represents ground frame angular position
            - `state[2, :]` represents body frame linear velocity
            - `state[3, :]` represents ground frame linear position

        Args:
            index (DRONE_INDEX): index

        Returns:
            np.ndarray: state
        """
        return self.drones[index].state

    def aux_state(self, index: int) -> np.ndarray:
        """Returns the auxiliary state for the indexed drone.

        This is typically an (n, ) vector, representing various attributes such as:
            - booster thrust settings
            - fuel remaining
            - control surfaces deflection magnitude
            - etc...

        Args:
            index (DRONE_INDEX): index

        Returns:
            np.ndarray: auxiliary state
        """
        return self.drones[index].aux_state

    @property
    def all_states(self) -> list[np.ndarray]:
        """Returns a list of states for all drones in the environment.

        This is a `num_drones` list of (4, 3) arrays, where each element in the list corresponds to the i-th drone state.

        Similar to the `state` property, the states contain information corresponding to:
            - `state[0, :]` represents body frame angular velocity
            - `state[1, :]` represents ground frame angular position
            - `state[2, :]` represents body frame linear velocity
            - `state[3, :]` represents ground frame linear position

        This function is not very optimized, if you want the state of a single drone, do `state(i)`.

        Returns:
            np.ndarray: list of states
        """
        states = []
        for drone in self.drones:
            states.append(drone.state)

        return states

    @property
    def all_aux_states(self) -> list[np.ndarray]:
        aux_states = []
        for drone in self.drones:
            aux_states.append(drone.aux_state)

        return aux_states

    def print_all_bodies(self):
        bodies = dict()
        for i in range(self.getNumBodies()):
            bodies[i] = self.getBodyInfo(i)[-1].decode("UTF-8")

        from pprint import pprint

        pprint(bodies)

    def set_mode(self, flight_modes: int | list[int]):
        if isinstance(flight_modes, list):
            assert len(flight_modes) == len(
                self.drones
            ), f"Expected {len(self.drones)} flight_modes, got {len(flight_modes)}."
            for drone, mode in zip(self.drones, flight_modes):
                drone.set_mode(mode)
        else:
            for drone in self.drones:
                drone.set_mode(flight_modes)

    def set_setpoint(self, index: int, setpoint: np.ndarray):
        """Sets the setpoint of one drone in the environment.

        Args:
            index (DRONE_INDEX): index
            setpoint (np.ndarray): setpoint
        """
        self.drones[index].setpoint = setpoint

    def set_all_setpoints(self, setpoints: np.ndarray):
        """Sets the setpoints of each drone in the environment.

        Args:
            setpoints (np.ndarray): list of setpoints
        """
        for i, drone in enumerate(self.drones):
            drone.setpoint = setpoints[i]

    def step(self):
        # compute rtf if we're rendering
        if self.render:
            elapsed = time.time() - self.now
            self.now = time.time()

            self._sim_elapsed += self.update_period * self.updates_per_step
            self._frame_elapsed += elapsed

            time.sleep(max(self._sim_elapsed - self._frame_elapsed, 0.0))

            # print RTF every 0.5 seconds, this actually adds considerable overhead
            if self._frame_elapsed >= 0.5:
                # calculate real time factor based on realtime/simtime
                RTF = self._sim_elapsed / (self._frame_elapsed + 1e-6)
                self._sim_elapsed = 0.0
                self._frame_elapsed = 0.0

                self.rtf_debug_line = self.addUserDebugText(
                    text=f"RTF: {RTF:.3f}",
                    textPosition=[0, 0, 0],
                    textColorRGB=[1, 0, 0],
                    replaceItemUniqueId=self.rtf_debug_line,
                )

        # reset collisions
        self.contact_array &= False

        # step the environment enough times for one control loop of the slowest controller
        for step in range(self.updates_per_step):
            # update onboard avionics conditionally
            [
                drone.update_control()
                for drone in self.drones
                if step % drone.physics_control_ratio == 0
            ]

            # update physics and state
            [drone.update_physics() for drone in self.drones]
            [drone.update_state() for drone in self.drones]

            # advance pybullet
            self.stepSimulation()

            # splice out collisions
            for collision in self.getContactPoints():
                self.contact_array[collision[1], collision[2]] = True
                self.contact_array[collision[2], collision[1]] = True

            # increment the number of physics steps
            self.physics_steps += 1
            self.elapsed_time = self.physics_steps / self.physics_hz

        # update the last components of the drones, this is usually limited to cameras only
        [drone.update_last() for drone in self.drones]

        self.aviary_steps += 1
