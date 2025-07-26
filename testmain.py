import numpy as np
from PyFlyt.core import Aviary

start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup
env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True, drone_type="quadx")

# set to position control
env.set_mode(0)

action = np.array([0, 0, 0, 0])

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    action += 1
    env.set_setpoint(0, action)
    env.step()