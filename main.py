import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
from environment.env.config import ConfigJSON

config = ConfigJSON.from_json("environment/formations/train_1.json")

start_pos = np.array([[0.0, 0.0, 0.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

NUM_POINTS = 100
time_offsets = np.random.rand(NUM_POINTS)

rr.init("rerun_example_my_data", spawn=True)
rr.set_time("stable_time", duration=0)

# asset = rr.Asset3D(path="drone.obj")

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
rr.log("world/drone", rr.Points3D([[0,0,0]]))

for i in range(1000):
    time = i * 0.01
    times = np.repeat(time, NUM_POINTS) + time_offsets
    rr.set_time("stable_time", duration=time)
    
    pos = config.init_agent_pos[0]

    # rr.log("world/drone", rr.Transform3D(quaternion=[1, 0, 0, 1], scale=0.1, translation=pos))
