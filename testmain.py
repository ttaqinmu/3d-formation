import gymnasium
from environment.multi_quadcopter_formation import MultiQuadcopterFormation
import numpy as np
import rerun as rr

env = MultiQuadcopterFormation.from_json("environment/formations/train_5.json")
observations, infos = env.reset()

rr.init("multi_quadcopter_formation", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
asset = rr.Asset3D(path="environment/data/drone.obj")

for i, agent in enumerate(env.agents):
    rr.log(
        f"world/agent/{agent}",
        asset,
        rr.Transform3D(
            quaternion=[1, 0, 0, 1],
            scale=[0.01, 0.01, 0.01],
            translation=env.start_pos[i],
        ),
    )

for i, target in enumerate(env.target_pos):
    rr.log(
        f"world/target/target_{i}",
        rr.Points3D(positions=target, radii=0.05),
    )


while env.agents:
    actions = {
        agent: np.insert(env.target_pos[i], 2, 0) for i, agent in enumerate(env.agents)
    }

    observations, rewards, terminations, truncations, infos = env.step(actions)

    for agent, obs in observations.items():
        rr.log(
            f"world/agent/{agent}",
            rr.Transform3D(
                quaternion=[1, 0, 0, 1], scale=[0.01, 0.01, 0.01], translation=obs[0:3]
            ),
        )


env.close()
