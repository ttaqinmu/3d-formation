import gymnasium
from environment.multi_quadcopter_formation import MultiQuadcopterFormation
import numpy as np

env = MultiQuadcopterFormation.from_json("environment/formations/train_1.json")
observations, infos = env.reset()


def gen_action(agent_idx, target_pos):
    return np.insert(target_pos, 2, 0)


while env.agents:
    # this is where you would insert your policy
    actions = {
        agent: gen_action(i, env.target_pos[i]) for i, agent in enumerate(env.agents)
    }

    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
