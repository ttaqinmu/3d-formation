import gymnasium
from onpolicy.envs.quadcopter_formation import MultiQuadcopterFormation
import numpy as np


env = MultiQuadcopterFormation.from_json("train_1.json", 7, None)

env.reset()

total_reward = 0
num_env_steps = 0
episode_length = 1
n_rollout_threads = 1
n = 0


while env.agents:
    actions = {
        agent: np.insert(env.target_pos[i], 2, 0) for i, agent in enumerate(env.agents)
    }

    observations, rewards, terminations, infos = env.step(actions)

    total_reward += sum(rewards)
    n += 1

    if any(terminations):
        break

print(total_reward)
print(n)

env.close()
