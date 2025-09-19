import gymnasium
from onpolicy.envs.quadcopter_formation import MultiQuadcopterFormation
import numpy as np
import time

# env = MultiQuadcopterFormation(
#     num_targets=3,
#     render="human",
#     control_mode=1,
# )

env = MultiQuadcopterFormation.from_json(
    filename="basic_1.json",
    control_mode=7,
    default_render="human"
)

env.reset()

total_reward = 0
num_env_steps = 0
episode_length = 1
n_rollout_threads = 1
n = 0


while env.agents:
    actions = {
        # agent: np.array([1.0, 0.0, 0.0, 0.01*n]) for i, agent in enumerate(env.agents)
        # agent: np.array([1, 0, 0]) for i, agent in enumerate(env.agents)
        agent: np.insert(env.target_pos[i], 2, 0) for i, agent in enumerate(env.agents)
    }

    observations, shared_obs, rewards, terminations, infos, [] = env.step(actions)
    
    if n%100 == 0:
        print("obs", infos["uav_0"])
        # print("mean reward", np.array(rewards).mean())

    # time.sleep(1)

    total_reward += sum(rewards)
    n += 1

    if any(terminations) or n >= 250:
        break

print(total_reward)
print(n)

env.close()
