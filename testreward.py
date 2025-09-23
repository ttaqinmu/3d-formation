import gymnasium
from onpolicy.envs.quadcopter_formation import MultiQuadcopterFormation
import numpy as np
import time
import pybullet as p

def get_action(body_id, mode):
    pos, orn = p.getBasePositionAndOrientation(body_id)
    euler = p.getEulerFromQuaternion(orn)  # p,q,r
    lin_vel, ang_vel = p.getBaseVelocity(body_id)  # vx,vy,vz ; vp,vq,vr
    rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
    lin_vel_local = rot_mat.T @ np.array(lin_vel)  # u,v,w

    x, y, z = pos
    p_, q, r = euler
    vp, vq, vr = ang_vel
    u, v, w = lin_vel_local
    vx, vy, vz = lin_vel

    # switch mode
    if mode == 0: return [vp, vq, vr, 0] # failed
    elif mode == 1: return [p_, q, r, vz] # komponen z error
    elif mode == 2: return [vp, vq, vr, z]  # error
    elif mode == 3: return [p_, q, r, z]    # just ok
    elif mode == 4: return [u, v, vr, z]    # ok
    elif mode == 5: return [u, v, vr, vz]
    elif mode == 6: return [vx, vy, vr, vz]
    elif mode == 7: return [x, y, r, z]
    return [0,0,0,0]   # failed

# env = MultiQuadcopterFormation(
#     num_targets=5,
#     render="human",
#     control_mode=7,
# )

env = MultiQuadcopterFormation.from_json(
    filename="debug.json",
    control_mode=7,
    default_render=None
)

env.reset()

total_reward = 0
num_env_steps = 0
episode_length = 1
n_rollout_threads = 1
n = 0

acts = []


while env.agents:
    actions = {
        # agent: np.array([1.0, 0.0, 0.0, 0.01*n]) for i, agent in enumerate(env.agents)
        # agent: np.array([1, 0, 0]) for i, agent in enumerate(env.agents)
        # agent: np.array([1,0,0,0]) for i, agent in enumerate(env.agents)
        agent: np.insert(env.target_pos[i], 2, 0) for i, agent in enumerate(env.agents)
        # agent: np.array([1,0,0,1]) for i, agent in enumerate(env.agents)
    }

    observations, shared_obs, rewards, terminations, infos, [] = env.step(actions)

    _id = env.aviary.drones[0].Id

    acts.append(get_action(_id, 5))

    if n%100 == 0 or n == 0:
        # print("obs", infos["uav_0"])
        # print(env.aviary.state(0))
        print("motors: ", env.aviary.drones[0].motors.get_states())
        print("outside: ", acts[-1])

    # time.sleep(1)

    total_reward += sum(rewards)
    n += 1

    if n >= 500:
        break

print(total_reward)
print(n)

env.close()

########

env = MultiQuadcopterFormation.from_json(
    filename="debug.json",
    control_mode=5,
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
        agent: np.array(acts[n]) for i, agent in enumerate(env.agents)
        # agent: np.insert(env.target_pos[i], 2, 0) for i, agent in enumerate(env.agents)
        # agent: np.array([1,0,0,1]) for i, agent in enumerate(env.agents)
    }

    observations, shared_obs, rewards, terminations, infos, [] = env.step(actions)

    if n%100 == 0 or n == 0:
        # print("obs", infos["uav_0"])
        # print(env.aviary.state(0))
        # print("motos: ", env.aviary.drones[0].motors.get_states())
        print("act: ", acts[n])
        # print(get_action(_id, mode=5))
        # print("mean reward", np.array(rewards).mean())

    # time.sleep(1)

    total_reward += sum(rewards)
    n += 1

    if n >= 500:
        break

print(total_reward)
print(n)
