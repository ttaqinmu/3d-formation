import numpy as np
from new_environment import MultiQuadFormation, ExtraConfig, ActionEnum
import pybullet as p
from PyFlyt.core import Aviary

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
    if mode == 0: return [vp, vq, vr, 1] # failed
    elif mode == 1: return [p_, q, r, vz] # komponen z error
    elif mode == 2: return [vp, vq, vr, z]  # error
    elif mode == 3: return [p_, q, r, z]    # just ok
    elif mode == 4: return [u, v, vr, z]    # ok
    elif mode == 5: return [u, v, vr, vz]
    elif mode == 6: return [vx, vy, vr, vz]
    elif mode == 7: return [x, y, r, z]
    return [0,0,0,0]   # failed


def via_aviary():
    d = []

    env = Aviary(
        start_orn=np.array([[0,0,0]]),
        start_pos=np.array([[0,0,0]]),
        drone_type="quadx",
        render=True,
    )

    env.set_mode(7)

    for i in range(800):
        env.set_all_setpoints(np.array([[0.5,1,0,1]]))
        d.append(get_action(env.drones[0].Id, 5))

        env.step()

        # if i % 10 == 0:
            # print(d[-1])
        #     print("pwm: ",env.aviary.drones[0].pwm)
        #     print("motor", env.aviary.drones[0].motors.get_states())
            # print(env.aviary.state(0)[3])

    env.reset()
    env.set_mode(5)

    for i in range(800):
        env.set_all_setpoints(np.array([d[i]]))
        env.step()

        # if i % 10 == 0:
        #     print("input: ",d[i])
        #     print("pwm: ",env.aviary.drones[0].pwm)
        #     print("motor", env.aviary.drones[0].motors.get_states())

        # if i % 10 == 0:
            # print(env.aviary.state(0)[3])

def via_env():
    d = []

    env = MultiQuadFormation(
        start_positions=np.array([[0,0,0]]),
        target_positions=np.array([[0.5, 1, 1]]),
        render="human",
    )

    env.aviary.set_mode(7)

    for i in range(800):
        env.step(np.array([[0.5,1,0,1]]))
        d.append(get_action(env.aviary.drones[0].Id, 5))

    env.reset()
    env.aviary.set_mode(5)

    for i in range(800):
        env.step(np.array([d[i]]))

via_env()
