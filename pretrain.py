"""
get obs from reset

loop
    get action
    apply action
        compute reward
        compute obs
        compute share obs
    get obs reward



"""

import pandas as pd
import pybullet as p
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Any, Dict, List
from onpolicy.envs.quadcopter_formation.multi_quadcopter_formation import MultiQuadcopterFormation
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from random import randint
from gym.spaces import Box
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim


components = [
    "vp", "vq", "vr",       # angular velocity
    "p", "q", "r",          # roll pitch yaw
    "u", "v", "w",          # local linear velocity
    "x", "y", "z",          # position global
    "ca_x1", "ca_y1", "ca_z1",  # closest agent
    "ca_x2", "ca_y2", "ca_z2",
    "ca_x3", "ca_y3", "ca_z3",
    "ct_x1", "ct_y1", "ct_z1",  # closest target
    "ct_x2", "ct_y2", "ct_z2",
    "ct_x3", "ct_y3", "ct_z3",
]


data: Dict[str, List[Any]] = {
    "a_u": [],
    "a_v": [],
    "a_vr": [],
    "a_z": [],
    "num_agents": [],
    "start_pos": [],
    "target_pos": [],
    "target_reached": [],
    "crowding": [],
    "collision": [],
    "out_of_bounds": [],
    "approach_same_target": [],
    "reward": [],
}


for comp in components:
    data[comp] = []


def get_action(body_id, mode) -> List[float]:
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

def assingment_strategy(start_pos: np.ndarray, target_pos: np.ndarray) -> list:
    assert len(start_pos) == len(target_pos)

    n = len(start_pos)
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = np.linalg.norm(start_pos[i] - target_pos[j])

    _, target_index = linear_sum_assignment(cost_matrix)
    return target_index


def collect_data(outfile="dataset.csv"):
    action_mode = 4
    max_step = 1000
    max_episode = 100

    for episode in range(max_episode):
        num_agents = randint(5,10)

        env = MultiQuadcopterFormation(
            num_targets=num_agents,
            control_mode=7,
            random_when_reset=True,
            render=None
        )

        obs, _, _, = env.reset()

        target_pos = env.target_pos
        start_pos = env.start_pos

        pairwise = assingment_strategy(start_pos, target_pos)

        for step in range(max_step):
            actions = {
                f"uav_{i}": np.insert(target_pos[pairwise[i]], 2, 0)
                for i in range(len(start_pos))
            }

            for i in range(num_agents):
                for o, comp in enumerate(components):
                    data[comp].append(obs[i][o])

                data["num_agents"].append(num_agents)
                data["start_pos"].append(start_pos[i])
                data["target_pos"].append(target_pos[pairwise[i]])

            obs, _, rews, _, infos, [] = env.step(actions)

            for i in range(num_agents):
                data["target_reached"].append(infos[f"uav_{i}"]['target_reached'])
                data["crowding"].append(infos[f"uav_{i}"]['crowding'])
                data["collision"].append(infos[f"uav_{i}"]['collision'])
                data["out_of_bounds"].append(infos[f"uav_{i}"]['out_of_bounds'])
                data["approach_same_target"].append(infos[f"uav_{i}"]['approach_same_target'])
                data["reward"].append(rews[i])

                body_id = env.aviary.drones[i].Id
                actions = get_action(body_id, action_mode)
                data["a_u"].append(actions[0])
                data["a_v"].append(actions[1])
                data["a_vr"].append(actions[2])
                data["a_z"].append(actions[3])


        print(f"Episode {episode+1}/{max_episode} collected.")

        env.close()

    df = pd.DataFrame(data)
    df.to_csv(outfile, index=False)


def train(args, infile="dataset.csv", outfile="pretrained_actor.pth"):
    df = pd.read_csv(infile, header=0, index_col=False)

    obs = Box(-np.inf, np.inf, shape=(len(components),), dtype=np.float32)
    action = Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    dataset = TensorDataset(
        torch.tensor(df[components].to_numpy(), dtype=torch.float32),
        torch.tensor(df[["x", "y", "r", "z"]].to_numpy(), dtype=torch.float32)
    )

    print(len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda")
    actor = R_Actor(args, obs, action, device)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    epochs = 20
    total_loss = 0
    for epoch in range(epochs):
        # ---- Training ----
        actor.train()
        total_loss = 0
        for batch_states, batch_actions in train_loader:
            i = batch_states.to(device)
            a, _, _ = actor(i, [], [], None, True)

            o = batch_actions.to(device)
            loss = criterion(a, o)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)

        # ---- Validation ----
        actor.eval()
        val_loss = 0
        with torch.no_grad():
            for val_states, val_actions in val_loader:
                i = val_states.to(device)
                a, _, _ = actor(i, [], [], None, True)
                
                o = val_actions.to(device)
                loss = criterion(a, o)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              )

    torch.save(actor.state_dict(), outfile)


def test(args, inputfile):
    state_dict = torch.load(inputfile)

    obs = Box(-np.inf, np.inf, shape=(len(components),), dtype=np.float32)
    action = Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    env = MultiQuadcopterFormation(
        num_targets=5,
        control_mode=7,
        render="human",
    )

    actor = R_Actor(args, obs, action, torch.device("cuda"))
    actor.eval()

    obs, _, _, = env.reset()
    
    with torch.no_grad():
        for step in range(500):
            actions = {}
            for i in range(env.num_agents):
                o = np.array(obs[i], dtype=np.float32)
                o = torch.tensor(o).unsqueeze(0)
                a, _, _ = actor(o, [], [], None, True)
                a = a.squeeze(0).detach().to("cpu").numpy()
                actions[f"uav_{i}"] = a * 100

                for o, comp in enumerate(components):
                    data[comp].append(obs[i][o])

            if step % 10 == 0:
                print("output: ", actions["uav_0"])
                print("real: ", data["x"][-1], data["y"][-1], data["r"][-1], data["z"][-1])

            # print(actions)
            obs, _, rews, _, infos, [] = env.step(actions)


class Args:
    def __init__(self):
        self.use_feature_normalization = True
        self.use_orthogonal = True
        self.use_ReLU = True
        self.stacked_frames = 1
        self.layer_N = 2
        self.hidden_size = 256
        self.gain = 0.01
        self.use_policy_active_masks = False
        self.use_naive_recurrent_policy = False
        self.use_recurrent_policy = False
        self.recurrent_N = 1
        self.algorithm_name = "rmappo"

# collect_data("dataset_1.csv")
# train(Args(), "dataset_1.csv", "actor_1.pth")
test(Args(), "actor_1.pth")
