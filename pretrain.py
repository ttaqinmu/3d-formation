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


obs_components = [
    "vp", "vq", "vr",       # angular velocity
    "p", "q", "r",          # roll pitch yaw
    "u", "v", "w",          # local linear velocity
    "x", "y", "z",          # position global
    "ct_x1", "ct_y1", "ct_z1",  # closest target
    "ct_x2", "ct_y2", "ct_z2",
    "ct_x3", "ct_y3", "ct_z3",
    "ct_x4", "ct_y4", "ct_z4",
    "ca_x1", "ca_y1", "ca_z1",  # closest agent
    "ca_x2", "ca_y2", "ca_z2",
    "ca_x3", "ca_y3", "ca_z3",
    "acq_x1", "acq_y1", "acq_z1", "acq_x2",  # acquired target
]


data: Dict[str, List[Any]] = {
    "m4_a1": [],
    "m4_a2": [],
    "m4_a3": [],
    "m4_a4": [],
    "m5_a1": [],
    "m5_a2": [],
    "m5_a3": [],
    "m5_a4": [],
    "m6_a1": [],
    "m6_a2": [],
    "m6_a3": [],
    "m6_a4": [],
    "m7_a1": [],
    "m7_a2": [],
    "m7_a3": [],
    "m7_a4": [],
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


for comp in obs_components:
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
            cost_matrix[i, j] = np.linalg.norm(target_pos[j] - start_pos[i])

    _, target_index = linear_sum_assignment(cost_matrix)
    return target_index


def validate_strategy():
    env = MultiQuadcopterFormation(
        num_targets=5,
        control_mode=7,
        random_when_reset=True,
        render="human",
        max_target_neighbor=3,
        max_agent_neighbor=3,
    )

    obs, _, _, = env.reset()

    target_pos = env.target_pos
    start_pos = env.start_pos

    pairwise = assingment_strategy(start_pos, target_pos)

    print(pairwise)

    for step in range(1000):
        actions = {
            f"uav_{i}": np.insert(target_pos[pairwise[i]], 2, 0)
            for i in range(len(start_pos))
        }

        obs, _, _, _, _, _ = env.step(actions)

        # if step % 100 == 0:
            # print(obs[0])


def collect_data(outfile="dataset.csv"):
    max_step = 500
    max_episode = 1000

    for episode in range(max_episode):
        num_agents = randint(5,10)

        env = MultiQuadcopterFormation(
            num_targets=num_agents,
            control_mode=7,
            random_when_reset=True,
            render=None,
            max_target_neighbor=3,
            max_agent_neighbor=3,
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
                for o, comp in enumerate(obs_components):
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

                for am in [4,5,6,7]:
                    actions = get_action(body_id, am)
                    data[f"m{am}_a1"].append(actions[0])
                    data[f"m{am}_a2"].append(actions[1])
                    data[f"m{am}_a3"].append(actions[2])
                    data[f"m{am}_a4"].append(actions[3])


        print(f"Episode {episode+1}/{max_episode} collected.")

        env.close()

    df = pd.DataFrame(data)
    df.to_csv(outfile, index=False)


def train(args, infile="dataset.csv", outfile="pretrained_actor.pth", m=4, pretrain=None):
    def compute_loss(dist, mean, expert_actions, alpha=0.5, beta=0.5):
        mse_loss = torch.nn.MSELoss()(mean, expert_actions)
        log_probs_expert = dist.log_probs(expert_actions)
        logp = log_probs_expert.view(-1).mean()
        nll_loss = -1* logp
        total_loss = alpha * mse_loss + beta * nll_loss
        return total_loss, mse_loss, nll_loss


    df = pd.read_csv(infile, header=0, index_col=False)
    df['acq_x2'] = 0.5
    df["ct_x4"] = 5
    df["ct_y4"] = 5
    df["ct_z4"] = 5

    obs = Box(-np.inf, np.inf, shape=(len(obs_components),), dtype=np.float32)
    action = Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    dataset = TensorDataset(
        torch.tensor(df[obs_components].to_numpy(), dtype=torch.float32),
        torch.tensor(
            df[[f"m{m}_a1", f"m{m}_a2", f"m{m}_a3", f"m{m}_a4"]].to_numpy(), 
            dtype=torch.float32
        )
    )

    print(len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda")
    actor = R_Actor(args, obs, action, device)

    if pretrain:
        state_dict = torch.load(pretrain)
        actor.load_state_dict(state_dict)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        # ---- Training ----
        actor.train()
        total, total_mse, total_nll = 0, 0, 0
        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()
            
            i = batch_states.to(device)
            dist, _, _ = actor(
                obs=i, 
                rnn_states=[],
                masks=[], 
                available_actions=None, 
                deterministic=False,
                supervised=True
            )
            mean = dist.mean

            loss, mse, nll = compute_loss(dist, mean.to(device), batch_actions.to(device), 0.5, 0.5)

            loss.backward()
            optimizer.step()

            total += loss.item()
            total_mse += mse.item()
            total_nll += nll.item()

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

        print(f"[Epoch {epoch+1}] "
              f"Loss: {total/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f} | "
              f"MSE: {total_mse/len(train_loader):.4f}, "
              f"NLL: {total_nll/len(train_loader):.4f}")

    torch.save(actor.state_dict(), outfile)


def test(args, inputfile, m):
    state_dict = torch.load(inputfile)

    obs = Box(-np.inf, np.inf, shape=(len(obs_components),), dtype=np.float32)
    action = Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    # env = MultiQuadcopterFormation.from_json(
    #     "debug.json",
    #     m,
    #     "human"
    # )

    env = MultiQuadcopterFormation(
        num_targets=4,
        control_mode=m,
        render="human",
    )

    actor = R_Actor(args, obs, action, torch.device("cuda"))
    actor.load_state_dict(state_dict)
    actor.eval()

    pairwise = assingment_strategy(env.start_pos, env.target_pos)

    obs, _, _, = env.reset()

    rew = 0

    takeover = False
    
    with torch.no_grad():
        for step in range(5000):
            actions = {}

            if takeover:
                for i in range(env.num_agents):
                    actions[f"uav_{i}"] = np.insert(env.target_pos[pairwise[i]], 2, 0)
            else:
                for i in range(env.num_agents):
                    o = np.array(obs[i], dtype=np.float32)
                    o = torch.tensor(o).unsqueeze(0)
                    a, _, _ = actor(
                        obs=o, 
                        rnn_states=[],
                        masks=[], 
                        available_actions=None, 
                        deterministic=False,
                        supervised=True,
                    )
                    a = a.mode().squeeze(0).detach().to("cpu").numpy()
                    actions[f"uav_{i}"] = a

            # print(actions)
            obs, _, rews, _, infos, [] = env.step(actions)

            rew += rews[0]

            if step % 100 == 0:
                # print("target: ", env.target_pos)
                # print("obs: ", obs)
                print("output: ", actions["uav_0"])
                # print("infos: ", infos)

            # if step > 300 and not takeover:
                # print("takeover")
                # takeover = True
                # env.aviary.set_mode(7)

    print("Total reward: ", rew)

class Args:
    def __init__(self):
        self.use_feature_normalization = True
        self.use_orthogonal = True
        self.use_ReLU = True
        self.stacked_frames = 1
        self.layer_N = 2
        self.hidden_size = 128
        self.gain = 1
        self.use_policy_active_masks = False
        self.use_naive_recurrent_policy = False
        self.use_recurrent_policy = False
        self.recurrent_N = 1
        self.algorithm_name = "mappo"

# validate_strategy()
# collect_data("dataset_all_2_big.csv")
# train(Args(), "dataset_all_2_big.csv", "actor_m7.pth", 7, "actor.pt")
test(Args(), "actor.pt",7)
