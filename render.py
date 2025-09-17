#!/usr/bin/env python
import sys
import os
import wandb
import socket
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.quadcopter_formation import MultiQuadcopterFormation
from onpolicy.envs.env_wrappers import DummyVecEnv


def make_render_env(all_args):
    def get_env():
        filename = all_args.formation_filename
        # return MultiQuadcopterFormation.from_json(
        #     filename,
        #     all_args.control_mode,
        #     "human"
        # )
        return MultiQuadcopterFormation(
            num_targets=all_args.num_agents,
            render="human",
        )

    return DummyVecEnv([get_env])


def parse_args(args, parser):
    parser.add_argument('--formation_filename', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int,
                        default=10, help="number of players")
    parser.add_argument('--control_mode', type=int,
                        default=10, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    device = torch.device("cuda:0")
    torch.set_num_threads(all_args.n_training_threads)
    if all_args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.model_dir is None or all_args.model_dir == "":
        print("model_dir should be provided to load the pre-trained model")
        return

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_render_env(all_args)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    # if all_args.share_policy:
    from onpolicy.runner.shared.quadcopter_runner import QuadcopterRunner as Runner
    # else:
        # from onpolicy.runner.separated.quadcopter_runner import QuadcopterRunner as Runner

    runner = Runner(config)
    runner.render()


if __name__ == "__main__":
    main(sys.argv[1:])
