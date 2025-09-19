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
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv

"""Train script for MPEs."""

def make_train_env(all_args):
    def get_env():
        filename = all_args.formation_filename
        
        if all_args.scenario_name == "random":
            return MultiQuadcopterFormation(
                num_targets=all_args.num_agents,
                control_mode=all_args.control_mode,
                random_when_reset=True,
            )

        return MultiQuadcopterFormation.from_json(
            filename=filename,
            control_mode=all_args.control_mode,
        )

    return ShareSubprocVecEnv([get_env for _ in range(all_args.n_training_threads)])


def make_eval_env(all_args):
    def get_env():
        return MultiQuadcopterFormation(
            num_targets=all_args.num_agents,
            control_mode=all_args.control_mode,
        )
    return ShareSubprocVecEnv([get_env for _ in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--formation_filename', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int,
                        default=10, help="number of players")
    parser.add_argument('--control_mode', type=int,
                        default=-1, help="control mode")

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

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    all_args.use_wandb = True

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    all_args.use_eval = False
    envs = make_train_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    # if all_args.share_policy:
    from onpolicy.runner.shared.quadcopter_runner import QuadcopterRunner as Runner
    # else:
        # from onpolicy.runner.separated.quadcopter_runner import QuadcopterRunner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
