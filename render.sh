#!/bin/sh
env="MultiQuadcopter Formation"
scenario="random" 
num_agents=5
algo="rmappo"
exp="render_1"
seed_max=1
run_dir="sukses"
formation_filename="train_1.json"
control_mode=7
model_dir="/home/ttaqinmu/work/results/MultiQuadcopter/train_1/rmappo/mode_7_reward_2/wandb/run-20250921_064605-cbg0pihw/files"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python render.py --model_dir ${model_dir} --formation_filename ${formation_filename} --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --use_render \
    --control_mode ${control_mode} \
    --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --cuda True --use_wandb False \
    --run_dir ${run_dir} --hidden_size 256 --layer_N 2 --n_rollout_threads 1 --render_episode 1 \
    --episode_length 500
done
