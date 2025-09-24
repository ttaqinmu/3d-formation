#!/bin/sh
env="MultiQuadcopter Formation"
scenario="random" 
formation_filename="random"
num_agents=5
algo="mappo"
exp="render_1"
seed_max=1
run_dir="sukses"
control_mode=7
model_dir="/home/ttaqinmu/work/results/MultiQuadcopter/random/mappo/mode_7_reward_1/wandb/run-20250924_202707-kbul1apf/files"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    python render.py --model_dir ${model_dir} --formation_filename ${formation_filename} --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --use_render \
    --control_mode ${control_mode} \
    --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --cuda True --use_wandb False \
    --run_dir ${run_dir} --hidden_size 128 --layer_N 2 --n_rollout_threads 1 --render_episode 1 \
    --episode_length 1000
done
