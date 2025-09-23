#!/bin/sh
seed_max=1
env="MultiQuadcopter Formation"
algo="mappo"
num_agents=5
scenario="random" 
formation_filename="random"
exp="mode_7_reward_1"
control_mode=5
# model_dir="/home/ttaqinmu/work/results/MultiQuadcopter/train_4/mappo/mode_7_reward_3/wandb/run-20250922_152543-mvmakvxh/files"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train.py --formation_filename ${formation_filename} --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --seed ${seed} \
    --n_training_threads 12 --n_rollout_threads 12 --num_mini_batch 2 --share_policy --episode_length 250 \
    --num_env_steps 5000000 --ppo_epoch 5 --use_ReLU --gain 1 --lr 1e-4 --entropy_coef 0.01 --critic_lr 3e-4 \
    --hidden_size 128 --layer_N 2 \
    --save_interval 1 \
    --use_eval --eval_interval 100 --n_eval_rollout_threads 2 --eval_episodes 300 \
    --control_mode ${control_mode} \
    --user_name "imammuttaqin98-universitas-gadjah-mada-library" --cuda True
    # --model_dir ${model_dir}
done
