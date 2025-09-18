#!/bin/sh
env="MultiQuadcopter Formation"
scenario="random" 
num_agents=5
algo="rmappo"
exp="mode_7_shared"
seed_max=1
formation_filename="random"
control_mode=7
# model_dir="/home/ttaqinmu/work/results/MultiQuadcopter/train_1/rmappo/test_1_mode_7/wandb/run-20250917_153226-l42xxjk9/files"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train.py --formation_filename ${formation_filename} --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --seed ${seed} \
    --n_training_threads 12 --n_rollout_threads 12 --num_mini_batch 1 --share_policy --episode_length 250 \
    --num_env_steps 2000000 --ppo_epoch 5 --use_ReLU --gain 0.01 --lr 7e-4 --entropy_coef 0.01 --critic_lr 5e-4 \
    --hidden_size 256 --layer_N 2 \
    --save_interval 1 \
    --use_eval --eval_interval 100 --n_eval_rollout_threads 2 --eval_episodes 300 \
    --control_mode ${control_mode} \
    --user_name "imammuttaqin98-universitas-gadjah-mada-library" --cuda True
    # --model_dir ${model_dir}
done
