#!/bin/sh
env="MultiQuadcopter Formation"
scenario="auto_formation" 
num_agents=10
algo="mappo"
exp="train_random_3"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} \
    --n_training_threads 16 --n_rollout_threads 16 --num_mini_batch 1 --episode_length 100 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --hidden_size 1024 --layer_N 4 --wandb_name "xxx" --user_name "imammuttaqin98-universitas-gadjah-mada-library" --cuda True
done
