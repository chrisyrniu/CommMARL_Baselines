#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --env_name grf \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 1000 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --critic_lrate 0.0015 \
  --actor_lrate 0.0015 \
  --max_steps 100 \
  --commnet \
  --recurrent \
  --save \
  --save_every 10 \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --seed 0 \
  | tee train.log

#  --render \

#   --plot \
#   --plot_env grf_tar_ic3net_scoring_hid_128_adv_0_seed0 \
#   --plot_port 8009 \