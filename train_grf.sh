#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --env_name grf \
  --nagents 3 \
  --nprocesses 4 \
  --num_epochs 1000 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.0015 \
  --max_steps 100 \
  --gcomm \
  --gnn_type gcn \
  --directed \
  --recurrent \
  --save \
  --save_every 200 \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --seed 0 \
  --plot \
  --plot_env grf_gcomm_gcn_scoring_hid_128_adv_0_seed0_run1 \
  --plot_port 8097 \
  | tee train_grf.log

#  --render \
#   --plot \
#   --plot_env grf_tar_ic3net_scoring_hid_128_adv_0_seed0 \
#   --plot_port 8009 \