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
  --value_coeff 0.5 \
  --lrate 0.0008 \
  --max_steps 100 \
  --ic3net \
  --tarcomm \
  --recurrent \
  --save \
  --save_every 200 \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --seed 0 \
  --plot \
  --plot_env grf_ac_shared_tar_ic3_scoring_hid_128_adv_0_seed0_run3 \
  --plot_port 8009 \
  | tee train_grf.log

#  --render \
#   --plot \
#   --plot_env grf_tar_ic3net_scoring_hid_128_adv_0_seed0 \
#   --plot_port 8009 \