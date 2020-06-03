#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --env_name predator_prey \
  --nagents 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 4 \
  --num_epochs 1000 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --ic3net \
  --tarcomm \
  --recurrent \
  --save \
  --save_every 200 \
  --plot \
  --plot_env pp_medium_cf_tar_ic3_hid_128_seed0_run1 \
  --plot_port 8009 \
  | tee train_pp.log

#   --plot \
#   --plot_env pp_medium_gacomm_individual_rew_hid_128_seed3780 \
#   --plot_port 8009 \

  ## easy
  # --nagents 3 \
  # --dim 5 \
  # --max_steps 20 \
  # --vision 0 \

  ## medium
  # --nagents 5 \
  # --dim 10 \
  # --max_steps 40 \
  # --vision 1 \

  ## hard
  # --nagents 10 \
  # --dim 20 \
  # --max_steps 80 \
  # --vision 1 \

