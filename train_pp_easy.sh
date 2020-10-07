#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --env_name predator_prey \
  --nagents 3 \
  --dim 5 \
  --max_steps 20 \
  --vision 0 \
  --nprocesses 16 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.001 \
  --gacomm \
  --recurrent \
  --seed 3780 \
  --save \
  --save_every 1000 \
  --plot \
  --plot_env exp_pp_easy_gacomm_seed3780_lr0.001 \
  --plot_port 8009 \
  | tee train_pp_easy.log


#   --plot \
#   --plot_env pp_medium_gcomm_gat_hid_128_seed0_run11 \
#   --plot_port 8097 \

#   --save \
#   --save_every 200 \

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

