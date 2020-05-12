#!/bin/bash

python -u main.py \
  --env_name predator_prey \
  --nagents 3 \
  --dim 5 \
  --max_steps 20 \
  --vision 0 \
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
  --save_every 50 \
  | tee train.log

  # --plot \
  # --plot_env predator_prey \

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

