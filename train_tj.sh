#!/bin/bash

python -u main.py \
  --env_name traffic_junction \
  --nagents 5 \
  --dim 6 \
  --max_steps 20 \
  --add_rate_min 0.1 \
  --add_rate_max 0.3 \
  --difficulty easy \
  --vision 0 \
  --nprocesses 1 \
  --num_epochs 2000 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --ic3net \
  --recurrent \
  --curr_start 250 \
  --curr_end 1250 \
  --save \
  --save_every 100 \
  --seed 0 \
  --use_gpu \
  | tee train_tj.log

#   --plot \
#   --plot_env tj_gacomm_hid_128_seed0 \
#   --plot_port 8009 \

  ## easy
  # --nagents 5 \
  # --dim 6 \
  # --max_steps 20 \
  # --add_rate_min 0.1 \
  # --add_rate_max 0.3 \
  # --difficulty easy \

  ## medium
  # --nagents 10 \
  # --dim 14 \
  # --max_steps 40 \
  # --add_rate_min 0.05 \
  # --add_rate_max 0.2 \
  # --difficulty medium \

  ## hard
  # --nagents 20 \
  # --dim 18 \
  # --max_steps 80 \
  # --add_rate_min 0.02 \
  # --add_rate_max 0.05 \
  # --difficulty hard \

