#!/bin/bash

python -u test.py \
  --env_name grf \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 200 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --max_steps 100 \
  --commnet \
  --recurrent \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --run_num 1 \
  --ep_num 0 \
  --render \
  | tee test.log

#  --render \
