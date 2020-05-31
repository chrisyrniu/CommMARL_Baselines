#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --env_name traffic_junction \
  --nagents 10 \
  --dim 14 \
  --max_steps 40 \
  --add_rate_min 0.1 \
  --add_rate_max 0.1 \
  --difficulty medium \
  --vision 0 \
  --nprocesses 4 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --value_coeff 0.5 \
  --lrate 0.0005 \
  --ic3net \
  --tarcomm \
  --recurrent \
  --curr_start 500 \
  --curr_end 500 \
  --save \
  --plot \
  --plot_env tj_medium_ac_shared_tar_ic3_hid_128_seed0_no_curriculum_0.1 \
  --plot_port 8009 \
  | tee train_tj.log

  # --plot \
  # --plot_env traffic_juction \
  # --plot_port 8009 \

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

