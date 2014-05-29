#!/bin/bash

# Copyright 2014  Vimal Manohar

. path.sh
. cmd.sh

set -e
set -u
set -o pipefail

datadir=data_noisy_fbank/train_noisy
dir=exp/irm_nnet

irm_scp=data_noisy_fbank/train_noisy/irm_targets.scp
pnorm_input_dim=3000
pnorm_output_dim=300
num_hidden_layers=5
dnn_init_learning_rate=0.008
dnn_final_learning_rate=0.0008
dnn_gpu_parallel_opts=(--minibatch-size 512 --max-change 40 --num-jobs-nnet 4 --num-threads 1 \
  --parallel-opts "-l gpu=1" --cmd "queue.pl -l arch=*64 -l mem_free=2G,ram_free=1G")
stage=-100

. parse_options.sh

#dnn_init_learning_rate=`perl -e "print $dnn_init_learning_rate / 26"`
#dnn_final_learning_rate=`perl -e "print $dnn_final_learning_rate / 26"`

steps/nnet2/train_irm_nnet.sh \
  "${dnn_gpu_parallel_opts[@]}" \
  --pnorm-input-dim $pnorm_input_dim \
  --pnorm-output-dim $pnorm_output_dim \
  --num-hidden-layers $num_hidden_layers \
  --initial-learning-rate $dnn_init_learning_rate \
  --final-learning-rate $dnn_final_learning_rate \
  --irm_scp $irm_scp \
  --stage $stage --cleanup false \
  $datadir $dir || exit 1 
