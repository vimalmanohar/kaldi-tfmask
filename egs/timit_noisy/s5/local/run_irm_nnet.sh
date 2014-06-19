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
  --parallel-opts "-l gpu=1" --cmd "queue.pl -l arch=*64 -l mem_free=4G,ram_free=2G")
stage=-100
use_subset=true

. parse_options.sh

if $use_subset; then
  datadir_orig=$datadir
  datadir=${datadir_orig}.10p
  if [ ! -s $datadir/irm_targets.scp ]; then
    numutts_keep=`perl -e 'print int($ARGV[0]/10)' "$(wc -l < $datadir_orig/feats.scp)"`
    subset_data_dir.sh $datadir_orig $numutts_keep $datadir
    filter_scp.pl $datadir/feats.scp $datadir_orig/irm_targets.scp > $datadir/irm_targets.scp
    irm_scp=$datadir/irm_targets.scp
  fi
fi

steps/nnet2/train_irm_nnet.sh \
  "${dnn_gpu_parallel_opts[@]}" \
  --pnorm-input-dim $pnorm_input_dim \
  --pnorm-output-dim $pnorm_output_dim \
  --num-hidden-layers $num_hidden_layers \
  --initial-learning-rate $dnn_init_learning_rate \
  --final-learning-rate $dnn_final_learning_rate \
  --irm_scp $irm_scp --nj 64 \
  --stage $stage --cleanup false \
  $datadir $dir || exit 1 
