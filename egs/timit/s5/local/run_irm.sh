#!/bin/bash

# Copyright 2014  Vimal Manohar

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e
set -u
set -o pipefail

echo $*

# Prepare data to train IRM nnet
local/timit_prepare_irm_data.sh

# Train IRM predictor neural net. Run this outside this script.
local/run_irm_nnet.sh --irm-scp data_noisy_fbank/train_noisy/irm_targets.scp \
  --datadir data_noisy_fbank/train_noisy

mkdir -p irm 
mkdir -p masked_fbank

# Make masked fbank features by feed-forward propagating the noisy features
# through the IRM predictor neural net and applying the mask.
local/make_masked_fbank.sh data_noisy_fbank/train_multi \
  data_noisy_fbank/train_multi_masked exp/irm_nnet \
  exp/make_masked_fbank/train_multi irm masked_fbank
utils/compute_cmvn_stats.sh --fake data_noisy_fbank/train_multi_masked \
  exp/make_masked_fbank/train_multi masked_fbank

# Make masked fbank features for each of the test and dev set noise conditions
##
while read noise_type <&3; do
  while read snr <&4; do
    for x in dev test; do
      x=${x}_noisy_${noise_type}_snr_$snr
      local/make_masked_fbank.sh data_fbank/${x} data_fbank/${x}_masked exp/irm_nnet exp/make_masked_fbank/$x irm masked_fbank
      utils/compute_cmvn_stats.sh --fake data_fbank/${x}_masked exp/make_masked_fbank/$x masked_fbank
    done
  done 4< snr.list
done 3< noisetypes.list

for x in dev test; do
  x=${x}_clean
  local/make_masked_fbank.sh data_fbank/${x} data_fbank/${x}_masked exp/irm_nnet exp/make_masked_fbank/$x irm masked_fbank
  utils/compute_cmvn_stats.sh --fake data_fbank/${x}_masked exp/make_masked_fbank/$x masked_fbank
done
##

# Compute MFCC features from the noisy and masked fbank features for training ASR
for y in data_noisy_fbank/train_multi_masked data_noisy_fbank/train_multi; do
  x=`basename $y`
  local/make_mfcc_from_fbank.sh ${y} data/${x} exp/make_fbank_mfcc/$x fbank_mfcc
  utils/compute_cmvn_stats.sh data_noisy_fbank/${x} exp/make_fbank_mfcc/$x fbank_mfcc 
done
# Compute concatenated MFCC features from the noisy and masked fbank features for training ASR
local/make_concat_feats.sh data_noisy_fbank/train_multi_concat data_noisy_fbank/train_multi data_noisy_fbank/train_multi_masked exp/make_fbank_mfcc/train_multi_concat fbank_mfcc
utils/compute_cmvn_stats.sh data_noisy_fbank/train_multi_concat exp/make_fbank_mfcc/train_multi_concat fbank_mfcc

# Compute noisy, masked and concat MFCC features for each noise conditions for test and dev speech
##
while read noise_type <&3; do
  while read snr <&4; do
    for x in dev test; do
      x=${x}_noisy_${noise_type}_snr_$snr
      local/make_mfcc_from_fbank.sh data_fbank/${x} data/${x} \
        exp/make_fbank_mfcc/$x fbank_mfcc
      local/make_mfcc_from_fbank.sh data_fbank/${x}_masked data/${x}_masked \
        exp/make_fbank_mfcc/${x}_masked fbank_mfcc
      local/make_concat_feats.sh data_fbank/${x}_concat data_fbank/${x} data_fbank_${x}_masked exp/make_fbank_mfcc/${x}_concat fbank_mfcc
      utils/compute_cmvn_stats.sh data_fbank/${x}_concat exp/make_fbank_mfcc/${x}_concat fbank_mfcc
    done
  done 4< snr.list
done 3< noisetypes.list

for x in dev test; do
  x=${x}_clean
  local/make_mfcc_from_fbank.sh data_fbank/${x} data/${x} \
    exp/make_fbank_mfcc/$x fbank_mfcc
  local/make_mfcc_from_fbank.sh data_fbank/${x}_masked data/${x}_masked \
    exp/make_fbank_mfcc/${x}_masked fbank_mfcc
  local/make_concat_feats.sh data_fbank/${x}_concat data_fbank/${x} data_fbank_${x}_masked exp/make_fbank_mfcc/${x}_concat fbank_mfcc
  utils/compute_cmvn_stats.sh data_fbank/${x}_concat exp/make_fbank_mfcc/${x}_concat fbank_mfcc
done
##
