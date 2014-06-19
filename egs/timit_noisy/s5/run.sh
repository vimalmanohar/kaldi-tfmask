T!/bin/bash

# Copyright 2013  Bagher BabaAli
# Copyright 2014  Vimal Manohar

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

# Acoustic model parameters
numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000
numLeavesSAT=2500
numGaussSAT=15000
numGaussUBM=400
numLeavesSGMM=7000
numGaussSGMM=9000

decode_nj=16
train_nj=32

rate=8k

echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================

timit=/home/vmanoha1/workspace_tfmask/data/timit/TIMIT

false && {
local/timit_data_prep.sh $timit
}

[ ! -f noisetypes.list ] && echo "Unable to find noise types for which data needs to be prepared" && exit 1

false && {
local/timit_noisy_data_prep.sh --rate ${rate} ${timit}_clean clean
}

false && {
while read noise_type <&3; do
  while read snr <&4; do
    local/timit_noisy_data_prep.sh --rate ${rate} ${timit}_noisy_${noise_type}_snr_${snr} noisy_${noise_type}_snr_${snr}
  done 4< snr.list
done 3< noisetypes.list
}

false && {
local/timit_prepare_dict.sh

# Caution below: we insert optional-silence with probability 0.5, which is the
# default, but this is probably not appropriate for this setup, since silence
# appears also as a word in the dictionary and is scored.  We could stop this
# by using the option --sil-prob 0.0, but apparently this makes results worse.
utils/prepare_lang.sh --position-dependent-phones false --num-sil-states 3 \
 data/local/dict "sil" data/local/lang_tmp data/lang
}

false && {
local/timit_format_data.sh
}

false && {
local/timit_noisy_format_data.sh --prefix clean
}

false && {
while read noise_type <&3; do
  while read snr <&4; do
    local/timit_noisy_format_data.sh --prefix noisy_${noise_type}_snr_${snr}
  done 4< snr.list
done 3< noisetypes.list
}

echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set           "
echo ============================================================================

# Now make MFCC features.
mfccdir=mfcc
fbankdir=fbank

mkdir -p data_fbank

false && {
for x in train dev test; do 
  x=${x}_clean
  cp -rT data/$x data_fbank/$x
  set +e
  rm data_fbank/$x/{feats.scp,cmvn.scp}
  set -e
  steps/make_fbank.sh --cmd "$train_cmd" --nj 30 data_fbank/$x exp/make_fbank/$x $fbankdir
  steps/compute_cmvn_stats.sh --fake data_fbank/$x exp/make_fbank/$x $fbankdir
done
}

true && {
for x in train dev test; do 
  x=${x}_clean
  local/make_mfcc_from_fbank.sh --cmd "$train_cmd" --nj $decode_nj data_fbank/$x data/$x exp/make_fbank_mfcc/$x fbank_mfcc
  steps/compute_cmvn_stats.sh data/$x exp/make_fbank_mfcc/$x fbank_mfcc
done
}

false && {
while read noise_type <&3; do
  while read snr <&4; do
    for x in train dev test; do 
      x=${x}_noisy_${noise_type}_snr_${snr}
      cp -rT data/$x data_fbank/$x
      set +e
      rm data_fbank/$x/{feats.scp,cmvn.scp}
      set -e
      steps/make_fbank.sh --cmd "$train_cmd" --nj 30 data_fbank/$x exp/make_fbank/$x $fbankdir
      steps/compute_cmvn_stats.sh --fake data_fbank/$x exp/make_fbank/$x $fbankdir
    done
  done 4< snr.list
done 3< noisetypes.list
}

false && {
local/run_irm.sh
}

echo ============================================================================
echo "                     MonoPhone Training & Decoding                        "
echo ============================================================================

train=data/train_clean
dev=data/dev_clean
test=data/test_clean

devid=`basename $dev`
testid=`basename $test`

steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" $train data/lang exp/mono

utils/mkgraph.sh --mono data/lang_test_bg exp/mono exp/mono/graph

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/mono/graph $dev exp/mono/decode_${devid}

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/mono/graph $test exp/mono/decode_${testid}

echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas Training & Decoding               "
echo ============================================================================

steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
 $train data/lang exp/mono exp/mono_ali

# Train tri1, which is deltas + delta-deltas, on train data.
steps/train_deltas.sh --cmd "$train_cmd" \
 $numLeavesTri1 $numGaussTri1 $train data/lang exp/mono_ali exp/tri1

utils/mkgraph.sh data/lang_test_bg exp/tri1 exp/tri1/graph

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri1/graph $dev exp/tri1/decode_${devid}

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri1/graph $test exp/tri1/decode_${testid}

echo ============================================================================
echo "                 tri2 : LDA + MLLT Training & Decoding                    "
echo ============================================================================

steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
  $train data/lang exp/tri1 exp/tri1_ali

steps/train_lda_mllt.sh --cmd "$train_cmd" \
 --splice-opts "--left-context=3 --right-context=3" \
 $numLeavesMLLT $numGaussMLLT $train data/lang exp/tri1_ali exp/tri2

utils/mkgraph.sh data/lang_test_bg exp/tri2 exp/tri2/graph

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri2/graph $dev exp/tri2/decode_${devid}

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri2/graph $test exp/tri2/decode_${testid}

echo ============================================================================
echo "              tri3 : LDA + MLLT + SAT Training & Decoding                 "
echo ============================================================================

# Align tri2 system with train data.
steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
 --use-graphs true $train data/lang exp/tri2 exp/tri2_ali

# From tri2 system, train tri3 which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
 $numLeavesSAT $numGaussSAT $train data/lang exp/tri2_ali exp/tri3

utils/mkgraph.sh data/lang_test_bg exp/tri3 exp/tri3/graph

steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri3/graph $dev exp/tri3/decode_${devid}

steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri3/graph $test exp/tri3/decode_${testid}

echo "Basic Models done!" && exit 0

echo ============================================================================
echo "                        SGMM2 Training & Decoding                         "
echo ============================================================================

steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
 $train data/lang exp/tri3 exp/tri3_ali

steps/train_ubm.sh --cmd "$train_cmd" \
 $numGaussUBM $train data/lang exp/tri3_ali exp/ubm4

steps/train_sgmm2.sh --cmd "$train_cmd" $numLeavesSGMM $numGaussSGMM \
 $train data/lang exp/tri3_ali exp/ubm4/final.ubm exp/sgmm2_4

utils/mkgraph.sh data/lang_test_bg exp/sgmm2_4 exp/sgmm2_4/graph

steps/decode_sgmm2.sh --nj "$decode_nj" --cmd "$decode_cmd"\
 --transform-dir exp/tri3/decode_${devid} exp/sgmm2_4/graph $dev \
 exp/sgmm2_4/decode_${devid}

steps/decode_sgmm2.sh --nj "$decode_nj" --cmd "$decode_cmd"\
 --transform-dir exp/tri3/decode_${testid} exp/sgmm2_4/graph $test \
 exp/sgmm2_4/decode_${testid}

echo ============================================================================
echo "                    MMI + SGMM2 Training & Decoding                       "
echo ============================================================================

steps/align_sgmm2.sh --nj "$train_nj" --cmd "$train_cmd" \
 --transform-dir exp/tri3_ali --use-graphs true --use-gselect true $train \
 data/lang exp/sgmm2_4 exp/sgmm2_4_ali

steps/make_denlats_sgmm2.sh --nj "$train_nj" --sub-split "$train_nj" --cmd "$decode_cmd"\
 --transform-dir exp/tri3_ali $train data/lang exp/sgmm2_4_ali \
 exp/sgmm2_4_denlats

steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" \
 --transform-dir exp/tri3_ali --boost 0.1 --zero-if-disjoint true \
 $train data/lang exp/sgmm2_4_ali exp/sgmm2_4_denlats \
 exp/sgmm2_4_mmi_b0.1

for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3/decode_${devid} data/lang_test_bg $dev \
   exp/sgmm2_4/decode_${devid} exp/sgmm2_4_mmi_b0.1/decode_${devid}_it$iter

  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3/decode_${testid} data/lang_test_bg $test \
   exp/sgmm2_4/decode_${testid} exp/sgmm2_4_mmi_b0.1/decode_${testid}_it$iter
done

echo "SGMM Models done" && exit 0

echo ============================================================================
echo "                    DNN Hybrid Training & Decoding                        "
echo ============================================================================

# DNN hybrid system training parameters
dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

steps/train_nnet_cpu.sh --mix-up 5000 --initial-learning-rate 0.015 \
  --final-learning-rate 0.002 --num-hidden-layers 2 --num-parameters 1500000 \
  --num-jobs-nnet "$train_nj" --cmd "$train_cmd" "${dnn_train_extra_opts[@]}" \
  $train data/lang exp/tri3_ali exp/tri4_nnet

decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G")
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" \
  --transform-dir exp/tri3/decode_${devid} exp/tri3/graph $dev \
  exp/tri4_nnet/decode_${devid} | tee exp/tri4_nnet/decode_${devid}/decode.log

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" \
  --transform-dir exp/tri3/decode_${testid} exp/tri3/graph $test \
  exp/tri4_nnet/decode_${testid} | tee exp/tri4_nnet/decode_${testid}/decode.log

echo ============================================================================
echo "                    System Combination (DNN+SGMM)                         "
echo ============================================================================

for iter in 1 2 3 4; do
  local/score_combine.sh --cmd "$decode_cmd" \
   $dev data/lang_test_bg exp/tri4_nnet/decode_${devid} \
   exp/sgmm2_4_mmi_b0.1/decode_${devid}_it$iter exp/combine_2/decode_${devid}_it$iter

  local/score_combine.sh --cmd "$decode_cmd" \
   $test data/lang_test_bg exp/tri4_nnet/decode_${testid} \
   exp/sgmm2_4_mmi_b0.1/decode_${testid}_it$iter exp/combine_2/decode_${testid}_it$iter
done


echo ============================================================================
echo "                    Getting Results [see RESULTS file]                    "
echo ============================================================================

for x in exp/*/decode*; do
  [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh
done 

echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0
