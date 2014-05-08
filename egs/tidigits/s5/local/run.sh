cp -r data/train_fbank data/train_fbank_mfcc
rm data/train_fbank_mfcc/{feats.scp,cmvn.scp}

utils/split_data.sh data/train_fbank $train_nj
$train_cmd JOB=1:$train_nj exp/make_fbank_mfcc/compute_mfcc.JOB.log \
  compute-mfcc-from-fbank --config=conf/mfcc.conf \
  scp:$sdata/JOB/feats.scp \
  ark,scp:fbank_mfcc/train_clean.JOB.ark,fbank_mfcc/train_clean.JOB.scp || exit 1

for n in `seq $train_nj`; do 
  cat fbank_mfcc/train_clean.$n.scp
done | sort > data/train_fbank_mfcc/feats.scp

steps/compute_cmvn_stats.sh data/train_fbank_mfcc exp/make_fbank_mfcc fbank_mfcc

utils/subset_data_dir.sh $train 1000 ${train}_1k

steps/train_mono.sh  --nj 4 --cmd "$train_cmd" \
  ${train}_1k data/lang exp/mono0a

steps/align_si.sh --nj 4 --cmd "$train_cmd" \
   ${train} data/lang exp/mono0a exp/mono0a_ali

steps/train_deltas.sh --cmd "$train_cmd" \
    300 3000 ${train} data/lang exp/mono0a_ali exp/tri1

steps/align_si.sh --nj 4 --cmd "$train_cmd" \
   ${train} data/lang exp/tri1 exp/tri1_ali

#dir=exp/tri1_nnet
#local/run_nnet.sh $train $dir
