src_datadir=data/train_noisy
datadir=${src_datadir}_fbank_mfcc
train_nj=4
decode_nj=10

src_testdir=data/test_noisy
testdir=${src_testdir}_fbank_mfcc

dirid=`basename $src_datadir`
affix=_$(echo $dirid | awk -F'_' '{print $NF}')

cp -r $src_datadir $datadir
rm $datadir/{feats.scp,cmvn.scp}

utils/split_data.sh $src_datadir $train_nj
sdata=$src_datadir/split$train_nj
$train_cmd JOB=1:$train_nj exp/make_fbank_mfcc/compute_mfcc.JOB.log \
  compute-mfcc-feats-from-fbank --config=conf/mfcc.conf \
  scp:$sdata/JOB/feats.scp \
  ark,scp:fbank_mfcc/$dirid.JOB.ark,fbank_mfcc/$dirid.JOB.scp || exit 1

for n in `seq $train_nj`; do 
  cat fbank_mfcc/$dirid.$n.scp
done | sort > $datadir/feats.scp

steps/compute_cmvn_stats.sh $datadir exp/make_fbank_mfcc fbank_mfcc

utils/subset_data_dir.sh $datadir 1000 ${datadir}_1k

steps/train_mono.sh  --nj $train_nj --cmd "$train_cmd" \
  ${datadir}_1k data/lang exp/mono0a${affix}_fbank_mfcc

steps/align_si.sh --nj 4 --cmd "$train_cmd" \
   ${datadir} data/lang exp/mono0a${affix}_fbank_mfcc exp/mono0a${affix}_fbank_mfcc_ali

steps/train_deltas.sh --cmd "$train_cmd" \
    300 3000 ${datadir} data/lang exp/mono0a${affix}_fbank_mfcc_ali exp/tri1${affix}_fbank_mfcc

steps/align_si.sh --nj 4 --cmd "$train_cmd" \
   ${datadir} data/lang exp/tri1${affix}_fbank_mfcc exp/tri1${affix}_fbank_mfcc_ali

cp -r $src_testdir $testdir
rm $testdir/{feats.scp,cmvn.scp}

utils/split_data.sh $src_testdir $decode_nj
sdata=$src_testdir/split$decode_nj

dirid=`basename $src_testdir`
affix=_$(echo $dirid | awk -F'_' '{print $NF}')

$decode_cmd JOB=1:$decode_nj exp/make_fbank_mfcc/compute_mfcc.JOB.log \
  compute-mfcc-feats-from-fbank --config=conf/mfcc.conf \
  scp:$sdata/JOB/feats.scp \
  ark,scp:fbank_mfcc/$dirid.JOB.ark,fbank_mfcc/$dirid.JOB.scp || exit 1

for n in `seq $decode_nj`; do 
  cat fbank_mfcc/$dirid.$n.scp
done | sort > $testdir/feats.scp

steps/compute_cmvn_stats.sh $testdir exp/make_fbank_mfcc fbank_mfcc

