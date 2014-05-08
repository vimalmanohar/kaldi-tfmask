. path.sh
. cmd.sh

set -e

datadir=data/train_noisy
nnet_dir=exp/irm_nnet
nj=10
cmd=run.pl 

. parse_options.sh

dirid=`basename $datadir`

sdata=$datadir/split$nj
utils/split_data.sh $datadir $nj

mkdir -p irm

$cmd JOB=1:$nj exp/make_irm/make_irm_$dirid.JOB.log \
  nnet2-compute --raw=true $nnet_dir/final.nnet scp:$sdata/JOB/feats.scp \
  ark,scp:irm/irm_$dirid.JOB.ark,irm/irm_$dirid.JOB.scp

mkdir -p masked_fbank

$cmd JOB=1:$nj exp/make_irm/mask_$dirid.JOB.log \
  matrix-mul-elements scp:$sdata/JOB/feats.scp \
  scp:irm/irm_$dirid.JOB.scp \
  ark,scp:masked_fbank/masked_fbank_$dirid.JOB.ark,masked_fbank/masked_fbank_$dirid.JOB.scp

cp -rT $datadir ${datadir}_masked
rm ${datadir}_masked/{feats.scp,cmvn.scp}

for n in `seq $nj`; do 
  cat masked_fbank/masked_fbank_$dirid.$n.scp
done | sort > ${datadir}_masked/feats.scp

steps/compute_cmvn_stats.sh --fake ${datadir}_masked exp/make_irm masked_fbank


cp -rT ${datadir}_masked ${datadir}_masked_fbank_mfcc
rm ${datadir}_masked_fbank_mfcc/{feats.scp,cmvn.scp}

sdata=${datadir}_masked/split$nj
utils/split_data.sh ${datadir}_masked $nj

dirid=${dirid}_masked

mkdir -p masked_mfcc
mkdir -p exp/make_masked_mfcc

utils/split_data.sh ${datadir}_fbank_mfcc $nj

$cmd JOB=1:$nj exp/make_masked_mfcc/compute_mfcc.JOB.log \
  compute-mfcc-feats-from-fbank --config=conf/mfcc.conf \
  scp:$sdata/JOB/feats.scp ark:- \| paste-feats scp:${datadir}_fbank_mfcc/split$nj/JOB/feats.scp ark:- \
  ark,scp:masked_mfcc/raw_${dirid}_masked_mfcc.JOB.ark,masked_mfcc/raw_${dirid}_masked_mfcc.JOB.scp || exit 1

for n in `seq $nj`; do 
  cat masked_mfcc/raw_${dirid}_masked_mfcc.$n.scp
done | sort > ${datadir}_masked_fbank_mfcc/feats.scp

steps/compute_cmvn_stats.sh ${datadir}_masked_fbank_mfcc exp/make_fbank_mfcc fbank_mfcc
