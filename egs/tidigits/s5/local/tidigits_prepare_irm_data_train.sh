# Copyright 2014  Vimal Manohar

set -o pipefail

. path.sh
. cmd.sh 

train_nj=10

. parse_options.sh

mkdir -p exp/noisy_data_train/log

utils/split_scp.pl data/train/wav.scp \
  $(for n in `seq 1 $train_nj`; do echo "exp/noisy_data_train/clean_split_wav.$n.scp"; done | tr '\n' ' ')
 
mkdir -p data/train_fbank

for f in wav.scp text utt2spk; do 
  [ -f data/train/$f ] && cp data/train/$f data/train_fbank
done

set -e 

sleep 10

echo "Creating Mel filterbank features for clean speech in data/train_fbank..."
steps/make_fbank.sh --cmd "$train_cmd" --nj $train_nj \
  data/train_fbank exp/make_fbank fbank

for snr in 10 20 5 0; do
  mkdir -p exp/noisy_data_train/${snr}dB

  echo "Creating noisy data with SNR $snr dB..."
  
  $train_cmd JOB=1:$train_nj exp/noisy_data_train/log/make_noisy_data_train_${snr}dB.JOB.log \
    wav-add-noise --snr=$snr scp:exp/noisy_data_train/clean_split_wav.JOB.scp \
    ark,scp:exp/noisy_data_train/${snr}dB/noisy_wav.JOB.ark,exp/noisy_data_train/${snr}dB/noisy_wav.JOB.scp \
    ark,scp:exp/noisy_data_train/${snr}dB/noise_wav.JOB.ark,exp/noisy_data_train/${snr}dB/noise_wav.JOB.scp
 
  mkdir -p data/train_noisy_${snr}dB
  mkdir -p data/train_noise_${snr}dB
  
  echo "Merging wav.scp in $train_nj splits..."

  for n in `seq 1 $train_nj`; do 
    cat exp/noisy_data_train/${snr}dB/noisy_wav.$n.scp
  done | sort > data/train_noisy_${snr}dB/wav.scp

  for n in `seq 1 $train_nj`; do 
    cat exp/noisy_data_train/${snr}dB/noise_wav.$n.scp
  done | sort > data/train_noise_${snr}dB/wav.scp
  
  set +e
  cp -rT data/train_fbank data/train_fbank_${snr}dB
  set -e

  echo "Making copy of wav.scp and feats.scp in data/train_fbank_${snr}dB..."
  
  cat data/train_fbank/wav.scp| awk '{key=$1"-'${snr}'dB"; for (i=2; i<=NF; i++) key=key" "$i; print key}' \
    | sort > data/train_fbank_${snr}dB/wav.scp
  cat data/train_fbank/feats.scp| awk '{key=$1"-'${snr}'dB"; for (i=2; i<=NF; i++) key=key" "$i; print key}' \
    | sort > data/train_fbank_${snr}dB/feats.scp
  
  echo "Making copy of text in data/train_noisy_${snr}dB..."

  cat data/train_fbank/text | awk '{key=$1"-'${snr}'dB"; for (i=2; i<=NF; i++) key=key" "$i; print key}' \
    | sort > data/train_noisy_${snr}dB/text
  cp data/train_noisy_${snr}dB/text data/train_noise_${snr}dB/text
  cp data/train_noisy_${snr}dB/text data/train_fbank_${snr}dB/text
  
  echo "Making copy of utt2spk in data/train_noisy_${snr}dB..."

  cat data/train_fbank/utt2spk | awk '{print $1"-'$snr'dB "$2"-'$snr'dB"}' \
    | sort > data/train_noisy_${snr}dB/utt2spk
  cp data/train_noisy_${snr}dB/utt2spk data/train_noise_${snr}dB/utt2spk
  cp data/train_noisy_${snr}dB/utt2spk data/train_fbank_${snr}dB/utt2spk

  echo "Making spk2utt..."

  for d in fbank noisy noise; do 
    utils/utt2spk_to_spk2utt.pl data/train_${d}_${snr}dB/utt2spk \
      > data/train_${d}_${snr}dB/spk2utt
  done

  steps/make_fbank.sh --cmd "$train_cmd" --nj $train_nj \
    data/train_noisy_${snr}dB exp/make_fbank fbank
  
  steps/make_fbank.sh --cmd "$train_cmd" --nj $train_nj \
    data/train_noise_${snr}dB exp/make_fbank fbank

  utils/split_data.sh data/train_fbank_${snr}dB $train_nj
  utils/split_data.sh data/train_noisy_${snr}dB $train_nj
  utils/split_data.sh data/train_noise_${snr}dB $train_nj

  $train_cmd JOB=1:$train_nj exp/noisy_data_train/log/make_irm_targets_${snr}dB.JOB.log \
    compute-irm-targets scp:data/train_fbank_${snr}dB/split$train_nj/JOB/feats.scp \
    scp:data/train_noise_${snr}dB/split$train_nj/JOB/feats.scp \
    ark,scp:exp/noisy_data_train/${snr}dB/irm.JOB.ark,exp/noisy_data_train/${snr}dB/irm.JOB.scp
  
  for n in `seq 1 $train_nj`; do 
    cat exp/noisy_data_train/${snr}dB/irm.$n.scp
  done | sort > data/train_noisy_${snr}dB/irm.scp

done

utils/combine_data.sh data/train_noisy $(for snr in 10; do echo data/train_noisy_${snr}dB; done | tr '\n' ' ')
utils/combine_data.sh data/train_noise $(for snr in 10; do echo data/train_noise_${snr}dB; done | tr '\n' ' ')
  
for snr in 10; do 
  cat data/train_noisy_${snr}dB/irm.scp
done > data/train_noisy/irm.scp

rm -rf data/train_noise_*dB
rm -rf data/train_noisy_*dB
rm -rf data/train_fbank_*dB
    
utils/utt2spk_to_spk2utt.pl data/train_fbank/utt2spk \
  > data/train_fbank/spk2utt

steps/compute_cmvn_stats.sh --fake data/train_fbank exp/make_fbank fbank
steps/compute_cmvn_stats.sh --fake data/train_noisy exp/make_fbank fbank
steps/compute_cmvn_stats.sh --fake data/train_noise exp/make_fbank fbank

cp -r data/train_fbank data/train_fbank_mfcc
rm data/train_fbank_mfcc/{feats.scp,cmvn.scp}

mkdir -p exp/make_fbank_mfcc fbank_mfcc

utils/split_data.sh data/train_fbank $train_nj

sdata=data/train_fbank/split$train_nj

$train_cmd JOB=1:$train_nj exp/make_fbank_mfcc/compute_mfcc.JOB.log \
  compute-mfcc-feats-from-fbank --config=conf/mfcc.conf \
  scp:$sdata/JOB/feats.scp \
  ark,scp:fbank_mfcc/raw_train_fbank_mfcc.JOB.ark,fbank_mfcc/raw_train_fbank_mfcc.JOB.scp || exit 1

for n in `seq $train_nj`; do 
  cat fbank_mfcc/raw_train_fbank_mfcc.$n.scp
done | sort > data/train_fbank_mfcc/feats.scp

steps/compute_cmvn_stats.sh data/train_fbank_mfcc exp/make_fbank_mfcc fbank_mfcc
