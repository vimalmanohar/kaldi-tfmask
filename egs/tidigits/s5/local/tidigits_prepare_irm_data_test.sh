# Copyright 2014  Vimal Manohar

set -o pipefail

. path.sh
. cmd.sh 

decode_nj=20

. parse_options.sh

mkdir -p exp/noisy_data_test/log

utils/split_scp.pl data/test/wav.scp \
  $(for n in `seq 1 $decode_nj`; do echo "exp/noisy_data_test/clean_split_wav.$n.scp"; done | tr '\n' ' ')
 
mkdir -p data/test_fbank

for f in wav.scp text utt2spk; do 
  [ -f data/test/$f ] && cp data/test/$f data/test_fbank
done

set -e 

sleep 10

echo "Creating Mel filterbank features for clean speech in data/test_fbank..."
steps/make_fbank.sh --cmd "$decode_cmd" --nj $decode_nj \
  data/test_fbank exp/make_fbank fbank

for snr in 10; do
  mkdir -p exp/noisy_data_test/${snr}dB

  echo "Creating noisy data with SNR $snr dB..."
  
  $decode_cmd JOB=1:$decode_nj exp/noisy_data_test/log/make_noisy_data_test_${snr}dB.JOB.log \
    wav-add-noise scp:exp/noisy_data_test/clean_split_wav.JOB.scp \
    ark,scp:exp/noisy_data_test/${snr}dB/noisy_wav.JOB.ark,exp/noisy_data_test/${snr}dB/noisy_wav.JOB.scp \
    ark,scp:exp/noisy_data_test/${snr}dB/noise_wav.JOB.ark,exp/noisy_data_test/${snr}dB/noise_wav.JOB.scp
 
  mkdir -p data/test_noisy_${snr}dB
  mkdir -p data/test_noise_${snr}dB
  
  echo "Merging wav.scp in $decode_nj splits..."

  for n in `seq 1 $decode_nj`; do 
    cat exp/noisy_data_test/${snr}dB/noisy_wav.$n.scp
  done | sort > data/test_noisy_${snr}dB/wav.scp

  for n in `seq 1 $decode_nj`; do 
    cat exp/noisy_data_test/${snr}dB/noise_wav.$n.scp
  done | sort > data/test_noise_${snr}dB/wav.scp
  
  set +e
  cp -rT data/test_fbank data/test_fbank_${snr}dB
  set -e

  echo "Making copy of wav.scp and feats.scp in data/test_fbank_${snr}dB..."
  
  cat data/test_fbank/wav.scp| awk '{key=$1"-'${snr}'dB"; for (i=2; i<=NF; i++) key=key" "$i; print key}' \
    | sort > data/test_fbank_${snr}dB/wav.scp
  cat data/test_fbank/feats.scp| awk '{key=$1"-'${snr}'dB"; for (i=2; i<=NF; i++) key=key" "$i; print key}' \
    | sort > data/test_fbank_${snr}dB/feats.scp
  
  echo "Making copy of text in data/test_noisy_${snr}dB..."

  cat data/test_fbank/text | awk '{key=$1"-'${snr}'dB"; for (i=2; i<=NF; i++) key=key" "$i; print key}' \
    | sort > data/test_noisy_${snr}dB/text
  cp data/test_noisy_${snr}dB/text data/test_noise_${snr}dB/text
  cp data/test_noisy_${snr}dB/text data/test_fbank_${snr}dB/text
  
  echo "Making copy of utt2spk in data/test_noisy_${snr}dB..."

  cat data/test_fbank/utt2spk | awk '{key=$1"-'${snr}'dB"; for (i=2; i<=NF; i++) key=key" "$i; print key}' \
    | sort > data/test_noisy_${snr}dB/utt2spk
  cp data/test_noisy_${snr}dB/utt2spk data/test_noise_${snr}dB/utt2spk
  cp data/test_noisy_${snr}dB/utt2spk data/test_fbank_${snr}dB/utt2spk

  echo "Making spk2utt..."

  for d in fbank noisy noise; do 
    utils/utt2spk_to_spk2utt.pl data/test_${d}_${snr}dB/utt2spk \
      > data/test_${d}_${snr}dB/spk2utt
  done

  steps/make_fbank.sh --cmd "$decode_cmd" --nj $decode_nj \
    data/test_noisy_${snr}dB exp/make_fbank fbank
  
  steps/make_fbank.sh --cmd "$decode_cmd" --nj $decode_nj \
    data/test_noise_${snr}dB exp/make_fbank fbank

  utils/split_data.sh data/test_fbank_${snr}dB $decode_nj
  utils/split_data.sh data/test_noisy_${snr}dB $decode_nj
  utils/split_data.sh data/test_noise_${snr}dB $decode_nj

  $decode_cmd JOB=1:$decode_nj exp/noisy_data_test/log/make_irm_targets_${snr}dB.JOB.log \
    compute-irm-targets scp:data/test_fbank_${snr}dB/split$decode_nj/JOB/feats.scp \
    scp:data/test_noise_${snr}dB/split$decode_nj/JOB/feats.scp \
    ark,scp:exp/noisy_data_test/${snr}dB/irm.JOB.ark,exp/noisy_data_test/${snr}dB/irm.JOB.scp
  
  for n in `seq 1 $decode_nj`; do 
    cat exp/noisy_data_test/${snr}dB/irm.$n.scp
  done | sort > data/test_noisy_${snr}dB/irm.scp

done

utils/combine_data.sh data/test_noisy $(for snr in 10; do echo data/test_noisy_${snr}dB; done | tr '\n' ' ')
utils/combine_data.sh data/test_noise $(for snr in 10; do echo data/test_noise_${snr}dB; done | tr '\n' ' ')
  
for snr in 10; do 
  cat data/test_noisy_${snr}dB/irm.scp
done > data/test_noisy/irm.scp

rm -rf data/test_noise_*dB
rm -rf data/test_noisy_*dB
rm -rf data/test_fbank_*dB
    
utils/utt2spk_to_spk2utt.pl data/test_fbank/utt2spk \
  > data/test_fbank/spk2utt

steps/compute_cmvn_stats.sh --fake data/test_fbank exp/make_fbank fbank
steps/compute_cmvn_stats.sh --fake data/test_noisy exp/make_fbank fbank
steps/compute_cmvn_stats.sh --fake data/test_noise exp/make_fbank fbank
