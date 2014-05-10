# Copyright 2014  Vimal Manohar
# This script prepares the data for some experiments using TF-Masking

. path.sh
. cmd.sh 

set -o pipefail
set -u

nj=20
dataset_dir=data/train
noisy_datadir_prefix=exp/noisy_data    # The noisy data will be stored as exp/noisy_data_train etc.
snr_list="10 20 5 0"
cmd=run.pl
stage=-1
make_duplicate_datadir=false

. parse_options.sh

dataset_id=`basename $dataset_dir`

noisy_data=${noisy_datadir_prefix}_${dataset_id}

echo "$0: Creating directories to store noisy data in ${noisy_data}..."

mkdir -p ${noisy_datadir_prefix}_${dataset_id}
mkdir -p ${noisy_datadir_prefix}_${dataset_id}/log

echo "$0: Making temporary scp of clean wav files $noisy_data/clean_wav.*.scp..."

utils/split_scp.pl ${dataset_dir}/wav.scp \
  $(for n in `seq 1 $nj`; do echo "$noisy_data/clean_wav.$n.scp"; done | tr '\n' ' ')

if [ $stage -le 0 ]; then
  cp -rT ${dataset_dir} ${dataset_dir}_clean
  rm ${dataset_dir}_clean/{feats.scp,cmvn.scp}
  rm -rf ${dataset_dir}_clean/split*

  sleep 10

  echo "$0: Creating Mel filterbank features for clean speech in ${dataset_dir}_clean..."
  steps/make_fbank.sh --cmd "$cmd" --nj $nj \
    ${dataset_dir}_clean exp/make_fbank fbank
fi

set -e 

count=0

for snr in $snr_list; do
  mkdir -p $noisy_data/${snr}dB

  echo "$0: Creating noisy data for $dataset_id with SNR $snr dB..."
 
  if [ $stage -le $[1+count*5] ]; then
    $cmd JOB=1:$nj $noisy_data/log/make_noisy_data_${dataset_id}_${snr}dB.JOB.log \
      cat $noisy_data/clean_wav.JOB.scp \| \
      awk "{key=\"snr"${snr}"dB-\"\$1; for (i=2; i<=NF; i++) key=key\" \"\$i; print key}" \| \
      wav-add-noise --snr=$snr scp:- \
      ark,scp:$noisy_data/${snr}dB/noisy_wav.JOB.ark,$noisy_data/${snr}dB/noisy_wav.JOB.scp \
      ark,scp:$noisy_data/${snr}dB/noise_wav.JOB.ark,$noisy_data/${snr}dB/noise_wav.JOB.scp
  fi

  mkdir -p ${dataset_dir}_noisy_${snr}dB
  mkdir -p ${dataset_dir}_noise_${snr}dB
  
  echo "$0: Merging wav.scp in $nj splits..."

  # Make temporary data direcotories for clean, noise and noisy
  # at each noise level

  if [ $stage -le $[2+count*5] ]; then
    for n in `seq 1 $nj`; do 
      cat $noisy_data/${snr}dB/noisy_wav.$n.scp
    done | sort -k 1,1 > ${dataset_dir}_noisy_${snr}dB/wav.scp

    for n in `seq 1 $nj`; do 
      cat $noisy_data/${snr}dB/noise_wav.$n.scp
    done | sort -k 1,1 > ${dataset_dir}_noise_${snr}dB/wav.scp

    set +e
    cp -rT ${dataset_dir}_clean ${dataset_dir}_clean_${snr}dB
    rm -rf ${dataset_dir}_clean_${snr}dB/split*
    set -e

    echo "$0: Making copy of wav.scp and feats.scp in ${dataset_dir}_clean_${snr}dB..."

    cat ${dataset_dir}_clean/wav.scp | awk '{key="snr'${snr}'dB-"$1; for (i=2; i<=NF; i++) key=key" "$i; print key}' \
      | sort -k 1,1 > ${dataset_dir}_clean_${snr}dB/wav.scp
    cat ${dataset_dir}_clean/feats.scp | awk '{key="snr'${snr}'dB-"$1; for (i=2; i<=NF; i++) key=key" "$i; print key}' \
      | sort -k 1,1 > ${dataset_dir}_clean_${snr}dB/feats.scp

    echo "$0: Making copy of text in ${dataset_dir}_*_${snr}dB..."

    cat ${dataset_dir}_clean/text | awk '{key="snr'${snr}'dB-"$1; for (i=2; i<=NF; i++) key=key" "$i; print key}' \
      | sort -k 1,1 > ${dataset_dir}_noisy_${snr}dB/text
    cp ${dataset_dir}_noisy_${snr}dB/text ${dataset_dir}_noise_${snr}dB/text
    cp ${dataset_dir}_noisy_${snr}dB/text ${dataset_dir}_clean_${snr}dB/text

    echo "$0: Making copy of utt2spk in ${dataset_dir}_*_${snr}dB..."

    cat ${dataset_dir}_clean/utt2spk | awk '{print "snr'${snr}'dB-"$1" snr'${snr}'dB-"$2}' \
      | sort -k 1,1 > ${dataset_dir}_noisy_${snr}dB/utt2spk
    cp ${dataset_dir}_noisy_${snr}dB/utt2spk ${dataset_dir}_noise_${snr}dB/utt2spk
    cp ${dataset_dir}_noisy_${snr}dB/utt2spk ${dataset_dir}_clean_${snr}dB/utt2spk

    echo "$0: Making spk2utt in ${dataset_dir}_*_${snr}dB..."

    for d in clean noisy noise; do 
      utils/utt2spk_to_spk2utt.pl ${dataset_dir}_${d}_${snr}dB/utt2spk \
        | sort -k 1,1 > ${dataset_dir}_${d}_${snr}dB/spk2utt
    done
  fi


  if [ $stage -le $[3+count*5] ]; then
    echo "$0: Creating Mel filterbank features for noisy speech in ${dataset_dir}_noisy_${snr}dB..."
    steps/make_fbank.sh --cmd "$cmd" --nj $nj \
      ${dataset_dir}_noisy_${snr}dB exp/make_fbank fbank
  fi

  if [ $stage -le $[4+count*5] ]; then
    echo "$0: Creating Mel filterbank features for noise in ${dataset_dir}_noise_${snr}dB..."
    steps/make_fbank.sh --cmd "$train_cmd" --nj $nj \
      ${dataset_dir}_noise_${snr}dB exp/make_fbank fbank
  fi

  if [ $stage -le $[5+count*5] ]; then

    echo "$0: Split data directories into $nj jobs..."
    utils/split_data.sh ${dataset_dir}_clean_${snr}dB $nj
    utils/split_data.sh ${dataset_dir}_noisy_${snr}dB $nj
    utils/split_data.sh ${dataset_dir}_noise_${snr}dB $nj
    
    echo "$0: Compute IRM targets using noise and clean Mel filterbank features for ${snr}dB..."
    $cmd JOB=1:$nj $noisy_data/log/make_irm_targets_${snr}dB.JOB.log \
      compute-irm-targets scp:${dataset_dir}_clean_${snr}dB/split$nj/JOB/feats.scp \
      scp:${dataset_dir}_noise_${snr}dB/split$nj/JOB/feats.scp \
      ark,scp:$noisy_data/${snr}dB/irm.JOB.ark,$noisy_data/${snr}dB/irm.JOB.scp

    for n in `seq 1 $nj`; do 
      cat $noisy_data/${snr}dB/irm.$n.scp
    done | sort -k1,1 > ${dataset_dir}_noisy_${snr}dB/irm.scp

  fi

  count=$[count+1]
done

if [ $stage -le $[1+count*5] ]; then
  echo "$0: Combining different noise levels into single directory..."
  utils/combine_data.sh ${dataset_dir}_noisy $(for snr in $snr_list; do echo ${dataset_dir}_noisy_${snr}dB; done | tr '\n' ' ')
  utils/combine_data.sh ${dataset_dir}_noise $(for snr in $snr_list; do echo ${dataset_dir}_noise_${snr}dB; done | tr '\n' ' ')
  
  for snr in $snr_list; do 
    cat ${dataset_dir}_noisy_${snr}dB/irm.scp
  done > ${dataset_dir}_noisy/irm.scp

  utils/utt2spk_to_spk2utt.pl ${dataset_dir}_clean/utt2spk \
    > ${dataset_dir}_clean/spk2utt

fi

    
if [ $stage -le $[2+count*5] ]; then
  steps/compute_cmvn_stats.sh --fake ${dataset_dir}_clean exp/make_fbank fbank
  steps/compute_cmvn_stats.sh --fake ${dataset_dir}_noisy exp/make_fbank fbank
  steps/compute_cmvn_stats.sh --fake ${dataset_dir}_noise exp/make_fbank fbank
fi

if [ $stage -le $[3+count*5] ]; then
  for d in clean noisy; do
    set +e
    cp -rT ${dataset_dir}_${d} ${dataset_dir}_${d}_fbank_mfcc
    rm -rf ${dataset_dir}_${d}_fbank_mfcc/split*
    set -e

    rm ${dataset_dir}_${d}_fbank_mfcc/{feats.scp,cmvn.scp}
    mkdir -p fbank_mfcc

    utils/split_data.sh ${dataset_dir}_${d} $nj
    sdata=${dataset_dir}_${d}/split$nj
    $cmd JOB=1:$nj exp/make_fbank_mfcc/make_fbank_mfcc_${dataset_id}.JOB.log \
      compute-mfcc-feats-from-fbank --config=conf/mfcc.conf \
      scp:$sdata/JOB/feats.scp \
      ark,scp:fbank_mfcc/raw_${dataset_id}_${d}_fbank_mfcc.JOB.ark,fbank_mfcc/raw_${dataset_id}_${d}_fbank_mfcc.JOB.scp

    for n in `seq $nj`; do 
      cat fbank_mfcc/raw_${dataset_id}_${d}_fbank_mfcc.$n.scp
    done | sort -k 1,1 > ${dataset_dir}_${d}_fbank_mfcc/feats.scp

    steps/compute_cmvn_stats.sh ${dataset_dir}_${d}_fbank_mfcc exp/make_fbank_mfcc fbank_mfcc
  done
fi

if $make_duplicate_datadir && [ $stage -le $[4+count*5] ]; then
  dir_list=""

  for snr in $snr_list; do 
    utils/copy_data_dir.sh --spk-prefix "snr${snr}dB-" --utt-prefix "snr${snr}dB-" \
      ${dataset_dir}_clean_fbank_mfcc ${dataset_dir}_clean_fbank_mfcc_${snr}dB
    dir_list="$dir_list${dataset_dir}_clean_fbank_mfcc_${snr}dB "
  done

  utils/combine_data.sh ${dataset_dir}_clean_duplicated_fbank_mfcc $dir_list

fi

rm -rf ${dataset_dir}_noise_*dB
rm -rf ${dataset_dir}_noisy_*dB
rm -rf ${dataset_dir}_clean_*dB

