. path.sh
. cmd.sh

src_datadir=data/train_clean
train_nj=4
decode_nj=10
align_duplicated=false

. parse_options.sh

datadir=${src_datadir}_fbank_mfcc

dirid=`basename $src_datadir`
affix=_$(echo $dirid | awk -F'_' '{print $NF}')

utils/subset_data_dir.sh $datadir 1000 ${datadir}_1k

steps/train_mono.sh  --nj $train_nj --cmd "$train_cmd" \
  ${datadir}_1k data/lang exp/mono0a${affix}_fbank_mfcc

steps/align_si.sh --nj 4 --cmd "$train_cmd" \
   ${datadir} data/lang exp/mono0a${affix}_fbank_mfcc exp/mono0a${affix}_fbank_mfcc_ali

steps/train_deltas.sh --cmd "$train_cmd" \
    300 3000 ${datadir} data/lang exp/mono0a${affix}_fbank_mfcc_ali exp/tri1${affix}_fbank_mfcc

steps/align_si.sh --nj 4 --cmd "$train_cmd" \
   ${datadir} data/lang exp/tri1${affix}_fbank_mfcc exp/tri1${affix}_fbank_mfcc_ali

if $align_duplicated && [ -d ${src_datadir}_duplicated_fbank_mfcc ]; then
  steps/align_si.sh --nj 4 --cmd "$train_cmd" \
    ${src_datadir}_duplicated_fbank_mfcc data/lang exp/tri1${affix}_fbank_mfcc exp/tri1${affix}_duplicated_fbank_mfcc_ali
fi

