. path.sh
. cmd.sh

train_stage=-100
transform_dir=exp/tri3_ali

. parse_options.sh

if [ $# -ne 1 ]; then
  echo "Usage: local/run_nnet.sh <data-dir>"
  echo "e.g. : local/run_nnet.sh data/train_clean"
fi

train=$1

trainid=`basename $train`


set -e
set -o pipefail

echo ============================================================================
echo "                    DNN Hybrid Training & Decoding                        "
echo ============================================================================

# DNN hybrid system training parameters
  dnn_gpu_parallel_opts=(--minibatch-size 512 --max-change 40 --num-jobs-nnet 4 --num-threads 1 \
                         --parallel-opts "-l gpu=1" --cmd "queue.pl -l arch=*64 -l mem_free=2G,ram_free=1G")

steps/nnet2/train_pnorm.sh --mix-up 5000 \
  --initial-learning-rate 0.008 \
  --final-learning-rate 0.0008 \
  --num-hidden-layers 3 \
  --pnorm-input-dim 2000 \
  --pnorm-output-dim 200 \
  --num-jobs-nnet "$train_nj" --cmd "$train_cmd" \
  "${dnn_gpu_parallel_opts[@]}" \
  --stage $train_stage \
  --transform-dir "$transform_dir" \
  $train data/lang exp/tri3_ali exp/tri4_nnet_$trainid
