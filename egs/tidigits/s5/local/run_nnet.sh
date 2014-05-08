dir=exp/tri1_nnet

pnorm_input_dim=800
pnorm_output_dim=200
dnn_init_learning_rate=0.008
dnn_final_learning_rate=0.0008
dnn_gpu_parallel_opts=(--minibatch-size 512 --max-change 40 --num-jobs-nnet 4 --num-threads 1 \
  --parallel-opts "-l gpu=1" --cmd "queue.pl -l arch=*64 -l mem_free=2G,ram_free=1G")
dnn_cpu_parallel_opts=(--minibatch-size 128 --max-change 10 --num-jobs-nnet 8 --num-threads 16 \
  --parallel-opts "-pe smp 16" --cmd "queue.pl -l arch=*64 -l mem_free=2G,ram_free=1G")
stage=-100
dnn_beam=16.0
dnn_lat_beam=8.5
minimize=true
decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G")
train_data_dir=data/train
test_data_dir=data/test
alidir=exp/tri1_ali

. parse_options.sh

steps/nnet2/train_pnorm.sh \
  "${dnn_cpu_parallel_opts[@]}" \
  --pnorm-input-dim $pnorm_input_dim \
  --pnorm-output-dim $pnorm_output_dim \
  --num-hidden-layers 2 \
  --initial-learning-rate $dnn_init_learning_rate \
  --final-learning-rate $dnn_final_learning_rate \
  --stage $stage --cleanup false \
  ${train_data_dir} data/lang $alidir $dir || exit 1 

utils/mkgraph.sh data/lang $dir $dir/graph

decode=$dir/decode_$(basename $test_data_dir)
steps/nnet2/decode.sh \
  --minimize $minimize --cmd "$decode_cmd" --nj 10 \
  --beam $dnn_beam --lat-beam $dnn_lat_beam \
  --skip-scoring false "${decode_extra_opts[@]}" \
  $dir/graph ${test_data_dir} $decode
