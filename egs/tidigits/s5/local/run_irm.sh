dir=exp/irm_nnet

pnorm_input_dim=800
pnorm_output_dim=200
dnn_init_learning_rate=0.008
dnn_final_learning_rate=0.0008
dnn_gpu_parallel_opts=(--minibatch-size 512 --max-change 40 --num-jobs-nnet 4 --num-threads 1 \
  --parallel-opts "-l gpu=1" --cmd "queue.pl -l arch=*64 -l mem_free=2G,ram_free=1G")
stage=-100

. parse_options.sh

dnn_init_learning_rate=`perl -e "print $dnn_init_learning_rate / 26"`
dnn_final_learning_rate=`perl -e "print $dnn_final_learning_rate / 26"`

steps/nnet2/train_irm_nnet.sh \
  "${dnn_gpu_parallel_opts[@]}" \
  --pnorm-input-dim $pnorm_input_dim \
  --pnorm-output-dim $pnorm_output_dim \
  --num-hidden-layers 2 \
  --initial-learning-rate $dnn_init_learning_rate \
  --final-learning-rate $dnn_final_learning_rate \
  --irm_scp data/train_noisy/irm.scp \
  --stage $stage --cleanup false \
  data/train_noisy $dir || exit 1 
