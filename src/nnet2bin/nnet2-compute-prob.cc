// nnet2bin/nnet2-compute-prob.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet2/nnet-randomize.h"
#include "nnet2/train-nnet.h"
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-update.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Computes and prints the average log-prob per frame of the given data with a\n"
        "neural net.  The input of this is the output of e.g. nnet22-get-egs\n"
        "Aside from the logging output, which goes to the standard error, this program\n"
        "prints the average log-prob per frame to the standard output.\n"
        "Also see nnet2-logprob, which produces a matrix of log-probs for each utterance.\n"
        "By default reads/writes model file (.mdl) but with --raw=true,\n"
        "reads/writes raw-nnet.\n"
        "\n"
        "Usage:  nnet2-compute-prob [options] <model-in> <training-examples-in>\n"
        "e.g.: nnet2-compute-prob 1.nnet ark:valid.egs\n";
    
    bool raw = false;
    NnetUpdaterConfig updater_config;

    ParseOptions po(usage);
    po.Register("raw", &raw,
                "If true, read/write raw neural net rather than .mdl");

    updater_config.Register(&po);

    po.Read(argc, argv);
   
    KALDI_LOG << po.NumArgs();
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2);

    TransitionModel trans_model;
    AmNnet am_nnet;
    Nnet nnet;
    if (!raw) {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    } else { 
      ReadKaldiObject(nnet_rxfilename, &nnet);
    }

    std::vector<NnetExample> examples;
    double tot_like = 0;
    int64 num_examples = 0;
    SequentialNnetExampleReader example_reader(examples_rspecifier);
    for (; !example_reader.Done(); example_reader.Next(), num_examples++) {
      if (examples.size() == 1000) {
        tot_like += ComputeNnetObjf((raw ? nnet : am_nnet.GetNnet()), examples, updater_config);
        examples.clear();
      }
      examples.push_back(example_reader.Value());
      if (num_examples % 5000 == 0 && num_examples > 0)
        KALDI_LOG << "Saw " << num_examples << " examples, average "
                  << "probability is " << (tot_like / num_examples) << " with "
                  << "total weight " << num_examples;
    }
    if (!examples.empty()) {
      tot_like += ComputeNnetObjf((raw ? nnet : am_nnet.GetNnet()), examples, updater_config);
    }

    KALDI_LOG << "Saw " << num_examples << " examples, average "
              << "probability is " << (tot_like / num_examples) << " with "
              << "total weight " << num_examples;
    
    std::cout << (tot_like / num_examples) << "\n";
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


