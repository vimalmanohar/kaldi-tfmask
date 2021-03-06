// nnet2bin/nnet2-info.cc

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
#include "nnet2/am-nnet.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Print human-readable information about the neural network\n"
        "to the standard output\n"
        "By default reads model file (.mdl) but with --raw=true,\n"
        "reads/writes raw-nnet.\n"
        "Usage:  nnet2-info [options] <nnet-in>\n"
        "e.g.:\n"
        " nnet2-info 1.nnet\n";
        
    bool raw = false;

    ParseOptions po(usage);
    po.Register("raw", &raw,
                "If true, read/write raw neural net rather than .mdl");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1);
    
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

    if (!raw) {
      std::cout << am_nnet.Info();
    } else {
      std::cout << nnet.Info();
    }

    KALDI_LOG << "Printed info about " << nnet_rxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}




