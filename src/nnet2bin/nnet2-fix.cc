// nnet2bin/nnet2-fix.cc

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
#include "nnet2/nnet-fix.h"
#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy a (cpu-based) neural net and its associated transition model,\n"
        "but modify it to remove certain pathologies.  We use the average\n"
        "derivative statistics stored with the layers derived from\n"
        "NonlinearComponent.  Note: some processes, such as nnet2-combine-fast,\n"
        "may not process these statistics correctly, and you may have to recover\n"
        "them using the --stats-from option of nnet2-copy before you use.\n"
        "this program.\n"
        "By default reads/writes model file (.mdl) but with --raw=true,\n"
        "reads/writes raw-nnet.\n"
        "\n"
        "Usage:  nnet2-fix [options] <nnet-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet2-fix 1.mdl 1_fixed.mdl\n"
        "or:\n"
        " nnet2-fix --get-counts-from=1.gradient 1.mdl 1_shrunk.mdl\n";

    bool binary_write = true;
    NnetFixConfig config;
    bool raw = false;
    
    ParseOptions po(usage);
    po.Register("raw", &raw,
                "If true, read/write raw neural net rather than .mdl");
    po.Register("binary", &binary_write, "Write output in binary mode");
    config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        nnet_wxfilename = po.GetArg(2);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    Nnet nnet;
    if (!raw) {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    } else {
      ReadKaldiObject(nnet_rxfilename, &nnet);
    }

    FixNnet(config, (raw ? &nnet : &am_nnet.GetNnet()));
    
    if (!raw) {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    } else {
      WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
    }
    
    KALDI_LOG << "Copied neural net from " << nnet_rxfilename
              << " to " << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
