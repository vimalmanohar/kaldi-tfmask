// featbin/compute-mfcc-feats.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)
//           2014       Vimal Manohar

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
#include "feat/feature-mfcc.h"
#include "feat/feature-fbank.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Create MFCC feature files from Fbank feature files.\n"
        "Usage:  compute-mfcc-feats-from-fbank [options...] <fbank-feat-rspecifier> <feats-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);
    MfccOptions mfcc_opts;
    bool subtract_mean = false;
    // Define defaults for gobal options
    std::string output_format = "kaldi";

    // Register the MFCC option struct
    mfcc_opts.Register(&po);

    // Register the options
    po.Register("output-format", &output_format, "Format of the output "
                "files [kaldi, htk]");
    po.Register("subtract-mean", &subtract_mean, "Subtract mean of each "
                "feature file [CMS]; not recommended to do it this way. ");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fbank_rspecifier = po.GetArg(1);
    std::string output_wspecifier = po.GetArg(2);

    Mfcc mfcc(mfcc_opts);

    SequentialBaseFloatMatrixReader fbank_reader(fbank_rspecifier);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.
    TableWriter<HtkMatrixHolder> htk_writer;

    if (output_format == "kaldi") {
      if (!kaldi_writer.Open(output_wspecifier))
        KALDI_ERR << "Could not initialize output with wspecifier "
                  << output_wspecifier;
    } else if (output_format == "htk") {
      if (!htk_writer.Open(output_wspecifier))
        KALDI_ERR << "Could not initialize output with wspecifier "
                  << output_wspecifier;
    } else {
      KALDI_ERR << "Invalid output_format string " << output_format;
    }

    int32 num_utts = 0, num_success = 0;
    for (; !fbank_reader.Done(); fbank_reader.Next()) {
      num_utts++;
      std::string utt = fbank_reader.Key();
      const Matrix<BaseFloat> fbank = fbank_reader.Value();

      Matrix<BaseFloat> features;
      try {
        mfcc.ComputeFromFbank(fbank, &features);
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance "
                   << utt;
        continue;
      }
      if (subtract_mean) {
        Vector<BaseFloat> mean(features.NumCols());
        mean.AddRowSumMat(1.0, features);
        mean.Scale(1.0 / features.NumRows());
        for (int32 i = 0; i < features.NumRows(); i++)
          features.Row(i).AddVec(-1.0, mean);
      }
      if (output_format == "kaldi") {
        kaldi_writer.Write(utt, features);
      } else {
        std::pair<Matrix<BaseFloat>, HtkHeader> p;
        p.first.Resize(features.NumRows(), features.NumCols());
        p.first.CopyFromMat(features);
        HtkHeader header = {
          features.NumRows(),
          100000,  // 10ms shift
          static_cast<int16>(sizeof(float)*(features.NumCols())),
          static_cast<uint16>( 006 | // MFCC
          (mfcc_opts.use_energy ? 0100 : 020000)) // energy; otherwise c0
        };
        p.second = header;
        htk_writer.Write(utt, p);
      }
      if (num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }
    KALDI_LOG << " Done " << num_success << " out of " << num_utts
              << " utterances.";
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


