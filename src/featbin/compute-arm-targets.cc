// featbin/compute-arm-targets.cc

// Copyright 2014 Vimal Manohar

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
#include "matrix/kaldi-matrix.h"
#include <limits>

namespace kaldi {
  BaseFloat sigmoid(BaseFloat x) {
    return 1 / ( 1 + Exp(-x) );
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
      "Compute soft targets for Apparent Ratio Mask used for "
      "TF-Masking. Uses clean and noisy log-fbank features.\n"
      "Give --from-noisy=true to compute from clean and noisy fbank features\n"
      "The soft function maps the ARM to a sigmoid"
      "d(t,f) = 1/1+exp(-alpha(ARM(t,f) - beta))"
      "Usage: compute-arm-targets [options] (<clean-rspecifier> <noisy-rspecifier> <target-wspecifier>) | (<clean-wxfilename> <noisy-rxfilename> <target-wxfilename>)\n"
      "e.g.: compute-arm-targets scp:clean_feats.scp scp:noisy_feats.scp ark,scp:arm_targets.ark,arm_targets.scp\n"
      "e.g.: compute-arm-targets --from-noisy=true scp:clean_feats.scp scp:noisy_feats.scp ark,scp:arm_targets.ark,arm_targets.scp\n";

    ParseOptions po(usage);
    bool binary = true;
    BaseFloat beta = -2.6;      // in dB
    BaseFloat arm_span = 8.0;    // in dB

    po.Register("binary", &binary, "Binary-mode output (not relevant if writing "
        "to archive)");
    po.Register("beta", &beta, "Shift the target sigmoid to be centered at beta (in dB)");
    po.Register("arm-span", &arm_span, "The difference between ARM values that correspond to target values of 0.05 and 0.95");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (arm_span <= 0) {
      KALDI_ERR << "--arm-span is expected to be > 0. But it is given " << arm_span;
    }

    BaseFloat alpha = 2 * Log(19.0) / Log(10.0) / arm_span;

    int32 num_done = 0, num_missing = 0, num_mismatch = 0;

    // Copying tables of features.
    std::string clean_rspecifier = po.GetArg(1);
    std::string noisy_rspecifier = po.GetArg(2);
    std::string target_wspecifier = po.GetArg(3);

    BaseFloatMatrixWriter target_writer(target_wspecifier);
    SequentialBaseFloatMatrixReader clean_reader(clean_rspecifier);
    RandomAccessBaseFloatMatrixReader noisy_reader(noisy_rspecifier);

    for (; !clean_reader.Done(); clean_reader.Next()) {
      std::string key = clean_reader.Key();
      Matrix<BaseFloat> clean_feats = clean_reader.Value();
      if (!noisy_reader.HasKey(key)) {
        KALDI_WARN << "Missing noisy features for utterance " << key;
        num_missing++;
        continue;
      }

      int32 num_frames = clean_feats.NumRows();
      int32 dim = clean_feats.NumCols();

      Matrix<BaseFloat> noisy_feats(noisy_reader.Value(key));

      if (num_frames != noisy_feats.NumRows()) {
        KALDI_WARN << "Mismatch in number of frames for clean and noisy features for utterance " << key << ": \n"
          << num_frames << " vs " << noisy_feats.NumRows() << ". " 
          << "Skippking utterance";
        num_mismatch++;
        continue;
      }

      if (dim != noisy_feats.NumCols()) {
        KALDI_WARN << "Mismatch in feature dimension for clean and noisy features for utterance " << key << ": \n"
          << dim << " vs " << noisy_feats.NumCols() << ". " 
          << "Skippking utterance";
        num_mismatch++;
        continue;
      }

      Matrix<BaseFloat> target_arm(num_frames, dim);

      clean_feats.Scale(10/Log(10.0));  // To convert log fbank feats to dB
      noisy_feats.Scale(10/Log(10.0));  // To convert log fbank feats to dB

      for (int32 i = 0; i < num_frames; i++) {
        for (int32 j = 0; j < dim; j++) {
          if (target_arm(i,j) == 0.0) {
            target_arm(i,j) = sigmoid( alpha * (clean_feats(i,j) - noisy_feats(i,j) - beta) );
          }
        }
      }

      target_writer.Write(key, target_arm);
      num_done++;
    }
    KALDI_LOG << "Computed ARM targets for " << num_done << " feature matrices, " 
      << "missing noisy features for " << num_missing << " feature matrices, "
      << "mismatch of noisy features for " << num_mismatch << " feature matrices.";
    return (num_done > num_missing + num_mismatch ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


