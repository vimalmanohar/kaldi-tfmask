// featbin/arm-targets-to-arm.cc

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
      "Convert targets for arm training to actual arm\n"
      "Usage: arm-targets-to-arm [options] <in-rspecifier> <out-wspecifier>\n"
      "e.g.: arm-targets-to-arm ark:- ark,scp:foo.ark,foo.scp\n";

    ParseOptions po(usage);
    BaseFloat beta = -2.6;      // in dB
    BaseFloat arm_span = 8.0;  // in dB
    bool apply_log = true;
    
    po.Register("beta", &beta, "Shift the target sigmoid to be centered at beta (in dB)");
    po.Register("arm-span", &arm_span, "The difference between ARM values that correspond to target values of 0.05 and 0.95");
    po.Register("apply-log", &apply_log, "Output logarithm of the arm");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    if (arm_span <= 0) {
      KALDI_ERR << "--arm-span is expected to be > 0. But it is given " << arm_span;
    }
    
    BaseFloat alpha = 2 * Log(19.0) / Log(10.0) / arm_span;
    
    int32 num_done = 0;

    // Copying tables of features.
    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    for (; !kaldi_reader.Done(); kaldi_reader.Next(), num_done++) {
      Matrix<BaseFloat> mat = kaldi_reader.Value();
      for (int32 t = 0; t < mat.NumRows(); t++) 
        for (int32 f = 0; f < mat.NumCols(); f++) {
          mat(t,f) = ( -Log( 1/mat(t,f) - 1 )/alpha + beta ) * Log(10.0) / 10;
        }
      if (!apply_log) mat.ApplyExp();
      kaldi_writer.Write(kaldi_reader.Key(), mat);
    }
    KALDI_LOG << "Converted " << num_done << " arm targets to arm feats.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

