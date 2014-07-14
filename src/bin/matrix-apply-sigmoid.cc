// bin/matrix-apply-sigmoid.cc

// Copyright 2014  Vimal Manohar

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

namespace kaldi {
  BaseFloat sigmoid(BaseFloat x) {
    return 1 / ( 1 + Exp(-x) );
  }
  
  BaseFloat inverse_sigmoid(BaseFloat x) {
    return -Log(1 / x - 1);
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Apply a sigmoid transformation or an inverse transformation on a set of matrices in a Table (useful for feature matrices)\n"
        "Usage: matrix-apply-sigmoid [options] <in-rspecifier> <out-wspecifier>\n";

    ParseOptions po(usage);

    bool inverse = false;

    po.Register("inverse", &inverse, "Apply inverse sigmoid transformation");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    BaseFloatMatrixWriter mat_writer(wspecifier);

    SequentialBaseFloatMatrixReader mat_reader(rspecifier);
    for (; !mat_reader.Done(); mat_reader.Next()) {
      Matrix<BaseFloat> mat(mat_reader.Value());

      for (int32 i = 0; i < mat.NumRows(); i++) {
        for (int32 j = 0; j < mat.NumCols(); j++) {
          if (!inverse) 
            mat(i,j) = sigmoid(mat(i,j));
          else
            mat(i,j) = inverse_sigmoid(mat(i,j));
        }
      }
      mat_writer.Write(mat_reader.Key(), mat);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

