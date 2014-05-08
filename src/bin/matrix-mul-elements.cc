// bin/matrix-mul-elements.cc

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;  

    const char *usage =
        "Takes two archives of matrices (typically representing features)\n"
        "and for each utterance, outputs the element-wise product.\n"
        "Useful for applying Time-Frequency masks.\n"
        "Usage: matrix-mul-elements matrix-rspecifier1 matrix-rspecifier2 matrix-wspecifier\n";
    
    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
      
    std::string matrix_rspecifier1 = po.GetArg(1),
        matrix_rspecifier2 = po.GetArg(2),
        matrix_wspecifier = po.GetArg(3);

    kaldi::SequentialBaseFloatMatrixReader matrix_reader1(matrix_rspecifier1);
    kaldi::RandomAccessBaseFloatMatrixReader matrix_reader2(matrix_rspecifier2);
    kaldi::BaseFloatMatrixWriter matrix_writer(matrix_wspecifier); 
    
    int32 num_done = 0, num_err = 0;
    
    for (; !matrix_reader1.Done(); matrix_reader1.Next()) {
      std::string key = matrix_reader1.Key();
      const Matrix<BaseFloat> &matrix1 = matrix_reader1.Value();
      if (!matrix_reader2.HasKey(key)) {
        KALDI_WARN << "No matrix for utterance " << key << " in second table.";
        num_err++;
      } else {
        const Matrix<BaseFloat> &matrix2 = matrix_reader2.Value(key);
        if (matrix1.NumRows() != matrix2.NumRows()) {
          KALDI_WARN << "Rows mismatch for utterance " << key
                     << " : " << matrix1.NumRows() << " vs. " << matrix2.NumRows();
          num_err++;
          continue;
        }
        if (matrix1.NumCols() != matrix2.NumCols()) {
          KALDI_WARN << "Cols mismatch for utterance " << key
                     << " : " << matrix1.NumCols() << " vs. " << matrix2.NumCols();
          num_err++;
          continue;
        }

        Matrix<BaseFloat> product(matrix1);
        product.MulElements(matrix2);
        matrix_writer.Write(key, product);
        num_done++;
      }
    }
    KALDI_LOG << "Done computing element-wise products of " << num_done
              << " matrices; errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


