// featbin/compute-irm-targets.cc

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

  BaseFloat LogDiffExp(BaseFloat a, BaseFloat b) { 
    // log(exp(a)-exp(b))
    if (a <= b) {
      return (std::numeric_limits<BaseFloat>::min());
    }

    return (a + Log(1 - Exp(b - a)));
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compute soft targets for Ideal Ratio Mask used for "
        "TF-Masking. Uses clean and noise log-fbank features.\n"
        "Give --from-noisy=true to compute from clean and noisy fbank features\n"
        "The soft function maps the SNR to a sigmoid"
        "d(t,f) = 1/1+exp(-alpha(SNR(t,f) - beta))"
        "Usage: compute-irm-targets [options] (<clean-rspecifier> <noise-rspecifier> <target-wspecifier>) | (<clean-wxfilename> <noise-rxfilename> <target-wxfilename>)\n"
        "e.g.: compute-irm-targets scp:clean_feats.scp scp:noise_feats.scp ark,scp:irm_targets.ark,irm_targets.scp\n"
        "e.g.: compute-irm-targets --from-noisy=true scp:clean_feats.scp scp:noisy_feats.scp ark,scp:irm_targets.ark,irm_targets.scp\n";

    ParseOptions po(usage);
    bool binary = true;
    bool compress = false;
    BaseFloat beta = -6;      // in dB
    BaseFloat snr_span = 35;  // in dB
    bool from_noisy = false;   
    std::string snr_out;

    po.Register("binary", &binary, "Binary-mode output (not relevant if writing "
                "to archive)");
    po.Register("compress", &compress, "If true, write output in compressed form"
                "(only currently supported for wxfilename, i.e. archive/script,"
                "output)");
    po.Register("beta", &beta, "Shift the target sigmoid to be centered at beta (in dB)");
    po.Register("snr-span", &snr_span, "The difference between SNR values that correspond to target values of 0.05 and 0.95");
    po.Register("from-noisy", &from_noisy, "Compute from noisy fbank features instead of the actual noise features. Used when noise features corresponding to parallel clean and noisy data are not available");
    po.Register("snr-out", &snr_out, "Write SNR");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (snr_span <= 0) {
      KALDI_ERR << "--snr-span is expected to be > 0. But it is given " << snr_span;
    }

    BaseFloat alpha = 2 * Log(19.0) / Log(10.0) / snr_span;

    int32 num_done = 0, num_missing = 0, num_mismatch = 0;
    
    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier) {
      // Copying tables of features.
      std::string clean_rspecifier = po.GetArg(1);
      std::string noise_rspecifier = po.GetArg(2);
      std::string target_wspecifier = po.GetArg(3);
      
      if (!compress) {
        BaseFloatMatrixWriter target_writer(target_wspecifier);
        SequentialBaseFloatMatrixReader clean_reader(clean_rspecifier);
        RandomAccessBaseFloatMatrixReader noise_reader(noise_rspecifier);
        BaseFloatMatrixWriter snr_writer(snr_out);

        for (; !clean_reader.Done(); clean_reader.Next()) {
          std::string key = clean_reader.Key();
          Matrix<BaseFloat> clean_feats = clean_reader.Value();
          if (!noise_reader.HasKey(key)) {
            KALDI_WARN << "Missing noise features for utterance " << key;
            num_missing++;
            continue;
          }
          
          int32 num_frames = clean_feats.NumRows();
          int32 dim = clean_feats.NumCols();
        
          Matrix<BaseFloat> noise_feats(noise_reader.Value(key));

          if (num_frames != noise_feats.NumRows()) {
            KALDI_WARN << "Mismatch in number of frames for clean and noise features for utterance " << key << ": \n"
              << num_frames << " vs " << noise_feats.NumRows() << ". " 
              << "Skippking utterance";
            num_mismatch++;
            continue;
          }

          if (dim != noise_feats.NumCols()) {
            KALDI_WARN << "Mismatch in feature dimension for clean and noise features for utterance " << key << ": \n"
              << dim << " vs " << noise_feats.NumCols() << ". " 
              << "Skippking utterance";
            num_mismatch++;
            continue;
          }
          
          Matrix<BaseFloat> target_irm(num_frames, dim);
          target_irm.SetZero();
          
          if (from_noisy) {
            for (int32 i = 0; i < noise_feats.NumRows(); i++) {
              for (int32 j = 0; j < noise_feats.NumCols(); j++) {
                if (clean_feats(i,j) > noise_feats(i,j)) { 
                  // Clean is larger than noisy. Here we assume infinite SNR 
                  target_irm(i,j) = 1.0;
                } 
                // Assume noise_feats = noisy_feats - clean_feats for each time-frequency bin
                // independent of other bins
                noise_feats(i,j) = LogDiffExp(noise_feats(i,j), clean_feats(i,j));
              }
            }
          }
          
          clean_feats.Scale(10/Log(10.0));  // To convert log fbank feats to dB
          noise_feats.Scale(10/Log(10.0));  // To convert log fbank feats to dB

          if (snr_out != "") {
            Matrix<BaseFloat> snr(clean_feats);
            snr.AddMat(-1.0, noise_feats);
            snr_writer.Write(key, snr);
          }

          for (int32 i = 0; i < num_frames; i++) {
            for (int32 j = 0; j < dim; j++) {
              if (target_irm(i,j) == 0.0) {
                target_irm(i,j) = sigmoid( alpha * (clean_feats(i,j) - noise_feats(i,j) - beta) );
              }
            }
          }
          
          target_writer.Write(key, target_irm);
          num_done++;
        }
      } else {
        CompressedMatrixWriter target_writer(target_wspecifier);
        SequentialBaseFloatMatrixReader clean_reader(clean_rspecifier);
        RandomAccessBaseFloatMatrixReader noise_reader(noise_rspecifier);
        BaseFloatMatrixWriter snr_writer(snr_out);

        for (; !clean_reader.Done(); clean_reader.Next()) {
          std::string key = clean_reader.Key();
          Matrix<BaseFloat> clean_feats = clean_reader.Value();
          if (!noise_reader.HasKey(key)) {
            KALDI_WARN << "Missing noise features for utterance " << key;
            num_missing++;
            continue;
          }
          
          int32 num_frames = clean_feats.NumRows();
          int32 dim = clean_feats.NumCols();
          
          Matrix<BaseFloat> noise_feats(noise_reader.Value(key));
          
          if (num_frames != noise_feats.NumRows()) {
            KALDI_WARN << "Mismatch in number of frames for clean and noise features for utterance " << key << ": \n"
              << num_frames << " vs " << noise_feats.NumRows() << ". " 
              << "Skippking utterance";
            num_mismatch++;
            continue;
          }

          if (dim != noise_feats.NumCols()) {
            KALDI_WARN << "Mismatch in feature dimension for clean and noise features for utterance " << key << ": \n"
              << dim << " vs " << noise_feats.NumCols() << ". " 
              << "Skippking utterance";
            num_mismatch++;
            continue;
          }
            
          Matrix<BaseFloat> target_irm(num_frames, dim);
          target_irm.SetZero();
          
          if (from_noisy) {
            for (int32 i = 0; i < noise_feats.NumRows(); i++) {
              for (int32 j = 0; j < noise_feats.NumCols(); j++) {
                if (clean_feats(i,j) > noise_feats(i,j)) { 
                  // Clean is larger than noisy. Here we assume infinite SNR 
                  target_irm(i,j) = 1.0;
                } 
                // Assume noise_feats = noisy_feats - clean_feats for each time-frequency bin
                // independent of other bins
                noise_feats(i,j) = LogDiffExp(noise_feats(i,j), clean_feats(i,j));
              }
            }
          }
          
          clean_feats.Scale(10/Log(10.0));  // To convert log fbank feats to dB
          noise_feats.Scale(10/Log(10.0));  // To convert log fbank feats to dB

          if (snr_out != "") {
            Matrix<BaseFloat> snr(clean_feats);
            snr.AddMat(-1.0, noise_feats);
            snr_writer.Write(key, snr);
          }

          for (int32 i = 0; i < num_frames; i++) {
            for (int32 j = 0; j < dim; j++) {
              if (target_irm(i,j) == 0.0) {
                target_irm(i,j) = sigmoid( alpha * (clean_feats(i,j) - noise_feats(i,j) - beta) );
              }
            }
          }
          
          target_writer.Write(key, CompressedMatrix(target_irm));
          num_done++;
        }
      }
      KALDI_LOG << "Computed IRM targets for " << num_done << " feature matrices, " 
        << "missing noise features for " << num_missing << " feature matrices, "
        << "mismatch of noise features for " << num_mismatch << " feature matrices.";
      return (num_done > num_missing + num_mismatch ? 0 : 1);
    } else {
      KALDI_ASSERT(!compress && "Compression not yet supported for single files");
      
      std::string clean_rxfilename = po.GetArg(1),
        noise_rxfilename = po.GetArg(2),
        target_wxfilename = po.GetArg(3);

      Matrix<BaseFloat> clean_matrix;
      ReadKaldiObject(clean_rxfilename, &clean_matrix);
      Matrix<BaseFloat> noise_matrix;
      ReadKaldiObject(noise_rxfilename, &noise_matrix);
          
      int32 num_frames = clean_matrix.NumRows();
      int32 dim = clean_matrix.NumCols();

      if (num_frames != noise_matrix.NumRows()) {
        KALDI_ERR << "Mismatch in number of frames for clean and noise feature matrices: \n"
          << num_frames << " vs " << noise_matrix.NumRows() << ".";
      }

      if (dim != noise_matrix.NumCols()) {
        KALDI_ERR << "Mismatch in feature dimension for clean and noise feature matrices: \n"
          << dim << " vs " << noise_matrix.NumCols() << ".";
      }
       
      Matrix<BaseFloat> target_irm(num_frames, dim);
      target_irm.SetZero();

      if (from_noisy) {
        for (int32 i = 0; i < noise_matrix.NumRows(); i++) {
          for (int32 j = 0; j < noise_matrix.NumCols(); j++) {
            if (clean_matrix(i,j) > noise_matrix(i,j)) { 
              // Clean is larger than noisy. Here we assume infinite SNR 
              target_irm(i,j) = 1.0;
            } 
            // Assume noise_feats = noisy_feats - clean_feats for each time-frequency bin
            // independent of other bins
            noise_matrix(i,j) = LogDiffExp(noise_matrix(i,j), clean_matrix(i,j));
          }
        }
      }

      clean_matrix.Scale(10/Log(10.0));  // To convert log fbank feats to dB
      noise_matrix.Scale(10/Log(10.0));  // To convert log fbank feats to dB

      if (snr_out != "") {
        Matrix<BaseFloat> snr(clean_matrix);
        snr.AddMat(-1.0, noise_matrix);
        WriteKaldiObject(snr, snr_out, binary);
      }

      for (int32 i = 0; i < num_frames; i++) {
        for (int32 j = 0; j < dim; j++) {
          if (target_irm(i,j) == 0.0) {
            target_irm(i,j) = sigmoid( alpha * (clean_matrix(i,j) - noise_matrix(i,j) - beta) );
          }
        }
      }

      WriteKaldiObject(target_irm, target_wxfilename, binary);
      KALDI_LOG << "Computed IRM target from " 
        << "clean matrix " << clean_rxfilename << " and "
        << "noise matrix " << noise_rxfilename << ". "
        << "Wrote target to " << target_wxfilename;

      return 0;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


