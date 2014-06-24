// featbin/wav-difference.cc

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
#include "feat/wave-reader.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Compute the difference between two archives wave files.\n"
        "Typically used to get the noise from noisy and clean wav archives.\n"
        "\n"
        "Usage:  wav-difference [options...] <wav-rspecifier1> <wav-rspecifier2> <wav-wspecifier>\n"
        "e.g. wav-difference scp:noisy.scp scp:clean.scp ark:-\n"
        "See also: compute-irm-targets\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier1 = po.GetArg(1),
        wav_rspecifier2 = po.GetArg(2),
        wav_wspecifier = po.GetArg(3);

    int32 num_done = 0, num_missing = 0, num_mismatch = 0;
    
    SequentialTableReader<WaveHolder> wav_reader1(wav_rspecifier1);
    RandomAccessTableReader<WaveHolder> wav_reader2(wav_rspecifier2);
    TableWriter<WaveHolder> wav_writer(wav_wspecifier);

    for (; !wav_reader1.Done(); wav_reader1.Next()) {
      std::string key = wav_reader1.Key();
      
      if (! wav_reader2.HasKey(key)) {
        KALDI_WARN << "Second wave archive does not have key " << key;
        num_missing++;
        continue;
      }
      
      WaveData wav1 = wav_reader1.Value();
      int32 num_channels = wav1.NumChannels();
      int32 num_samples = wav1.NumSamples();
      BaseFloat samp_freq = wav1.SampFreq();
      
      WaveData wav2 = wav_reader2.Value(key);

      if (num_channels != wav2.NumChannels()) {
        KALDI_WARN << "Mismatch in Number of Channels for key " << key
          << ", " << num_channels << " vs " << wav2.NumChannels();
        num_mismatch++;
        continue;
      }
      
      if (num_samples != wav2.NumSamples()) {
        KALDI_WARN << "Mismatch in Number of Samples for key " << key
          << ", " << num_samples << " vs " << wav2.NumSamples();
        num_mismatch++;
        continue;
      }
      if (samp_freq != wav2.SampFreq()) {
        KALDI_WARN << "Mismatch in Sampling Frequency for key " << key
          << ", " << samp_freq << " vs " << wav2.SampFreq();
        num_mismatch++;
        continue;
      }
    
      Matrix<BaseFloat> diff_data(wav1.Data());
      Matrix<BaseFloat> data2(wav2.Data());

      diff_data.AddMat(-1.0, data2);

      WaveData diff_wav(samp_freq, diff_data);
      
      wav_writer.Write(key, diff_wav);
      num_done++;
    }
    KALDI_LOG << "Computed wav difference for " << num_done 
      << " wave files; missing " << num_missing << " wave files, "
      << "mismatch in " << num_mismatch << " wave files.\n";
    return (num_done > (num_mismatch + num_missing) ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

