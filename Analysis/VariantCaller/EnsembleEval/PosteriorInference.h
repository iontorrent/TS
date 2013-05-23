/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef POSTERIORINFERENCE_H
#define POSTERIORINFERENCE_H

#include "api/BamReader.h"

#include "../Analysis/file-io/ion_util.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>

#include "ExtendedReadInfo.h"
#include "CrossHypotheses.h"

#include "ExtendParameters.h"
#include "ShortStack.h"

using namespace std;



class PosteriorInference{
  public:
  // important inferred quantities
  vector<float> log_posterior_by_frequency;
  vector<float> eval_at_frequency;
  float max_freq;
  float max_ll; // current best setup
  int max_index; // location of max
  float params_ll; // likelihood offset for fitted parameters
  bool scan_done;
  float data_reliability;
  
  PosteriorInference();
    void FindMaxFrequency(bool update_frequency); // from posterior likelihood
  float LogDefiniteIntegral(float alpha, float beta);
  void InterpolateFrequencyScan(ShortStack &total_theory, bool update_frequency, int strand_key);
  // update for frequency
  void  DoPosteriorFrequencyScan(ShortStack &total_theory, bool update_frequency, int strand_key);
  void UpdateMaxFreqFromResponsibility(ShortStack &total_theory, int strand_key);
  void StartAtNull(ShortStack &total_theory, bool update_frequency);
  void StartAtHardClassify(ShortStack &total_theory, bool update_frequency, float start_frequency);
  unsigned int ResizeToMatch(ShortStack &total_theory);
  void QuickUpdateStep(ShortStack &total_theory);
  void DetailedUpdateStep(ShortStack &total_theory, bool update_frequency);
  float ReturnMaxLL(){ return(max_ll+params_ll);};
};



#endif // POSTERIORINFERENCE_H
