/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef POSTERIORINFERENCE_H
#define POSTERIORINFERENCE_H

#include "api/BamReader.h"

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ExtendedReadInfo.h"
#include "CrossHypotheses.h"

#include "ExtendParameters.h"
#include "ShortStack.h"

using namespace std;


// holds optimization of frequencies
class FreqMaster{
public:
  //  float max_freq;
    vector<float> max_hyp_freq;

    // prior pseudo-observations to bias our inferences when we lack data
    float germline_prior_strength; // "total pseudo-observations"
    float germline_log_prior_normalization; // for comparing LL across different priors
    vector<float> prior_frequency_weight;

    float data_reliability;
    FreqMaster();
    void SetHypFreq(const vector<float> & local_freq);
    void SetPriorStrength(const vector<float> & local_freq);
    void UpdateFrequencyAgainstOne(vector<float> &tmp_freq, float local_freq, int source_state);
    bool Compare(vector <float> &original, int numreads, float threshold);
};



class ScanSpace{
public:
  // important inferred quantities
  // integrate across pair of hypotheses
  vector<float> log_posterior_by_frequency;
  vector<float> eval_at_frequency;

  float max_ll; // current best setup
  int max_index; // location of max

  // if I am doing one allele vs another for genotyping
  vector<int> freq_pair;
  float freq_pair_weight;
  bool scan_done;
  unsigned int min_detail_level_for_fast_scan;

  ScanSpace();
  float LogDefiniteIntegral(float alpha, float beta);
  float FindMaxFrequency();
  void UpdatePairedFrequency(vector <float > &tmp_freq, FreqMaster &base_clustering, float local_freq);
  unsigned int ResizeToMatch(ShortStack &total_theory, unsigned max_detail_level = 0);
  void  DoPosteriorFrequencyScan(ShortStack &total_theory, FreqMaster &base_clustering, bool update_frequency, int strand_key, bool scan_ref, int max_detail_level = 0);
  void SetDebug(bool debug){ debug_ = debug;};

private:
  //  Calculate the posterior for just one hyp frequency
  void DoPosteriorFrequencyScanOneHypFreq_(unsigned int i_eval);
  // functions for fast scan
  void DoFastScan_();
  void DoFineScan_(unsigned int i_left, unsigned int i_right, unsigned int i_middle);
  void DoInterpolation_();
  void LinearInterpolation_(unsigned int i_1, unsigned int i_2, unsigned int i_intp);
  unsigned int FibonacciSearchMax_(unsigned int i_left, unsigned int i_right);
  unsigned int FibonacciSearchMax_(unsigned int i_left, unsigned int i_right, unsigned int i_middle);

  // variables for fast scan
  vector<bool> is_scanned_;
  unsigned int argmax_log_posterior_scanned_;
  float max_log_posterior_scanned_;
  float coarse_freq_resolution_;
  unsigned int num_of_fibonacci_blocks;
  float fine_log_posterior_cutoff_gap_;
  float fine_freq_search_increment_;
  float min_fine_log_posterior_gap_;
  float fine_scan_penalty_order_;

  // pointers for DoPosteriorFrequencyScanOneHypFreq_
  ShortStack *ptr_total_theory_ = NULL;
  FreqMaster *ptr_base_clustering_ = NULL;
  int *ptr_strand_key_ = NULL;
  bool *ptr_scan_ref_ = NULL;
  bool debug_ = false;
};

class PosteriorInference{
  public:

  ScanSpace ref_vs_all; // evidence for variant vs ref

  ScanSpace gq_pair; // genotype quality for a vs b

  FreqMaster clustering;

  float params_ll; // likelihood offset for fitted parameters

  PosteriorInference();

  void FindMaxFrequency(bool update_frequency); // from posterior likelihood

//  void InterpolateFrequencyScan(ShortStack &total_theory, bool update_frequency, int strand_key);
  // update for frequency

  void UpdateMaxFreqFromResponsibility(ShortStack &total_theory, int strand_key);
//  void StartAtNull(ShortStack &total_theory, bool update_frequency);
  void StartAtHardClassify(ShortStack &total_theory, bool update_frequency, const vector<float> &start_frequency);
  void QuickUpdateStep(ShortStack &total_theory);
//  void DetailedUpdateStep(ShortStack &total_theory, bool update_frequency);
  float ReturnMaxLL(){ return(ref_vs_all.max_ll+params_ll);};
  float ReturnJustLL(){return(ref_vs_all.max_ll);};

  void SetDebug(bool debug){
	  ref_vs_all.SetDebug(debug);
	  gq_pair.SetDebug(debug);
  };
};



#endif // POSTERIORINFERENCE_H
