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

    // Historically FreqMaster stored and used typical_prob = (1.0f - outlier_prob).
    // It can cause singular outlier_prob (i.e. outlier_prob = 0.0f) which is calculated by outlier_prob from outlier_prob = (1.0f - typical_prob) in the downstream process if typical_prob is too close to 1.
    float outlier_prob;
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
  bool scan_pair_done;
  bool scan_ref_done;
  unsigned int min_detail_level_for_fast_scan;
  unsigned int max_detail_level;
  int DEBUG;

  ScanSpace();
  float LogDefiniteIntegral(float alpha, float beta);
  float FindMaxFrequency();
  void UpdatePairedFrequency(vector <float > &tmp_freq, FreqMaster &base_clustering, float local_freq);
  unsigned int ResizeToMatch(ShortStack &total_theory, unsigned max_detail_level = 0);
  void  DoPosteriorFrequencyScan(ShortStack &total_theory, FreqMaster &base_clustering, bool update_frequency, int strand_key, bool scan_ref);
  void SetTargetMinAlleleFreq(const ExtendParameters& my_param, const vector<VariantSpecificParams>& variant_specific_params);

private:
  //  Calculate the posterior for just one hyp frequency
  void DoPosteriorFrequencyScanOneHypFreq_(unsigned int i_eval);
  // Scan all frequencies
  void DoFullScan_();
  // Update max_log_posterior_scanned_ and argmax_log_posterior_scanned_
  void UpdateMaxPosteior_(unsigned int i_eval);
  // functions for fast scan
  void DoFastScan_();
  void DoFineScan_(unsigned int i_left, unsigned int i_right, unsigned int i_middle);
  void DoInterpolation_();
  unsigned int FibonacciSearchMax_(unsigned int i_left, unsigned int i_right);
  unsigned int FibonacciSearchMax_(unsigned int i_left, unsigned int i_right, unsigned int i_middle);

  // variables for fast scan
  vector<bool> is_scanned_; // is_scanned_[i_eval] = Have I scanned for log_posterior_by_frequency[i_eval]?
  vector<float> scan_more_frequencies_;
  unsigned int argmax_log_posterior_scanned_;
  float max_log_posterior_scanned_;

  // hard coded parameters for fast scan
  const static unsigned int kNumOfFibonacciBlocks = 3; // The maximums for Hom, Het, Hom
  const static constexpr float kCoarseFreqResolution_ = 0.01f;
  const static constexpr float kFineLogPosteriorCutoffGap_ = -log(0.00001f);
  const static constexpr float kFineFreqSearchIncrement_ = 0.02f;
  const static constexpr float kMinFineLogPosteriorGap_ = 0.01f;
  const static constexpr float kFineScanPenaltyOrder_ = 1.5f;

  // pointers for DoPosteriorFrequencyScanOneHypFreq_
  ShortStack *ptr_total_theory_ = NULL;
  FreqMaster *ptr_base_clustering_ = NULL;
  int *ptr_strand_key_ = NULL;
  bool *ptr_scan_ref_ = NULL;
};

class PosteriorInference{
  public:

  ScanSpace ref_vs_all; // evidence for variant vs ref

  ScanSpace gq_pair; // genotype quality for a vs b

  FreqMaster clustering;

  float params_ll; // likelihood offset for fitted parameters

  int DEBUG;

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

};



#endif // POSTERIORINFERENCE_H
