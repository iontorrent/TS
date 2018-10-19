/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantAssist.h
//! @ingroup  VariantCaller
//! @brief    Utilities for output of variants


#ifndef VARIANTASSIST_H
#define VARIANTASSIST_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <Variant.h>
#include "MiscUtil.h"
#include "CrossHypotheses.h"
#include "RandSchrange.h"
#include "ExtendParameters.h"

using namespace std;

class EvalFamily;
class CrossHypotheses;

class MultiBook {
public:
  vector<int> tag_similar_counts; // counts for similar molecular tags
  vector<float> lod; // limit of detection for tag seq

  MultiBook();
  void Allocate(int num_hyp);
  int NumAltAlleles() const {return _num_hyp_not_null - 1;}; // not counting Ref allele
  void ResetCounter();
  float GetFailedReadRatio() const;
  int GetDepth(int i_strand, int i_alt) const {return GetAlleleCount(i_strand, 0) + GetAlleleCount(i_strand, i_alt + 1);};
  int GetAlleleCount(int strand_key, int i_hyp) const {return _my_book[strand_key + 1][i_hyp];};
  int TotalCount(int strand_key) const;
  float OldStrandBias(int i_alt, float tune_bias) const ;
  float StrandBiasPval(int i_alt, float tune_bias) const ;
  float GetXBias(int i_alt, float tune_bias) const;
  void AssignStrandToHardClassifiedReads(const vector<int> &strand_id, const vector<int> &read_id, const vector<int>& left, const vector<int>& right);
  void FillBiDirFamBook(const vector<vector<unsigned int> >& alt_fam_indices, const vector<EvalFamily>& my_eval_families, const vector<CrossHypotheses>& my_hypotheses);
  float PositionBias(int i_alt);
  float GetPositionBiasPval(int i_alt);
  float GetPositionBias(int i_alt);
  bool IsBiDirUMT() const {return _is_bi_dir_umt;};
 private:
  vector< vector<int> > _my_book;  // my_book[0]: bi-dir, my_book[1]: FWD, my_book[2]: REV. alleles = 0 (ref), 1,2,... alts
  vector<int> allele_index; // alleles = 0 (ref), 1,2,... alts
  int _invalid_reads;
  bool _is_bi_dir_umt;
  int _num_hyp_not_null; // Number of (reference + alternatives) alleles.

  // BI-DIR STB related
  struct BiDirCov {
	  int fwd_total_cov;
	  int rev_total_cov;
	  int fwd_var_cov;
	  int rev_var_cov;
  };
  vector< vector<BiDirCov> > _my_bidir_fam_book;
  float OldStrandBiasBiDirFam_(int i_alt, float tune_bias) const;


  // Position Bias related
  struct position_bias {
    float rho;
    float pval;
  };
  vector<position_bias> my_position_bias;
  vector<int> to_left; // distance of allele from softclips in each read
  vector<int> to_right; // distance of allele from softclips in each read


  void ComputePositionBias(int i_allele);
};


class VariantOutputInfo {
  public:
    bool isFiltered;
    float variant_qual_score;
    float gt_quality_score;
    // int genotype_call;
    vector<string>  filterReason; // make sure we can push back as many as we need
    string infoString;


    VariantOutputInfo() {
      isFiltered = false;
      variant_qual_score = 0.0;
      gt_quality_score = 0.0f;
      //genotype_call = -1;
      //filterReason = "";
      //infoString = "";

    }
};


float ComputeXBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float var_zero);
float ComputeTunedXBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float proportion_zero);
float BootstrapStrandBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float tune_fish);
float ComputeTransformStrandBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float tune_fish);
inline float ComputeStrandBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth);

namespace VariantAssist {
  // helper functions for ComputePositionBias
  // todo: move to Stats.h
  inline float median(const std::vector<float>& values);
  inline void randperm(vector<unsigned int> &v, RandSchrange& rand_generator);
  inline double partial_sum(const vector<double> &v, size_t n);
  void tiedrank(vector<double> &vals);
  double MannWhitneyURho(vector<float> &ref, vector<float> &var, bool debug);
  inline double MannWhitneyU(vector<float> &ref, vector<float> &var, bool debug);

  struct mycomparison
  {
    bool operator() (double* lhs, double* rhs) {return (*lhs) < (*rhs);}
  };
}

namespace LodAssist {
double LogComplementFromLogP(double log_p);
double LinearInterpolation(double x, double x1, double y1, double x2, double y2);
double LogSumFromLogIndividual(const vector<double>& log_vec);
inline double LinearToPhread(double x);
inline double PhreadToLinear(double x);

class BinomialUtils{
private:
	vector<double> precompute_log_factorial_;
	double StirlingNamesApprox_(int x) const;

public:
	BinomialUtils();
	void PreComputeLogFactorial(int precompute_size);
	double LogNChooseK(int n, int k) const;
	double LogFactorial(int x) const;
	double LogBinomialPmf(int n, int k, double log_p, double log_q = 1.0) const;
	double LogBinomialCdf(int x, int n, double log_p, double log_q = 1.0) const;
};
}

class LodManager{
private:
	LodAssist::BinomialUtils binomial_utils_;
	int min_var_coverage_;
	double min_allele_freq_;
	double min_variant_score_;
	double min_callable_prob_;
	bool do_smoothing_;

	double CalculateCallableProb_(int dp, double af, int min_callable_ao, double qual_plus = -1.0, double qual_minus = -1.0) const;
	void CalculateMinCallableAo_(int dp, int& min_callable_ao, double& qual_plus, double& qual_minus) const;

public:
	LodManager();
	void SetParameters(int min_var_coverage, double min_allele_freq, double min_variant_score, double min_callable_prob);
	void DoSmoothing(bool do_smoothing) {do_smoothing_ = do_smoothing; };
	double CalculateLod(int dp) const;
};

#endif //VARIANTASSIST_H
