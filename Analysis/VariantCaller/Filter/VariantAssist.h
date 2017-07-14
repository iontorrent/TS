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
#include "api/api_global.h"
#include "api/BamAux.h"
#include "api/BamConstants.h"
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "api/SamReadGroup.h"
#include "api/SamReadGroupDictionary.h"
#include "api/SamSequence.h"
#include "api/SamSequenceDictionary.h"

#include "sys/types.h"
#include "sys/stat.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include <Variant.h>
#include "MiscUtil.h"
#include "RandSchrange.h"
// ugly, too many headers
#include "ExtendParameters.h"

using namespace std;
using namespace BamTools;

class MultiBook {
public:
  // strand = 0/1
  // alleles = 0 (ref), 1,2,... alts
  vector< vector<int> > my_book;
  int invalid_reads;
  vector<int> tag_similar_counts; // counts for similar molecular tags
  vector<float> lod; // limit of detection for tag seq

  MultiBook();
  void Allocate(int num_hyp);
  float GetFailedReadRatio();
  void SetCount(int i_strand, int i_hyp, int count);
  int GetDepth(int i_strand, int i_alt);
  float OldStrandBias(int i_alt, float tune_bias);
  float StrandBiasPval(int i_alt, float tune_bias);
  float GetXBias(int i_alt, float tune_bias);
  int GetAlleleCount(int strand_key, int i_hyp);
  int TotalCount(int strand_key);
  int NumAltAlleles();
  void ResetCounter();
  void AssignStrandToHardClassifiedReads(const vector<bool> &strand_id, const vector<int> &read_id);
  void AssignPositionFromEndToHardClassifiedReads(const vector<int> &read_id, const vector<int> &left, const vector<int> &right);
  float PositionBias(int i_alt);
  float GetPositionBiasPval(int i_alt);
  float GetPositionBias(int i_alt);

 private:
  struct position_bias {
    float rho;
    float pval;
  };
  vector<position_bias> my_position_bias;

  // distance of allele from softclips in each read
  vector<int> allele_index; // alleles = 0 (ref), 1,2,... alts
  vector<int> to_left;
  vector<int> to_right;

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
  inline float median(std::vector<float>& values);
  inline void randperm(vector<unsigned int> &v, RandSchrange& rand_generator);
  inline double partial_sum(vector<double> &v, size_t n);
  void tiedrank(vector<double> &vals);
  double MannWhitneyURho(vector<float> &ref, vector<float> &var, bool debug);
  inline double MannWhitneyU(vector<float> &ref, vector<float> &var, bool debug);

  struct mycomparison
  {
    bool operator() (double* lhs, double* rhs) {return (*lhs) < (*rhs);}
  };
}

#endif //VARIANTASSIST_H
