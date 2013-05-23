/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantCallUsingFlowDist.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "VariantCallUsingFlowDist.h"


#define NDX_MIN_FREQUENCY_SMALL_PEAK 0
#define NDX_MIN_FLOW_PEAK_SHORTHP_DISTANCE 1
#define NDX_MIN_FLOW_PEAK_LONGHP_DISTANCE 2
#define NDX_SHORT_HP_FOR_PEAK 3
#define NDX_MAX_PEAK_DEVIATION 4
#define NDX_PARAM_FIVE_NOT_USED 5
#define NDX_CALL_USING_EM_METHOD 6
#define NDX_PARAM_SEVEN_NOT_USED 7
#define NDX_STRAND_BIAS 8

float compute_bayesian_score(float maxProb) {
  float BayesianScore;
  if (maxProb >= 1)
    BayesianScore = 100.0f;
  else
    if (maxProb >= 0.5f)
      BayesianScore = -10.0f*log10(1.0f-maxProb);
    else
      BayesianScore = 0.0f;
  return(BayesianScore);
}

void compute_maxProb(float *peak_finding_results, int refHpLen, float &maxProb, bool &isReferenceCall, float refBias) {
    (void)refBias;
    if (peak_finding_results[0] > peak_finding_results[2]) { //unimodal distribution
    maxProb = peak_finding_results[0];
    if (fabs(peak_finding_results[1]-refHpLen*100) < 50)
      isReferenceCall = true;
  } else {
    maxProb = peak_finding_results[2];
  }
  //cout << "Checking for reference call : peak finder mean = " << peak_finding_results[1] << " ref len  " << refHpLen << " isRef " << isReferenceCall << endl;
}

// this should be a method for flowDist?
void UpdateFlowDistWithOrdinaryVariant(FlowDist *flowDist, bool strand, float refLikelihood, float minDistToRead, float distDelta) {
 
  bool happy_variant = (refLikelihood<distDelta && minDistToRead <15.0f);
  //bool sufficient_information_to_call = (distDelta>15.0f);

  flowDist->summary_stats.UpdateSummaryStats(strand,happy_variant, 0.0f);

  if (happy_variant /*&& sufficient_information_to_call*/) {
    flowDist->getReferenceLikelihoods()->push_back(refLikelihood);
    flowDist->getVariantLikelihoods()->push_back(distDelta);
  }
}



// this should be a method for flowDist?
void UpdateFlowDistWithLongVariant(FlowDist *flowDist, bool strand, float delta, AlleleIdentity &variant_identity, LocalReferenceContext seq_context) {
  float hundred_delta = delta*100.0f;
  int trunc_delta = (int)hundred_delta;

  if (hundred_delta > 0.0f && hundred_delta < MAXSIGDEV)
    flowDist->getHomPolyDist()[trunc_delta]++;

  bool variant_evidence = (fabs(delta - seq_context.my_hp_length.at(0)) > fabs(delta - (seq_context.my_hp_length.at(0) - variant_identity.inDelLength)));

  flowDist->summary_stats.UpdateSummaryStats(strand,variant_evidence, trunc_delta);

}


void CalculateOrdinaryScore(FlowDist *flowDist, AlleleIdentity &variant_identity,
                            vcf::Variant ** candidate_variant, ControlCallAndFilters &my_controls,
                            bool *isFiltered, int DEBUG) {
  vector<float>* reflikelihoods = flowDist->getReferenceLikelihoods();
  vector<float>* varlikelihoods = flowDist->getVariantLikelihoods();
  int totalReads = reflikelihoods->size();
  const int totalHypotheses = 2;
  float **scoreLikelihoods;
  float refLikelihoods;
  float varLikelihoods;
  float scores[totalHypotheses]  = {0};
  int counts[totalHypotheses] = {0};
  float minDiff = 2.0;
  float minLikelihoods = 3.0;
  allocateArray(&scoreLikelihoods, totalReads, totalHypotheses);

  for (int i = 0; i < totalReads; i++) {
    refLikelihoods = reflikelihoods->at(i);
    varLikelihoods = varlikelihoods->at(i);
    if (DEBUG)
      cout << "ref likelihood = " << refLikelihoods << endl;
    scoreLikelihoods[i][0] = refLikelihoods;
    scoreLikelihoods[i][1] = varLikelihoods;
  }

  if (variant_identity.status.isSNP || variant_identity.status.isMNV) {
    minDiff = 1.0;
    minLikelihoods = 0.5;
  }

  calc_score_hyp(totalReads, totalHypotheses, scoreLikelihoods, scores, counts, minDiff, minLikelihoods);
  float BayesianScore;

  BayesianScore = scores[1];
  //string *filterReason = new string();

  //cout << "Bayesian Score = " << BayesianScore << endl;
  float stdBias = flowDist->summary_stats.getStrandBias();
  float refBias = flowDist->summary_stats.getRefStrandBias();
  //float baseStdBias = flowDist->summary_stats.getBaseStrandBias();



  /* moving filter operation to DecisionTree
  *isFiltered = filterVariants(filterReason, variant_identity.status.isSNP, variant_identity.status.isMNV, variant_identity.status.isIndel, variant_identity.status.isHPIndel,
                               false, BayesianScore, my_controls, 0, 0, 0, stdBias, refBias, baseStdBias,
                               flowDist->summary_stats.getAltAlleleFreq(), variant_identity.refHpLen, abs(variant_identity.inDelLength));

  flowDist->summary_info.isFiltered= *isFiltered;
  */

  flowDist->summary_info.alleleScore = BayesianScore;

  //flowDist->summary_info.filterReason = *filterReason;

  //now if the allele is a SNP and a possible overcall/undercall FP SNP, evaluate the score for HP lengths on either side of SNP
  //we move filtering to final stage so calculate confidence of HP length for all overcall/undercall snps
  if (variant_identity.status.isSNP && variant_identity.status.isOverCallUnderCallSNP &&
      (variant_identity.underCallLength+1 > 11 || variant_identity.overCallLength-1 > 11) ) {
    flowDist->summary_info.isFiltered = true;
    flowDist->summary_info.filterReason = "Overcall/Undercall_HP_SNP";
    flowDist->summary_info.alleleScore  = 0.0;
  }
  if (!flowDist->summary_info.isFiltered && variant_identity.status.isSNP && variant_identity.status.isOverCallUnderCallSNP) {
    float overCallHPScore = 0.0f;
    float underCallHPScore = 0.0f;
    bool isUnderCallRef = false;
    bool isOverCallRef = false;
    float underCallFreq = 0.0;
    float overCallFreq = 0.0;
    float maxProb = 0;
    int * peak_finding_tuning_parameters = new int[9];
    peak_finding_tuning_parameters[NDX_MIN_FREQUENCY_SMALL_PEAK] = (int)(my_controls.filter_hp_indel.min_allele_freq *100);
    peak_finding_tuning_parameters[NDX_MIN_FLOW_PEAK_SHORTHP_DISTANCE] = 85;
    peak_finding_tuning_parameters[NDX_MIN_FLOW_PEAK_LONGHP_DISTANCE] = 85;
    peak_finding_tuning_parameters[NDX_SHORT_HP_FOR_PEAK] = 8;
    peak_finding_tuning_parameters[NDX_MAX_PEAK_DEVIATION] = my_controls.control_peak.fpe_max_peak_deviation;
    peak_finding_tuning_parameters[NDX_PARAM_FIVE_NOT_USED] = 0;
    peak_finding_tuning_parameters[NDX_CALL_USING_EM_METHOD] = 0;
    peak_finding_tuning_parameters[NDX_PARAM_SEVEN_NOT_USED] = 0;
    peak_finding_tuning_parameters[NDX_STRAND_BIAS] = (int)(0.5*100);

    float * peak_finding_results = new float[13];

    int optimization_start, optimization_end;

    optimization_start = max((variant_identity.underCallLength-3)*100, 0);
    optimization_end = min((variant_identity.underCallLength+3)*100, MAXSIGDEV);
    int variation_allowed = variant_identity.underCallLength;

    runLMS((int*) flowDist->getHomPolyDist(), MAXSIGDEV, peak_finding_tuning_parameters, peak_finding_results,
           optimization_start, optimization_end,
           variant_identity.underCallLength+1, variation_allowed, DEBUG);




    compute_maxProb(peak_finding_results, variant_identity.underCallLength+1, maxProb, isUnderCallRef, refBias);

    underCallHPScore = compute_bayesian_score(maxProb);
    underCallFreq = peak_finding_results[5];

    delete[] peak_finding_results;

    //now evaluate the overcall HP length

    maxProb = 0;
    peak_finding_results = new float[13];
    for (size_t i = 0; i < 13; i++ )
      peak_finding_results[i] = 0;


    optimization_start = max((variant_identity.overCallLength-3)*100, 0);
    optimization_end = min((variant_identity.overCallLength+3)*100, MAXSIGDEV);
    variation_allowed = variant_identity.overCallLength;

    runLMS((int*) flowDist->getHomPolyDist(), MAXSIGDEV, peak_finding_tuning_parameters, peak_finding_results,
           optimization_start, optimization_end,
           variant_identity.overCallLength-1, variation_allowed, DEBUG);

    delete[] peak_finding_tuning_parameters;


    compute_maxProb(peak_finding_results, variant_identity.overCallLength-1, maxProb, isOverCallRef, refBias);

    overCallHPScore = compute_bayesian_score(maxProb);
    overCallFreq = peak_finding_results[5];

    //not sure how to move this part to decision tree, leaving it here for now.
    if (isUnderCallRef || isOverCallRef
        || underCallHPScore < 5 || overCallHPScore < 5
        || overCallFreq < my_controls.filter_snps.min_allele_freq || underCallFreq < my_controls.filter_snps.min_allele_freq) {
      //filter the variant as possible overcall undercall FP
      flowDist->summary_info.isFiltered = true;
      flowDist->summary_info.filterReason = "Overcall/Undercall_HP_SNP";
      flowDist->summary_info.alleleScore  = 0.0;
    }

  }

  stringstream infoss;


  infoss << "Score= " << scores[1] << " | STDBIAS= "<< stdBias;

  flowDist->summary_info.infoString = infoss.str();

  //InsertGenericInfoTag(candidate_variant, infoss);


  //if (filterReason!=NULL)
  //  delete filterReason;

  deleteArray(&scoreLikelihoods, totalReads, totalHypotheses);
}





void RunPeakFinderOnFlowDist(FlowDist *flowDist, MultiAlleleVariantIdentity& multi_variant, unsigned int allele_idx,
                             ControlCallAndFilters &my_controls, bool *isFiltered, int DEBUG,
                             string *filterReason, stringstream &infoss, float &BayesianScore, float &maxProb) {
  // CK: Replaced in fuction call: AlleleIdentity &variant_identity, vcf::Variant ** candidate_variant,

  float stdBias = flowDist->summary_stats.getStrandBias();
  float refBias = flowDist->summary_stats.getRefStrandBias();
  //float baseStdBias = flowDist->summary_stats.getBaseStrandBias();

  bool isReferenceCall = false;

  int * peak_finding_tuning_parameters = new int[9];
  peak_finding_tuning_parameters[NDX_MIN_FREQUENCY_SMALL_PEAK] = (int)(my_controls.filter_hp_indel.min_allele_freq  *100);
  peak_finding_tuning_parameters[NDX_MIN_FLOW_PEAK_SHORTHP_DISTANCE] = 85;
  peak_finding_tuning_parameters[NDX_MIN_FLOW_PEAK_LONGHP_DISTANCE] = 85;
  peak_finding_tuning_parameters[NDX_SHORT_HP_FOR_PEAK] = 8;
  peak_finding_tuning_parameters[NDX_MAX_PEAK_DEVIATION] = my_controls.control_peak.fpe_max_peak_deviation;
  peak_finding_tuning_parameters[NDX_PARAM_FIVE_NOT_USED] = 0;
  peak_finding_tuning_parameters[NDX_CALL_USING_EM_METHOD] = 0;
  peak_finding_tuning_parameters[NDX_PARAM_SEVEN_NOT_USED] = 0;
  peak_finding_tuning_parameters[NDX_STRAND_BIAS] = (int)(stdBias*100);

  float * peak_finding_results = new float[13];
  //initialize results
  for (size_t i = 0; i < 13; i++)
    peak_finding_results[i] = 0.0;

  int optimization_start, optimization_end;

  optimization_start = max((multi_variant.seq_context.my_hp_length.at(0)-3)*100, 0);
  optimization_end = min((multi_variant.seq_context.my_hp_length.at(0)+3)*100, MAXSIGDEV);
  int variation_allowed = (multi_variant.seq_context.my_hp_length.at(0) - multi_variant.allele_identity_vector[allele_idx].inDelLength);

  runLMS((int*) flowDist->getHomPolyDist(), MAXSIGDEV, peak_finding_tuning_parameters, peak_finding_results,
         optimization_start, optimization_end,
         multi_variant.seq_context.my_hp_length.at(0), variation_allowed, DEBUG);

  delete[] peak_finding_tuning_parameters;


  compute_maxProb(peak_finding_results, multi_variant.seq_context.my_hp_length.at(0), maxProb, isReferenceCall, refBias);

  BayesianScore = compute_bayesian_score(maxProb);

  /* If peak finder calls for a reference call then set the variant status to reference call */
  multi_variant.allele_identity_vector[allele_idx].status.isReferenceCall = isReferenceCall;
  //cout << "VariantCallingUsingFlowDist is Reference = " << variant_identity.status.isReferenceCall << endl;

  /*MOVE this filtering to Decision tree
  if (flowDist->summary_stats.getDepth() < my_controls.min_cov_hp_del) {
    variant_identity.status.isNoCallVariant = true;

  }

  *isFiltered = filterVariants(filterReason, false, false, variant_identity.status.isIndel,
                               variant_identity.status.isHPIndel, isReferenceCall, BayesianScore,
                               my_controls, peak_finding_results[7], peak_finding_results[8], peak_finding_results[9],
                               stdBias, refBias, baseStdBias, peak_finding_results[5], variant_identity.refHpLen, 1);

  */

  flowDist->summary_stats.setAltAlleleFreq((float)peak_finding_results[5]);

  //cout << " Output of filter variants " << *isFiltered << endl;
  //(*candidate_variant)->quality = BayesianScore;
  /*MOVED to decision tree
  if (*isFiltered) {
    infoss << "FAIL-" << *filterReason;
    //(*candidate_variant)->filter = "FAIL";
  } else {
    infoss << "PASS-";
    //(*candidate_variant)->filter = "PASS";
  }
  */
  if (peak_finding_results != NULL) {

      infoss << "ProbUniModal=" << peak_finding_results[0];
      infoss << "UniModalMean=" << peak_finding_results[1];
      infoss << "UniModalStd=" << peak_finding_results[7];
      infoss << "ProbBiModal=" << peak_finding_results[2];
      infoss << "BiModalMean1=" << peak_finding_results[3];
      infoss << "BiModalMean2=" << peak_finding_results[4];
      infoss << "BiModalStd1=" << peak_finding_results[8];
      infoss << "BiModalStd2=" << peak_finding_results[9];
      infoss << "AlleleFreq1=" << peak_finding_results[5];
      infoss << "AlleleFreq2=" << peak_finding_results[6];

  }

  if (peak_finding_results != NULL)
    delete[] peak_finding_results;
}

void CalculatePeakFindingScore(FlowDist *flowDist, MultiAlleleVariantIdentity& multi_variant, unsigned int allele_idx,
		                       ControlCallAndFilters &my_controls, bool *isFiltered, int DEBUG) {
  //CK: Replaced in function call: AlleleIdentity &variant_identity, vcf::Variant ** candidate_variant,

  float maxProb = 0.0f;

  stringstream infoss;
  string * filterReason = new string();
  float BayesianScore=0.0f;
  if (DEBUG)
    cout << "Calling LMS using Ref Length = " << multi_variant.seq_context.my_hp_length.at(0) << " Var Length = " << (multi_variant.seq_context.my_hp_length.at(0) - multi_variant.allele_identity_vector[allele_idx].inDelLength) << endl;

  if (multi_variant.seq_context.my_hp_length.at(0) <= 11 && !PrefilterSummaryStats(flowDist->summary_stats, my_controls, isFiltered, filterReason,infoss)) {
    RunPeakFinderOnFlowDist(flowDist, multi_variant, allele_idx, my_controls, isFiltered, DEBUG, filterReason,
                            infoss, BayesianScore, maxProb);
  } else {
    if (multi_variant.seq_context.my_hp_length.at(0) > 11)
      *filterReason = "HP LENGTH > 11";

    flowDist->summary_info.alleleScore = 0.0;
    flowDist->summary_info.isFiltered = true;
    flowDist->summary_info.filterReason = *filterReason;
    flowDist->summary_info.infoString = infoss.str();

    if (filterReason != NULL)
      delete filterReason;

    return;

  }

  //infoss << maxProb << "|" <<  flowDist->summary_stats.getStrandBias() << "|";

  flowDist->summary_info.alleleScore = BayesianScore;
  flowDist->summary_info.isFiltered = *isFiltered;
  flowDist->summary_info.filterReason = *filterReason;
  flowDist->summary_info.infoString = infoss.str();



  if (filterReason!=NULL)
    delete filterReason;

}



