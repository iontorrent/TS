/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "EnsembleGlue.h"

// glue function temporarily here

void GlueOneRead(CrossHypotheses &my_set, ExtendedReadInfo &current_read, AlleleIdentity &variant_identity,
                 string refAllele, const string &local_contig_sequence, int DEBUG) {
  //int DEBUG = 0;

  vector<string> my_tmp_hypotheses;

  // this function call is of course horrible and needs refactoring and/or replacement
  int evaluate_splicing = HypothesisSpliceVariant(my_tmp_hypotheses,
                          current_read,
                          variant_identity,
                          refAllele,
                          local_contig_sequence,
                          DEBUG);

  // routine outputs in the wrong order
  my_set.instance_of_read_by_state.resize(3);
  // if failed to splice, >no tmp hypotheses<
  if (evaluate_splicing == 0) {
    my_set.instance_of_read_by_state[0] = my_tmp_hypotheses[2];  // read
    my_set.instance_of_read_by_state[1] = my_tmp_hypotheses[0];  // reference
    my_set.instance_of_read_by_state[2] = my_tmp_hypotheses[1];  // variant
  }
  else {
    my_set.instance_of_read_by_state[0] = current_read.alignment.QueryBases;  // CK: Why???
    my_set.instance_of_read_by_state[1] = current_read.alignment.QueryBases;
    my_set.instance_of_read_by_state[2] = current_read.alignment.QueryBases;
  }
  my_set.success = (evaluate_splicing == 0);

}


void BootUpAllAlleles(EnsembleEval &my_ensemble, const string & local_contig_sequence, int DEBUG) {
  // set up all variants
  for (unsigned int i_allele = 0; i_allele < my_ensemble.multi_allele_var.allele_identity_vector.size(); i_allele++) {

    // build a unique identifier -> Unnecessary variable duplication
    my_ensemble.allele_eval[i_allele].variant_position = my_ensemble.multi_allele_var.allele_identity_vector[i_allele].modified_start_pos;
    my_ensemble.allele_eval[i_allele].ref_allele = (*(my_ensemble.multi_allele_var.variant))->ref;
    my_ensemble.allele_eval[i_allele].var_allele = (*(my_ensemble.multi_allele_var.variant))->alt[i_allele];
    // unique identifier built

    // generate null+two hypotheses per read
    my_ensemble.allele_eval[i_allele].total_theory.my_hypotheses.resize(my_ensemble.my_data.read_stack.size());
    for (unsigned int i_read = 0; i_read < my_ensemble.allele_eval[i_allele].total_theory.my_hypotheses.size(); i_read++) {
      // Wrapper for legacy splicing function
      GlueOneRead(my_ensemble.allele_eval[i_allele].total_theory.my_hypotheses[i_read], my_ensemble.my_data.read_stack[i_read],
                  my_ensemble.multi_allele_var.allele_identity_vector[i_allele],
                  (*(my_ensemble.multi_allele_var.variant))->ref, local_contig_sequence, DEBUG);
    }
    // fill in quantities derived from predictions
    my_ensemble.allele_eval[i_allele].InitForInference(my_ensemble.my_data);

  }
}

// handle the unfortunate case in which we must try multiple alleles to be happy
// try only ref vs alt allele here
// leave ensemble in ref vs alt state
int TrySolveAllAllelesVsRef(EnsembleEval &my_ensemble, const string & local_contig_sequence, int DEBUG) {


  BootUpAllAlleles(my_ensemble, local_contig_sequence, DEBUG);
  // join information across multi-alleles

  //my_ensemble.UnifyTestFlows();

  // do inference

  my_ensemble.ExecuteInferenceAllAlleles();
  // now we're in the guaranteed state of best index
  return(my_ensemble.DetectBestAllele());
}

//@TODO: move this into DecisionTree and read from the tag RBI instead
void SpecializedFilterFromLatentVariables(HypothesisStack &hypothesis_stack, DecisionTreeData &my_decision, float bias_radius, int _allele) {

  float bias_threshold;
  // likelihood threshold
  if (bias_radius < 0.0f)
    bias_threshold = 100.0f; // oops, wrong variable - should always be positive
  else
    bias_threshold = bias_radius; // fine now

  float radius_bias = hypothesis_stack.cur_state.bias_generator.RadiusOfBias();

  if (radius_bias > bias_threshold) {
    stringstream filterReasonStr;
    filterReasonStr << "PREDICTIONSHIFTx" ;
    filterReasonStr << radius_bias;
    string my_tmp_string = filterReasonStr.str();
    my_decision.OverrideFilter(my_tmp_string, _allele);
  }
}

void SpecializedFilterFromHypothesisBias(HypothesisStack &hypothesis_stack, AlleleIdentity allele_identity, DecisionTreeData &my_decision, float deletion_bias, float insertion_bias, int _allele) {



  float ref_bias = hypothesis_stack.cur_state.bias_generator.latent_bias_v[0];
  float var_bias = hypothesis_stack.cur_state.bias_generator.latent_bias_v[1];

  if (allele_identity.status.isHPIndel) {
    if (allele_identity.status.isDeletion) {
      if ((ref_bias > 0 && fabs(ref_bias) > deletion_bias) || (var_bias > 0 && fabs(var_bias) > deletion_bias)) {
        stringstream filterReasonStr;
        filterReasonStr << "PREDICTIONHypSHIFTx" ;
        filterReasonStr << var_bias;
        string my_tmp_string = filterReasonStr.str();
        my_decision.OverrideFilter(my_tmp_string, _allele);
      }
    }
    else if (allele_identity.status.isInsertion) {
      if ((ref_bias > 0 && fabs(ref_bias) > insertion_bias) || (var_bias > 0 && fabs(var_bias) > insertion_bias)) {
              stringstream filterReasonStr;
              filterReasonStr << "PREDICTIONHypSHIFTx" ;
              filterReasonStr << var_bias;
              string my_tmp_string = filterReasonStr.str();
              my_decision.OverrideFilter(my_tmp_string, _allele);
            }
    }
  }

}

void EnsembleSummaryStats(EnsembleEval &my_ensemble, DecisionTreeData &my_decision) {
  // demonstrate this item
  vector<int> read_id;
  vector<bool> strand_id;
  // pretend we can classify reads across multiple alleles
  my_ensemble.ApproximateHardClassifierForReads(read_id, strand_id);
  // each read goes to a summary place for the right alternative
  for (unsigned int i_read = 0; i_read < read_id.size(); i_read++) {
    int _allele_index = read_id[i_read];
    for (unsigned int i_alt = 0; i_alt < my_decision.summary_stats_vector.size(); i_alt++) {
      if (_allele_index == (int)(i_alt + 1))
        my_decision.summary_stats_vector[i_alt].UpdateSummaryStats(strand_id[i_read], true, 0.0f);
      else
        if (_allele_index == 0)
          my_decision.summary_stats_vector[i_alt].UpdateSummaryStats(strand_id[i_read], false, 0.0f);
      // otherwise no deal - doesn't count for this alternate at all
    }
  }
}

void SummarizeInfoFieldsFromEnsemble(EnsembleEval &my_ensemble, vcf::Variant ** candidate_variant, int _best_allele_index) {
  //my_ensemble.allele_eval[_best_allele_index].
  float mean_flows_disrupted;
  float max_flow_discrimination;
  float mean_ll_delta;

  my_ensemble.ScanSupportingEvidence(mean_ll_delta, mean_flows_disrupted, max_flow_discrimination, 2.0f, _best_allele_index);
//   (*candidate_variant)->info["MFDT"].push_back(convertToString(mean_flows_disrupted));
//   (*candidate_variant)->info["MXFD"].push_back(convertToString(max_flow_discrimination));
  (*candidate_variant)->info["MLLD"].push_back(convertToString(mean_ll_delta));

//   float biasLL =  my_ensemble.allele_eval[_best_allele_index].cur_state.bias_generator.BiasLL();
//   (*candidate_variant)->info["BLL"].push_back(convertToString(biasLL));

  float radius_bias = my_ensemble.allele_eval[_best_allele_index].cur_state.bias_generator.RadiusOfBias();
  (*candidate_variant)->info["RBI"].push_back(convertToString(radius_bias));

  float fwd_bias = my_ensemble.allele_eval[_best_allele_index].cur_state.bias_generator.latent_bias[0];
  float rev_bias = my_ensemble.allele_eval[_best_allele_index].cur_state.bias_generator.latent_bias[1];
  float ref_bias = my_ensemble.allele_eval[_best_allele_index].cur_state.bias_generator.latent_bias_v[0];
  float var_bias = my_ensemble.allele_eval[_best_allele_index].cur_state.bias_generator.latent_bias_v[1];
  (*candidate_variant)->info["FWDB"].push_back(convertToString(fwd_bias));
  (*candidate_variant)->info["REVB"].push_back(convertToString(rev_bias));
  (*candidate_variant)->info["REFB"].push_back(convertToString(ref_bias));
  (*candidate_variant)->info["VARB"].push_back(convertToString(var_bias));
  
}


void GlueOutputVariant(EnsembleEval &my_ensemble, ExtendParameters *parameters, int _best_allele_index) {

  DecisionTreeData my_decision;

  my_decision.SetupFromMultiAllele(my_ensemble.multi_allele_var);
  my_decision.SetupSummaryStatsFromCandidate(my_ensemble.multi_allele_var.variant);
  EnsembleSummaryStats(my_ensemble, my_decision);

  for (unsigned int _alt_allele_index = 0; _alt_allele_index < my_decision.multi_allele.allele_identity_vector.size(); _alt_allele_index++) {
    // for each alt allele, do my best
    // thresholds here can vary by >type< of allele
    float local_min_allele_freq = FreqThresholdByType(my_ensemble.multi_allele_var.allele_identity_vector[_alt_allele_index], parameters->my_controls);

    my_ensemble.allele_eval[_alt_allele_index].CallGermline(local_min_allele_freq,
        my_decision.summary_info_vector[_alt_allele_index].genotype_call,
        my_decision.summary_info_vector[_alt_allele_index].gt_quality_score,
        my_decision.summary_info_vector[_alt_allele_index].alleleScore);

    // if something is strange here
    SpecializedFilterFromLatentVariables(my_ensemble.allele_eval[_alt_allele_index], my_decision, parameters->my_eval_control.filter_unusual_predictions, _alt_allele_index); // unusual filters
    SpecializedFilterFromHypothesisBias(my_ensemble.allele_eval[_alt_allele_index], my_ensemble.multi_allele_var.allele_identity_vector[_alt_allele_index], my_decision, parameters->my_eval_control.filter_deletion_bias, parameters->my_eval_control.filter_insertion_bias, _alt_allele_index);
  }
  // alleleScore does not 
  my_decision.best_allele_index = _best_allele_index;
  my_decision.best_allele_set = true;
  // extract useful fields from the evaluator
  SummarizeInfoFieldsFromEnsemble(my_ensemble, my_ensemble.multi_allele_var.variant, _best_allele_index);

  my_decision.DecisionTreeOutputToVariant(my_ensemble.multi_allele_var.variant, parameters);

}

//----------------------output some diagnostic information below---------------

void DiagnosticWriteJson(const Json::Value & json, const std::string& filename_json) {
  std::ofstream outJsonFile(filename_json.c_str(), std::ios::out);
  if (outJsonFile.good())
    outJsonFile << json.toStyledString();
  else
    std::cerr << "[tvc] diagnostic unable to write JSON file " << filename_json << std::endl;
  outJsonFile.close();
}

void DiagnosticJsonReadStack(Json::Value &json, StackPlus &read_stack) {
  json["FlowOrder"] = read_stack.flow_order;
}

void DiagnosticJsonFrequency(Json::Value &json, PosteriorInference &cur_posterior) {

  json["MaxFreq"] = cur_posterior.max_freq;
  json["MaxLL"] = cur_posterior.max_ll;
  json["ParamLL"] = cur_posterior.params_ll;
  for (unsigned int i_val = 0; i_val < cur_posterior.log_posterior_by_frequency.size(); i_val++) {
    json["LogPosterior"][i_val] = cur_posterior.log_posterior_by_frequency[i_val];
    json["EvalFrequency"][i_val] = cur_posterior.eval_at_frequency[i_val];
  }
}

void DiagnosticJsonCrossHypotheses(Json::Value &json, CrossHypotheses &my_cross) {
  // output relevant data from the individual read hypothesis tester

  json["strand"] = my_cross.strand_key;
  json["success"] = my_cross.success ? 1 : 0;
  json["heavy"] = my_cross.heavy_tailed;
  json["lastrelevantflow"] = my_cross.max_last_flow;

// difference between allele predictions for this read
  for (unsigned int i_flow = 0; i_flow < my_cross.delta.size(); i_flow++) {
    json["delta"][i_flow] = my_cross.delta[i_flow];
  }

  // active flows over which we are testing
  for (unsigned int i_test = 0; i_test < my_cross.test_flow.size(); i_test++)
    json["testflows"][i_test] = my_cross.test_flow[i_test];

// hold some intermediates size data matrix hyp * nFlows (should be active flows)

  for (unsigned int i_hyp = 0; i_hyp < my_cross.predictions.size(); i_hyp++) {
    for (unsigned int i_flow = 0; i_flow < my_cross.predictions[0].size(); i_flow++) {
      json["predictions"][i_hyp][i_flow] = my_cross.predictions[i_hyp][i_flow];
      json["normalized"][i_hyp][i_flow] = my_cross.normalized[i_hyp][i_flow];
      json["residuals"][i_hyp][i_flow] = my_cross.residuals[i_hyp][i_flow];
      json["sigma"][i_hyp][i_flow] = my_cross.sigma_estimate[i_hyp][i_flow];
      json["basiclikelihoods"][i_hyp][i_flow] = my_cross.basic_likelihoods[i_hyp][i_flow];
    }
  }
  // sequence, responsibility (after clustering), etc
  for (unsigned int i_hyp = 0; i_hyp < my_cross.responsibility.size(); i_hyp++) {
    json["instancebystate"][i_hyp] = my_cross.instance_of_read_by_state[i_hyp];
    json["responsibility"][i_hyp] = my_cross.responsibility[i_hyp];
    json["loglikelihood"][i_hyp] = my_cross.log_likelihood[i_hyp];
    json["scaledlikelihood"][i_hyp] = my_cross.scaled_likelihood[i_hyp];
  }
}

void DiagnosticJsonCrossStack(Json::Value &json, HypothesisStack &hypothesis_stack) {
  for (unsigned int i_read = 0; i_read < hypothesis_stack.total_theory.my_hypotheses.size(); i_read++) {
    DiagnosticJsonCrossHypotheses(json["Cross"][i_read], hypothesis_stack.total_theory.my_hypotheses[i_read]);
  }
}

void DiagnosticJsonBias(Json::Value &json, BasicBiasGenerator &bias_generator) {
  for (unsigned int i_latent = 0; i_latent < bias_generator.latent_bias.size(); i_latent++) {
    json["latentbias"][i_latent] = bias_generator.latent_bias[i_latent];
  }
}

void DiagnosticJsonSkew(Json::Value &json, BasicSkewGenerator &skew_generator) {
  for (unsigned int i_latent = 0; i_latent < skew_generator.latent_skew.size(); i_latent++) {
    json["latentskew"][i_latent] = skew_generator.latent_skew[i_latent];
  }
}

void DiagnosticJsonSigma(Json::Value &json, BasicSigmaGenerator &sigma_generator) {
  for (unsigned int i_latent = 0; i_latent < sigma_generator.latent_sigma.size(); i_latent++) {
    json["latentsigma"][i_latent] = sigma_generator.latent_sigma[i_latent];
    json["priorsigma"][i_latent] = sigma_generator.prior_latent_sigma[i_latent];
  }
}

void DiagnosticJsonMisc(Json::Value &json, LatentSlate &cur_state) {
  json["iterdone"] = cur_state.iter_done;
  json["maxiterations"] = cur_state.max_iterations;
  for (unsigned int i_iter = 0; i_iter < cur_state.ll_at_stage.size(); i_iter++) {
    json["llatstage"][i_iter] = cur_state.ll_at_stage[i_iter];
  }
  json["startfreq"] = cur_state.start_freq_of_winner;
}

void RichDiagnosticOutput(StackPlus &my_data, HypothesisStack &hypothesis_stack, string  &out_dir) {
  string outFile;
  Json::Value diagnostic_json;

  outFile = out_dir + hypothesis_stack.variant_contig + "."
            + convertToString(hypothesis_stack.variant_position) + "."
            + hypothesis_stack.ref_allele + "." + hypothesis_stack.var_allele + ".diagnostic.json";

  diagnostic_json["MagicNumber"] = 10;
  DiagnosticJsonFrequency(diagnostic_json["TopLevel"], hypothesis_stack.cur_state.cur_posterior);
  DiagnosticJsonFrequency(diagnostic_json["FwdStrand"], hypothesis_stack.cur_state.fwd_posterior);
  DiagnosticJsonFrequency(diagnostic_json["RevStrand"], hypothesis_stack.cur_state.rev_posterior);
  DiagnosticJsonReadStack(diagnostic_json["ReadStack"], my_data);
  DiagnosticJsonCrossStack(diagnostic_json["CrossHypotheses"], hypothesis_stack);
  DiagnosticJsonBias(diagnostic_json["Latent"], hypothesis_stack.cur_state.bias_generator);
  DiagnosticJsonSigma(diagnostic_json["Latent"], hypothesis_stack.cur_state.sigma_generator);
  DiagnosticJsonSkew(diagnostic_json["Latent"], hypothesis_stack.cur_state.skew_generator);
  DiagnosticJsonMisc(diagnostic_json["Misc"], hypothesis_stack.cur_state);

//  DiagnosticJsonFrequency(diagnostic_json["RefLevel"], hypothesis_stack.ref_state.cur_posterior);
//  DiagnosticJsonFrequency(diagnostic_json["VarLevel"], hypothesis_stack.var_state.cur_posterior);
  DiagnosticWriteJson(diagnostic_json, outFile);
}

void JustOneDiagnosis(EnsembleEval &my_ensemble, string &out_dir) {
  //diagnose one particular variant
  // check against a list?
  // only do this if using a small VCF for input of variants
  for (unsigned int i_allele = 0; i_allele < my_ensemble.allele_eval.size(); i_allele++)
    RichDiagnosticOutput(my_ensemble.my_data, my_ensemble.allele_eval[i_allele], out_dir);

}

