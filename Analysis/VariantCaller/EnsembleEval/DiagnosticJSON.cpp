/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DiagnosticJson.h"
//----------------------output some diagnostic information below---------------

void DiagnosticWriteJson(const Json::Value & json, const std::string& filename_json) {
  std::ofstream outJsonFile(filename_json.c_str(), std::ios::out);
  if (outJsonFile.good())
    outJsonFile << json.toStyledString();
  else
    std::cerr << "[tvc] diagnostic unable to write JSON file " << filename_json << std::endl;
  outJsonFile.close();
}

void DiagnosticJsonReadStack(Json::Value &json, const StackPlus &read_stack) {
  json["FlowOrder"] = read_stack.flow_order;
  for (unsigned int i_read = 0; i_read < read_stack.read_stack.size(); i_read++) {
    if ( ! read_stack.read_stack.at(i_read).well_rowcol.empty() ) {
      json["Row"][i_read] = read_stack.read_stack.at(i_read).well_rowcol.at(0);
      json["Col"][i_read] = read_stack.read_stack.at(i_read).well_rowcol.at(1);
    }
    json["MapQuality"][i_read] = read_stack.read_stack.at(i_read).map_quality;
  }
}

void DiagnosticJsonFrequency(Json::Value &json, const PosteriorInference &cur_posterior) {

  json["MaxFreq"] = cur_posterior.clustering.max_hyp_freq.at(0); // reference tentry

  for (unsigned int i_val=0; i_val<cur_posterior.clustering.max_hyp_freq.size(); i_val++){
    json["AllFreq"][i_val] = cur_posterior.clustering.max_hyp_freq.at(i_val);
  }

  json["MaxLL"] = cur_posterior.ref_vs_all.max_ll;
  json["ParamLL"] = cur_posterior.params_ll;

  for (unsigned int i_val = 0; i_val < cur_posterior.ref_vs_all.log_posterior_by_frequency.size(); i_val++) {
    json["LogPosterior"][i_val] = cur_posterior.ref_vs_all.log_posterior_by_frequency.at(i_val);
    json["EvalFrequency"][i_val] = cur_posterior.ref_vs_all.eval_at_frequency.at(i_val);
  }

  for (unsigned int i_val=0; i_val<cur_posterior.gq_pair.freq_pair.size(); i_val++)
    json["GQ"]["Allele"][i_val] = cur_posterior.gq_pair.freq_pair.at(i_val);

  for (unsigned int i_val = 0; i_val < cur_posterior.gq_pair.log_posterior_by_frequency.size(); i_val++) {
    json["GQ"]["LogPosterior"][i_val] = cur_posterior.gq_pair.log_posterior_by_frequency.at(i_val);
    json["GQ"]["EvalFrequency"][i_val] = cur_posterior.gq_pair.eval_at_frequency.at(i_val);
  }

  for (unsigned int i_hyp=0; i_hyp<cur_posterior.clustering.prior_frequency_weight.size(); i_hyp++){
    json["PriorFreq"][i_hyp]= cur_posterior.clustering.prior_frequency_weight.at(i_hyp);
  }
  json["PriorStrength"] = cur_posterior.clustering.germline_prior_strength;
  json["PriorLL"] = cur_posterior.clustering.germline_log_prior_normalization;
}

void DiagnosticJsonCrossHypotheses(Json::Value &json, const CrossHypotheses &my_cross) {
  // output relevant data from the individual read hypothesis tester

  json["strand"] = my_cross.strand_key;
  json["success"] = my_cross.success ? 1 : 0;
  json["usecorr"] = my_cross.use_correlated_likelihood ? 1: 0;

  json["heavy"] = my_cross.heavy_tailed;
  json["lastrelevantflow"] = my_cross.max_last_flow;
  json["correlation"] = my_cross.delta_state.delta_correlation;

// difference between allele predictions for this read
  for (unsigned int i_flow = 0; i_flow < my_cross.delta_state.delta.at(0).size(); i_flow++) {
    json["delta"][i_flow] = my_cross.delta_state.delta.at(0).at(i_flow);
  }
  // new-style delta
  for (unsigned int i_alt = 0; i_alt < my_cross.delta_state.delta.size(); i_alt++){
    for (unsigned int i_flow = 0; i_flow<my_cross.delta_state.delta.at(i_alt).size(); i_flow++){
      json["deltabase"][i_alt][i_flow] = my_cross.delta_state.delta.at(i_alt).at(i_flow);
    }
  }

  // active flows over which we are testing
  for (unsigned int i_test = 0; i_test < my_cross.test_flow.size(); i_test++)
    json["testflows"][i_test] = my_cross.test_flow.at(i_test);

// hold some intermediates size data matrix hyp * nFlows (should be active flows)

  for (unsigned int i_hyp = 0; i_hyp < my_cross.predictions.size(); i_hyp++) {
    for (unsigned int i_flow = 0; i_flow < my_cross.predictions.at(0).size(); i_flow++) {
      json["predictions"][i_hyp][i_flow] = my_cross.predictions.at(i_hyp).at(i_flow);
      json["modpred"][i_hyp][i_flow] = my_cross.mod_predictions.at(i_hyp).at(i_flow);
      json["normalized"][i_hyp][i_flow] = my_cross.normalized.at(i_hyp).at(i_flow);
      json["residuals"][i_hyp][i_flow] = my_cross.residuals.at(i_hyp).at(i_flow);
      json["sigma"][i_hyp][i_flow] = my_cross.sigma_estimate.at(i_hyp).at(i_flow);
      json["basiclikelihoods"][i_hyp][i_flow] = my_cross.basic_likelihoods.at(i_hyp).at(i_flow);
    }
  }
  // sequence, responsibility (after clustering), etc
  for (unsigned int i_hyp = 0; i_hyp < my_cross.responsibility.size(); i_hyp++) {
    json["instancebystate"][i_hyp] = my_cross.instance_of_read_by_state.at(i_hyp);
    json["responsibility"][i_hyp] = my_cross.responsibility.at(i_hyp);
    json["loglikelihood"][i_hyp] = my_cross.log_likelihood.at(i_hyp);
    json["scaledlikelihood"][i_hyp] = my_cross.scaled_likelihood.at(i_hyp);
  }
}

void TinyDiagnosticJsonCrossHypotheses(Json::Value &json, const CrossHypotheses &my_cross) {

json["strand"] = my_cross.strand_key;
 json["success"] = my_cross.success ? 1 : 0;

 // active flows over which we are testing
 for (unsigned int i_test = 0; i_test < my_cross.test_flow.size(); i_test++){
   json["testflows"][i_test] = my_cross.test_flow.at(i_test);
   json["testdelta"][i_test] = my_cross.delta_state.delta.at(0).at(my_cross.test_flow.at(i_test));
 }
}

void DiagnosticJsonCrossStack(Json::Value &json, const HypothesisStack &hypothesis_stack) {
  for (unsigned int i_read = 0; i_read < hypothesis_stack.total_theory.my_hypotheses.size(); i_read++) {
    DiagnosticJsonCrossHypotheses(json["Cross"][i_read], hypothesis_stack.total_theory.my_hypotheses.at(i_read));
  }
}

void TinyDiagnosticJsonCrossStack(Json::Value &json, const HypothesisStack &hypothesis_stack) {
  for (unsigned int i_read = 0; i_read < hypothesis_stack.total_theory.my_hypotheses.size(); i_read++) {
    TinyDiagnosticJsonCrossHypotheses(json["Cross"][i_read], hypothesis_stack.total_theory.my_hypotheses.at(i_read));
  }
}


void DiagnosticJsonBias(Json::Value &json, const BasicBiasGenerator &bias_generator) {
  for (unsigned int i_latent = 0; i_latent < bias_generator.latent_bias.size(); i_latent++) {
    json["latentbias"][i_latent] = bias_generator.latent_bias.at(i_latent).at(0);
  }
  for (unsigned int i_strand=0; i_strand<bias_generator.latent_bias.size(); i_strand++){
    for (unsigned int i_alt=0; i_alt<bias_generator.latent_bias.at(i_strand).size(); i_alt++)
      json["allbias"][i_strand][i_alt] = bias_generator.latent_bias.at(i_strand).at(i_alt);
  }
}

void DiagnosticJsonSkew(Json::Value &json, const BasicSkewGenerator &skew_generator) {
  for (unsigned int i_latent = 0; i_latent < skew_generator.latent_skew.size(); i_latent++) {
    json["latentskew"][i_latent] = skew_generator.latent_skew.at(i_latent);
  }
}

void DiagnosticJsonSigma(Json::Value &json, const BasicSigmaGenerator &sigma_generator) {
  for (unsigned int i_latent = 0; i_latent < sigma_generator.latent_sigma.size(); i_latent++) {
    json["latentsigma"][i_latent] = sigma_generator.latent_sigma.at(i_latent);
    json["priorsigma"][i_latent] = sigma_generator.prior_latent_sigma.at(i_latent);
  }
}

void DiagnosticJsonMisc(Json::Value &json, const LatentSlate &cur_state) {
  json["iterdone"] = cur_state.iter_done;
  json["maxiterations"] = cur_state.max_iterations;
  for (unsigned int i_iter = 0; i_iter < cur_state.ll_at_stage.size(); i_iter++) {
    json["llatstage"][i_iter] = cur_state.ll_at_stage.at(i_iter);
  }
  for (unsigned int i_freq=0; i_freq<cur_state.start_freq_of_winner.size(); i_freq++)
    json["startfreq"][i_freq] = cur_state.start_freq_of_winner.at(i_freq);
}

void DiagnosticJsonHistory(Json::Value &json, const HypothesisStack &hypothesis_stack){
  for (unsigned int i_start=0; i_start<hypothesis_stack.ll_record.size(); i_start++){
    json["LLrecord"][i_start] = hypothesis_stack.ll_record.at(i_start);
  }
}

void TinyDiagnosticOutput(const StackPlus &my_data, const HypothesisStack &hypothesis_stack, string &out_dir){
  string outFile;
  Json::Value diagnostic_json;

  outFile = out_dir + hypothesis_stack.variant_contig + "."
            + convertToString(hypothesis_stack.variant_position) + "."
            + hypothesis_stack.ref_allele + "." + hypothesis_stack.var_allele + ".tiny.json";
  // just a little bit of data
  DiagnosticJsonReadStack(diagnostic_json["ReadStack"], my_data);
  TinyDiagnosticJsonCrossStack(diagnostic_json["CrossHypotheses"], hypothesis_stack);
  DiagnosticJsonBias(diagnostic_json["Latent"], hypothesis_stack.cur_state.bias_generator);
  // write it out
  DiagnosticWriteJson(diagnostic_json, outFile);
}

void RichDiagnosticOutput(const StackPlus &my_data, const HypothesisStack &hypothesis_stack, string  &out_dir) {
  string outFile;
  Json::Value diagnostic_json;

  outFile = out_dir + hypothesis_stack.variant_contig + "."
            + convertToString(hypothesis_stack.variant_position) + "."
            + hypothesis_stack.ref_allele + "." + hypothesis_stack.var_allele + ".diagnostic.json";

  diagnostic_json["MagicNumber"] = 12;
  DiagnosticJsonFrequency(diagnostic_json["TopLevel"], hypothesis_stack.cur_state.cur_posterior);

  DiagnosticJsonReadStack(diagnostic_json["ReadStack"], my_data);
  DiagnosticJsonCrossStack(diagnostic_json["CrossHypotheses"], hypothesis_stack);
  DiagnosticJsonBias(diagnostic_json["Latent"], hypothesis_stack.cur_state.bias_generator);
  DiagnosticJsonSigma(diagnostic_json["Latent"], hypothesis_stack.cur_state.sigma_generator.fwd);
  DiagnosticJsonSigma(diagnostic_json["Latent"]["SigmaRev"], hypothesis_stack.cur_state.sigma_generator.rev);
    DiagnosticJsonSkew(diagnostic_json["Latent"], hypothesis_stack.cur_state.skew_generator);
  DiagnosticJsonMisc(diagnostic_json["Misc"], hypothesis_stack.cur_state);
  DiagnosticJsonHistory(diagnostic_json["History"],hypothesis_stack);

  DiagnosticWriteJson(diagnostic_json, outFile);
}

void JustOneDiagnosis(const EnsembleEval &my_ensemble, string &out_dir, bool rich_diag) {
  //diagnose one particular variant
  // check against a list?
  // only do this if using a small VCF for input of variants

    if (rich_diag)
      RichDiagnosticOutput(my_ensemble.my_data, my_ensemble.allele_eval, out_dir);
  else
      TinyDiagnosticOutput(my_ensemble.my_data, my_ensemble.allele_eval, out_dir);


}

