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

void DiagnosticJsonReadStack(Json::Value &json, const vector<const Alignment *>& read_stack, const InputStructures &global_context) {

  bool multiple_flow_orders = false;
  for (unsigned int iFO=0; iFO < global_context.flow_order_vector.size(); iFO++) {
    json["FlowOrder"][iFO] = global_context.flow_order_vector.at(iFO).str();
    if (not multiple_flow_orders and iFO > 0)
      multiple_flow_orders = true;
  }

  for (unsigned int i_read = 0; i_read < read_stack.size(); i_read++) {
    if ( ! read_stack[i_read]->well_rowcol.empty() ) {
      json["Row"][i_read] = read_stack[i_read]->well_rowcol[0];
      json["Col"][i_read] = read_stack[i_read]->well_rowcol[1];
      if (multiple_flow_orders)
        json["FlowOrderIndex"][i_read] = read_stack[i_read]->flow_order_index;
    }
    json["MapQuality"][i_read] = read_stack[i_read]->alignment.MapQuality;
  }
}

void DiagnosticJsonFrequency(Json::Value &json, const PosteriorInference &cur_posterior) {

  json["MaxFreq"] = cur_posterior.clustering.max_hyp_freq[0]; // reference tentry

  for (unsigned int i_val=0; i_val<cur_posterior.clustering.max_hyp_freq.size(); i_val++){
    json["AllFreq"][i_val] = cur_posterior.clustering.max_hyp_freq[i_val];
  }

  json["MaxLL"] = cur_posterior.ref_vs_all.max_ll;
  json["ParamLL"] = cur_posterior.params_ll;

  for (unsigned int i_val = 0; i_val < cur_posterior.ref_vs_all.log_posterior_by_frequency.size(); i_val++) {
    json["LogPosterior"][i_val] = cur_posterior.ref_vs_all.log_posterior_by_frequency[i_val];
    json["EvalFrequency"][i_val] = cur_posterior.ref_vs_all.eval_at_frequency[i_val];
  }

  for (unsigned int i_val=0; i_val<cur_posterior.gq_pair.freq_pair.size(); i_val++)
    json["GQ"]["Allele"][i_val] = cur_posterior.gq_pair.freq_pair[i_val];

  for (unsigned int i_val = 0; i_val < cur_posterior.gq_pair.log_posterior_by_frequency.size(); i_val++) {
    json["GQ"]["LogPosterior"][i_val] = cur_posterior.gq_pair.log_posterior_by_frequency[i_val];
    json["GQ"]["EvalFrequency"][i_val] = cur_posterior.gq_pair.eval_at_frequency[i_val];
  }

  for (unsigned int i_hyp=0; i_hyp<cur_posterior.clustering.prior_frequency_weight.size(); i_hyp++){
    json["PriorFreq"][i_hyp]= cur_posterior.clustering.prior_frequency_weight[i_hyp];
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
  for (unsigned int i_flow = 0; i_flow < my_cross.delta_state.delta[0].size(); i_flow++) {
    json["delta"][i_flow] = my_cross.delta_state.delta[0][i_flow];
  }
  // new-style delta
  for (unsigned int i_alt = 0; i_alt < my_cross.delta_state.delta.size(); i_alt++){
    for (unsigned int i_flow = 0; i_flow<my_cross.delta_state.delta[i_alt].size(); i_flow++){
      json["deltabase"][i_alt][i_flow] = my_cross.delta_state.delta[i_alt][i_flow];
    }
  }

  // active flows over which we are testing
  for (unsigned int i_test = 0; i_test < my_cross.test_flow.size(); i_test++)
    json["testflows"][i_test] = my_cross.test_flow[i_test];

// hold some intermediates size data matrix hyp * nFlows (should be active flows)

  for (unsigned int i_hyp = 0; i_hyp < my_cross.predictions.size(); i_hyp++) {
    for (unsigned int i_flow = 0; i_flow < my_cross.predictions[0].size(); i_flow++) {
      json["predictions"][i_hyp][i_flow] = my_cross.predictions[i_hyp][i_flow];
      json["modpred"][i_hyp][i_flow] = my_cross.mod_predictions[i_hyp][i_flow];
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

void TinyDiagnosticJsonCrossHypotheses(Json::Value &json, const CrossHypotheses &my_cross) {

json["strand"] = my_cross.strand_key;
 json["success"] = my_cross.success ? 1 : 0;

 // active flows over which we are testing
 for (unsigned int i_test = 0; i_test < my_cross.test_flow.size(); i_test++){
   json["testflows"][i_test] = my_cross.test_flow[i_test];
   json["testdelta"][i_test] = my_cross.delta_state.delta[0][my_cross.test_flow[i_test]];
 }
}

void DiagnosticJsonCrossStack(Json::Value &json, const HypothesisStack &hypothesis_stack) {
  for (unsigned int i_read = 0; i_read < hypothesis_stack.total_theory.my_hypotheses.size(); i_read++) {
    DiagnosticJsonCrossHypotheses(json["Cross"][i_read], hypothesis_stack.total_theory.my_hypotheses[i_read]);
  }
}

void TinyDiagnosticJsonCrossStack(Json::Value &json, const HypothesisStack &hypothesis_stack) {
  for (unsigned int i_read = 0; i_read < hypothesis_stack.total_theory.my_hypotheses.size(); i_read++) {
    TinyDiagnosticJsonCrossHypotheses(json["Cross"][i_read], hypothesis_stack.total_theory.my_hypotheses[i_read]);
  }
}


void DiagnosticJsonBias(Json::Value &json, const BasicBiasGenerator &bias_generator) {
  for (unsigned int i_latent = 0; i_latent < bias_generator.latent_bias.size(); i_latent++) {
    json["latentbias"][i_latent] = bias_generator.latent_bias[i_latent][0];
  }
  for (unsigned int i_strand=0; i_strand<bias_generator.latent_bias.size(); i_strand++){
    for (unsigned int i_alt=0; i_alt<bias_generator.latent_bias[i_strand].size(); i_alt++)
      json["allbias"][i_strand][i_alt] = bias_generator.latent_bias[i_strand][i_alt];
  }
}

void DiagnosticJsonSkew(Json::Value &json, const BasicSkewGenerator &skew_generator) {
  for (unsigned int i_latent = 0; i_latent < skew_generator.latent_skew.size(); i_latent++) {
    json["latentskew"][i_latent] = skew_generator.latent_skew[i_latent];
  }
}

void DiagnosticJsonSigma(Json::Value &json, const BasicSigmaGenerator &sigma_generator) {
  for (unsigned int i_latent = 0; i_latent < sigma_generator.latent_sigma.size(); i_latent++) {
    json["latentsigma"][i_latent] = sigma_generator.latent_sigma[i_latent];
    json["priorsigma"][i_latent] = sigma_generator.prior_latent_sigma[i_latent];
  }
}

void DiagnosticJsonMisc(Json::Value &json, const LatentSlate &cur_state) {
  json["iterdone"] = cur_state.iter_done;
  json["maxiterations"] = cur_state.max_iterations;
  for (unsigned int i_iter = 0; i_iter < cur_state.ll_at_stage.size(); i_iter++) {
    json["llatstage"][i_iter] = cur_state.ll_at_stage[i_iter];
  }
  for (unsigned int i_freq=0; i_freq<cur_state.start_freq_of_winner.size(); i_freq++)
    json["startfreq"][i_freq] = cur_state.start_freq_of_winner[i_freq];
}

void DiagnosticJsonHistory(Json::Value &json, const HypothesisStack &hypothesis_stack){
  for (unsigned int i_start=0; i_start<hypothesis_stack.ll_record.size(); i_start++){
    json["LLrecord"][i_start] = hypothesis_stack.ll_record[i_start];
  }
}

void TinyDiagnosticOutput(const vector<const Alignment *>& read_stack, const HypothesisStack &hypothesis_stack,
    const string& variant_contig, int variant_position, const string& ref_allele, const string& var_allele,
    const InputStructures &global_context, const string &out_dir){
  string outFile;
  Json::Value diagnostic_json;

  outFile = out_dir + variant_contig + "."
            + convertToString(variant_position) + "."
            + ref_allele + "." + var_allele + ".tiny.json";
  // just a little bit of data
  DiagnosticJsonReadStack(diagnostic_json["ReadStack"], read_stack, global_context);
  TinyDiagnosticJsonCrossStack(diagnostic_json["CrossHypotheses"], hypothesis_stack);
  DiagnosticJsonBias(diagnostic_json["Latent"], hypothesis_stack.cur_state.bias_generator);
  // write it out
  DiagnosticWriteJson(diagnostic_json, outFile);
}

void RichDiagnosticOutput(const vector<const Alignment *>& read_stack, const HypothesisStack &hypothesis_stack,
    const string& variant_contig, int variant_position, const string& ref_allele, const string& var_allele,
    const InputStructures &global_context, const string  &out_dir) {
  string outFile;
  Json::Value diagnostic_json;

  outFile = out_dir + variant_contig + "."
            + convertToString(variant_position) + "."
            + ref_allele + "." + var_allele + ".diagnostic.json";

  diagnostic_json["MagicNumber"] = 12;
  DiagnosticJsonFrequency(diagnostic_json["TopLevel"], hypothesis_stack.cur_state.cur_posterior);

  DiagnosticJsonReadStack(diagnostic_json["ReadStack"], read_stack, global_context);
  DiagnosticJsonCrossStack(diagnostic_json["CrossHypotheses"], hypothesis_stack);
  DiagnosticJsonBias(diagnostic_json["Latent"], hypothesis_stack.cur_state.bias_generator);
  DiagnosticJsonSigma(diagnostic_json["Latent"], hypothesis_stack.cur_state.sigma_generator.fwd);
  DiagnosticJsonSigma(diagnostic_json["Latent"]["SigmaRev"], hypothesis_stack.cur_state.sigma_generator.rev);
    DiagnosticJsonSkew(diagnostic_json["Latent"], hypothesis_stack.cur_state.skew_generator);
  DiagnosticJsonMisc(diagnostic_json["Misc"], hypothesis_stack.cur_state);
  DiagnosticJsonHistory(diagnostic_json["History"],hypothesis_stack);

  DiagnosticWriteJson(diagnostic_json, outFile);
}

void JustOneDiagnosis(const EnsembleEval &my_ensemble, const InputStructures &global_context,
    const string &out_dir, bool rich_diag)
{
  //diagnose one particular variant
  // check against a list?
  // only do this if using a small VCF for input of variants

  // build a unique identifier to write out diagnostics
  int variant_position = my_ensemble.variant->position;
  string ref_allele = my_ensemble.variant->ref;
  string var_allele = my_ensemble.variant->alt[0];
  for (unsigned int i_allele=1; i_allele<my_ensemble.variant->alt.size(); i_allele++) {
    var_allele += ',';
    var_allele += my_ensemble.variant->alt[i_allele];
  }
  string variant_contig =  my_ensemble.variant->sequenceName;

  if (rich_diag)
    RichDiagnosticOutput(my_ensemble.read_stack, my_ensemble.allele_eval,
        variant_contig, variant_position, ref_allele, var_allele, global_context, out_dir);
  else
    TinyDiagnosticOutput(my_ensemble.read_stack, my_ensemble.allele_eval,
        variant_contig, variant_position, ref_allele, var_allele, global_context, out_dir);
}

