/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "StackEngine.h"


void LatentSlate::PropagateTuningParameters(EnsembleEvalTuningParameters &my_params) {
  // prior reliability for outlier read frequency
  cur_posterior.data_reliability = my_params.DataReliability();
  rev_posterior.data_reliability = my_params.DataReliability();
  fwd_posterior.data_reliability = my_params.DataReliability();

  // prior precision and likelihood penalty for moving off-center
  bias_generator.damper_bias = my_params.prediction_precision;
  bias_generator.pseudo_sigma_base = my_params.pseudo_sigma_base;

  // prior variance-by-intensity relationship
  sigma_generator.prior_sigma_regression[0] = my_params.magic_sigma_base;
  sigma_generator.prior_sigma_regression[1] = my_params.magic_sigma_slope;
  sigma_generator.prior_weight = my_params.sigma_prior_weight;

  // not actually used at this point
  skew_generator.dampened_skew = my_params.prediction_precision;
}

// see how quick we can make this
void LatentSlate::LocalExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma, float start_frequency) {
  if (detailed_integral) {
    DetailedExecuteInference(total_theory, update_frequency, update_sigma);
  }
  else {
    FastExecuteInference(total_theory, update_frequency, update_sigma, start_frequency);
  }
}

void LatentSlate::DetailedStep(ShortStack &total_theory, bool update_frequency, bool update_sigma) {

  bias_generator.DoStepForBias(total_theory); // update bias estimate-> residuals->likelihoods
  cur_posterior.DetailedUpdateStep(total_theory, update_frequency);

  if (update_sigma) {
    sigma_generator.DoStepForSigma(total_theory); // update sigma estimate
    cur_posterior.DetailedUpdateStep(total_theory, update_frequency);
  }
}

void LatentSlate::DetailedExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma) {

  cur_posterior.StartAtNull(total_theory, update_frequency);
  //StartAtHardClassify(local_posterior, update_frequency);

  float old_ll = cur_posterior.max_ll - 1.0f; // always try at least one step
  iter_done = 0;
  bool keep_optimizing = true;
  ll_at_stage.resize(0);
  ll_at_stage.push_back(cur_posterior.max_ll);
  while ((iter_done < max_iterations) & (keep_optimizing)) {
    iter_done++;
    //cout << i_count << " max_ll " << max_ll << endl;
    old_ll = cur_posterior.max_ll; // see if we improve over this cycle

    DetailedStep(total_theory, update_frequency, update_sigma);
    ll_at_stage.push_back(cur_posterior.max_ll);
    if (old_ll > cur_posterior.max_ll)
      keep_optimizing = false;

  }
  // evaluate likelihood of current parameter set
  // currently only done for bias
  cur_posterior.params_ll = bias_generator.BiasLL();
}


void LatentSlate::FastStep(ShortStack &total_theory, bool update_frequency, bool update_sigma) {

  bias_generator.DoStepForBias(total_theory); // update bias estimate-> residuals->likelihoods
  if (update_frequency)
    cur_posterior.QuickUpdateStep(total_theory);

  if (update_sigma) {
    sigma_generator.DoStepForSigma(total_theory); // update sigma estimate
    if (update_frequency)
      cur_posterior.QuickUpdateStep(total_theory);

  }
}

void LatentSlate::FastExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma, float start_frequency) {
  // start us out estimating frequency
  cur_posterior.StartAtHardClassify(total_theory, update_frequency, start_frequency);
  FastStep(total_theory, false, false);

  float old_ll = cur_posterior.max_ll; // always try at least one step
  iter_done = 0;
  bool keep_optimizing = true;
  ll_at_stage.resize(0);
  ll_at_stage.push_back(cur_posterior.max_ll);
  while ((iter_done < max_iterations) & keep_optimizing) {
    iter_done++;
    //cout << i_count << " max_ll " << max_ll << endl;
    old_ll = cur_posterior.max_ll; // see if we improve over this cycle

    FastStep(total_theory, update_frequency, update_sigma);
    ll_at_stage.push_back(cur_posterior.max_ll);
    if (old_ll > cur_posterior.max_ll)
      keep_optimizing = false;

  }
  // evaluate likelihood of current parameter set
  // currently only done for bias
  cur_posterior.params_ll = bias_generator.BiasLL();
}


void LatentSlate::SolveForFixedFrequency(ShortStack &total_theory, float test_freq) {
  total_theory.ResetNullBias();
  // look at what just reference gives us for parameters
  cur_posterior.max_freq = test_freq;
  FastExecuteInference(total_theory, false, false, test_freq); // just update bias terms
}

void LatentSlate::ScanStrandPosterior(ShortStack &total_theory) {
  // keep information on distribution by strand
  if (!cur_posterior.scan_done)
    cur_posterior.DoPosteriorFrequencyScan(total_theory, true, ALL_STRAND_KEY);
  fwd_posterior.DoPosteriorFrequencyScan(total_theory, true, 0);
  rev_posterior.DoPosteriorFrequencyScan(total_theory, true, 1);
};


void HypothesisStack::DefaultValues() {

  ref_allele = "X";
  var_allele = "X";
  variant_position = -1;
  try_alternatives = true;
}

float HypothesisStack::ReturnMaxLL() {
  return(cur_state.cur_posterior.ReturnMaxLL());
}

void HypothesisStack::InitForInference(StackPlus &my_data) {
  PropagateTuningParameters(); // sub-objects need to know

  total_theory.FindValidIndexes();
  // predict given hypotheses per read
  total_theory.FillInPredictions(my_data);
  total_theory.InitTestFlow();
};


void HypothesisStack::ExecuteInference() {

  // now with unrestrained inference
  ExecuteFullInference();

  if (try_alternatives) {
    ExecuteExtremeInferences();
  }
}

void HypothesisStack::SetAlternateFromMain() {
  // make sure things are populated

//  ref_state = cur_state;
//  var_state = cur_state;
}

void HypothesisStack::ExecuteFullInference() {
  total_theory.ResetQualities();
  cur_state.detailed_integral = false;
  cur_state.LocalExecuteInference(total_theory, true, true, 0.5f);

  cur_state.ScanStrandPosterior(total_theory);
}

// try altenatives to the main function to see if we have a better fit somewhere else
void HypothesisStack::ExecuteExtremeInferences() {

  LatentSlate tmp_state;

  tmp_state = cur_state;
  tmp_state.detailed_integral = false;
  tmp_state.start_freq_of_winner = 1.0f;
  total_theory.ResetQualities();
  tmp_state.LocalExecuteInference(total_theory, true, true, 1.0f); // start at reference
  tmp_state.ScanStrandPosterior(total_theory);

  if (cur_state.cur_posterior.ReturnMaxLL() < tmp_state.cur_posterior.ReturnMaxLL()) {
    cur_state = tmp_state; // update to the better solution, hypothetically
  }
  tmp_state = cur_state;
  tmp_state.detailed_integral = false;
  tmp_state.start_freq_of_winner = 0.0f;
  total_theory.ResetQualities();
  tmp_state.LocalExecuteInference(total_theory, true, true, 0.0f); // start at variant
  tmp_state.ScanStrandPosterior(total_theory);

  if (cur_state.cur_posterior.ReturnMaxLL() < tmp_state.cur_posterior.ReturnMaxLL()) {
    cur_state = tmp_state; // update to the better solution, hypothetically
  }
  RestoreFullInference(); // put total_theory to be consistent with whoever won
}

void HypothesisStack::RestoreFullInference() {
//  cur_state = backup_state;
  cur_state.bias_generator.ResetActiveBias(total_theory);
  // in theory, need to update sigma & skew, but since not fitting for particular variants, don't worry about it at the moment.
  cur_state.sigma_generator.UpdateSigmaEstimates(total_theory);
  total_theory.UpdateRelevantLikelihoods();
  //DoPosteriorFrequencyScan(cur_posterior, true);
  total_theory.UpdateResponsibility(cur_state.cur_posterior.max_freq, cur_state.cur_posterior.data_reliability); // once bias, sigma, max_freq established, reset responsibilities is forced
}


// subobjects need to know their tuning
void HypothesisStack::PropagateTuningParameters() {
  total_theory.PropagateTuningParameters(my_params);
  // number of pseudo-data points at no bias
  cur_state.PropagateTuningParameters(my_params);
}



// tool for combining items at differing log-levels
float log_sum(float a, float b) {
  float max_ab = max(a, b);
  float log_sum_val = max_ab + log(exp(a - max_ab) + exp(b - max_ab));
  return(log_sum_val);
}

void update_genotype_interval(vector<float> &genotype_interval, vector<float> &interval_cuts, PosteriorInference &local_posterior) {
  for (unsigned int i_cut = 0; i_cut < genotype_interval.size(); i_cut++) {
    genotype_interval[i_cut] = log_sum(genotype_interval[i_cut], local_posterior.LogDefiniteIntegral(interval_cuts[i_cut+1], interval_cuts[i_cut]) + local_posterior.params_ll);
  }
}

// must have scan done to be workable
bool HypothesisStack::CallGermline(float hom_safety, int &genotype_call, float &quasi_phred_quality_score, float &reject_status_quality_score) {
  // divide MAF into three zones = 0/1/2 variant alleles

  // make sure we're safe based on the number of reads
  int num_reads = cur_state.cur_posterior.eval_at_frequency.size();
  // if we don't have enough reads might be crazy near mid
  float mid_cutoff = 0.5f - 0.5f / (num_reads + 1.0f);
  // if we don't have any variants, still can't exclude 3/num_reads frequency
  // so no sense testing that low
  float low_cutoff = 1.0f / (num_reads);
  // if we have a small number of reads, these ranges may conflict
  float real_safety = min(mid_cutoff, max(low_cutoff, hom_safety));

  bool safety_active_flag = false;
  if (fabs(real_safety - hom_safety) > 0.5f / (num_reads + 1.0f))
    safety_active_flag = true;

  vector<float> interval_cuts(4);
  interval_cuts[0] = 1.0f;
  interval_cuts[1] = 1.0f - real_safety;
  interval_cuts[2] = real_safety;
  interval_cuts[3] = 0.0f;


  vector<float> genotype_interval(3);
  for (unsigned int i_cut = 0; i_cut < genotype_interval.size(); i_cut++) {
    genotype_interval[i_cut] = cur_state.cur_posterior.LogDefiniteIntegral(interval_cuts[i_cut+1], interval_cuts[i_cut]) + cur_state.cur_posterior.params_ll;
  }

  // check alternate hypotheses in case they contribute meaningfully to posterior inference
  // we implicitly assume that prior on parameters reflects relative likelihood of parameters and captures possibly multimodal influences
  // i.e. assume the path we take to the points in parameter space evaluated can be safely neglected
  // because integrating over the whole parameter distribution and induced likelihoods is expensive
  // but we do need to handle the basic cases of "pure states"
  // Note: because likelihoods are so sensitive to sigma, keep same variance across extreme hypotheses
  // we're checking that we can't juggle around "read labeling" and visit a far-away portion of parameter space
  /*  if (try_alternatives) {
      update_genotype_interval(genotype_interval, interval_cuts, ref_state.cur_posterior);
      update_genotype_interval(genotype_interval, interval_cuts, var_state.cur_posterior);
    }*/
  // because obviously our odds of mis-calling may be dominated by systematic errors in prediction

  //@TODO: do as paired and sort, for clean code
  // best zone = call
  unsigned int best_call = 0;
  float best_val = genotype_interval[0];
  unsigned int worst_call = 0;
  float worst_val = genotype_interval[0];
  for (unsigned int i_geno = 1; i_geno < genotype_interval.size(); i_geno++) {
    if (best_val < genotype_interval[i_geno]) {
      best_val = genotype_interval[i_geno];
      best_call = i_geno;
    }
    if (worst_val > genotype_interval[i_geno]) {
      worst_val = genotype_interval[i_geno];
      worst_call = i_geno;
    }
  }
  float middle_val = 0.0f;
  for (unsigned int i_geno = 0; i_geno < genotype_interval.size(); i_geno++) {
    if ((i_geno != worst_call) & (i_geno != best_call)) {
      middle_val = genotype_interval[i_geno];
    }
  }
  // most likely interval
  genotype_call = best_call;

  // quality score
  float log_alternative = middle_val + log(1 + exp(worst_val - middle_val)); // total mass on alternative intervals
  float log_all = best_val + log(1 + exp(log_alternative - best_val)); // total mass on all intervals

  // output
  quasi_phred_quality_score = 10 * (log_all - log_alternative) / log(10); // -10*log(error), base 10

  if (isnan(quasi_phred_quality_score)) {
    cout << "Warning: quality score NAN " << variant_contig << "." << variant_position << endl;
    quasi_phred_quality_score = 0.0f;
  }

  // reject ref = quality of rejection call
  float log_ref = 0.0f;
  if (genotype_call == 0) {
    // if reference, how strongly can we reject the outer interval
    log_ref = log_alternative;
  }
  else {
    // if var, how strongly can we reject ref
    log_ref = genotype_interval[0];
  }
  reject_status_quality_score = 10 * (log_all - log_ref) / log(10); // how much mass speaks against the pure reference state
  if (isnan(reject_status_quality_score)) {
    cout << "Warning: reject ref score NAN " << variant_contig << "." << variant_position << endl;
    reject_status_quality_score = 0.0f;
  }

  return(safety_active_flag);
};

// hyper-sensitive to any faint issues, but no edge cases
void HypothesisStack::CallByMAP(int &genotype_call, float &quasi_phred_quality_score) {
  PosteriorInference tmp_inference = cur_state.cur_posterior;

  // combine information across possible cases
  /*  if (try_alternatives) {
      for (unsigned int i_eval = 0; i_eval < tmp_inference.log_posterior_by_frequency.size(); i_eval++) {
        tmp_inference.log_posterior_by_frequency[i_eval] = log_sum(tmp_inference.log_posterior_by_frequency[i_eval], ref_state.cur_posterior.log_posterior_by_frequency[i_eval]);
        tmp_inference.log_posterior_by_frequency[i_eval] = log_sum(tmp_inference.log_posterior_by_frequency[i_eval], var_state.cur_posterior.log_posterior_by_frequency[i_eval]);
      }
    }*/
  tmp_inference.FindMaxFrequency(true);
  // three hypotheses: pure ref, pure var, mixed pop at MAF
  int ref_locus = tmp_inference.log_posterior_by_frequency.size() - 1;
  float second_ll = 0.0f;
  if (tmp_inference.max_index == ref_locus) {
    genotype_call = 0;
    second_ll = tmp_inference.log_posterior_by_frequency[0];
  }
  //float ref_ll = tmp_inference.log_posterior_by_frequency[ref_locus];
  if ((tmp_inference.max_index < ref_locus) & (tmp_inference.max_index > 0)) {
    genotype_call = 1;
    second_ll = max(tmp_inference.log_posterior_by_frequency[ref_locus], tmp_inference.log_posterior_by_frequency[0]);
  }
  if (tmp_inference.max_index == 0) {
    genotype_call = 2;
    second_ll = tmp_inference.log_posterior_by_frequency[ref_locus];
  }
  quasi_phred_quality_score = 10 * (tmp_inference.max_ll - second_ll) / log(10);
}


void EnsembleEval::SetupHypothesisChecks(ExtendParameters *parameters) {
  for (unsigned int i_allele = 0; i_allele < allele_eval.size(); i_allele++) {

    allele_eval[i_allele].my_params = parameters->my_eval_control;
    allele_eval[i_allele].variant_contig = (*(multi_allele_var.variant))->sequenceName;
  }
}

void EnsembleEval::ScanSupportingEvidence(float &mean_ll_delta, float &mean_supporting_flows, float &mean_max_discrimination, float threshold, int i_allele) {
  mean_supporting_flows = 0.0f;
  mean_max_discrimination = 0.0f;
  mean_ll_delta = 0.0f;
  int count = 0;
  for (unsigned int i_read = 0; i_read < allele_eval[i_allele].total_theory.my_hypotheses.size(); i_read++) {
    if (allele_eval[i_allele].total_theory.my_hypotheses[i_read].success) {
      // measure disruption
      float max_fld;
      int support_flows;
      allele_eval[i_allele].total_theory.my_hypotheses[i_read].ComputeLocalDiscriminationStrength(threshold, max_fld, support_flows);
      mean_max_discrimination += max_fld;
      mean_supporting_flows += support_flows;
      mean_ll_delta += allele_eval[i_allele].total_theory.my_hypotheses[i_read].ComputeLLDifference();
      count++;
    }
  }
  mean_max_discrimination /= (count + 0.01f);
  mean_supporting_flows /= (count + 0.01f);
  mean_ll_delta /= (count + 0.01f);
  mean_ll_delta = 10.0f * mean_ll_delta / log(10.0f); // phred-scaled
}

//@TODO: this is a hack to make downstream code think we have "hard classified reads"
// will be replaced by something more subtle later
// just get the output statistics up and running
void EnsembleEval::ApproximateHardClassifierForReads(vector<int> &read_allele_id, vector<bool> &strand_id) {
  // return something to get summary statistics from
  read_allele_id.assign(my_data.read_stack.size(), -1);
  for (unsigned int i_read = 0; i_read < read_allele_id.size(); i_read++) {
    vector<float> allele_test;
    allele_test.resize(allele_eval.size());

    // "softmax responsibility
    // the reference likelihood (100% ref) vs the read likelihood under the best global hypothesis this allele (max_freq)
    // who is most clearly separated from the reference (which is common to all) conditional on the null hypothesis (outlier)
    // i.e. most responsibility relative to the reference [because this is effectively scaled likelihood under the max-frequency estimate
    // and in theory we are common across all reads

    for (unsigned int i_alt = 0; i_alt < allele_test.size(); i_alt++) {
      allele_test[i_alt] = allele_eval[i_alt].total_theory.my_hypotheses[i_read].responsibility[2] - allele_eval[i_alt].total_theory.my_hypotheses[i_read].responsibility[1];
//      allele_test[i_alt] = allele_eval[i_alt].total_theory.my_hypotheses[i_read].ComputePosteriorLikelihood(allele_eval[i_alt].cur_state.cur_posterior.max_freq, allele_eval[i_alt].my_params.DataReliability());
//      allele_test[i_alt] -= allele_eval[i_alt].total_theory.my_hypotheses[i_read].ComputePosteriorLikelihood(1.0f, allele_eval[i_alt].my_params.DataReliability());
    }

    int max_alt_state = 0;
    float best_ll = allele_test[0];
    for (unsigned int i_alt = 0; i_alt < allele_test.size(); i_alt++) {
      if (allele_test[i_alt] > best_ll) {
        max_alt_state = i_alt;
        best_ll = allele_test[i_alt];
      }
    }

    // check: does this give sensible results
    if (best_ll > 0) {
      // there exists at least one read willing to take responsibility
      read_allele_id[i_read] = max_alt_state + 1;  // which alternate am I
    }
    else {
      read_allele_id[i_read] = 0; // reference
    }

    // problem: what to do with outlier/failed reads?
    if (!allele_eval[max_alt_state].total_theory.my_hypotheses[i_read].success) {
      read_allele_id[i_read] = -1; // failure!
    }

    if (allele_eval[max_alt_state].total_theory.my_hypotheses[i_read].responsibility[0] >
        max(allele_eval[max_alt_state].total_theory.my_hypotheses[i_read].responsibility[2], allele_eval[max_alt_state].total_theory.my_hypotheses[i_read].responsibility[1])) {
      read_allele_id[i_read] = -1; // failure! dominated by outlier probability even at best LL across all bivariate checks
    }

  }
  strand_id.assign(my_data.read_stack.size(), false);
  for (unsigned int i_read = 0; i_read < strand_id.size(); i_read++) {
    strand_id[i_read] = my_data.read_stack[i_read].is_forward_strand;
  }
}

void EnsembleEval::UnifyTestFlows() {
  // don't bother if we only have one alternate
  if (allele_eval.size() > 1) {
    // make sure we evaluate log-likelihood over a unified set of test flows for each read for each alternate hypothesis
    for (unsigned int i_read = 0; i_read < my_data.read_stack.size(); i_read++) {
      // accumulate test flows across all alternate alleles
      vector<int> all_flows;
      for (unsigned int i_alt = 0; i_alt < allele_eval.size(); i_alt++) {
        unsigned int num_test_flows = allele_eval[i_alt].total_theory.my_hypotheses[i_read].test_flow.size();
        // of course don't bother unless splicing succeeded in this read
        bool go_ahead = allele_eval[i_alt].total_theory.my_hypotheses[i_read].success;
        for (unsigned int i_test = 0; (i_test < num_test_flows) & go_ahead; i_test++) {
          all_flows.push_back(allele_eval[i_alt].total_theory.my_hypotheses[i_read].test_flow[i_test]);
        }
      }
      // now we have all flows tested in each ref/alt set
      // make unique
      std::vector<int>::iterator it;
      it = std::unique(all_flows.begin(), all_flows.end());
      all_flows.resize(std::distance(all_flows.begin(), it));
      std::sort(all_flows.begin(), all_flows.end());
      // set each individual read to use the same test_flows

      for (unsigned int i_alt = 0; i_alt < allele_eval.size(); i_alt++) {
        allele_eval[i_alt].total_theory.my_hypotheses[i_read].test_flow = all_flows;
      }
    }
  }
}

// guess what: we need this to be compatible with the hard-classify hack
// especially as the likelihood is not quite consistent across binary checks since different flows are used.
int EnsembleEval::DetectBestAlleleHardClassify() {
  vector<int> read_id;
  vector<bool> strand_id;

  ApproximateHardClassifierForReads(read_id, strand_id);

  // post-process
  // if we hard classify as reference, any allele should call reference so take that responsibility within that hypothesis
  // if we hard classify as that allele, take responsibility for variant
  // if classify as outlier, of course no-one is happy
  vector<float> best_allele_test;
  best_allele_test.assign(allele_eval.size(), 0.0f);
  for (unsigned int i_read = 0; i_read < read_id.size(); i_read++) {
    int my_alt = read_id[i_read];
    if (my_alt > -1) {
      if (my_alt == 0) {
        // reference counts for everyone
        for (unsigned int i_alt = 0; i_alt < best_allele_test.size(); i_alt++) {
          best_allele_test[i_alt] += allele_eval[i_alt].total_theory.my_hypotheses[i_read].responsibility[1];
        }
      }
      else {
        // just update my one state
        int alt_ndx = my_alt-1;
        best_allele_test[alt_ndx] += allele_eval[alt_ndx].total_theory.my_hypotheses[i_read].responsibility[2];
      }
    }
  }
  int best_ndx = 0;
  float best_qual = best_allele_test[0];
  for (unsigned int i_alt = 1; i_alt < best_allele_test.size(); i_alt++) {
    if (best_qual < best_allele_test[i_alt]) {
      best_ndx = i_alt;
      best_qual = best_allele_test[i_alt];
    }
  }
// 0 based for best allele model
  return(best_ndx);
}

int EnsembleEval::DetectBestAllele() {
  //return(DetectBestAlleleML());
  return(DetectBestAlleleHardClassify());
}


int EnsembleEval::DetectBestAlleleML() {
  int best_idx = 0;

  float max_ll = -9999999999.0f;
  for (unsigned int i_allele = 0; i_allele < allele_eval.size(); i_allele++) {
    if (allele_eval[i_allele].ReturnMaxLL() > max_ll) {
      best_idx = i_allele;
      max_ll = allele_eval[i_allele].ReturnMaxLL();
    }
  }
  return(best_idx);
}

void EnsembleEval::ExecuteInferenceAllAlleles() {
  for (unsigned int i_allele = 0; i_allele < multi_allele_var.allele_identity_vector.size(); i_allele++) {

    allele_eval[i_allele].ExecuteInference();
  }
}
