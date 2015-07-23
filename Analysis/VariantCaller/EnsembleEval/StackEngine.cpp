/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "StackEngine.h"

// utility function for comparisons
bool compare_best_response(pair<int,float> a, pair<int,float> b){
  return (a.second>b.second);
}


void LatentSlate::PropagateTuningParameters(EnsembleEvalTuningParameters &my_params, int num_hyp_no_null) {
  // prior reliability for outlier read frequency
  cur_posterior.clustering.data_reliability = my_params.DataReliability();
  cur_posterior.clustering.germline_prior_strength = my_params.germline_prior_strength;
  //rev_posterior.data_reliability = my_params.DataReliability();
  //fwd_posterior.data_reliability = my_params.DataReliability();

  // prior precision and likelihood penalty for moving off-center
  bias_generator.InitForStrand(num_hyp_no_null-1); // num_alt = num_hyp_no_null-1
  bias_generator.damper_bias = my_params.prediction_precision;
  bias_generator.pseudo_sigma_base = my_params.pseudo_sigma_base;

  // check my biases after fit
  bias_checker.Init(num_hyp_no_null);
  bias_checker.damper_bias = my_params.prediction_precision;
  bias_checker.soft_clip = my_params.soft_clip_bias_checker;

  // prior variance-by-intensity relationship
  sigma_generator.fwd.prior_sigma_regression[0] = my_params.magic_sigma_base;
  sigma_generator.fwd.prior_sigma_regression[1] = my_params.magic_sigma_slope;
  sigma_generator.fwd.prior_weight = my_params.sigma_prior_weight;
  sigma_generator.fwd.k_zero = my_params.k_zero;
  sigma_generator.rev=sigma_generator.fwd;

  // not actually used at this point
  skew_generator.dampened_skew = my_params.prediction_precision;
}

// see how quick we can make this
void LatentSlate::LocalExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma, vector<float> &start_frequency) {
  if (detailed_integral) {
    //DetailedExecuteInference(total_theory, update_frequency, update_sigma);
    cout << "obsolete in multiallele world" << endl;
  }
  else {
    FastExecuteInference(total_theory, update_frequency, update_sigma, start_frequency);
  }
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

void LatentSlate::FastExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma, vector<float> &start_frequency) {
  // start us out estimating frequency
  cur_posterior.StartAtHardClassify(total_theory, update_frequency, start_frequency);
  FastStep(total_theory, false, false);

  float epsilon_ll = 0.01f; // make sure LL steps move us quickly instead of just spinning wheels
  float old_ll = cur_posterior.ReturnJustLL(); // always try at least one step
  iter_done = 0;
  bool keep_optimizing = true;
  ll_at_stage.resize(0);
  ll_at_stage.push_back(cur_posterior.ReturnJustLL());
  while ((iter_done < max_iterations) & keep_optimizing) {
    iter_done++;
    //cout << i_count << " max_ll " << max_ll << endl;
    old_ll = cur_posterior.ReturnJustLL(); // see if we improve over this cycle

    FastStep(total_theory, update_frequency, update_sigma);
    ll_at_stage.push_back(cur_posterior.ReturnJustLL());
    if ((old_ll+epsilon_ll) > cur_posterior.ReturnJustLL())
      keep_optimizing = false;

  }
  // now we've iterated to frustration bias/variance
  // but our responsibilities and allele frequency may not have converged to the latest values
  keep_optimizing=true;
  vector <float> old_hyp_freq;

  int nreads = total_theory.my_hypotheses.size(); //
  // always do a little cleanup on frequency even if hit max iterations
  int post_max = max(2,max_iterations-iter_done)+iter_done;
  while ((iter_done<post_max) & keep_optimizing){
    iter_done++;
    old_ll = cur_posterior.ReturnJustLL(); // do we improve?
    old_hyp_freq = cur_posterior.clustering.max_hyp_freq;
    cur_posterior.QuickUpdateStep(total_theory);  // updates max_ll as well
    ll_at_stage.push_back(cur_posterior.ReturnJustLL());
    if (cur_posterior.clustering.Compare(old_hyp_freq,nreads,0.5f))  // if total change less than 1/2 read worth
       keep_optimizing=false;
    if ((old_ll+epsilon_ll) > cur_posterior.ReturnJustLL())
      keep_optimizing = false;
  }

  // evaluate likelihood of current parameter set
  // currently only done for bias
  cur_posterior.params_ll = bias_generator.BiasLL();
}


void LatentSlate::ScanStrandPosterior(ShortStack &total_theory,bool vs_ref, int max_detail_level) {
  // keep information on distribution by strand
  if (!cur_posterior.ref_vs_all.scan_done){

    cur_posterior.ref_vs_all.DoPosteriorFrequencyScan(total_theory, cur_posterior.clustering, true, ALL_STRAND_KEY, vs_ref, max_detail_level);
    cur_posterior.gq_pair = cur_posterior.ref_vs_all; // pairwise analysis identical
  }

}

void LatentSlate::ResetToOrigin(){
  bias_generator.ResetUpdate();
  sigma_generator.ResetSigmaGenerator();
}


void HypothesisStack::DefaultValues()
{
  try_alternatives = true;
}

void HypothesisStack::AllocateFrequencyStarts(int num_hyp_no_null){
 // int num_hyp = 2; // ref + alt, called doesn't count as a "start"
  int num_start = num_hyp_no_null+1;
  if (!try_alternatives)
    num_start = 1;
  ll_record.assign(num_start,0);
  try_hyp_freq.resize(num_start);
  // reset whole matrix
  for (unsigned int i_try=0; i_try<try_hyp_freq.size(); i_try++){
    try_hyp_freq[i_try].assign(num_hyp_no_null,0.0f);
  }
  // try first at uniform distribution
    try_hyp_freq[0].assign(num_hyp_no_null,1.0f/num_hyp_no_null);
  // try pure frequencies
    if (try_alternatives){
  for (unsigned int i_hyp=0; i_hyp<(unsigned int)num_hyp_no_null; i_hyp++){
    try_hyp_freq[i_hyp+1][i_hyp]=1.0f;
  }
    }
}

float HypothesisStack::ReturnMaxLL() {
  return(cur_state.cur_posterior.ReturnMaxLL());
}

void HypothesisStack::InitForInference(PersistingThreadObjects &thread_objects, vector<const Alignment *>& read_stack, const InputStructures &global_context, int num_hyp_no_null) {
  PropagateTuningParameters(num_hyp_no_null); // sub-objects need to know

  // predict given hypotheses per read
  total_theory.FillInPredictions(thread_objects, read_stack, global_context);
  total_theory.InitTestFlow();
  total_theory.FindValidIndexes();
  // how many alleles?
  AllocateFrequencyStarts(num_hyp_no_null);
}


void HypothesisStack::ExecuteInference(int max_detail_level) {
  // now with unrestrained inference
  ExecuteExtremeInferences(max_detail_level);
  // set up our filter
  cur_state.bias_checker.UpdateBiasChecker(total_theory);
  if(max_detail_level>0) cur_state.ScanStrandPosterior(total_theory,true, max_detail_level);
}

void HypothesisStack::SetAlternateFromMain() {
  // make sure things are populated

//  ref_state = cur_state;
//  var_state = cur_state;
}



float HypothesisStack::ExecuteOneRestart(vector<float> &restart_hyp, int max_detail_level){
  LatentSlate tmp_state;

  tmp_state = cur_state;
  tmp_state.detailed_integral = false;
  tmp_state.start_freq_of_winner =restart_hyp;

  total_theory.ResetQualities();  // clean slate to begin again
  tmp_state.ResetToOrigin(); // everyone back to starting places

  tmp_state.LocalExecuteInference(total_theory, true, true, restart_hyp); // start at reference
  if(max_detail_level<1) tmp_state.ScanStrandPosterior(total_theory,true);
  float restart_LL=tmp_state.cur_posterior.ReturnMaxLL();

  if (cur_state.cur_posterior.ReturnMaxLL() <restart_LL) {
    cur_state = tmp_state; // update to the better solution, hypothetically
  }
  return(restart_LL);
}

void HypothesisStack::TriangulateRestart(){
  // cur_state contains the best remaining guys
  // take the top 3, try by pairs because of diploid theories of the world
  if (cur_state.cur_posterior.clustering.max_hyp_freq.size()>2){
  vector<float> tmp_freq = cur_state.cur_posterior.clustering.max_hyp_freq;
  vector< pair<int,float> > best_freq_test;
  best_freq_test.resize(tmp_freq.size());
  for (unsigned int i_alt=0; i_alt<tmp_freq.size(); i_alt++){
    best_freq_test[i_alt].first = i_alt;
    best_freq_test[i_alt].second = tmp_freq[i_alt];
  }
  sort(best_freq_test.begin(), best_freq_test.end(), compare_best_response);

  // top 3 elements now in vector
  // try by pairs
  // AB, AC, BC
  for (int i_zero=0; i_zero<2; i_zero++){
    for (int i_one=i_zero+1; i_one<3; i_one++){
        // try a restart
      tmp_freq.assign(tmp_freq.size(), 0.0f);
      tmp_freq[best_freq_test[i_zero].first]=0.5f;
      tmp_freq[best_freq_test[i_one].first]=0.5f;
      float try_LL = ExecuteOneRestart(tmp_freq);
      ll_record.push_back(try_LL); // adding one to the number we try
    }
  }
  }
}

// try altenatives to the main function to see if we have a better fit somewhere else
void HypothesisStack::ExecuteExtremeInferences(int max_detail_level) {
  for (unsigned int i_start=0; i_start<try_hyp_freq.size(); i_start++){
    ll_record[i_start] = ExecuteOneRestart(try_hyp_freq[i_start], max_detail_level);
  }
  //TriangulateRestart();
  RestoreFullInference(); // put total_theory to be consistent with whoever won
}

void HypothesisStack::RestoreFullInference() {
//  cur_state = backup_state;
  cur_state.bias_generator.ResetActiveBias(total_theory);
  // in theory, need to update sigma & skew, but since not fitting for particular variants, don't worry about it at the moment.
  cur_state.sigma_generator.UpdateSigmaEstimates(total_theory);
  total_theory.UpdateRelevantLikelihoods();
  //DoPosteriorFrequencyScan(cur_posterior, true);
  total_theory.UpdateResponsibility(cur_state.cur_posterior.clustering.max_hyp_freq, cur_state.cur_posterior.clustering.data_reliability); // once bias, sigma, max_freq established, reset responsibilities is forced
}


// subobjects need to know their tuning
void HypothesisStack::PropagateTuningParameters(int num_hyp_no_null) {
  total_theory.PropagateTuningParameters(my_params);
  // number of pseudo-data points at no bias
  cur_state.PropagateTuningParameters(my_params, num_hyp_no_null);
}



// tool for combining items at differing log-levels
float log_sum(float a, float b) {
  float max_ab = max(a, b);
  float log_sum_val = max_ab + log(exp(a - max_ab) + exp(b - max_ab));
  return(log_sum_val);
}

void update_genotype_interval(vector<float> &genotype_interval, vector<float> &interval_cuts, PosteriorInference &local_posterior) {
  for (unsigned int i_cut = 0; i_cut < genotype_interval.size(); i_cut++) {
    genotype_interval[i_cut] = log_sum(genotype_interval[i_cut], local_posterior.gq_pair.LogDefiniteIntegral(interval_cuts[i_cut+1], interval_cuts[i_cut]) + local_posterior.params_ll);
  }
}

void GenotypeByIntegral(vector<float> genotype_interval, int &genotype_call, float &quasi_phred_quality_score){
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
    cout << "Warning: quality score NAN "  << endl;
    quasi_phred_quality_score = 0.0f;
  }
}

void DoGenotypeByIntegral(PosteriorInference &cur_posterior, float real_safety, int &genotype_call, float &quasi_phred_quality_score){
  vector<float> interval_cuts(4);
  interval_cuts[0] = 1.0f;
  interval_cuts[1] = 1.0f - real_safety;
  interval_cuts[2] = real_safety;
  interval_cuts[3] = 0.0f;

  vector<float> genotype_interval(3);
  for (unsigned int i_cut = 0; i_cut < genotype_interval.size(); i_cut++) {
    genotype_interval[i_cut] = cur_posterior.gq_pair.LogDefiniteIntegral(interval_cuts[i_cut+1], interval_cuts[i_cut]);
  }

  GenotypeByIntegral(genotype_interval, genotype_call, quasi_phred_quality_score);

}

bool RejectionByIntegral(vector<float> dual_interval, float &reject_status_quality_score){
  // reject ref = quality of rejection call
  bool is_ref = false;
  float log_ref = 0.0f;
  int top = 0;
  int bottom = 1;
  if (dual_interval[1]>dual_interval[0]) {
    // if reference, how strongly can we reject the outer interval
    log_ref = dual_interval[0];
    is_ref = false;
    top=1;
    bottom=0;
  }
  else {
    // if var, how strongly can we reject ref
    log_ref = dual_interval[1];
    is_ref = true;
    top=0;
    bottom=1;
  }

  float log_all = dual_interval[top]+ log(1+exp(dual_interval[bottom]-dual_interval[top]));
  reject_status_quality_score = 10 * (log_all - log_ref) / log(10); // how much mass speaks against the pure reference state
  if (isnan(reject_status_quality_score)) {
    cout << "Warning: reject ref score NAN " << endl;
    reject_status_quality_score = 0.0f;
  }
  return(is_ref);
}

bool DoRejectionByIntegral(PosteriorInference &cur_posterior, float real_safety, float &reject_status_quality_score){
  vector<float> variant_cuts(3);
  variant_cuts[0] = 1.0f; // all reference
  variant_cuts[1] = 1.0f-real_safety; // all variant
  variant_cuts[2] = 0.0f;

  vector<float> dual_interval(2);
  for (unsigned int i_cut = 0; i_cut < dual_interval.size(); i_cut++) {
    dual_interval[i_cut] = cur_posterior.ref_vs_all.LogDefiniteIntegral(variant_cuts[i_cut+1], variant_cuts[i_cut]);
  }

  return( RejectionByIntegral(dual_interval, reject_status_quality_score));
}

// must have scan done to be workable
// must have set which pair of hypotheses are being checked
bool CallByIntegral(PosteriorInference &cur_posterior, float hom_safety, int &genotype_call, float &quasi_phred_quality_score, float &reject_status_quality_score) {
  // divide MAF into three zones = 0/1/2 variant alleles

  // make sure we're safe based on the number of reads
  int detail_level = cur_posterior.ref_vs_all.eval_at_frequency.size();
  // if we don't have enough reads might be crazy near mid
  float fine_scale = 0.5f / (detail_level + 1.0f);
  float mid_cutoff = 0.5f - fine_scale;
  // if we don't have any variants, still can't exclude 3/num_reads frequency
  // so no sense testing that low
  float low_cutoff = 1.0f / detail_level;
  // if we have a small number of reads, these ranges may conflict
  float real_safety = min(mid_cutoff, max(low_cutoff, hom_safety));

  bool safety_active_flag = false;
  if (fabs(real_safety - hom_safety) > fine_scale)
    safety_active_flag = true;

  // bool isref =
  DoRejectionByIntegral(cur_posterior, real_safety, reject_status_quality_score);

  // in dual allele case, do not need to check "is-ref" before making some quality assessment
  DoGenotypeByIntegral(cur_posterior, real_safety, genotype_call, quasi_phred_quality_score);

  //cout << genotype_call << "\t" << reject_status_quality_score << "\t" << quasi_phred_quality_score << endl;

  return(safety_active_flag);
}

bool HypothesisStack::CallGermline(float hom_safety, int &genotype_call, float &quasi_phred_quality_score, float &reject_status_quality_score){
  bool retval = CallByIntegral(cur_state.cur_posterior, hom_safety, genotype_call, quasi_phred_quality_score, reject_status_quality_score);
  return(retval);
}



// evidence for i_allele vs ref
void EnsembleEval::ScanSupportingEvidence(float &mean_ll_delta,  int i_allele) {

  mean_ll_delta = 0.0f;
  int count = 0;
  int ref_hyp = 1;
  int alt_hyp = i_allele + 2;  // alt_alleles = 0->n not counting ref >or< null, therefore alt-allele 0 = 2

  for (unsigned int i_read = 0; i_read < allele_eval.total_theory.my_hypotheses.size(); i_read++) {
    if (allele_eval.total_theory.my_hypotheses[i_read].success) {
      // measure disruption

      mean_ll_delta += allele_eval.total_theory.my_hypotheses[i_read].ComputeLLDifference(ref_hyp, alt_hyp);
      count++;
    }
  }
  mean_ll_delta /= (count + 0.01f);
  mean_ll_delta = 10.0f * mean_ll_delta / log(10.0f); // phred-scaled
}

void EnsembleEval::ApproximateHardClassifierForReadsFromMultiAlleles(vector<int> &read_allele_id, vector<bool> &strand_id, vector<int> &dist_to_left, vector<int> &dist_to_right){
  // in fact this is a lot easier
  read_allele_id.assign(read_stack.size(), -1);
  for (unsigned int i_read = 0; i_read < read_allele_id.size(); i_read++) {
    read_allele_id[i_read]=allele_eval.total_theory.my_hypotheses[i_read].MostResponsible()-1; // -1 = null, 0 = ref , ...
    if (!allele_eval.total_theory.my_hypotheses[i_read].success) {
      read_allele_id[i_read] = -1; // failure = outlier
    }
  }
  strand_id.assign(read_stack.size(), false);
  for (unsigned int i_read = 0; i_read < strand_id.size(); i_read++) {
    strand_id[i_read] = not read_stack[i_read]->is_reverse_strand;
  }

  // for each variant, calculate its' position within the soft clipped read
  // distance to left and distance to right
  dist_to_left.assign(read_stack.size(), -1);
  dist_to_right.assign(read_stack.size(), -1);

  int position0 = variant->position -1; // variant->position 1-base: vcflib/Variant.h
  for (unsigned int i_read = 0; i_read < strand_id.size(); i_read++) {
    if (read_allele_id[i_read] > -1){
      //fprintf(stdout, "position0 =%d, read_stack[i_read]->align_start = %d, read_stack[i_read]->align_end = %d, read_stack[i_read]->left_sc = %d, read_stack[i_read]->right_sc = %d\n", (int)position0, (int)read_stack[i_read]->align_start, (int)read_stack[i_read]->align_end, (int)read_stack[i_read]->left_sc, (int)read_stack[i_read]->right_sc);
      //fprintf(stdout, "dist_to_left[%d] = =%d, dist_to_right[%d] = %d\n", (int)i_read, (int)(position0 - read_stack[i_read]->align_start), (int)i_read, (int)(read_stack[i_read]->align_end - position0));

      dist_to_left[i_read] = position0 - read_stack[i_read]->align_start;
      assert ( dist_to_left[i_read] >=0 );
      dist_to_right[i_read] = read_stack[i_read]->align_end - position0;
      assert ( dist_to_right[i_read] >=0 );
    }
  }
}

void EnsembleEval::ApproximateHardClassifierForReads(vector<int> &read_allele_id, vector<bool> &strand_id, vector<int> &dist_to_left, vector<int> &dist_to_right)
{
  ApproximateHardClassifierForReadsFromMultiAlleles(read_allele_id, strand_id, dist_to_left, dist_to_right);
}



int EnsembleEval::DetectBestMultiAllelePair()
{
  vector<int> read_id;        // vector of allele ids per read, -1 = outlier, 0 = ref, >0 real allele
  vector<bool> strand_id;     // vector of forward (true) or reverse (false) per read
  vector<int> dist_to_left;   // vector of distances from allele position to left scoft clip per read
  vector<int> dist_to_right;  // vector of distances from allele position to left scoft clip per read

  int best_alt_ndx = 0; // forced choice with ref

  ApproximateHardClassifierForReads(read_id, strand_id, dist_to_left, dist_to_right);

  //@TODO: just get the plane off the ground
  //@TODO: do the top pair by responsibility
  vector< pair<int,float> > best_allele_test;
  int num_hyp_no_null = allele_eval.total_theory.my_hypotheses[0].responsibility.size()-1;
  best_allele_test.resize(num_hyp_no_null); // null can never be a winner in "best allele" sweepstakes
  for (unsigned int i_alt=0; i_alt<best_allele_test.size(); i_alt++){
    best_allele_test[i_alt].first = i_alt;
    best_allele_test[i_alt].second = 0.0f;
  }
  // take responsibility
  for (unsigned int i_read=0; i_read<read_id.size(); i_read++){
    int my_alt = read_id[i_read];
    if (my_alt> -1){
      best_allele_test[my_alt].second += allele_eval.total_theory.my_hypotheses[i_read].responsibility[my_alt+1];
    } // otherwise count for nothing
  }
  // pick my pair of best alleles
  sort(best_allele_test.begin(), best_allele_test.end(), compare_best_response);
  // // not-null choices
  diploid_choice[0]= best_allele_test[0].first; // index of biggest weight
  diploid_choice[1]= best_allele_test[1].first; // index of second-biggest weight
  // problematic cases:
  // 2 alleles & ref, ref + 1 allele zero, want ref as the comparison
  // all ref, don't care about the second allele for genotype?
  // all zero implies what?
  //cout << best_allele_test.at(0).first << "\t" << best_allele_test.at(0).second << "\t" << best_allele_test.at(1).first << "\t" << best_allele_test.at(1).second << endl;
  if (diploid_choice[0]==0)
    best_alt_ndx = diploid_choice[1]-1;
  else
    best_alt_ndx = diploid_choice[0]-1;

  // sort as final step to avoid best_alt_ndx reflecting a worse allele
  sort(diploid_choice.begin(),diploid_choice.end()); // now in increasing allele order as needed

  return(best_alt_ndx);
}



void EnsembleEval::ComputePosteriorGenotype(int _alt_allele_index,float local_min_allele_freq,
                                            int &genotype_call, float &gt_quality_score, float &reject_status_quality_score)
{
  allele_eval.CallGermline(local_min_allele_freq,
     genotype_call,
     gt_quality_score,
     reject_status_quality_score);
}

void EnsembleEval::MultiAlleleGenotype(float local_min_allele_freq, vector<int> &genotype_component, float &gt_quality_score, float &reject_status_quality_score, int max_detail_level)
{
  // detect best allele hard classify
 // DetectBestAlleleHardClassify(); //diploid choice set
  DetectBestMultiAllelePair(); // diploid_choice set by posterior responsibility

  // set diploid in gq_pair
  allele_eval.cur_state.cur_posterior.gq_pair.freq_pair = diploid_choice;

  // scan gq_pair
  allele_eval.cur_state.cur_posterior.gq_pair.DoPosteriorFrequencyScan(allele_eval.total_theory,
                                                                             allele_eval.cur_state.cur_posterior.clustering,
                                                                             true, ALL_STRAND_KEY, false, max_detail_level);

  //call-germline
  int genotype_call;
  allele_eval.CallGermline(local_min_allele_freq, genotype_call, gt_quality_score, reject_status_quality_score);
  // set the outputs

  // start at "het" by choice
  genotype_component[0]=diploid_choice[0];
  genotype_component[1]=diploid_choice[1];
  if (genotype_call==2)
    genotype_component[0] = diploid_choice[1];  //hom var
  if (genotype_call==0)
    genotype_component[1] = diploid_choice[0];  //hom ref
}
