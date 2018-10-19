/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "StackEngine.h"
#include "MiscUtil.h"
#include "DecisionTreeData.h"

// utility function for comparisons
bool compare_best_response(pair<int,float> a, pair<int,float> b){
  return (a.second>b.second);
}


void LatentSlate::PropagateTuningParameters(EnsembleEvalTuningParameters &my_params, int num_hyp_no_null) {
  // prior reliability for outlier read frequency
  cur_posterior.clustering.outlier_prob = my_params.outlier_prob;
  cur_posterior.clustering.germline_prior_strength = my_params.germline_prior_strength;
  cur_posterior.gq_pair.max_detail_level = (unsigned int) my_params.max_detail_level;
  cur_posterior.gq_pair.min_detail_level_for_fast_scan = (unsigned int) my_params.min_detail_level_for_fast_scan;
  cur_posterior.ref_vs_all.min_detail_level_for_fast_scan = (unsigned int) my_params.min_detail_level_for_fast_scan;
  cur_posterior.ref_vs_all.max_detail_level = (unsigned int) my_params.max_detail_level;

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

void LatentSlate::SetAndPropagateDebug(int debug) {
	DEBUG = debug;
	cur_posterior.DEBUG = debug;
	cur_posterior.ref_vs_all.DEBUG = debug;
	cur_posterior.gq_pair.DEBUG = debug;
	bias_generator.DEBUG = debug;
	sigma_generator.DEBUG = debug;
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
 if(DEBUG > 0){
   cout << "  - Starting one iteration with allele_freq = " << PrintIteratorToString(cur_posterior.clustering.max_hyp_freq.begin(), cur_posterior.clustering.max_hyp_freq.end()) << (update_frequency? "." : ", update_frequency = false.") << endl;
  }

  bias_generator.DoStepForBias(total_theory); // update bias estimate-> residuals->likelihoods
  if (update_frequency)
    cur_posterior.QuickUpdateStep(total_theory);

  if (update_sigma) {
    sigma_generator.DoStepForSigma(total_theory); // update sigma estimate
    if (update_frequency)
      cur_posterior.QuickUpdateStep(total_theory);
  }
  if(DEBUG > 0){
	  cout << "  - One iteration done: ref_vs_all.max_ll = "<< cur_posterior.ReturnJustLL() <<endl;
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


void LatentSlate::ScanStrandPosterior(ShortStack &total_theory,bool vs_ref) {
  // keep information on distribution by strand
  if (!cur_posterior.ref_vs_all.scan_ref_done){
    cur_posterior.ref_vs_all.DoPosteriorFrequencyScan(total_theory, cur_posterior.clustering, true, ALL_STRAND_KEY, vs_ref);
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
  DEBUG = 0;
  variant = NULL;
}

// if try_alternatives and my_params.try_few_restart_freq, then I
// skip the pure frequencies of snp or mnp alt alleles,
// also skip the pure frequency of ref allele if "all" alt alleles are snp or mnp
void HypothesisStack::AllocateFrequencyStarts(int num_hyp_no_null, vector<AlleleIdentity> &allele_identity_vector){
   // int num_hyp = 2; // ref + alt, called doesn't count as a "start"
    //int num_start = num_hyp_no_null + 1;
    vector<float> try_me(num_hyp_no_null);
    float safety_zero = 0.001f;

    // I try at most (1 + num_hyp_no_null) frequencies: one uniform allele freq and num_hyp_no_null pure freq
    try_hyp_freq.reserve(1 + num_hyp_no_null);

    // Always try uniform hyp_freq
    try_me.assign(num_hyp_no_null, 1.0f / (float) num_hyp_no_null);
    try_hyp_freq.push_back(try_me);

    vector<int> fake_hs_alleles;
    fake_hs_alleles.reserve(num_hyp_no_null - 1);

    // try pure frequencies for the alleles
    if(try_alternatives){
    	// try pure frequencies for alt alleles
    	for(int i_hyp = 1; i_hyp < num_hyp_no_null; ++i_hyp){
    		int i_alt = i_hyp - 1;
   			if (allele_identity_vector[i_alt].status.isFakeHsAllele){
   				// I don't want to try the pure freq for a fake hs allele/
   				fake_hs_alleles.push_back(i_hyp);
   				continue;
   			}
   			if(my_params.try_few_restart_freq and (allele_identity_vector[i_alt].ActAsSNP() or allele_identity_vector[i_alt].ActAsMNP())){
   				// skip the pure frequency of a snp or mnp alt allele.
   				continue;
    		}

   			try_me.assign(num_hyp_no_null, safety_zero / float(num_hyp_no_null - 1));
   		    try_me[i_hyp] = 1.0f - safety_zero;
   		    try_hyp_freq.push_back(try_me);
    	}

    	bool has_ref_pure = false;
    	// try the pure frequency of the ref allele if we try at least one pure frequency of alt allele
    	if (try_hyp_freq.size() > 1){
   			try_me.assign(num_hyp_no_null, safety_zero / float(num_hyp_no_null - 1));
   		    try_me[0] = 1.0f - safety_zero;
   		    try_hyp_freq.push_back(try_me);
   		    has_ref_pure = true;
    	}

    	// Finally, try the uniform allele freq precluding all fake HS alleles.
    	if (not fake_hs_alleles.empty()){
        	int num_non_fake_hs_alleles = num_hyp_no_null - (int) fake_hs_alleles.size();
        	if (num_non_fake_hs_alleles > 1 or (num_non_fake_hs_alleles == 1 and (not has_ref_pure))){
        		try_me.assign(num_hyp_no_null, (1.0f - safety_zero) / (float) num_non_fake_hs_alleles);
        		float my_safety_zero = safety_zero / (float) fake_hs_alleles.size();
				for (unsigned int idx = 0; idx < fake_hs_alleles.size(); ++idx){
					try_me[fake_hs_alleles[idx]] = my_safety_zero;
				}
				try_hyp_freq.push_back(try_me);
        	}
    	}
    }

    ll_record.assign(try_hyp_freq.size(), 0.0f);
}

float HypothesisStack::ReturnMaxLL() {
  return cur_state.cur_posterior.ReturnMaxLL();
}

void HypothesisStack::InitForInference(PersistingThreadObjects &thread_objects, vector<const Alignment *>& read_stack, const InputStructures &global_context, vector<AlleleIdentity> &allele_identity_vector) {
  int num_hyp_no_null = (int) allele_identity_vector.size() + 1; // number of alt alleles + 1 for ref
  total_theory.num_hyp_not_null = num_hyp_no_null;
  PropagateTuningParameters(num_hyp_no_null); // sub-objects need to know
  // how many alleles?
  AllocateFrequencyStarts(num_hyp_no_null, allele_identity_vector);
  // predict given hypotheses per read
  total_theory.FillInPredictionsAndTestFlows(thread_objects, read_stack, global_context);
  // FlowDisruptiveOutlierFiltering must be done after FillInPredictionsAndTestFlows and filling FD matrix.
  total_theory.FlowDisruptiveOutlierFiltering((unsigned int) my_params.outlier_pre_filter, false);  // Filter out obvious outlier reads using flow-disruption.
  if(not total_theory.GetIsMolecularTag()){
	  // If use mol tag, total_theory.FindValidIndexes() will be done in total_theory.InitializeMyEvalFamilies(unsigned int num_hyp)
	  total_theory.FindValidIndexes();
  }
  else{
	total_theory.InitializeMyEvalFamilies((unsigned int) num_hyp_no_null + 1);
	if(DEBUG > 0){
		cout << endl << "+ Initialized families on the read stack: "<< endl
			 <<"  - Number of reads on the read stack = " << total_theory.my_hypotheses.size() << endl
			 <<"  - Number of valid reads on the read stack = " << total_theory.valid_indexes.size() << endl
			 <<"  - Number of families on the read stack = "<< total_theory.my_eval_families.size() << endl
			 <<"  - Number of functional families on the read stack = "<< total_theory.GetNumFuncFamilies() << endl;
	}
  }
}


void HypothesisStack::ExecuteInference() {
  unsigned long t0 = clock();

  if(DEBUG > 0){
    cout << endl << "+ Execute inference for the variant (" << PrintVariant(*variant) << ")" << endl;
  }

  // now with unrestrained inference
  ExecuteExtremeInferences();
  // set up our filter
  cur_state.bias_checker.UpdateBiasChecker(total_theory);
  // @TODO: Now with fast scan, we can always do it
  if(cur_state.cur_posterior.ref_vs_all.max_detail_level > 0){
	  cur_state.ScanStrandPosterior(total_theory, true);
  }

  if(DEBUG > 0){
    vector<float> max_allele_freq  = cur_state.cur_posterior.clustering.max_hyp_freq;
    if(cur_state.cur_posterior.ref_vs_all.scan_ref_done){
      cur_state.cur_posterior.clustering.UpdateFrequencyAgainstOne(max_allele_freq, cur_state.cur_posterior.ref_vs_all.eval_at_frequency[cur_state.cur_posterior.ref_vs_all.max_index], 0);
    }
    cout << "+ Execute inference for the variant (" <<  PrintVariant(*variant) << ") done. "
    	 <<	"Processing time = " << (double) (clock() - t0) / 1E6 << " sec."<< endl
		 << "  - Winner of the initial allele_freq = " << PrintIteratorToString(cur_state.start_freq_of_winner.begin(), cur_state.start_freq_of_winner.end()) << endl;
    cur_state.bias_generator.PrintDebug(false);
	cur_state.sigma_generator.PrintDebug(false);
    cout << "  - params_ll = "<< cur_state.cur_posterior.params_ll << endl
	     << "  - ref_vs_all.max_ll + params_ll = "<< cur_state.cur_posterior.ReturnMaxLL() <<" @ allele_freq = " <<  PrintIteratorToString(max_allele_freq.begin(), max_allele_freq.end())
	     << (cur_state.cur_posterior.ref_vs_all.scan_ref_done? " from scan." : "from responsibility.") << endl << endl;
  }

}

void HypothesisStack::SetAlternateFromMain() {
  // make sure things are populated

//  ref_state = cur_state;
//  var_state = cur_state;
}



float HypothesisStack::ExecuteOneRestart(vector<float> &restart_hyp){
  LatentSlate tmp_state;
  tmp_state = cur_state;
  tmp_state.detailed_integral = false;
  tmp_state.start_freq_of_winner = restart_hyp;
  total_theory.ResetQualities(my_params.outlier_prob);  // clean slate to begin again
  tmp_state.ResetToOrigin(); // everyone back to starting places

  if (DEBUG > 0){
	  cout<< "+ Restart the EM algorithm with initial allele_freq = " << PrintIteratorToString(restart_hyp.begin(), restart_hyp.end()) << endl;
  }

  tmp_state.LocalExecuteInference(total_theory, true, true, restart_hyp); // start at reference

  if( tmp_state.cur_posterior.ref_vs_all.max_detail_level == 0){
	  if (DEBUG > 0){
		  cout<< "  - Scan posterior likelihood for the restart of the EM algorithm."<< endl;
	  }
	  tmp_state.ScanStrandPosterior(total_theory, true);
  }
  float restart_LL=tmp_state.cur_posterior.ReturnMaxLL();

  if (DEBUG > 0){
	  vector<float> max_allele_freq = tmp_state.cur_posterior.clustering.max_hyp_freq;
	  if(tmp_state.cur_posterior.ref_vs_all.scan_ref_done){
        tmp_state.cur_posterior.clustering.UpdateFrequencyAgainstOne(max_allele_freq, tmp_state.cur_posterior.ref_vs_all.eval_at_frequency[tmp_state.cur_posterior.ref_vs_all.max_index], 0);
	  }
	  cout << "+ Restart the EM algorithm with initial allele_freq = " << PrintIteratorToString(restart_hyp.begin(), restart_hyp.end()) << " done. "<< endl
	       << "  - params_ll = "<< tmp_state.cur_posterior.params_ll << endl
	       << "  - ref_vs_all.max_ll + params_ll = "<< restart_LL << " @ allele_freq = " << PrintIteratorToString(max_allele_freq.begin(), max_allele_freq.end()) << (tmp_state.cur_posterior.ref_vs_all.scan_ref_done? " from scan." : "from responsibility.") << endl;
  }

  if (cur_state.cur_posterior.ReturnMaxLL() <restart_LL) {
    cur_state = tmp_state; // update to the better solution, hypothetically
  }

  return restart_LL;
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
void HypothesisStack::ExecuteExtremeInferences() {
  if (DEBUG > 0){
	  cout<< "  - The EM algorithm will be restarted by trying "<< try_hyp_freq.size() << " different initial allele_freq."<< endl;
  }
  for (unsigned int i_start=0; i_start<try_hyp_freq.size(); i_start++){
    ll_record[i_start] = ExecuteOneRestart(try_hyp_freq[i_start]);
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
  total_theory.UpdateResponsibility(cur_state.cur_posterior.clustering.max_hyp_freq, cur_state.cur_posterior.clustering.outlier_prob); // once bias, sigma, max_freq established, reset responsibilities is forced
}


// subobjects need to know their tuning
void HypothesisStack::PropagateTuningParameters(int num_hyp_no_null) {
  total_theory.PropagateTuningParameters(my_params);
  // number of pseudo-data points at no bias
  cur_state.PropagateTuningParameters(my_params, num_hyp_no_null);
  if (my_params.outlier_pre_filter < 0){
	  my_params.outlier_pre_filter = total_theory.GetIsMolecularTag()? 1 : 0;
  }
}

// tool for combining items at differing log-levels
float log_sum(float a, float b) {
  float max_ab = max(a, b);
  float log_sum_val = max_ab + log(exp(a - max_ab) + exp(b - max_ab));
  return log_sum_val;
}

void update_genotype_interval(vector<float> &genotype_interval, vector<float> &interval_cuts, PosteriorInference &local_posterior) {
  for (unsigned int i_cut = 0; i_cut < genotype_interval.size(); i_cut++) {
    genotype_interval[i_cut] = log_sum(genotype_interval[i_cut], local_posterior.gq_pair.LogDefiniteIntegral(interval_cuts[i_cut+1], interval_cuts[i_cut]) + local_posterior.params_ll);
  }
}

void GenotypeByIntegral(vector<float> genotype_interval, int &genotype_call, float &quasi_phred_quality_score){
  //@TODO: do as paired and sort, for clean code

  // First normalize genotype_interval to enhance the floating point accuracy
  // It's better not to calculate best_val and worst_val together with the normalization step.
  float max_genotype_interval = genotype_interval[0];
  for (unsigned int i_geno = 1; i_geno < genotype_interval.size(); i_geno++)
	  max_genotype_interval = max(genotype_interval[i_geno], max_genotype_interval);
  for (unsigned int i_geno = 0; i_geno < genotype_interval.size(); i_geno++) {
	  genotype_interval[i_geno] -= max_genotype_interval;
  }

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
  float log_alternative = middle_val + log(1.0f + exp(worst_val - middle_val)); // total mass on alternative intervals
  float log_all = best_val + log(1.0f + exp(log_alternative - best_val)); // total mass on all intervals

  // output
  quasi_phred_quality_score = 10.0f * (log_all - log_alternative) / log(10.0f); // -10*log(error), base 10

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

  if (cur_posterior.DEBUG > 0){
	  int pair_0 = cur_posterior.gq_pair.freq_pair[0];
	  int pair_1 = cur_posterior.gq_pair.freq_pair[1];
	  float allele_ratio_cut_off = real_safety / (1.0f - real_safety);
	  cout << "+ Calling genotype: " << endl;
	  cout << "  - Baseline allele_freq = " << PrintIteratorToString(cur_posterior.clustering.max_hyp_freq.begin(), cur_posterior.clustering.max_hyp_freq.end()) << endl;
	  cout << "  - H(GT="<< pair_0 <<"/"<< pair_0
		   << ": allele_freq[" << pair_1 <<"]/(allele_freq["<< pair_0 << "]+allele_freq[" << pair_1 << "])"
		   << " < " << real_safety << "), log(definite integral of posterior) = " << genotype_interval[0] <<endl;
	  cout << "  - H(GT="<< pair_0 <<"/"<< pair_1
		   << ": " << real_safety <<  " <= allele_freq[" << pair_1 <<"]/(allele_freq["<< pair_0 << "]+allele_freq[" << pair_1 << "])" << " <= " << 1.0f - real_safety
		   << "), log(definite integral of posterior) = " << genotype_interval[1] <<endl;
	  cout << "  - H(GT="<< pair_1 <<"/"<< pair_1
		   << ": allele_freq[" << pair_1 <<"]/(allele_freq["<< pair_0 << "]+allele_freq[" << pair_1 << "])"
		   << " > " << real_safety << "), log(definite integral of posterior) = " << genotype_interval[2] << endl;
	  cout << "  - Call GT = ";
	  switch (genotype_call){
	      case 0:
	    	  cout << pair_0 <<"/"<< pair_0;
	    	  break;
	      case 1:
	    	  cout << pair_0 <<"/"<< pair_1;
	    	  break;
	      case 2:
	    	  cout << pair_1 <<"/"<< pair_1;
	    	  break;
	      default:
	    	  cout << "?/?";
	    	  break;
	  }
	  cout << ", GQ = " << quasi_phred_quality_score << endl;
  }

}

bool RejectionByIntegral(vector<float> dual_interval, float &reject_status_quality_score){
  // reject ref = quality of rejection call
  bool is_ref = dual_interval[0] >= dual_interval[1];
  double dual_interval_diff = abs((double) dual_interval[0] - (double) dual_interval[1]);  // Use double to enhance the numerical accuracy
  // QUAL = exp(min(dual_interval)) / ( exp(min(dual_interval)) + exp(max(dual_interval) ) ) in phred scale which is equivalent to the following expression.
  reject_status_quality_score = (float) (10.0 / log(10.0) * (dual_interval_diff + log(1.0 + exp(-dual_interval_diff))));
  reject_status_quality_score = max(reject_status_quality_score, 3.0103f); // Guard by the minimum possible QUAL score to prevent floating point error.

  return is_ref;
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

  bool is_ref = RejectionByIntegral(dual_interval, reject_status_quality_score);
  if (cur_posterior.DEBUG > 0){
	  cout << "+ Calling ref vs. (all alt): "<< endl;
	  cout << "  - Baseline allele_freq = " << PrintIteratorToString(cur_posterior.clustering.max_hyp_freq.begin(), cur_posterior.clustering.max_hyp_freq.end()) << endl;
	  float min_area = min(dual_interval[0], dual_interval[1]);
	  cout << "  - H(ref: sum(allele_freq[1:])/allele_freq[0] < "<< real_safety << "/" << (1.0f - real_safety)
		   << " = " << real_safety / (1.0f - real_safety) << "), log(definite integral of posterior) = " << min_area <<" + " << (dual_interval[0] - min_area) << endl;
	  cout << "  - H(all alt: sum(allele_freq[1:])/allele_freq[0] >= "<< real_safety << "/" << (1.0f - real_safety)
		   << " = " << real_safety / (1.0f - real_safety) << "), log(definite integral of posterior) = " <<  min_area <<" + " << (dual_interval[1] - min_area) << endl;
	  cout << "  - "<< (is_ref? "Call reference" : "Call variant")<< ", QUAL = " << reject_status_quality_score << endl;
  }
  return is_ref;
}

bool CalculateRealSafetyAfCutoff(float af_cutoff, int detail_level, float &real_safety){
  // make sure we're safe based on the number of reads
  // if we don't have enough reads might be crazy near mid
  float fine_scale = 0.5f / (detail_level + 1.0f);
  float mid_cutoff = 0.5f - fine_scale;
  // if we don't have any variants, still can't exclude 3/num_reads frequency
  // so no sense testing that low
  float low_cutoff = 1.0f / detail_level;
  // if we have a small number of reads, these ranges may conflict
  real_safety = min(mid_cutoff, max(low_cutoff, af_cutoff));
  bool safety_active_flag = false;
  if (fabs(real_safety- af_cutoff) > fine_scale)
	safety_active_flag = true;
  return safety_active_flag;
}

// must have scan done to be workable
// must have set which pair of hypotheses are being checked
bool HypothesisStack::CallByIntegral(float af_cutoff_rej, float af_cutoff_gt, vector<int> &genotype_component, float &quasi_phred_quality_score, float &reject_status_quality_score, int &qual_type) {
  // divide MAF into three zones = 0/1/2 variant alleles
  bool safety_active_flag = false;
  int detail_level = cur_state.cur_posterior.ref_vs_all.eval_at_frequency.size();

  float real_safety_rej = af_cutoff_rej;
  safety_active_flag += CalculateRealSafetyAfCutoff(af_cutoff_rej, detail_level, real_safety_rej);


  bool is_ref = DoRejectionByIntegral(cur_state.cur_posterior, real_safety_rej, reject_status_quality_score);
  qual_type = is_ref? 0 : 1;

  /* Less exception is better, though it is possible to see some non-trivial cases.
  // If I reject the variant, I force to make a reference call (i.e., GT=0/0) and output GQ=QUAL.
  // In this case, I don't even try to scan the gq pair and call DoGenotypeByIntegral.
  if (is_ref){
	  quasi_phred_quality_score = reject_status_quality_score;
	  genotype_component.assign(2, 0);
	  if (cur_state.cur_posterior.DEBUG > 0){
		  cout << "+ Set GT=0/0 and GQ=QUAL for a reference call." << endl;
	  }
	  return safety_active_flag;
  }
  */

  int genotype_call = 0;
  assert(not cur_state.cur_posterior.gq_pair.freq_pair.empty()); // gq_pair.freq_pair must be set.
  if (not cur_state.cur_posterior.gq_pair.scan_pair_done){
	  cur_state.cur_posterior.gq_pair.DoPosteriorFrequencyScan(total_theory, cur_state.cur_posterior.clustering, true, ALL_STRAND_KEY, false);
  }

  float real_safety_gt = af_cutoff_gt;
  safety_active_flag += CalculateRealSafetyAfCutoff(af_cutoff_gt, detail_level, real_safety_gt);

  // in dual allele case, do not need to check "is-ref" before making some quality assessment
  DoGenotypeByIntegral(cur_state.cur_posterior, real_safety_gt, genotype_call, quasi_phred_quality_score);

  genotype_component = cur_state.cur_posterior.gq_pair.freq_pair; // het diploid_choice[0]/diploid_choice[1]
  if(genotype_call == 2){ // hom diploid_choice[1]/diploid_choice[1]
	genotype_component[0] = genotype_component[1];
  }
  else if(genotype_call == 0){ // hom diploid_choice[0]/diploid_choice[0]
    genotype_component[1] = genotype_component[0];
  }
  return safety_active_flag;
}

// evidence for i_allele vs ref
void EnsembleEval::ScanSupportingEvidence(float &mean_ll_delta,  int i_allele) {

  mean_ll_delta = 0.0f;
  int count = 0;
  int ref_hyp = 1;
  int alt_hyp = i_allele + 2;  // alt_alleles = 0->n not counting ref >or< null, therefore alt-allele 0 = 2

  if (allele_eval.total_theory.GetIsMolecularTag()){
	  for (vector<EvalFamily>::iterator fam_it = allele_eval.total_theory.my_eval_families.begin(); fam_it != allele_eval.total_theory.my_eval_families.end(); ++fam_it){
		  if (fam_it->GetFuncFromValid()){
			  mean_ll_delta += fam_it->ComputeLLDifference(ref_hyp, alt_hyp);
			  ++count;
		  }
	  }
  }
  else{
	  for (vector<CrossHypotheses>::iterator read_it = allele_eval.total_theory.my_hypotheses.begin(); read_it != allele_eval.total_theory.my_hypotheses.end(); ++read_it){
		  if (read_it->success){
			  mean_ll_delta += read_it->ComputeLLDifference(ref_hyp, alt_hyp);
			  ++count;
		  }
	  }
  }

  mean_ll_delta /= ((float) count + 0.01f);
  mean_ll_delta = 10.0f * mean_ll_delta / log(10.0f); // phred-scaled
}

// Hard classify each read using its responsibility
// read_id_[i] = -1 means the i-th read is classified as an outlier.
// read_id_[i] = 0 means the i-th read is classified as ref.
// read_id_[i] = 1 means the i-th read is classified as the variant allele 1, and so on.
void EnsembleEval::ApproximateHardClassifierForReads()
{
    read_id_.assign(read_stack.size(), -1);
	strand_id_.assign(read_stack.size(), -1);
	dist_to_left_.assign(read_stack.size(), -1);
	dist_to_right_.assign(read_stack.size(), -1);

	int position0 = variant->position -1; // variant->position 1-base: vcflib/Variant.h
	for (unsigned int i_read = 0; i_read < read_stack.size(); ++i_read) {
		// compute read_id_
		if(allele_eval.total_theory.my_hypotheses[i_read].success){
			int most_responsibile_hyp = allele_eval.total_theory.my_hypotheses[i_read].MostResponsible();
	        read_id_[i_read] = most_responsibile_hyp - 1; // -1 = null, 0 = ref , ...
		}
		else{
	        read_id_[i_read] = -1; // failure = outlier
	    }

		// not an outlier
		if(read_id_[i_read] > -1){
		    //fprintf(stdout, "position0 =%d, read_stack[i_read]->align_start = %d, read_stack[i_read]->align_end = %d, read_stack[i_read]->left_sc = %d, read_stack[i_read]->right_sc = %d\n", (int)position0, (int)read_stack[i_read]->align_start, (int)read_stack[i_read]->align_end, (int)read_stack[i_read]->left_sc, (int)read_stack[i_read]->right_sc);
		    //fprintf(stdout, "dist_to_left[%d] = =%d, dist_to_right[%d] = %d\n", (int)i_read, (int)(position0 - read_stack[i_read]->align_start), (int)i_read, (int)(read_stack[i_read]->align_end - position0));
	        dist_to_left_[i_read] = position0 - read_stack[i_read]->align_start;
		    assert ( dist_to_left_[i_read] >=0 );
		    dist_to_right_[i_read] = read_stack[i_read]->align_end - position0;
		    assert ( dist_to_right_[i_read] >=0 );
	    }
	    //compute strand_id_
	    strand_id_[i_read] = read_stack[i_read]->is_reverse_strand? 1 : 0;
	}
}

struct PositionGroup{
	vector<const Alignment*> group_members;
	uint16_t mapq_sum;
	int group_size;
	int group_position;
	PositionGroup(){ mapq_sum = 0; group_size = 0; group_position = 0;};
	PositionGroup(const Alignment* rai, int p){
		group_members = {rai};
		mapq_sum = rai->alignment.MapQuality;
		group_size = rai->read_count;
		group_position = p;
	};
	void AddNew(const Alignment* rai){
		group_members.push_back(rai);
		mapq_sum += rai->alignment.MapQuality;
		group_size += rai->read_count;
	};
	// Compare two groups.
	// A group wins if it has more reads. If ties, the group with larger accumulated mapping quality wins.
	bool operator>(const PositionGroup& rhs){
	    if (group_size == rhs.group_size){
			return (mapq_sum > rhs.mapq_sum);
		}
		return (group_size > rhs.group_size);
	};
};

int GetConsensusPosition(EnsembleEval* my_ensemble, EvalFamily* my_fam, bool is_to_left){
	if (not my_fam->GetFuncFromValid()){
		return -1;
	}
	int position0 = my_ensemble->variant->position -1; // variant->position 1-base: vcflib/Variant.h
	vector<PositionGroup> position_groups_vec;
	position_groups_vec.reserve(my_fam->valid_family_members.size());
	for (vector<unsigned int>::iterator member_it = my_fam->valid_family_members.begin(); member_it != my_fam->valid_family_members.end(); ++member_it)
	{
		if (not my_ensemble->allele_eval.total_theory.my_hypotheses[*member_it].success){
			continue;
		}
		const Alignment* rai = my_ensemble->read_stack[*member_it];
		int my_position = is_to_left? position0 - rai->align_start : rai->align_end - position0;
		assert(my_position >= 0);
	    bool group_exists = false;
	    for (vector<PositionGroup>::iterator pos_group_it = position_groups_vec.begin(); pos_group_it != position_groups_vec.end(); ++pos_group_it){
	    	if (my_position == pos_group_it->group_position){
	    		group_exists = true;
	    		pos_group_it->AddNew(rai);
	    		break;
	    	}
	    }
	    if (not group_exists){
	    	position_groups_vec.push_back(PositionGroup(rai, my_position));
	    }
	}
	if (position_groups_vec.empty()){
		return -1;
	}
	vector<PositionGroup>::iterator best_pos_group_it = position_groups_vec.begin();
    for (vector<PositionGroup>::iterator pos_group_it = position_groups_vec.begin(); pos_group_it != position_groups_vec.end(); ++pos_group_it){
    	if (*pos_group_it > *best_pos_group_it){
    		best_pos_group_it = pos_group_it;
    	}
    }
    return best_pos_group_it->group_position;
}

// Hard classify each family using its family responsibility
// Compute read_id_ and strand_id_, which are similar to ApproximateHardClassifierForReads
// read_id_ is obtained from the most responsible hypothesis of the family
// read_id_ stores the hard classification results for "families" (can be non-functional)
// read_id_[i] = -1 means the i-th family is classified as an outlier.
// read_id_[i] = 0 means the i-th family is classified as ref.
// read_id_[i] = 1 means the i-th family is classified as the variant allele 1, and so on.
void EnsembleEval::ApproximateHardClassifierForFamilies(){
	if(not allele_eval.total_theory.GetIsMolecularTag()){
		cerr<<"Warning: Skip EnsembleEval::ApproximateHardClassifierForFamilies() because use_mol_tag is off!"<< endl;
		return;
	}

	unsigned int num_families = allele_eval.total_theory.my_eval_families.size();
	int position0 = variant->position -1; // variant->position 1-base: vcflib/Variant.h
	read_id_.assign(num_families, -1); // every family starts with outlier
	strand_id_.assign(num_families, -1);
	dist_to_left_.assign(num_families, -1);
	dist_to_right_.assign(num_families, -1);
	alt_fam_indices_.resize(allele_identity_vector.size());

	for(unsigned int fam_idx = 0; fam_idx < allele_eval.total_theory.my_eval_families.size(); ++fam_idx){
		EvalFamily* my_fam = &(allele_eval.total_theory.my_eval_families[fam_idx]);
		// Hard classify the family to allele_assigned (-1 = outlier, 0 = ref, 1 = alt1, etc)
		int allele_assigned = my_fam->GetFuncFromValid() ? my_fam->MostResponsible() - 1 : -1; // non-func familiy = outlier
		read_id_[fam_idx] = allele_assigned;
		strand_id_[fam_idx] = my_fam->strand_key;

		// Classify as an alt
		if (allele_assigned > 0){
			alt_fam_indices_[allele_assigned - 1].push_back(fam_idx);
		}

		// Assign position
		if (allele_assigned > -1){
			dist_to_left_[fam_idx] = GetConsensusPosition(this, my_fam, true);
			dist_to_right_[fam_idx] = GetConsensusPosition(this, my_fam, false);
		}
	}

    if (DEBUG > 0){
    	unsigned int max_num_print_per_allele = 100;
    	vector <vector <unsigned int> > print_families_per_allele(allele_identity_vector.size() + 1);
    	for (unsigned int i_allele = 0; i_allele < print_families_per_allele.size(); ++i_allele){
    		print_families_per_allele[i_allele].reserve(max_num_print_per_allele + 1);
    	}
    	for(unsigned int fam_idx = 0; fam_idx < read_id_.size(); ++fam_idx){
    		if (read_id_[fam_idx] >= 0){
    			if (print_families_per_allele[read_id_[fam_idx]].size() <= max_num_print_per_allele){
    				print_families_per_allele[read_id_[fam_idx]].push_back(fam_idx);
    			}
    		}
    	}
    	cout << "+ Hard-classifying "<< num_families << " families on read stack for the variant ("<< PrintVariant(*variant) << "):" << endl;
    	for (unsigned int i_allele = 0; i_allele < print_families_per_allele.size(); ++i_allele){
    		cout << "  - Families that are most responsible for allele " << i_allele <<": ";
    		for (unsigned int i = 0; i < min((unsigned int) print_families_per_allele[i_allele].size(), max_num_print_per_allele); ++i){
    			unsigned int i_fam = print_families_per_allele[i_allele][i];
    			cout <<  allele_eval.total_theory.my_eval_families[i_fam].family_barcode << ", ";
    		}
    		if ((unsigned int) print_families_per_allele[i_allele].size() > max_num_print_per_allele)
    			cout << "......";
    		cout << endl;
    	}
    }
}

void EnsembleEval::DetectPossiblePolyploidyAlleles(const vector<float>& allele_freq_est, const ControlCallAndFilters &my_controls, const vector<VariantSpecificParams>& variant_specific_params){
	assert(allele_freq_est.size() == allele_identity_vector.size() + 1);
	float min_of_min_allele_freq = 1.0f;
	is_possible_polyploidy_allele.assign(allele_freq_est.size(), false);

	if (DEBUG){
		cout << "+ Detecting Possible Polyploidy Alleles (PPA):" << endl
			 << "  - estimated_allele_freq = " << PrintIteratorToString(allele_freq_est.begin(), allele_freq_est.end()) << endl;
	}

	for (int i_allele = (int) allele_freq_est.size() - 1; i_allele >= 0; --i_allele){
		int i_alt = i_allele - 1;
		float my_min_allele_freq = (i_allele == 0)? min_of_min_allele_freq : 0.0f;
		if (i_allele > 0){
			my_min_allele_freq = FreqThresholdByType(allele_identity_vector[i_alt], my_controls, variant_specific_params[i_alt]);
			min_of_min_allele_freq = min(my_min_allele_freq, min_of_min_allele_freq);
		}
		is_possible_polyploidy_allele[i_allele] = allele_freq_est[i_allele] > my_min_allele_freq;

		if (DEBUG){
			cout << "  - Allele "<< i_allele << (is_possible_polyploidy_allele[i_allele]? " may be" : " is not") << " a PPA: estimated_allele_freq[" << i_allele << "] = " << allele_freq_est[i_allele] << (is_possible_polyploidy_allele[i_allele]? " > " : " <= ") << "min-allele-freq = " << my_min_allele_freq << endl;
		}
	}
}

int EnsembleEval::DetectBestMultiAllelePair(){
	vector<float> allele_freq_estimation;
	return DetectBestMultiAllelePair(allele_freq_estimation);
}


int EnsembleEval::DetectBestMultiAllelePair(vector<float>& allele_freq_estimation){
    int best_alt_ndx = 0; // forced choice with ref
    //@TODO: just get the plane off the ground
    //@TODO: do the top pair by responsibility
    vector< pair<int,float> > best_allele_test;
    int num_hyp_no_null = (int) allele_identity_vector.size() + 1;
    best_allele_test.resize(num_hyp_no_null); // null can never be a winner in "best allele" sweepstakes

    for (unsigned int i_alt=0; i_alt<best_allele_test.size(); i_alt++){
        best_allele_test[i_alt].first = i_alt;
        best_allele_test[i_alt].second = 0.0f;
    }

    // using molecular tagging
    // This does the similar things as the no tagging case
    if(allele_eval.total_theory.GetIsMolecularTag()){
	    if(read_id_.empty()){
	        // similar to getting read_id and strand_id by ApproximateHardClassifierForReads()
	        ApproximateHardClassifierForFamilies();
	    }

	    // take family responsibility
	    for(unsigned int i_family = 0; i_family < read_id_.size(); ++i_family){
	        int my_alt = read_id_[i_family];
		    if(my_alt > -1){
		        best_allele_test[my_alt].second += allele_eval.total_theory.my_eval_families[i_family].family_responsibility[my_alt + 1];
		    } // otherwise count for nothing
	    }
    }
    // not using molecular tagging
    else{
	    if(read_id_.empty()){
    	    ApproximateHardClassifierForReads();
	    }
	    // take responsibility
	    for(unsigned int i_read = 0; i_read < read_id_.size(); ++i_read){
	        int my_alt = read_id_[i_read];
		    if (my_alt > -1){
		        best_allele_test[my_alt].second += allele_eval.total_theory.my_hypotheses[i_read].weighted_responsibility[my_alt + 1];
		    } // otherwise count for nothing
	    }
    }

    // calculate allele_freq_estimation
	float total_weight = 0.0f;
	allele_freq_estimation.assign(num_hyp_no_null, 0.0f);
	for(int i = 0; i < num_hyp_no_null; ++i){
		total_weight += best_allele_test[i].second;
	}
	for(int i = 0; i < num_hyp_no_null; ++i){
		allele_freq_estimation[i] = best_allele_test[i].second / total_weight;
	}

    if (DEBUG > 0){
    	cout << "+ Detecting the best allele pair for the variant ("<< PrintVariant(*variant) << "):" << endl;
    	cout << "  - allele_freq = " <<	PrintIteratorToString(allele_freq_estimation.begin(), allele_freq_estimation.end()) << " from semi-hard classification result." << endl;
    }

    // pick my pair of best alleles
    sort(best_allele_test.begin(), best_allele_test.end(), compare_best_response);
    // not-null choices
    diploid_choice.assign(2, 0);
    diploid_choice[0] = best_allele_test[0].first; // index of biggest weight
    diploid_choice[1] = best_allele_test[1].first; // index of second-biggest weight
    // problematic cases:
    // 2 alleles & ref, ref + 1 allele zero, want ref as the comparison
    // all ref, don't care about the second allele for genotype?
    // all zero implies what?
    //cout << best_allele_test.at(0).first << "\t" << best_allele_test.at(0).second << "\t" << best_allele_test.at(1).first << "\t" << best_allele_test.at(1).second << endl;
    if(diploid_choice[0]==0)
        best_alt_ndx = diploid_choice[1]-1;
    else
        best_alt_ndx = diploid_choice[0]-1;

    // sort as final step to avoid best_alt_ndx reflecting a worse allele
    sort(diploid_choice.begin(),diploid_choice.end()); // now in increasing allele order as needed

    if (DEBUG > 0){
    	cout << "  - The best allele pair = ("<< best_allele_test[0].first<<", "<< best_allele_test[1].first << ")"<< endl << endl;
    }

    return best_alt_ndx;
}

void EnsembleEval::MultiAlleleGenotype(float af_cutoff_rej, float af_cutoff_gt, vector<int> &genotype_component, float &gt_quality_score, float &reject_status_quality_score, int &quality_type){
    if (diploid_choice.empty()){
    	// detect best allele pair if not done yet.
    	DetectBestMultiAllelePair();
    }

    // set diploid in gq_pair
    allele_eval.cur_state.cur_posterior.gq_pair.freq_pair = diploid_choice;

    // Call by integral
    allele_eval.CallByIntegral(af_cutoff_rej, af_cutoff_gt, genotype_component, gt_quality_score, reject_status_quality_score, quality_type);
}

// Rules of setting the effective min fam size:
// If override requested by HS, then effective_min_family_size = the maximum one among all overrides.
// If not override by HS, effective_min_family_size = min_family_size +
void EnsembleEval::SetEffectiveMinFamilySize(const ExtendParameters& parameters, const vector<VariantSpecificParams>& variant_specific_params){
	assert(allele_identity_vector.size() > 0);
	allele_eval.total_theory.effective_min_family_size = (unsigned int) parameters.tag_trimmer_parameters.min_family_size;
	allele_eval.total_theory.effective_min_fam_per_strand_cov = (unsigned int) parameters.tag_trimmer_parameters.min_fam_per_strand_cov;

	// Override min_tag_fam_size
	bool min_tag_fam_size_override = false;
	for (unsigned int i_alt = 0; i_alt < variant_specific_params.size(); ++i_alt){
		if (variant_specific_params[i_alt].min_tag_fam_size_override){
			if (variant_specific_params[i_alt].min_tag_fam_size < 1){
				cerr << "WARNING: Fail to override the parameter min_tag_fam_size by " << variant_specific_params[i_alt].min_tag_fam_size << " < 1." << endl;
				continue;
			}
			if (not min_tag_fam_size_override){
				// This is the first override.
				allele_eval.total_theory.effective_min_family_size = (unsigned int) variant_specific_params[i_alt].min_tag_fam_size;
			}else{
				allele_eval.total_theory.effective_min_family_size = max(allele_eval.total_theory.effective_min_family_size, (unsigned int) variant_specific_params[i_alt].min_tag_fam_size);
			}
			min_tag_fam_size_override = true;
		}
	}
	if (min_tag_fam_size_override){
		if (DEBUG){
			cout << "+ Override min_fam_size to " << allele_eval.total_theory.effective_min_family_size << endl;
		}
	}
	// Override min_fam_per_strand_cov
	bool min_fam_per_strand_cov_override = false;
	for (unsigned int i_alt = 0; i_alt < variant_specific_params.size(); ++i_alt){
		if (variant_specific_params[i_alt].min_fam_per_strand_cov_override){
			if (variant_specific_params[i_alt].min_fam_per_strand_cov < 0){
				cerr << "WARNING: Fail to override the parameter min_fam_per_strand_cov by " << variant_specific_params[i_alt].min_fam_per_strand_cov << " < 0." << endl;
				continue;
			}
			if (not min_fam_per_strand_cov_override){
				// This is the first override.
				allele_eval.total_theory.effective_min_fam_per_strand_cov = (unsigned int) variant_specific_params[i_alt].min_fam_per_strand_cov;
			}else{
				allele_eval.total_theory.effective_min_fam_per_strand_cov = max(allele_eval.total_theory.effective_min_fam_per_strand_cov, (unsigned int) variant_specific_params[i_alt].min_fam_per_strand_cov);
			}
			min_fam_per_strand_cov_override = true;
		}
	}
	if (min_fam_per_strand_cov_override){
		if (DEBUG){
			cout << "+ Override min_fam_per_strand_cov to " << allele_eval.total_theory.effective_min_fam_per_strand_cov << endl;
		}
	}

	if (min_fam_per_strand_cov_override or min_tag_fam_size_override){
		return;
	}

    // Increase min_fam_size if I found an allele is HP-INDEL.
	if (parameters.tag_trimmer_parameters.indel_func_size_offset > 0){
		for (unsigned int i_alt = 0; i_alt < allele_identity_vector.size(); ++i_alt){
			if (allele_identity_vector[i_alt].status.isHPIndel){
				allele_eval.total_theory.effective_min_family_size += (unsigned int) parameters.tag_trimmer_parameters.indel_func_size_offset;
				allele_eval.total_theory.effective_min_fam_per_strand_cov += (unsigned int) parameters.tag_trimmer_parameters.indel_func_size_offset;
				if (DEBUG){
					cout << "+ Found allele "<< i_alt + 1 << " is HP-INDEL." << endl
						 << "  - Increase min_fam_size from " << parameters.tag_trimmer_parameters.min_family_size << " to " << allele_eval.total_theory.effective_min_family_size  << endl
						 << "  - Increase min_fam_per_strand_cov from " << parameters.tag_trimmer_parameters.min_fam_per_strand_cov << " to " << allele_eval.total_theory.effective_min_fam_per_strand_cov  << endl;
				}
				return;
			}
		}
	}
}



void EnsembleEval::SetAndPropagateParameters(ExtendParameters* parameters, bool use_molecular_tag, const vector<VariantSpecificParams>& variant_specific_params){
    allele_eval.my_params = parameters->my_eval_control;
	// Set debug level
	DEBUG = parameters->program_flow.DEBUG;
    allele_eval.DEBUG = parameters->program_flow.DEBUG;
    allele_eval.cur_state.SetAndPropagateDebug(parameters->program_flow.DEBUG);
    allele_eval.total_theory.DEBUG = parameters->program_flow.DEBUG;
    // Set target min-allele-freq for fast scan
    allele_eval.cur_state.cur_posterior.ref_vs_all.SetTargetMinAlleleFreq(*parameters, variant_specific_params);
    allele_eval.cur_state.cur_posterior.gq_pair.SetTargetMinAlleleFreq(*parameters, variant_specific_params);
    // Set molecular tag related parameters
    allele_eval.total_theory.SetIsMolecularTag(use_molecular_tag);
    SetEffectiveMinFamilySize(*parameters, variant_specific_params);
    // only rich_json_diagnostic needs full data
    allele_eval.total_theory.preserve_full_data = parameters->program_flow.rich_json_diagnostic;
}

void EnsembleEval::FlowDisruptivenessInReadLevel(const InputStructures &global_context)
{
	for (unsigned int i_read = 0; i_read < read_stack.size(); ++i_read){
		allele_eval.total_theory.my_hypotheses[i_read].FillInFlowDisruptivenessMatrix(global_context.flow_order_vector.at(read_stack[i_read]->flow_order_index), *(read_stack[i_read]));
	}
}


// The rule for determining FD in the read stack level is the following.
// Suppose I want to determine the FD between i_hyp and j_hyp. Denote the set reads that support i_hyp by Reads(i_hyp).
// (Case 1): Both hypotheses have reads support, i.e., Reads(i_hyp) and Reads(j_hyp) are both non-empty
// (Case 1.a): I find reads in Reads(i_hyp) and reads in Reads(j_hyp) indicate that the pair (i_hyp, j_hyp) is FD. Then the pair (i_hyp, j_hyp) is FD.
// (Case 1.b): I find reads in Reads(i_hyp) shows (i_hyp, j_hyp) is FD, but no (or too few) reads in Reads(j_hyp) says (i_hyp, j_hyp) is FD. Then (i_hyp, j_hyp) is not FD, (i.e. FD bias)
// (Case 1.c): Reads(i_hyp) and Reads(j_hyp) have no read shows (i_hyp, j_hyp) is FD. Then (i_hyp, j_hyp) is not FD.
// (Case 2): i_hyp has reads support but j_hyp doesn't, i.e., Reads(i_hyp) is non-empty but Reads(j_hyp) is empty
// (Case 2.a): Reads(i_hyp) has reads indicate (i_hyp, j_hyp) is FD. Then (i_hyp, j_hyp) is FD.
// (Case 2.b): Reads(i_hyp) has no read indicate (i_hyp, j_hyp) is FD. Then (i_hyp, j_hyp) is not FD.
// (Case 3): Both i_hyp and j_hyp have no read support. Then the FD between the two hypotheses is indefinite.
// Usually, if any of i_hyp and j_hyp is obtained from the best allele pair, then case 3 should not happen.
// @TODO: Use the transitivity of HP-INDEL to determine the FD of the no coverage alleles.
void EnsembleEval::FlowDisruptivenessInReadStackLevel(float min_ratio_for_fd)
{
	unsigned int num_hyp_not_null = allele_identity_vector.size() + 1; // reference + alternatives
	global_flow_disruptive_matrix.assign(num_hyp_not_null, vector<int>(num_hyp_not_null, -1)); // Starts with indefinite
	float min_posterior_coverage = 0.9f;
	// Aggregating too many super low responsibility may exceed min_posterior_coverage. Use min_resp_cutoff to guard against the problem.
	float min_resp_cutoff = 0.1f;

	// postrior_coverage[i_hyp] = posterior coverage for i_hyp.
	vector<float> postrior_coverage(num_hyp_not_null, 0.0f);
	// posterior_fd_counts[i_hyp][j_hyp][i_type] = posterior counts for Reads(i_hyp) that indicates the FD-type of (i_hyp, j_hyp) is i_type.
	vector< vector< vector<float> > > posterior_fd_type_counts(num_hyp_not_null, vector< vector<float> >(num_hyp_not_null, {0.0f, 0.0f, 0.0f}));

    if (DEBUG > 0){
    	cout << "- Determine Flow-Disruptiveness (FD) of the variant:" << endl
    		 << "  - FD codes: 0 = HP-INDEL, 1 = (not HP-INDEL and not flow-disruptive), 2 = flow-disruptive" << endl
    		 << "  - min_ratio_for_fd = "<< min_ratio_for_fd << endl;
    }

	for (vector<CrossHypotheses>::iterator read_it = allele_eval.total_theory.my_hypotheses.begin(); read_it != allele_eval.total_theory.my_hypotheses.end(); ++read_it){
		if (not read_it->success){
			continue;
		}
		for (unsigned int i_hyp = 0; i_hyp < num_hyp_not_null; ++i_hyp){
			float resp = read_it->responsibility[i_hyp + 1];
			if (resp < min_resp_cutoff)
				continue;
			postrior_coverage[i_hyp] += resp;
			for (unsigned int j_hyp = 0; j_hyp < num_hyp_not_null; ++j_hyp){
				// Note that local_flow_disruptiveness_matrix contains the outlier hypothesis
				int fd_type = read_it->local_flow_disruptiveness_matrix[i_hyp + 1][j_hyp + 1];
				if (fd_type >= 0){
					posterior_fd_type_counts[i_hyp][j_hyp][fd_type] += resp;
				}
			}
		}
	}

    for (unsigned int i_hyp = 0; i_hyp < num_hyp_not_null; ++i_hyp){
    	global_flow_disruptive_matrix[i_hyp][i_hyp] = 0;
        for (unsigned int j_hyp = i_hyp + 1; j_hyp < num_hyp_not_null; ++j_hyp){
        	// Is it Case 1?
        	if (postrior_coverage[i_hyp] >= min_posterior_coverage and postrior_coverage[j_hyp] >= min_posterior_coverage){
        		// Check in the order of FD->(not FD and not HPINDEL)->(HPINDEL)
        		for (int type = 2; type >=0; --type){
        			if ( posterior_fd_type_counts[i_hyp][j_hyp][type] / postrior_coverage[i_hyp] >= min_ratio_for_fd
            				and posterior_fd_type_counts[j_hyp][i_hyp][type] / postrior_coverage[j_hyp] >= min_ratio_for_fd){
        				 global_flow_disruptive_matrix[i_hyp][j_hyp] = type;
        				 break;
        			}
        		}
        	}
        	// Is it Case 3?
        	else if (postrior_coverage[i_hyp] < min_posterior_coverage and postrior_coverage[j_hyp] < min_posterior_coverage){
        		// global_flow_disruptive_matrix[j_hyp][i_hyp] = -1; // Indefinite, and it was set by default.
        	}
        	// Case 2: one hyp has coverage but the other one doesn't
        	else{
        		int hyp_w_cov = i_hyp;
        		int hyp_wo_cov = j_hyp;
        		if (postrior_coverage[j_hyp] > postrior_coverage[i_hyp]){
        			hyp_w_cov = j_hyp;
        			hyp_wo_cov = i_hyp;
        		}
        		// Check in the order of (HPINDEL)->(not FD and not HPINDEL)->FD
        		// I check HPINDEL first because I don't believe in those HP-INDEL reads that support the hyp w/ coverage.
        		for (int type = 0; type < 3; ++type){
        		    if ( posterior_fd_type_counts[hyp_w_cov][hyp_wo_cov][type] / postrior_coverage[hyp_w_cov] >= min_ratio_for_fd){
        		    	global_flow_disruptive_matrix[i_hyp][j_hyp] = type;
        			    break;
        			}
        		}
        	}
        	global_flow_disruptive_matrix[j_hyp][i_hyp] = global_flow_disruptive_matrix[i_hyp][j_hyp];
        }
    }

    // Propagate the FD to downstream
    variant->info["FDVR"].clear();
    for (unsigned int i_alt = 0; i_alt < allele_identity_vector.size(); ++i_alt){
    	allele_identity_vector[i_alt].fd_level_vs_ref = global_flow_disruptive_matrix[0][i_alt + 1];
    	/*
    	// I make an exception according to the reference context!
    	if (allele_identity_vector[i_alt].status.isHPIndel and global_flow_disruptive_matrix[0][i_alt + 1] != 0){
    		if (DEBUG > 0){
    			cout << "  - Overwrite global_flow_disruptive_matrix[0][" << (i_alt + 1) << "] from "
					 << global_flow_disruptive_matrix[0][i_alt + 1] << " to 0 using the reference context." << endl;
    		}
    		allele_identity_vector[i_alt].fd_level_vs_ref = 0;
    		global_flow_disruptive_matrix[0][i_alt + 1] = 0;
    	}
    	*/
    	string fdvr = "-1";
    	if (global_flow_disruptive_matrix[0][i_alt + 1] == 0){
    		fdvr = "0";
    	}else if (global_flow_disruptive_matrix[0][i_alt + 1] == 1){
    		fdvr = "5";
    	}else if (global_flow_disruptive_matrix[0][i_alt + 1] == 2){
    		fdvr = "10";
    	}
    	variant->info["FDVR"].push_back(fdvr);
    }

    if (DEBUG > 0){
    	cout << "- Determine Flow-Disruptiveness (FD) of the variant:" << endl
    		 << "  - FD codes: 0 = HP-INDEL, 1 = (not HP-INDEL and not flow-disruptive), 2 = flow-disruptive" << endl
    		 << "  - min_ratio_for_fd = "<< min_ratio_for_fd << endl;
    	for (unsigned int i_hyp = 0; i_hyp < posterior_fd_type_counts.size(); ++i_hyp){
    		cout << "  - For the reads supporting Allele "<< i_hyp << ": " << "posterior coverage = " << postrior_coverage[i_hyp] << endl;
    		for (unsigned int j_hyp = 0; j_hyp < posterior_fd_type_counts[i_hyp].size(); ++j_hyp){
    			if (j_hyp == i_hyp){
    				continue; // trivial case
    			}
    			cout << "    - vs. Allele " << j_hyp << ": posterior coverage indexed by FD codes = [";
        		for (unsigned int fd_idx = 0; fd_idx < posterior_fd_type_counts[i_hyp][j_hyp].size(); ++fd_idx){
        			cout << posterior_fd_type_counts[i_hyp][j_hyp][fd_idx] << (fd_idx == posterior_fd_type_counts[i_hyp][j_hyp].size() - 1 ? "": ", ");
        		}
    			cout << "]" << endl;
    		}
    	}
    	cout << "  - FD matrix ('-': trivial, '/': indefinite)" << endl;
    	for (unsigned int i_hyp = 0; i_hyp < global_flow_disruptive_matrix.size(); ++i_hyp){
    		cout << (i_hyp == 0 ? "      = [[" : "         [");
        	for (unsigned int j_hyp = 0; j_hyp < global_flow_disruptive_matrix[i_hyp].size(); ++j_hyp){
        		if (i_hyp == j_hyp){
        			cout << "-";
        		}
        		else if (global_flow_disruptive_matrix[i_hyp][j_hyp] > -1){
            		cout << global_flow_disruptive_matrix[i_hyp][j_hyp];
        		}
        		else{
            		cout << "/";
        		}
        		cout << (j_hyp == global_flow_disruptive_matrix[i_hyp].size() - 1? "]": ", ");
        	}
    		cout << (i_hyp == global_flow_disruptive_matrix.size() -1 ? "]": ",") << endl;
    	}
    	cout<<endl;
    }
}

void EnsembleEval::ServeAfCutoff(const ControlCallAndFilters &my_controls, const vector<VariantSpecificParams>& variant_specific_params,
		float& af_cutoff_rej, float& af_cutoff_gt)
{
	af_cutoff_rej = 1.0f;
	// The old fashion scheme: choose the minimum one among all alleles
	if (not my_controls.use_fd_param ){
		// (TS-16940): min_allele_freq override has the top priority.
		bool has_override = false;
		for (unsigned int allele_idx = 0; allele_idx < allele_identity_vector.size(); ++allele_idx){
			if (variant_specific_params[allele_idx].min_allele_freq_override){
				has_override = true;
				af_cutoff_rej = min(af_cutoff_rej, variant_specific_params[allele_idx].min_allele_freq);
			}
		}
		if (has_override){
			af_cutoff_gt = af_cutoff_rej;
			return;
		}

		for (unsigned int allele_idx = 0; allele_idx < allele_identity_vector.size(); ++allele_idx){
			af_cutoff_rej = min(af_cutoff_rej, FreqThresholdByType(allele_identity_vector[allele_idx], my_controls, variant_specific_params[allele_idx]));
		}
		// The allele-freq-cutoff for calling GT is the same as the allele-freq-cutoff for Rej.
		af_cutoff_gt = af_cutoff_rej;
		return;
	}

	// The new approach:
	// af_cutoff_rej is the smallest min-allele-freq from the alleles in "diploid_choice" (aka best allele pair).
	// The reason of doing this is because I am not going to call any allele other than diploid_choice.
	// Therefore, it makes no sense that I use the min-allele-freq from the allele other than diploid_choice.
	for (vector<int>::iterator choice_it = diploid_choice.begin(); choice_it != diploid_choice.end(); ++choice_it){
		if (*choice_it == 0){
			// The reference allele is in best allele pair.
			continue;
		}
		af_cutoff_rej = min(af_cutoff_rej, FreqThresholdByType(allele_identity_vector[*choice_it - 1], my_controls, variant_specific_params[*choice_it - 1]));
	}

	// Get af_cutoff_gt
	if (diploid_choice[0] == 0 or diploid_choice[1] == 0){
		af_cutoff_gt = af_cutoff_rej;
		return;
	}

	int fd_diploid_choice = global_flow_disruptive_matrix[diploid_choice[0]][diploid_choice[1]];

	if (fd_diploid_choice == 2){
		af_cutoff_gt = my_controls.filter_fd_10.min_allele_freq;
	}
	else if (fd_diploid_choice == 1){
		af_cutoff_gt = my_controls.filter_fd_5.min_allele_freq;
	}
	else if (fd_diploid_choice == 0){
		af_cutoff_gt = my_controls.filter_fd_0.min_allele_freq;
	}
	else{
		// indefinite case
		af_cutoff_gt = my_controls.filter_fd_0.min_allele_freq;
	}
}

void EnsembleEval::VariantFamilySizeHistogram(){
	const int m = 2;
	// Basically, the bins will be (< (mode - m + 1)), .... mode - 1, mode, mode + 1, ..., (>(mode + m - 1))
	const int hist_x_num = 2 * m + 1;
	// Make sure ApproximateHardClassification is done.
	assert(alt_fam_indices_.size() == allele_identity_vector.size());
	for (unsigned int alt_idx = 0; alt_idx < alt_fam_indices_.size(); ++ alt_idx){
		if (alt_fam_indices_[alt_idx].empty()){
			// FAO = 0 means no histogram.
			variant->info["VFSH"].push_back(".");
			continue;
		}
		// Calculate the histogram
		map<int, unsigned int> fam_size_hist;
		for (unsigned int idx = 0; idx < alt_fam_indices_[alt_idx].size(); ++ idx){
			int fam_size = allele_eval.total_theory.my_eval_families[alt_fam_indices_[alt_idx][idx]].GetValidFamSize();
			++fam_size_hist[fam_size];
		}

		// Get the mode of the histogram
		unsigned int mode_counts = fam_size_hist.begin()->second;
		int mode_idx = 0;
		int hist_len = 0;
		for (map<int, unsigned int>::iterator it = fam_size_hist.begin(); it != fam_size_hist.end(); ++it, ++hist_len){
			if (it->second > mode_counts){
				mode_idx = hist_len;
				mode_counts = it->second;
			}
		}

		// See if I need to bin the begin, end of the histogram, respectively
		int x_remaining = hist_x_num - 1; // minus the index mode
		int x_start_cutoff = mode_idx;
		int x_end_cutoff = mode_idx;
		while (x_remaining > 0 and (not (x_start_cutoff == 0 and x_end_cutoff == hist_len - 1))){
			if (x_start_cutoff > 0){
				--x_remaining;
				--x_start_cutoff;
			}
			if (x_end_cutoff < hist_len - 1){
				--x_remaining;
				++x_end_cutoff;
			}
		}
		string hist_text = "(";
		map<int, unsigned int>::iterator it_start = fam_size_hist.begin();
		if (x_start_cutoff > 0){
			// bin the begin of the histogram
			int fam_counts = 0;
			for (int idx = 0; idx <= x_start_cutoff; ++idx, ++it_start){
				fam_counts += it_start->second;
			}
			--it_start;
			hist_text += string("(<") + to_string(it_start->first + 1) + string(",") + to_string(fam_counts) +"),";
			++it_start;
		}
		string last_hist_text = "";
		map<int, unsigned int>::iterator it_end = fam_size_hist.end();
		if (x_end_cutoff < hist_len - 1){
			// bin the end of the histogram
			int fam_counts = 0;
			--it_end;
			for (int idx = hist_len - 1; idx >= x_end_cutoff; --idx, --it_end){
				fam_counts += it_end->second;
			}
			++it_end;
			last_hist_text = string("(>") + to_string(it_end->first - 1) + string(",") + to_string(fam_counts) +"))";
		}

		// No need to bin the rest of the family size
		for (map<int, unsigned int>::iterator it = it_start; it != it_end; ++it){
			hist_text += string("(") + to_string(it->first) + string(",") + to_string(it->second) +"),";
		}

		if (last_hist_text.empty()){
			hist_text.back() = ')';
		}
		else{
			hist_text += last_hist_text;
		}
		variant->info["VFSH"].push_back(hist_text);
	}
}


// Check tag similarity if the molecular alternative coverage <= max_alt_cov and compute TGSM for each alternative allele.
// (Step 1): Determine pairwise family similarity
// I claim family A and family B are similar if
// a) Prefix A is partial similar to prefix B "and" suffix A is partial similar to suffix B (partial similar = allow 1 SNP + small HP-INDEL)
// or b) Prefix A is synchronized with prefix B or suffix A is synchronized with suffix B (synchronized = allow small HP-INDEL)
// The criterion a) is used to deal with PCR error occurs on tag. The criterion b) is used to against family cloning during sample prep.
// Note that pairwise family similarity has NO transitivity.
// (Step 2): Calculate TGSM
// I construct a graph by connecting the families using the relation defined by pairwise family similarity.
// TGSM = (number of families in the graph) - (number of isolated sub-graphs)
void EnsembleEval::CalculateTagSimilarity(const MolecularTagManager& mol_tag_manager, int max_alt_cov, int sample_idx)
{
	tag_similar_counts_.assign(allele_identity_vector.size(), 0);  // This is TGSM in the vcf record.
	unsigned int prefix_tag_len = mol_tag_manager.GetPrefixTagStruct(sample_idx).size();
	unsigned int suffix_tag_len = mol_tag_manager.GetSuffixTagStruct(sample_idx).size();

	for (unsigned int allele_idx = 0; allele_idx < alt_fam_indices_.size(); ++allele_idx){
		unsigned int allele_fam_cov = alt_fam_indices_[allele_idx].size();
		if ((int) allele_fam_cov > max_alt_cov or allele_fam_cov == 0){
			continue;
		}
		// Step 1: Determine pairwise tag similarity
		bool non_trivial_similar_tag_found = false; // Do I find any pairwise similar families?
		vector<vector<bool> > pairwise_tag_similar_matrix;
		pairwise_tag_similar_matrix.assign(allele_fam_cov, vector<bool>(allele_fam_cov, false));
		for (unsigned int fam_idx_1 = 0; fam_idx_1 < allele_fam_cov; ++fam_idx_1){
			pairwise_tag_similar_matrix[fam_idx_1][fam_idx_1] = true;  // pairwise tag similarity is a reflexive relation
			const string& tag_1 = allele_eval.total_theory.my_eval_families[alt_fam_indices_[allele_idx][fam_idx_1]].family_barcode;
			if (prefix_tag_len + suffix_tag_len != tag_1.size()){
				cerr << "ERROR: The molecular tag "<< tag_1 << " doesn't match the tag structure "<< mol_tag_manager.GetPrefixTagStruct(sample_idx) <<" + "<<mol_tag_manager.GetSuffixTagStruct(sample_idx) << endl;
				exit(-1);
			}
			string tag_1_prefix = tag_1.substr(0, prefix_tag_len);
			string tag_1_suffix = tag_1.substr(prefix_tag_len);
			for (unsigned int fam_idx_2 = fam_idx_1 + 1; fam_idx_2 < allele_fam_cov; ++fam_idx_2){
				const string& tag_2 = allele_eval.total_theory.my_eval_families[alt_fam_indices_[allele_idx][fam_idx_2]].family_barcode;
				string tag_2_prefix = tag_2.substr(0, prefix_tag_len);
				string tag_2_suffix = tag_2.substr(prefix_tag_len);
				// Determine Criterion b) in Step 1
				bool is_similar = mol_tag_manager.IsFlowSynchronizedTags(tag_1_prefix, tag_2_prefix, true) or mol_tag_manager.IsFlowSynchronizedTags(tag_1_suffix, tag_2_suffix, false);
				if (not is_similar){
					// Determine Criterion a) in Step 1
					is_similar += mol_tag_manager.IsPartialSimilarTags(tag_1_prefix, tag_2_prefix, true) and mol_tag_manager.IsPartialSimilarTags(tag_1_suffix, tag_2_suffix, false);
				}
				pairwise_tag_similar_matrix[fam_idx_1][fam_idx_2] = is_similar;
				pairwise_tag_similar_matrix[fam_idx_2][fam_idx_1] = is_similar; // pairwise tag similarity is a symmetric relation
				non_trivial_similar_tag_found += is_similar;
			}
		}
		// Step 2: Determine the number of isolated sub-graphs
		list<list<int> > subgraph_to_family;
		int num_isolated_subgraphs = 0;
		if (non_trivial_similar_tag_found){
			FindNodesInIsoSubGraph(pairwise_tag_similar_matrix, subgraph_to_family, true);
			num_isolated_subgraphs = (int) subgraph_to_family.size();
		}else{
			num_isolated_subgraphs = (int) allele_fam_cov;
		}
		tag_similar_counts_[allele_idx] = (int) allele_fam_cov - num_isolated_subgraphs;

		if (DEBUG > 0){
			cout <<"+ Checking tag similarity of the "<< allele_fam_cov <<" families that support allele " << allele_idx + 1 << endl;
			for (unsigned int fam_idx = 0; fam_idx < allele_fam_cov; ++fam_idx){
				const string& fam_barcode = allele_eval.total_theory.my_eval_families[alt_fam_indices_[allele_idx][fam_idx]].family_barcode;
				cout << "  - Family #" << alt_fam_indices_[allele_idx][fam_idx] << " \"" << fam_barcode.substr(0, prefix_tag_len) << "\" + \"" << fam_barcode.substr(prefix_tag_len) << "\" is similar to ";
				bool is_similar_to_else = false;
				for (unsigned int fam_idx_2 = 0; fam_idx_2 < allele_fam_cov; ++fam_idx_2){
					if (pairwise_tag_similar_matrix[fam_idx][fam_idx_2] and fam_idx != fam_idx_2){
						const string& sim_fam_barcode = allele_eval.total_theory.my_eval_families[alt_fam_indices_[allele_idx][fam_idx_2]].family_barcode;
						cout << "Family #" << alt_fam_indices_[allele_idx][fam_idx_2] << " \"" << sim_fam_barcode.substr(0, prefix_tag_len) << "\" + \"" << sim_fam_barcode.substr(prefix_tag_len) << "\", ";
						is_similar_to_else = true;
					}
				}
				cout << (is_similar_to_else? "": "none else.") << endl;
			}
			if (tag_similar_counts_[allele_idx] > 0){
				cout << "+ Found "<< num_isolated_subgraphs << " isolated subgraph(s) of tag-similar families for allele "<< allele_idx + 1 << ": " << endl;
				int group_idx = 0;
				for (list<list<int> >::iterator subgraph_it = subgraph_to_family.begin(); subgraph_it != subgraph_to_family.end(); ++subgraph_it, ++group_idx){
					cout << "  - Subgraph "<< group_idx << " consists of " << subgraph_it->size() <<" families: ";
					for (list<int>::iterator fam_idx_it = subgraph_it->begin(); fam_idx_it != subgraph_it->end(); ++fam_idx_it){
						cout << "#"<< alt_fam_indices_[allele_idx][*fam_idx_it] << ", ";
					}
					cout << endl;
				}
			}
			else{
				cout << "+ No tag-similar family found for allele " << allele_idx + 1 << endl;
			}
			cout << "+ Allele "<< allele_idx + 1 << " has TGSM = " << allele_fam_cov << " - " << num_isolated_subgraphs << " = " << tag_similar_counts_[allele_idx] << endl;
		}
	}
	if (DEBUG > 0) {
		cout << endl;
	}
}

// Compare the lists by comparing their first elements
bool CompareIntList(const list<int>& list_1, const list<int>& list_2){
    if (list_1.empty()){
        return false;
    }
    else if (list_2.empty()){
        return true;
    }
    return (*(list_1.begin()) < *(list_2.begin()));
}

// Consider a non-directed graph of N nodes where the connectivity between node i and j is determined by connectivity_matrix[i][j], i.e., true if connected.
// The function groups the nodes that are transitively connected.
// (Note): connectivity_matrix "must be" symmetric, i.e., connectivity_matrix[i][j] = connectivity_matrix[j][i]
// (Example): N = 5, connectivity_matrix[0][1] = 1, connectivity_matrix[1][2] = 1, connectivity_matrix[3][4] = 1, and all other upper triangle elements are 0.
//            Then there are two isolated subgraphs 0-1-2 and 3-4 => subgraph_to_nodes = {{0, 1, 2}, {3, 4}}.
void FindNodesInIsoSubGraph(const vector<vector<bool> >& connectivity_matrix, list<list<int> >& subgraph_to_nodes, bool sort_by_index){
    int num_of_nodes = (int) connectivity_matrix.size();
    int num_nodes_in_subgraphs = 0; // just for safety check
    list<list<int> > dummy_list;
    const list<list<int> >::iterator null_it = dummy_list.begin();
    vector<list<list<int> >::iterator> node_to_subgraph(num_of_nodes, null_it); // a look-up table to map from nodes to sub-graph
    subgraph_to_nodes.resize(0);

    for (int node_idx_1 = 0; node_idx_1 < num_of_nodes; ++node_idx_1){
        if (node_to_subgraph[node_idx_1] == null_it){
            // This is the first time fam_idx_1 shows up. Create a new subgraph for it.
            subgraph_to_nodes.push_back(list<int>(1, node_idx_1));
            node_to_subgraph[node_idx_1] = --(subgraph_to_nodes.end());
        }
        list<list<int> >::iterator& master_subgraph = node_to_subgraph[node_idx_1];
        // Now let the nodes connected to node_idx_1 join the subgraph where node_idx_1 belongs to.
        // No need to process node_idx_2 if node_idx_2 < node_idx_1, since it has been carried out previously.
        for (int node_idx_2 = node_idx_1 + 1; node_idx_2 < num_of_nodes; ++node_idx_2){
        	// Safety check: connectivity_matrix must be symmetric
        	assert(connectivity_matrix[node_idx_1][node_idx_2] == connectivity_matrix[node_idx_2][node_idx_1]);
            if (not connectivity_matrix[node_idx_1][node_idx_2]){
                continue;
            }
            if (node_to_subgraph[node_idx_2] == null_it){
                node_to_subgraph[node_idx_2] = master_subgraph;
                master_subgraph->push_back(node_idx_2);
            }else if (node_to_subgraph[node_idx_2] != master_subgraph){
                // store the last node in master_subgraph before merging
                list<int>::iterator last_node_in_master_before_merge = --(master_subgraph->end());
                // Let all nodes connected to node_idx_2 merge into master_subgraph
                master_subgraph->splice(master_subgraph->end(), *(node_to_subgraph[node_idx_2]));
                // Destroy the original subgraph where node_idx_2 belongs to.
                subgraph_to_nodes.erase(node_to_subgraph[node_idx_2]);
                // Update node_to_subgraph for the nodes that are newly merged into master_subgraph
                for (list<int>::iterator node_it = ++last_node_in_master_before_merge; node_it != master_subgraph->end(); ++node_it){
                    node_to_subgraph[*node_it] = master_subgraph;
                }
            }
        }
    }

    // Sort the nodes and sub-graphs
    for (list<list<int> >::iterator graph_it = subgraph_to_nodes.begin(); graph_it != subgraph_to_nodes.end(); ++graph_it){
        if (sort_by_index){
        	graph_it->sort();
        }
        num_nodes_in_subgraphs += (int) graph_it->size();
    }
    if (sort_by_index){
    	subgraph_to_nodes.sort(CompareIntList);
    }
    // Simple safety check.
    assert(num_nodes_in_subgraphs == num_of_nodes
           and (int) subgraph_to_nodes.size() <= num_of_nodes
           and (subgraph_to_nodes.size() > 0 or num_of_nodes == 0));
}

string PrintVariant(const vcf::Variant& variant){
	string spacing = "\t";
	string vcf_variant = variant.sequenceName + spacing
			           + convertToString(variant.position) + spacing
					   + variant.id + spacing
					   + variant.ref + spacing;
	for(unsigned int i_alt = 0; i_alt < variant.alt.size(); ++i_alt){
		vcf_variant += variant.alt[i_alt];
		if(i_alt != variant.alt.size() - 1){
			vcf_variant += ",";
		}
	}
	return vcf_variant;
}

class CompareAllelePositions{
public:
	CompareAllelePositions(vector<AlleleIdentity> const* allele_identity_vector = NULL){
		allele_identity_vector_ = allele_identity_vector;};
	~CompareAllelePositions(){};
	bool operator()(int lhs_allele_idx, int rhs_allele_idx) const {
		return allele_identity_vector_->at(lhs_allele_idx).start_variant_window < allele_identity_vector_->at(rhs_allele_idx).start_variant_window;
	};
private:
	vector<AlleleIdentity> const* allele_identity_vector_ = NULL;
};

class CompareAlleleGroups{
public:
	CompareAlleleGroups(vector<AlleleIdentity> const* allele_identity_vector = NULL){
			allele_identity_vector_ = allele_identity_vector;};
	~CompareAlleleGroups(){};
	bool operator()(const list<int>& lhs_group, const list<int>& rhs_group) const {
		return allele_identity_vector_->at(*(lhs_group.begin())).start_variant_window < allele_identity_vector_->at(*(rhs_group.begin())).start_variant_window;
	};
private:
	vector<AlleleIdentity> const*  allele_identity_vector_;
};


// Merge the groups that start at the same position, since it is required that the position of a vcf record should be unique.
void JoinGroupsStartAtTheSamePosition(const vector<AlleleIdentity>& allele_identity_vector, list<list<int> >& allele_groups, bool is_groups_sorted){
	// Sort alleles and groups if hasn't been done.
	if (not is_groups_sorted){
		// Used to sort the groups
		CompareAlleleGroups compare_groups(&allele_identity_vector);
		// Used to sort the alleles in a group
		CompareAllelePositions compare_positions(&allele_identity_vector);
		// Must sort the alleles in each group first (because compare_groups will use the pos of the first allele for sorting).
		for (list<list<int> >::iterator group_it = allele_groups.begin(); group_it != allele_groups.end(); ++group_it){
			group_it->sort(compare_positions);
		}
		// Then sort the groups of alleles.
		allele_groups.sort(compare_groups);
	}

	list<list<int> >::iterator last_group_it = allele_groups.begin();
	list<list<int> >::iterator group_it = allele_groups.begin();
	++group_it;
	while (group_it != allele_groups.end()){
		// Assume the groups are sorted by the minimum start positions among the alleles in the group.
		// i.e., the start position of the group is the start variant window of the first allele in the group.
		if (allele_identity_vector[*(group_it->begin())].start_variant_window == allele_identity_vector[*(last_group_it->begin())].start_variant_window){
			// last_group_it and group_it has the same start position.
			// Merge *group_it into *last_group_it.
			last_group_it->splice(last_group_it->end(), *group_it);
			// Remove *group_it since it is now empty after merging.
			group_it = allele_groups.erase(group_it);
		}
		else{
			// March to the next group.
			++last_group_it;
			++group_it;
		}
	}
}

// Finalize the splitting:
// 1) Sort the alleles and groups
// 2) Further split the groups if the group size is too large.
// 2.a) Stage 1: break the connectivity between long Fake HS and others; break the connectivity between Fake HS and HP-INDEL
// 2.b) Stage 2: break the connectivity between all Fake HS and others
// 2.c) Stage 3: break all connectivity
// 2.d) Stage 4: repeat stage 3 again (in case the alleles from the big group move to a small group and let the small group too big.).
void FinalizeSplitting(vector<vector<bool> >& allele_connectivity_matrix, const vector<AlleleIdentity>& allele_identity_vector, int max_group_size_allowed, list<list<int> > &allele_groups, unsigned int max_iteration = 4){
	int debug = allele_identity_vector[0].DEBUG;
	// Used to sort the groups
	CompareAlleleGroups compare_groups(&allele_identity_vector);
	// Used to sort the alleles in a group
	CompareAllelePositions compare_positions(&allele_identity_vector);
	unsigned int num_iter = 0;
	bool keep_itertate = true;
	bool final_splitting_applied = false;
	// Must sort the alleles in each group first (because compare_groups will use the pos of the first allele for sorting).
	for (list<list<int> >::iterator group_it = allele_groups.begin(); group_it != allele_groups.end(); ++group_it){
		group_it->sort(compare_positions);
	}
	// Then sort the groups of alleles.
	allele_groups.sort(compare_groups);
	// Merge the groups that start at the same position
	JoinGroupsStartAtTheSamePosition(allele_identity_vector, allele_groups, true);

	while (keep_itertate and num_iter < max_iteration){
		if (debug > 0){
			cout <<"+ Final variant splitting. Round "<< (num_iter + 1) << ":" << endl;
		}
		bool max_group_size_achieved = false;
		bool connectivity_changed = false;
		for (list<list<int> >::iterator group_it = allele_groups.begin(); group_it != allele_groups.end(); ++group_it){
			// See if I have a group that contains too many alleles.
			if ((int) group_it->size() > max_group_size_allowed){
				max_group_size_achieved = true;
				if (debug > 0){
					cout << "  - The group "<< PrintIteratorToString(group_it->begin(), group_it->end(), "{", "}", ", ", "alt")
						 << " contains "<< group_it->size() << " alleles (> max_alt_num = " << max_group_size_allowed <<"): splitting needed." << endl;
				}
				// I let the alleles in this group be isolated unless the other allele has the same start position.
				for (list<int>::iterator idx_it_1 = group_it->begin(); idx_it_1 != group_it->end(); ++idx_it_1){
					// At the first two rounds (num_iter < 2), I only want to break Fake HS.
					if (num_iter < 2 and allele_identity_vector[*idx_it_1].status.isFakeHsAllele){
						for (list<int>::iterator idx_it_2 = group_it->begin(); idx_it_2 != group_it->end(); ++idx_it_2){
							if (*idx_it_1 == *idx_it_2 or (not allele_connectivity_matrix[*idx_it_1][*idx_it_2])){
								continue;
							}
							if ( (num_iter == 0 and allele_identity_vector[*idx_it_2].status.isClearlyNonFD and allele_identity_vector[*idx_it_1].end_variant_window - allele_identity_vector[*idx_it_1].start_variant_window >= 8)
									or (num_iter == 1)){
								allele_connectivity_matrix[*idx_it_1][*idx_it_2] = false;
								allele_connectivity_matrix[*idx_it_2][*idx_it_1] = false;
								connectivity_changed = true;
								if (debug > 0){
									cout << "    - Breaking the connectivity between alt"<<  *idx_it_1 << ": "<<  allele_identity_vector[*idx_it_1].altAllele << "@[" << allele_identity_vector[*idx_it_1].position0 << ", " <<  allele_identity_vector[*idx_it_1].position0 + allele_identity_vector[*idx_it_1].ref_length
										 << ") and alt" << *idx_it_2 << ": " << allele_identity_vector[*idx_it_2].altAllele << "@[" << allele_identity_vector[*idx_it_2].position0 << ", " <<  allele_identity_vector[*idx_it_2].position0 + allele_identity_vector[*idx_it_2].ref_length << ")." << endl;
								}
							}
						}
					}
					// Then, I will break all alleles in the group.
					else if (num_iter >= 2){
						list<int>::iterator idx_it_2 = idx_it_1;
						++idx_it_2;
						for (; idx_it_2 != group_it->end(); ++idx_it_2){
							if (not allele_connectivity_matrix[*idx_it_1][*idx_it_2]){
								continue;
							}
							allele_connectivity_matrix[*idx_it_1][*idx_it_2] = false;
							allele_connectivity_matrix[*idx_it_2][*idx_it_1] = false;
							connectivity_changed = true;
							if (debug > 0){
								cout << "    - Breaking the connectivity between alt"<<  *idx_it_1 << ": "<< allele_identity_vector[*idx_it_1].altAllele << "@[" << allele_identity_vector[*idx_it_1].position0 << ", " <<  allele_identity_vector[*idx_it_1].position0 + allele_identity_vector[*idx_it_1].ref_length
									 << ") and alt" << *idx_it_2 << ": " << allele_identity_vector[*idx_it_2].altAllele << "@[" << allele_identity_vector[*idx_it_2].position0 << ", " <<  allele_identity_vector[*idx_it_2].position0 + allele_identity_vector[*idx_it_2].ref_length << "), "<< endl;
							}
						}
					}
				}
			}
		}

		if (max_group_size_achieved){
			// Split the variant using the modified allele_connectivity_matrix.
			if (connectivity_changed){
				FindNodesInIsoSubGraph(allele_connectivity_matrix, allele_groups, false);
				JoinGroupsStartAtTheSamePosition(allele_identity_vector, allele_groups, false);
			}
		}else{
			keep_itertate = false;
			if (debug > 0){
				cout <<"  - No splitting needed." << endl;
			}
		}
		++num_iter;
	}
}

void GetPaddingRemovedAlleleIdentityVector(const vector<AlleleIdentity>& allele_identity_vector, const ReferenceReader &ref_reader, vector<AlleleIdentity>& padding_removed_allele_identity_vector){
	int num_alt = (int) allele_identity_vector.size();
	map<pair<int, int>, LocalReferenceContext> contex_dict;

	padding_removed_allele_identity_vector.resize(num_alt);
	for (int i_alt = 0; i_alt < num_alt; ++i_alt){
		int num_padding = allele_identity_vector[i_alt].num_padding_added.first + allele_identity_vector[i_alt].num_padding_added.second;
		if (num_padding == 0){
			// Usually, calling this function means that there are padding bases added. Copying the original allele_identity would be rare.
			padding_removed_allele_identity_vector[i_alt] = allele_identity_vector[i_alt];
			continue;
		}
		padding_removed_allele_identity_vector[i_alt].DEBUG = allele_identity_vector[i_alt].DEBUG;
		pair< map<pair<int, int>, LocalReferenceContext>::iterator, bool> context_finder;
		// context_finder.first->second is the context for the padding pair allele_identity_vector[i_alt].num_padding_added.
		context_finder = contex_dict.insert(pair<pair<int, int>, LocalReferenceContext> (allele_identity_vector[i_alt].num_padding_added, LocalReferenceContext()));
		// context_finder.second indicates did I see this context before?
		if (context_finder.second){
			// Didn't see this context before. Need to detect context.
			context_finder.first->second.DetectContextAtPosition(ref_reader,
					allele_identity_vector[i_alt].chr_idx, allele_identity_vector[i_alt].start_variant_window,
					allele_identity_vector[i_alt].ref_length - num_padding);
		}
		if (not context_finder.first->second.context_detected){
			padding_removed_allele_identity_vector[i_alt].status.isProblematicAllele = true;
			continue;
		}
		padding_removed_allele_identity_vector[i_alt].altAllele = allele_identity_vector[i_alt].altAllele.substr(allele_identity_vector[i_alt].num_padding_added.first, (int) allele_identity_vector[i_alt].altAllele.size() - num_padding);
		padding_removed_allele_identity_vector[i_alt].position0 = allele_identity_vector[i_alt].start_variant_window;
		padding_removed_allele_identity_vector[i_alt].ref_length = allele_identity_vector[i_alt].ref_length - num_padding;
		padding_removed_allele_identity_vector[i_alt].chr_idx = allele_identity_vector[i_alt].chr_idx;
		padding_removed_allele_identity_vector[i_alt].status.isFakeHsAllele = allele_identity_vector[i_alt].status.isFakeHsAllele;
		padding_removed_allele_identity_vector[i_alt].status.isHotSpotAllele = allele_identity_vector[i_alt].status.isHotSpotAllele;

		if (not padding_removed_allele_identity_vector[i_alt].CharacterizeVariantStatus(context_finder.first->second, ref_reader)){
			padding_removed_allele_identity_vector[i_alt].status.isProblematicAllele = true;
			continue;
		}
		padding_removed_allele_identity_vector[i_alt].CalculateWindowForVariant(context_finder.first->second, ref_reader);
		padding_removed_allele_identity_vector[i_alt].status.isProblematicAllele += allele_identity_vector[i_alt].status.isProblematicAllele;
	}
}

// This function helps the candidate to split the multi-allele variant into smaller variants
void SplitAlleleIdentityVector(const vector<AlleleIdentity>& allele_identity_vector, list<list<int> >& allele_groups, const ReferenceReader &ref_reader, int max_group_size_allowed, bool padding_already_removed, unsigned int max_final_split_iteration = 4)
{
	int num_alt = (int) allele_identity_vector.size();
	map<pair<int, int>, LocalReferenceContext> contex_dict;
	vector<vector<bool> > allele_connectivity_matrix(num_alt, vector<bool>(num_alt));

	if (num_alt == 1){
		// Don't waste my time on the trivial splitting.
		allele_groups = {{0}};
		return;
	}

	// (Step 1): Get the padding removed allele_identity_vector
	// Note that allele connectivity should be determined using the "padding removed version" of the alt alleles.
	// In particular, I need the splicing window of padding removed allele.
	vector<AlleleIdentity> const * padding_removed_allele_identity_vector = &allele_identity_vector;
	vector<AlleleIdentity>* padding_removed_allele_identity_vector_temp = NULL;

	if (not padding_already_removed){
		padding_removed_allele_identity_vector_temp = new vector<AlleleIdentity>[1];
		GetPaddingRemovedAlleleIdentityVector(allele_identity_vector, ref_reader, *padding_removed_allele_identity_vector_temp);
		padding_removed_allele_identity_vector = padding_removed_allele_identity_vector_temp;
	}

	// (Step 2): Determine allele connectivity
	for (int i_alt = 0; i_alt < num_alt; ++i_alt){
		allele_connectivity_matrix[i_alt][i_alt] = true;
		for (int j_alt = i_alt + 1; j_alt < num_alt; ++j_alt){
			allele_connectivity_matrix[i_alt][j_alt] = IsAllelePairConnected(padding_removed_allele_identity_vector->at(i_alt), padding_removed_allele_identity_vector->at(j_alt));
			allele_connectivity_matrix[j_alt][i_alt] = allele_connectivity_matrix[i_alt][j_alt];
		}
	}

	// (Step 3): Split the alleles into groups
	FindNodesInIsoSubGraph(allele_connectivity_matrix, allele_groups, false);

	// (Step 4): Finalize the groups and alleles:
	// (4.a) Sort the alleles and groups
	// (4.b) Further split the groups that contain too many alleles.
	FinalizeSplitting(allele_connectivity_matrix, *padding_removed_allele_identity_vector, max_group_size_allowed, allele_groups, max_final_split_iteration);

	// (Step 5): Delete padding_removed_allele_identity_vector_temp
	if (padding_removed_allele_identity_vector_temp != NULL){
		delete [] padding_removed_allele_identity_vector_temp;
	}
}

// Given variant candidates, calculate the end of the look ahead window for candidate generator,
// where look ahead window = [variant_window_end, look_ahead_end).
// I.e., the candidate generator should make sure that there is NO other variant till the (0-based) position at (look_ahead_end_0 - 1), while a variant @ look_ahead_end_0 is fine.
int EnsembleEval::CalculateLookAheadEnd0(const ReferenceReader &ref_reader, int current_candidate_gen_window_end /*= -1*/){
	int chr_size = ref_reader.chr_size(seq_context.chr_idx); // The size of the chromosome.
	int non_fake_variant_window_end = (int) seq_context.position0;  // non_fake_variant_window_end is the largest variant_window_end of non-Fake-HS alleles.
	const int variant_window_end = (int) seq_context.position0 + (int) seq_context.reference_allele.size();
	current_candidate_gen_window_end = max(current_candidate_gen_window_end, variant_window_end);

	for (unsigned int i_alt = 0; i_alt < allele_identity_vector.size(); ++i_alt){
		if ((not allele_identity_vector[i_alt].status.isFakeHsAllele) and non_fake_variant_window_end < allele_identity_vector[i_alt].end_variant_window){
			non_fake_variant_window_end = allele_identity_vector[i_alt].end_variant_window;
			if (non_fake_variant_window_end == variant_window_end){
				break;
			}
		}
	}
	// (Note): If I see non_fake_variant_window_end == (int) seq_context.position0, it means that every allele is FakeHS.

	// (Step 1): Determine the at-least-end of the lookahead window, which may be pushed to the right later.
	// The look ahead window must cover (multiallele_window_end - 1) to make sure that the variant is not interfered by future variants
	// Note that multiallele_window_end can be dummy if there is a problematic. I will set look_ahead_end_0 to be variant_window_end.
	int look_ahead_end_0 = max(multiallele_window_end, current_candidate_gen_window_end);

	if (DEBUG){
		cout << "+ Investigating lookahead window (0-based) for the variant (1-based) (" << PrintVariant(*variant) <<"): "<< endl
			 << "  - Current multi-allele splicing window end = " << multiallele_window_end << ", which is the \"at least\" look ahead window end." << endl
			 << "  - Current variant window end = " << variant_window_end << endl
			 << "  - Current candidate gen window end = " << current_candidate_gen_window_end << endl
			 << "  - Current non-fake-variant window end = " << non_fake_variant_window_end << (non_fake_variant_window_end == (int) seq_context.position0? " (all alleles are FakeHS)" : "") << endl
	         << "  + Finding the smallest future position that won't be interfered by this variant (i.e., splicing lower bound@pos >= non-fake-variant window end):" << endl;
	}

	// Deal with exceptions:
	string exception_reason;
	if (non_fake_variant_window_end == (int) seq_context.position0){
		exception_reason =  " (All alleles are FakeHS which won't interfere others)."; // This may happen quite often!
	}else if (look_ahead_end_0 >= chr_size){
		look_ahead_end_0 = chr_size;
		exception_reason = " (reached the end of the chromosome)." ;
	}
	else if (look_ahead_end_0 < variant_window_end){
		exception_reason = " (problematic)";
	}

	if (not exception_reason.empty()){
		if (DEBUG){
				cout << "  - Lookahead window end = " << look_ahead_end_0 << exception_reason << endl;
		}
		return max(look_ahead_end_0, variant_window_end); // Safety: lookahead end = variant_window_end means no look ahead.
	}

	// (Step 2): Keep looking ahead until the non-Fake alleles will not interfere any variant at the future position.
	// I.e., the splicing lower bound at look_ahead_end_0 should >= non_fake_variant_window_end.
	// Note that all windows defined here are left-closed, right-open.
	LocalReferenceContext future_context; // The context at the future position
	// I assume that the candidate generator doesn't add right anchors to an alt allele for no reason. There must be another alt allele pushes the variant window to the right. (Of course it is not true for HS allele).
	// If the assumption doesn't hold, then the look ahead will be more conservative, not hurt.
	int splicing_lower_bound_at_look_ahead_end = -1;

	while (look_ahead_end_0 < chr_size){
		future_context.DetectContextAtPosition(ref_reader, seq_context.chr_idx, look_ahead_end_0, 1);
		// Calculate the splicing lower bound at look_ahead_end_0
		splicing_lower_bound_at_look_ahead_end = future_context.SplicingLeftBound(ref_reader);

		if (DEBUG){
			cout <<	"    - pos = " << look_ahead_end_0 << ", Splicing lower bound@pos = " << splicing_lower_bound_at_look_ahead_end << endl;
		}

		// I push the look ahead window end to the right until this variant won't interfere any variant outside the RHS of the look ahead window.
		if (splicing_lower_bound_at_look_ahead_end >= non_fake_variant_window_end){
			break;
		}
		++look_ahead_end_0;
	}

	if (DEBUG){
		if (look_ahead_end_0 < chr_size){
		    cout << "  - Lookahead window end = " << look_ahead_end_0 << ", splicing lower bound @(lookahead end) = " << splicing_lower_bound_at_look_ahead_end << endl;
		}else{
			cout << "  - Lookahead window end = " << look_ahead_end_0 << " (reached the end of the chromosome).";
		}
	}
	return look_ahead_end_0;
}

// Given the candidate alleles, determine the maximally possible split of the variant (or group of the alternative alleles) that can be correctly (i.e., w/o high FXX) evaluated by the evaluator.
// e.g. output: allele_groups = {{0,1,2}, {3, 4}, {5}}. Then alt[0], alt[1], alt[2] must be evaluated jointly; alt[3], alt[4] must be evaluated jointly; alt[5] can be evaluated individually.
void EnsembleEval::SplitMyAlleleIdentityVector(list<list<int> >& allele_groups, const ReferenceReader &ref_reader, int max_group_size_allowed){
	SplitAlleleIdentityVector(allele_identity_vector, allele_groups, ref_reader, max_group_size_allowed, false);

	if (DEBUG){
		cout << "+ Investigating variant splitting for the variant (" << PrintVariant(*variant) <<"): "<< endl
		     << "  - There are " << allele_groups.size() << " groups of alternative alleles identified." << endl;
		int group_idx = 0;
	    for (list<list<int> >::iterator g_it = allele_groups.begin(); g_it != allele_groups.end(); ++g_it, ++group_idx){
	    	cout << "  - Group #" << group_idx << " consists of "<< g_it->size() << " alternative alleles: "
	    		 << PrintIteratorToString(g_it->begin(), g_it->end(), "{", "}", ", ", "alt ") << endl;
	    }
	}
}

void EnsembleEval::FinalSplitReadyToGoAlleles(list<list<int> >& allele_groups_ready_to_go, const ReferenceReader &ref_reader, int max_group_size_allowed){
	unsigned int max_group_size = 0;
	unsigned int num_ready_to_go_alleles = 0;
	for (list<list<int> >::iterator g_it = allele_groups_ready_to_go.begin(); g_it != allele_groups_ready_to_go.end(); ++g_it){
		num_ready_to_go_alleles += g_it->size();
		if (g_it->size() > max_group_size){
			max_group_size = g_it->size() ;
		}
	}
	if ((int) max_group_size <= max_group_size_allowed){
		// No final splitting needed.
		return;
	}

	// Create a vector of AlleleIdentity objects that contains read-to-go alleles only.
	vector<AlleleIdentity> ready_to_go_allele_identity_vector;
	// Map the index from the ready_to_go_allele_identity_vector to allele_identity_vector.
	vector<int> read_to_go_index_to_original;
	ready_to_go_allele_identity_vector.reserve(num_ready_to_go_alleles);
	read_to_go_index_to_original.reserve(num_ready_to_go_alleles);
	for (list<list<int> >::iterator g_it = allele_groups_ready_to_go.begin(); g_it != allele_groups_ready_to_go.end(); ++g_it){
		for (list<int>::iterator a_it = g_it->begin(); a_it != g_it->end(); ++a_it){
			ready_to_go_allele_identity_vector.push_back(allele_identity_vector[*a_it]);
			read_to_go_index_to_original.push_back(*a_it);
		}
	}
	if (DEBUG){
		cout << "+ Final splitting ready-to-go alleles that contains a group of " << max_group_size << " alleles." <<endl;
	}

	// Final split ready-to-go alleles
	SplitAlleleIdentityVector(ready_to_go_allele_identity_vector, allele_groups_ready_to_go, ref_reader, max_group_size_allowed, false, 4);
	// Recover the index from ready_to_go_allele_identity_vector to allele_identity_vector.
	for (list<list<int> >::iterator g_it = allele_groups_ready_to_go.begin(); g_it != allele_groups_ready_to_go.end(); ++g_it){
		for (list<int>::iterator a_it = g_it->begin(); a_it != g_it->end(); ++a_it){
			*a_it = read_to_go_index_to_original[*a_it];
		}
	}
}

// (Inputs): ref_reader, current_candidate_gen_window_end_0.
// (Outputs): allele_groups_ready_to_go, alleles_on_hold, sliding_window_start_0, sliding_window_end_0
// current_candidate_gen_window_end_0: the end position of the current window for "de novo" candidate generation.
// allele_groups_ready_to_go: the list of allele groups that are safe for evaluation right now. That is, the variants outside current_look_ahead_window_end_0 won't interfere or be interfere the ready-to-go alleles.
// alleles_on_hold: the vector of the indices of alleles that may not be evaluated at this moment. I.e., they potentially interfere or be interfered by the variants outside the current candidate generation window.
// If the alleles in allele_groups_ready_to_go are output to the evaluator, then the candidate generator only needs to discover new variants in the look ahead "sliding" windows as follows.
// The candidate generator only needs to generating "novel" candidates in [sliding_window_start_0, sliding_window_end_0)
// The candidate generator only needs to generating "hotspots" candidates whose start position in [sliding_window_start_0, sliding_window_end_0)
// !!! Important !!! All the windows are defined as [win_start, win_end).
// !!! Important !!! If sliding_window_start_0 == sliding_window_end_0 == current_look_ahead_window_end_0, then all alleles are ready to go. No need to look ahead.
// !!! Important !!! It shall guarantee both conditions as follows: a) All on-hold alleles will be generated in the new sliding window. b) No ready-to-go alleles will be generated in the new sliding window.
void EnsembleEval::LookAheadSlidingWindow(int current_candidate_gen_window_end_0,
		const ReferenceReader &ref_reader,
		list<list<int> >& allele_groups_ready_to_go,
		vector<int>& alleles_on_hold,
		int& sliding_window_start_0,
		int& sliding_window_end_0,
		int max_group_size_allowed,
		const TargetsManager * const targets_manager){
	const int num_alt_alleles = (int) allele_identity_vector.size();
	int splicing_lower_bound_at_current_candidate_gen_window_end_0 = -1;
	LocalReferenceContext current_candidate_gen_window_context;
	vector<AlleleIdentity> padding_removed_allele_identity_vector;
	CompareAllelePositions compare_positions(&allele_identity_vector);
	// (Step 1.a): Initial Step
	allele_groups_ready_to_go.clear();
	alleles_on_hold.clear();

	// I first let start positions of the sliding window be their upper bound. They will be shrunk later.
	sliding_window_start_0 = current_candidate_gen_window_end_0;
	// I look ahead just 1bp every time.
	sliding_window_end_0 = min(current_candidate_gen_window_end_0 + 1, (int) ref_reader.chr_size(seq_context.chr_idx));

	// (Step 1.b): The trivial (and perhaps the most common) cases:
	// (1.b.1) Every allele is ready to go if the look ahead window can't be fully covered by any unmerged region.
	int current_merged_target_idx = targets_manager->FindMergedTargetIndex(seq_context.chr_idx, seq_context.position0);
	bool is_breaking_point = (current_merged_target_idx >= 0)? targets_manager->IsBreakingIntervalInMerged(current_merged_target_idx, seq_context.chr_idx, seq_context.position0, max(seq_context.position0, (long) sliding_window_end_0)) : false;

	// (1.b.2) If every allele and its end of the variant window hits the lookahead window end, then every allele is on hold.
	bool is_trivial_all_on_hold = true;
	for (int i_alt = 0; i_alt < num_alt_alleles; ++i_alt){
		if (allele_identity_vector[i_alt].end_variant_window < current_candidate_gen_window_end_0){
			is_trivial_all_on_hold = false;
			break;
		}
	}

	if (is_trivial_all_on_hold and (not is_breaking_point)){
		for (int i_alt = 0; i_alt < num_alt_alleles; ++i_alt){
			alleles_on_hold.push_back(i_alt);
			// Sort alleles_on_hold
			sort(alleles_on_hold.begin(), alleles_on_hold.end(), compare_positions);
			sliding_window_start_0 = min(sliding_window_start_0, allele_identity_vector[i_alt].start_variant_window);
		}

		if (DEBUG){
			cout << "+ Calculating the look ahead \"sliding\" window for (" << PrintVariant(*variant) <<"): "<< endl
				 << "  - Current candidate window end = " << current_candidate_gen_window_end_0 << endl
				 << "  - Current variant window = [" << seq_context.position0 << ", " << (int) seq_context.position0 + (int) seq_context.reference_allele.size() << ")" << endl
			     << "  - All alleles are on-hold, and none of them is ready to go." << endl
			     << "  - New look ahead sliding window = [" << sliding_window_start_0 << ", " << sliding_window_end_0 << ")" << endl;
		}
		return;
	}

	// (Step 2): Split the alleles into groups pretending every allele is ready to go.
	GetPaddingRemovedAlleleIdentityVector(allele_identity_vector, ref_reader, padding_removed_allele_identity_vector);
	if (DEBUG){
		cout << "+ Splitting the variant (" << PrintVariant(*variant) <<") for determining look ahead sliding window:" << endl;
	}

	// I don't do final splitting.
	SplitAlleleIdentityVector(padding_removed_allele_identity_vector, allele_groups_ready_to_go, ref_reader, num_alt_alleles, true, 0);

	// (Step 2.b): Trivial cases: No need to look ahead => all alleles are ready to go.
	// Case 1: hit the end of the chromosome
	// Case 2: breaking point in the merged region
	if (current_candidate_gen_window_end_0 == sliding_window_end_0
			or sliding_window_end_0 == (int) ref_reader.chr_size(seq_context.chr_idx)
			or is_breaking_point){
		sliding_window_start_0 = sliding_window_end_0;
		if (DEBUG){
			cout << "+ Calculating the look ahead \"sliding\" window for (" << PrintVariant(*variant) <<"): "<< endl
				 << "  - Current candidate window end = " << current_candidate_gen_window_end_0 << endl
				 << "  - Current variant window = [" << seq_context.position0 << ", " << (int) seq_context.position0 + (int) seq_context.reference_allele.size() << ")" << endl
			     << "  - All alleles are ready to go and no need to look ahead " << (is_breaking_point? "because it hits a breaking point in the merged region)." : ".") << endl;
		}
	    FinalSplitReadyToGoAlleles(allele_groups_ready_to_go, ref_reader, max_group_size_allowed);
		return;
	}

	// (Step 3): Remove on_hold_alleles from allele_groups_ready_to_go and determine sliding_window_start_0.
	// Specifically, an allele is ready to go if and only if all other alleles in the same group won't cause any interference for future variants.
	current_candidate_gen_window_context.DetectContextAtPosition(ref_reader, seq_context.chr_idx, current_candidate_gen_window_end_0, 1);
	splicing_lower_bound_at_current_candidate_gen_window_end_0 = current_candidate_gen_window_context.SplicingLeftBound(ref_reader);

	list<list<int> >::iterator group_it = allele_groups_ready_to_go.begin();
	while (group_it != allele_groups_ready_to_go.end()){
		bool is_group_ready_to_go = true;  // Everyone starts as ready-to-go.

		for (list<int>::iterator allele_it = group_it->begin(); allele_it != group_it->end(); ++allele_it){
			int my_splicing_end = max(padding_removed_allele_identity_vector[*allele_it].end_splicing_window, padding_removed_allele_identity_vector[*allele_it].end_variant_window); // In case not defined for problematic alleles.
			if (my_splicing_end > current_candidate_gen_window_end_0){
				// The allele is potentially interfered by future variants
				is_group_ready_to_go = false;
				break;
			}

			if (padding_removed_allele_identity_vector[*allele_it].end_variant_window > splicing_lower_bound_at_current_candidate_gen_window_end_0
					and (not padding_removed_allele_identity_vector[*allele_it].status.isFakeHsAllele)){
				// The allele is real and can potentially interfere future variants
				is_group_ready_to_go = false;
				break;
			}
		}
		// A group is ready to go if every allele in the group is ready to go.
		if (is_group_ready_to_go){
			++group_it;
		}else{
			// The group is not ready to go.
			// Put all alleles in the group in the on hold vector.
			for (list<int>::iterator allele_it = group_it->begin(); allele_it != group_it->end(); ++allele_it){
				alleles_on_hold.push_back(*allele_it);
				sliding_window_start_0 = min(sliding_window_start_0, padding_removed_allele_identity_vector[*allele_it].start_variant_window);
			}
			// Remove the group from allele_groups_ready_to_go.
			// I must remove the iterator carefully!
			group_it = allele_groups_ready_to_go.erase(group_it);
		}
	}

	// All alleles are ready to go. No need to look ahead.
	if (alleles_on_hold.empty()){
		sliding_window_end_0 = current_candidate_gen_window_end_0;
		sliding_window_start_0 = current_candidate_gen_window_end_0;
		if (DEBUG){
			cout << "+ Calculating the look ahead \"sliding\" window for (" << PrintVariant(*variant) <<"): "<< endl
				 << "  - Current candidate window end = " << current_candidate_gen_window_end_0 << endl
				 << "  - Current variant window = [" << seq_context.position0 << ", " << (int) seq_context.position0 + (int) seq_context.reference_allele.size() << ")" << endl
			     << "  - No need to look ahead and all alleles are ready to go!"  << endl;
		}
	    FinalSplitReadyToGoAlleles(allele_groups_ready_to_go, ref_reader, max_group_size_allowed);
		return;
	}

	// (Step 4): Move the ready-to-go alleles whose start position >= sliding_window_start_0 to be on hold.
	// I.e., I want to make sure that no ready-to-go allele will be generated in the new sliding window.
	// This will lose some ready-to-go alleles but I will have only sliding window.
	bool is_sliding_window_start_0_changed = false;
	do{
		is_sliding_window_start_0_changed = false;
		group_it = allele_groups_ready_to_go.begin();
		while (group_it != allele_groups_ready_to_go.end()){
			bool is_force_to_be_on_hold = false;
			for (list<int>::iterator allele_it = group_it->begin(); allele_it != group_it->end(); ++allele_it){
				if (allele_identity_vector[*allele_it].start_variant_window >= sliding_window_start_0
						or ((not allele_identity_vector[*allele_it].status.isFakeHsAllele) and allele_identity_vector[*allele_it].end_variant_window > sliding_window_start_0)){
					// This allele can be generated in the new slidgin window. Force the entire group to be on hold.
					is_force_to_be_on_hold = true;
					break;
				}
			}

			if (is_force_to_be_on_hold){
				for (list<int>::iterator allele_it = group_it->begin(); allele_it != group_it->end(); ++allele_it){
					// I need to make sure that the on-hold allele can be generated in the new sliding_window
					if (allele_identity_vector[*allele_it].start_variant_window < sliding_window_start_0){
						// Need to push sliding_window_start_0 to the left to generate this allele.
						is_sliding_window_start_0_changed = true;
						sliding_window_start_0 = allele_identity_vector[*allele_it].start_variant_window;
					}
					alleles_on_hold.push_back(*allele_it);
				}
				group_it = allele_groups_ready_to_go.erase(group_it);
			}else{
				++group_it;
			}
		}
	}while(is_sliding_window_start_0_changed);

	// Sanity check that makes sure no ready-to-go allele will be generated in the new sliding window.
	for (group_it = allele_groups_ready_to_go.begin(); group_it != allele_groups_ready_to_go.end(); ++group_it){
		for (list<int>::iterator allele_it = group_it->begin(); allele_it != group_it->end(); ++allele_it){
			assert(allele_identity_vector[*allele_it].start_variant_window < sliding_window_start_0);
			if (not allele_identity_vector[*allele_it].status.isFakeHsAllele){
				assert(allele_identity_vector[*allele_it].end_variant_window <= sliding_window_start_0);
			}
		}
	}

	// Sort alleles_on_hold
	sort(alleles_on_hold.begin(), alleles_on_hold.end(), compare_positions);
	// Final safey splitting for ready-to-go alleles.
    FinalSplitReadyToGoAlleles(allele_groups_ready_to_go, ref_reader, max_group_size_allowed);

	// Final debug message
	if (DEBUG){
		cout << "+ Calculating the look ahead \"sliding\" window for (" << PrintVariant(*variant) <<"): "<< endl
			 << "  - Current candidate window end = " << current_candidate_gen_window_end_0 << endl
			 << "  - Current variant window = [" << seq_context.position0 << ", " << (int) seq_context.position0 + (int) seq_context.reference_allele.size() << ")" << endl
		     << "  - New look ahead sliding window = [" << sliding_window_start_0 << ", " << sliding_window_end_0 << ")" << endl
			 << "  + Number of on-hold alleles = "<< alleles_on_hold.size() << endl
			 << "    - List of on-hold alleles = {" << PrintIteratorToString(alleles_on_hold.begin(), alleles_on_hold.end(), "", "", ", ", "alt ") << "}"<< endl
			 << "  + Number of ready-to-go groups = "<< allele_groups_ready_to_go.size() << endl
			 << "    - Total number of ready-to-go alleles = " << (num_alt_alleles - (int) alleles_on_hold.size()) << endl;
		int group_idx = 0;
	    for (list<list<int> >::iterator g_it = allele_groups_ready_to_go.begin(); g_it != allele_groups_ready_to_go.end(); ++g_it, ++group_idx){
	    	cout << "    - Group #" << group_idx << " consists of "<< g_it->size() << " alt alleles: "
	    		 << PrintIteratorToString(g_it->begin(), g_it->end(), "{", "}", ", ", "alt ") << endl;
	    }
	}
}


void EnsembleEval::SetupAllAlleles(const ExtendParameters &parameters,
                                                 const InputStructures  &global_context,
                                                 const ReferenceReader &ref_reader)
{
  seq_context.DetectContext(*variant, global_context.DEBUG, ref_reader);
  allele_identity_vector.resize(variant->alt.size());

  if (global_context.DEBUG > 0 and variant->alt.size() > 0) {
    cout << "Investigating variant candidate " << seq_context.reference_allele << " -> " << variant->alt[0];
    for (unsigned int i_allele = 1; i_allele < allele_identity_vector.size(); i_allele++)
      cout << ',' << variant->alt[i_allele];
    cout << endl;
  }

  // Make sure the vectors are initialized.
  assert(variant->alt.size() == variant->alt_orig_padding.size());
  assert(variant->alt.size() == variant->isAltHotspot.size());
  assert(variant->alt.size() == variant->isAltFakeHotspot.size());

  //now calculate the allele type (SNP/Indel/MNV/HPIndel etc.) and window for hypothesis calculation for each alt allele.
  for (unsigned int i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
    allele_identity_vector[i_allele].status.isHotSpot = variant->isHotSpot;
    allele_identity_vector[i_allele].status.isHotSpotAllele = variant->isAltHotspot[i_allele];
    allele_identity_vector[i_allele].status.isFakeHsAllele = variant->isAltFakeHotspot[i_allele];
    allele_identity_vector[i_allele].filterReasons.clear();
    allele_identity_vector[i_allele].DEBUG = global_context.DEBUG;

    allele_identity_vector[i_allele].indelActAsHPIndel = parameters.my_controls.filter_variant.indel_as_hpindel;
    allele_identity_vector[i_allele].getVariantType(variant->alt[i_allele], seq_context,
        global_context.ErrorMotifs,  parameters.my_controls.filter_variant, ref_reader, variant->alt_orig_padding[i_allele]);
    allele_identity_vector[i_allele].CalculateWindowForVariant(seq_context, ref_reader);
  }

  //GetMultiAlleleVariantWindow();
  multiallele_window_start = -1;
  multiallele_window_end   = -1;


  // Mark Ensemble for realignment if any of the possible variants should be realigned
  // TODO: Should we exclude already filtered alleles?
  for (unsigned int i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
    //if (!allele_identity_vector[i_allele].status.isNoCallVariant) {
    if (allele_identity_vector[i_allele].start_splicing_window < multiallele_window_start or multiallele_window_start == -1)
      multiallele_window_start = allele_identity_vector[i_allele].start_splicing_window;
    if (allele_identity_vector[i_allele].end_splicing_window > multiallele_window_end or multiallele_window_end == -1)
      multiallele_window_end = allele_identity_vector[i_allele].end_splicing_window;

    if (allele_identity_vector[i_allele].ActAsSNP() && parameters.my_controls.filter_variant.do_snp_realignment) {
      doRealignment = doRealignment or allele_identity_vector[i_allele].status.doRealignment;
    }
    if (allele_identity_vector[i_allele].ActAsMNP() && parameters.my_controls.filter_variant.do_mnp_realignment) {
      doRealignment = doRealignment or allele_identity_vector[i_allele].status.doRealignment;
    }
  }
  // pass allele windows back down the object
  for (unsigned int i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
    allele_identity_vector[i_allele].multiallele_window_start = multiallele_window_start;
    allele_identity_vector[i_allele].multiallele_window_end = multiallele_window_end;
  }


  if (global_context.DEBUG > 0) {
	cout << "Realignment for this candidate is turned " << (doRealignment ? "on" : "off") << endl;
    cout << "Final window for multi-allele: " << ": (" << multiallele_window_start << ") ";
    for (int p_idx = multiallele_window_start; p_idx < multiallele_window_end; p_idx++)
      cout << ref_reader.base(seq_context.chr_idx,p_idx);
    cout << " (" << multiallele_window_end << ") " << endl;
  }
}


void EnsembleEval::SpliceAllelesIntoReads(PersistingThreadObjects &thread_objects, const InputStructures &global_context,
                                          const ExtendParameters &parameters, const ReferenceReader &ref_reader)
{
  bool changed_alignment;
  unsigned int  num_valid_reads = 0;
  unsigned int  num_realigned = 0;
  int  num_hyp_no_null = allele_identity_vector.size()+1; // num alleles +1 for ref

  // generate null+ref+nr.alt hypotheses per read in the case of do_multiallele_eval
  allele_eval.total_theory.my_hypotheses.resize(read_stack.size());

  for (unsigned int i_read = 0; i_read < allele_eval.total_theory.my_hypotheses.size(); i_read++) {
    // --- New splicing function ---
    allele_eval.total_theory.my_hypotheses[i_read].success =
        SpliceVariantHypotheses(*read_stack[i_read],
                                *this,
                                seq_context,
                                thread_objects,
                                allele_eval.total_theory.my_hypotheses[i_read].splice_start_flow,
                                allele_eval.total_theory.my_hypotheses[i_read].splice_end_flow,
                                allele_eval.total_theory.my_hypotheses[i_read].instance_of_read_by_state,
                                allele_eval.total_theory.my_hypotheses[i_read].same_as_null_hypothesis,
                                changed_alignment,
                                global_context,
                                ref_reader);

    if (allele_eval.total_theory.my_hypotheses[i_read].success){
      num_valid_reads++;
      if (changed_alignment)
        num_realigned++;
    }

    // if we need to compare likelihoods across multiple possibilities
    if (num_hyp_no_null > 2)
      allele_eval.total_theory.my_hypotheses[i_read].use_correlated_likelihood = false;
  }

  // Check how many reads had their alignment modified
  std::ostringstream my_info;
  my_info.precision(4);
  if (doRealignment and num_valid_reads>0){
	float frac_realigned = (float)num_realigned / (float)num_valid_reads;
	// And re-do splicing without realignment if we exceed the threshold
	if (frac_realigned > parameters.my_controls.filter_variant.realignment_threshold){
      my_info << "SKIPREALIGNx" << frac_realigned;
      doRealignment = false;
      for (unsigned int i_read = 0; i_read < allele_eval.total_theory.my_hypotheses.size(); i_read++) {
          allele_eval.total_theory.my_hypotheses[i_read].success =
              SpliceVariantHypotheses(*read_stack[i_read],
                                      *this,
                                      seq_context,
                                      thread_objects,
                                      allele_eval.total_theory.my_hypotheses[i_read].splice_start_flow,
                                      allele_eval.total_theory.my_hypotheses[i_read].splice_end_flow,
                                      allele_eval.total_theory.my_hypotheses[i_read].instance_of_read_by_state,
                                      allele_eval.total_theory.my_hypotheses[i_read].same_as_null_hypothesis,
                                      changed_alignment,
                                      global_context,
                                      ref_reader);
      }
	}
	else {
      my_info << "REALIGNEDx" << frac_realigned;
	}
    info_fields.push_back(my_info.str());
  }
}

// Read and process records appropriate for this variant; positions are zero based
void EnsembleEval::StackUpOneVariant(const ExtendParameters &parameters, const PositionInProgress& bam_position, int sample_index)
{

  // Initialize random number generator for each stack -> ensure reproducibility
  RandSchrange RandGen(parameters.my_controls.RandSeed);

  read_stack.clear();  // reset the stack
  read_stack.reserve(parameters.my_controls.downSampleCoverage);
  int read_counter = 0;

  for (Alignment* rai = bam_position.begin; rai != bam_position.end; rai = rai->next) {

    // Check global conditions to stop reading in more alignments
    if (rai->original_position > multiallele_window_start)
      break;

    // filter reads belonging to other samples
    if (rai->sample_index != sample_index)
      continue;

    // TS-17069: The primer-trimmed read must fully cover the variant
    if (rai->alignment.Position > seq_context.position0 or rai->alignment.GetEndPosition() < (int) seq_context.position0 + (int) seq_context.reference_allele.size())
      continue;

    if (rai->filtered)
      continue;

    // TS-17069: The original read must fully cover the splicing window. (rai->original_positinon has been checked)
    if (rai->original_end_position < multiallele_window_end)
    	continue;

    // Reservoir Sampling
    if (read_stack.size() < (unsigned int)parameters.my_controls.downSampleCoverage) {
      read_counter++;
      read_stack.push_back(rai);
    } else {
      read_counter++;
      // produces a uniformly distributed test_position between [0, read_counter-1]
      unsigned int test_position = ((double)RandGen.Rand() / ((double)RandGen.RandMax + 1.0)) * (double)read_counter;
      if (test_position < (unsigned int)parameters.my_controls.downSampleCoverage)
        read_stack[test_position] = rai;
    }
  }
}

// The class is simply used for random_shuffle
class MyRandSchrange : private RandSchrange{
public:
    MyRandSchrange(int seed = 1) {SetSeed(seed);} ;
    int operator()(int upper_lim) {return Rand() % upper_lim;}; // return a random number between 0 and upper_lim-1
};


// Contains the information I need for downsampling with mol tagging
struct FamInfoForDownSample
{
	MolecularFamily* ptr_fam; // The pointer of the molecular family.
	unsigned int num_reads_remaining; // How many reads that are not picked up after down sampling?
	FamInfoForDownSample(MolecularFamily* const fam){
		ptr_fam = fam;
		num_reads_remaining = ptr_fam->valid_family_members.size(); // Initially, none of the reads is picked up.
	};
	bool operator<(const FamInfoForDownSample &rhs) const {
		return num_reads_remaining > rhs.num_reads_remaining;
	};
};

// Compare two func families for sorting.
// The family has a read with larger read count wins.
// valid_family_members_sorted must be sorted!
bool CompareFuncFamilies(const FamInfoForDownSample& fam_0, const FamInfoForDownSample& fam_1)
{
	if (fam_0.ptr_fam->valid_family_members[0]->read_count == 1 and fam_1.ptr_fam->valid_family_members[0]->read_count == 1){
		return fam_0.ptr_fam->valid_family_members.size() > fam_1.ptr_fam->valid_family_members.size();
	}

	unsigned int min_size = min(fam_0.ptr_fam->valid_family_members.size(), fam_1.ptr_fam->valid_family_members.size());
	for (unsigned int i_read = 0; i_read < min_size; ++i_read){
		if (fam_0.ptr_fam->valid_family_members[i_read]->read_count > fam_1.ptr_fam->valid_family_members[i_read]->read_count){
			return true;
		}
		else if (fam_0.ptr_fam->valid_family_members[i_read]->read_count < fam_1.ptr_fam->valid_family_members[i_read]->read_count){
			return false;
		}
	}
	return fam_0.ptr_fam->GetValidFamSize() > fam_1.ptr_fam->GetValidFamSize();
}

// Contains the information I need for downsampling with mol tagging
class FamInfoForBiDirDownSample
{
public:
	MolecularFamily* ptr_fam; // The pointer of the molecular family.
	vector<int> fwd_read_indicies;  // The indicies of the FWD reads in ptr_fam->valid_family_members
	vector<int> rev_read_indicies;  // The indicies of the REV reads in ptr_fam->valid_family_members
	// Picking the reads in fwd_read_indicies[0:min_func_fwd_idx + 1] and rev_read_indicies[0:min_func_rev_idx + 1] can make the family functional with as least number of reads as possible.
	int min_func_fwd_idx;
    int min_func_rev_idx;
	unsigned int num_fwd_reads_remaining; // How many reads that are not picked up after down sampling?
	unsigned int num_rev_reads_remaining; // How many reads that are not picked up after down sampling?

	FamInfoForBiDirDownSample(MolecularFamily* const fam, unsigned int min_fam_size, unsigned int min_fam_per_strand_cov);
	// Used to sort the objects by the number of reads remaining.
	bool operator<(const FamInfoForBiDirDownSample &rhs) const{
		return num_fwd_reads_remaining + num_rev_reads_remaining > rhs.num_fwd_reads_remaining + rhs.num_rev_reads_remaining;
	};
};

// Compare two functional Bi-Dir families for sorting.
// Note that the use case is that there is one consensus read on each strand, the rest of the reads are usually not consensus reads.
bool CompareFuncBiDirFamilies(const FamInfoForBiDirDownSample& lhs, const FamInfoForBiDirDownSample& rhs)
{
	assert(max(rhs.min_func_fwd_idx, rhs.min_func_rev_idx) >= 0 and max(lhs.min_func_fwd_idx, lhs.min_func_rev_idx) >= 0);
	bool is_rhs_efficient = rhs.min_func_fwd_idx < 1 and rhs.min_func_rev_idx < 1; // Can two reads or less make rhs functional?
	bool is_lhs_efficient = lhs.min_func_fwd_idx < 1 and lhs.min_func_rev_idx < 1; // Can two reads or less make lhs functional?

	// Deal with the most common case first.
	if (is_lhs_efficient and is_rhs_efficient){
		// Priority:
		// 1) min(ZR of the first FWD read, ZR of the first REV read), i.e., good coverage on each strand if I get one read on each strand
		// 2) max(ZR of the first FWD read, ZR of the first REV read), i.e., good coverage if I get one read on each strand
		// 3) Less total number of (consensus) reads
		// 4) Larger Family size

		// P.1
		// Note that min_func_fwd_idx and min_func_rev_idx can be -1 if min_fam_per_strand_cov = 0
		int min_lhs_1st_zr = min(lhs.ptr_fam->valid_family_members.at((lhs.min_func_fwd_idx < 0? lhs.min_func_rev_idx : lhs.min_func_fwd_idx))->read_count, lhs.ptr_fam->valid_family_members.at((lhs.min_func_rev_idx < 0? lhs.min_func_fwd_idx : lhs.min_func_rev_idx))->read_count);
		int min_rhs_1st_zr = min(rhs.ptr_fam->valid_family_members.at((rhs.min_func_fwd_idx < 0? rhs.min_func_rev_idx : rhs.min_func_fwd_idx))->read_count, rhs.ptr_fam->valid_family_members.at((rhs.min_func_rev_idx < 0? rhs.min_func_fwd_idx : rhs.min_func_rev_idx))->read_count);
		if (min_lhs_1st_zr > min_rhs_1st_zr){
			return true;
		}else if (min_lhs_1st_zr < min_rhs_1st_zr){
			return false;
		}else{
			// P.2
			int max_lhs_1st_zr = max(lhs.ptr_fam->valid_family_members.at((lhs.min_func_fwd_idx < 0? lhs.min_func_rev_idx : lhs.min_func_fwd_idx))->read_count, lhs.ptr_fam->valid_family_members.at((lhs.min_func_rev_idx < 0? lhs.min_func_fwd_idx : lhs.min_func_rev_idx))->read_count);
			int max_rhs_1st_zr = max(rhs.ptr_fam->valid_family_members.at((rhs.min_func_fwd_idx < 0? rhs.min_func_rev_idx : rhs.min_func_fwd_idx))->read_count, rhs.ptr_fam->valid_family_members.at((rhs.min_func_rev_idx < 0? rhs.min_func_fwd_idx : rhs.min_func_rev_idx))->read_count);
			if (max_lhs_1st_zr > max_rhs_1st_zr){
				return true;
			}else if (max_lhs_1st_zr < max_rhs_1st_zr){
				return false;
			}else{
				// P.3
				if (lhs.ptr_fam->valid_family_members.size() < rhs.ptr_fam->valid_family_members.size()){
					return true;
				}else if (lhs.ptr_fam->valid_family_members.size() > rhs.ptr_fam->valid_family_members.size()){
					return false;
				}
			}
		}
		// P.4
		return lhs.ptr_fam->GetValidFamSize() > rhs.ptr_fam->GetValidFamSize();
	}

	if (is_rhs_efficient and (not is_lhs_efficient)){
		return false;
	}
	if (is_lhs_efficient and (not is_rhs_efficient)){
		return true;
	}

	// if (not (is_rhs_efficient or is_lhs_efficient))
	// Priority:
	// a): min(min_func_fwd_idx + min_func_rev_idx), i.e.,needs less reads to be functional
	// b): Less total number of (consensus) reads
	// c): Larger family size

	// P.a
	if (lhs.min_func_fwd_idx + lhs.min_func_rev_idx < rhs.min_func_fwd_idx + rhs.min_func_rev_idx){
		return true;
	}else if (lhs.min_func_fwd_idx + lhs.min_func_rev_idx > rhs.min_func_fwd_idx + rhs.min_func_rev_idx){
		return false;
	}else{
		// P.b
		if (lhs.ptr_fam->valid_family_members.size() < rhs.ptr_fam->valid_family_members.size()){
			return true;
		}else if (lhs.ptr_fam->valid_family_members.size() > rhs.ptr_fam->valid_family_members.size()){
			return false;
		}
	}// else
	// P.c
	return lhs.ptr_fam->GetValidFamSize() > rhs.ptr_fam->GetValidFamSize();
}


FamInfoForBiDirDownSample::FamInfoForBiDirDownSample(MolecularFamily* const fam, unsigned int min_fam_size, unsigned int min_fam_per_strand_cov){
	// I require that the family must be functional and sorted.
	assert(fam->GetFuncFromValid());
	assert(fam->is_valid_family_members_sorted);
	ptr_fam = fam;
	fwd_read_indicies.reserve(ptr_fam->valid_family_members.size());
	rev_read_indicies.reserve(ptr_fam->valid_family_members.size());
	for (unsigned int read_idx = 0; read_idx != ptr_fam->valid_family_members.size(); ++read_idx){
		if (ptr_fam->valid_family_members[read_idx]->is_reverse_strand){
			rev_read_indicies.push_back(read_idx);
		}else{
			fwd_read_indicies.push_back(read_idx);
		}
	}
	min_func_fwd_idx = -1;
	min_func_rev_idx = -1;
	num_fwd_reads_remaining = fwd_read_indicies.size();
	num_rev_reads_remaining = rev_read_indicies.size();

	unsigned int current_fwd_cov = 0;
	unsigned int current_rev_cov = 0;
	// Get FWD reads to satisfy min_fam_per_strand_cov
	while (current_fwd_cov < min_fam_per_strand_cov){
		++min_func_fwd_idx;
		current_fwd_cov += (unsigned int) ptr_fam->valid_family_members[min_func_fwd_idx]->read_count;
	}
	// Get REV reads to satisfy min_fam_per_strand_cov
	while (current_rev_cov < min_fam_per_strand_cov){
		++min_func_rev_idx;
		current_rev_cov += (unsigned int) ptr_fam->valid_family_members[min_func_rev_idx]->read_count;
	}
	// Get reads to satisfy min_fam_size
	while (current_fwd_cov + current_rev_cov < min_fam_size){
		int next_fwd_count = (min_func_fwd_idx + 1 < (int) fwd_read_indicies.size())? ptr_fam->valid_family_members.at(fwd_read_indicies.at(min_func_fwd_idx + 1))->read_count : 0;
		int next_rev_count = (min_func_rev_idx + 1 < (int) rev_read_indicies.size())? ptr_fam->valid_family_members.at(rev_read_indicies.at(min_func_rev_idx + 1))->read_count : 0;
		// Safety check. Shouldn't happen.
		assert(max(next_fwd_count, next_rev_count) > 0);
		bool pick_fwd = false;
		// Prefer to get the read with a larger read count
		if (next_fwd_count > next_rev_count){
			pick_fwd = true;
		}else if (next_fwd_count < next_rev_count){
			pick_fwd = false;
		}else{
			// Tie. Prefer to get balanced coverage on both strands.
			if (current_fwd_cov > current_rev_cov){
				pick_fwd = false;
			}else if (current_fwd_cov > current_rev_cov){
				pick_fwd = true;
			}else{
				// Tie. Prefer to get balanced number of (consensus) reads on both strands.
				pick_fwd = min_func_fwd_idx < min_func_rev_idx;
			}
		}
		if (pick_fwd){
			++min_func_fwd_idx;
			current_fwd_cov += (unsigned int) next_fwd_count;
		}else{
			++min_func_rev_idx;
			current_rev_cov += (unsigned int) next_rev_count;
		}
	}
}

// This function does strategic downsampling for bi-directional UMT familis by assuming most of the families consist of two consensus reads, one on FWD strand and one on REV strand.
void EnsembleEval::DoDownSamplingBiDirMolTag(const ExtendParameters &parameters, unsigned int effective_min_fam_size,  unsigned int effective_min_fam_per_strand_cov, vector< vector<MolecularFamily> > &my_molecular_families,
			                            unsigned int num_reads_available, unsigned int num_func_fam, int strand_key)
{
	assert(strand_key == 0);
	MyRandSchrange my_rand_schrange(parameters.my_controls.RandSeed); 	// The random number generator that we use to guarantee reproducibility.
    unsigned int read_counter = 0;  // Number of reads on read stack
    unsigned int downSampleCoverage = (unsigned int) parameters.my_controls.downSampleCoverage;

	read_stack.clear();  // reset the stack
	allele_eval.total_theory.my_eval_families.clear();

	// (Case 1): I can keep all the reads in all functional families :D
	if (num_reads_available <= downSampleCoverage){
		allele_eval.total_theory.my_eval_families.reserve(num_func_fam);
		read_stack.reserve(num_reads_available);

		for (vector<MolecularFamily>::iterator family_it = my_molecular_families[strand_key].begin();
				family_it != my_molecular_families[strand_key].end(); ++family_it){
			if (family_it->SetFuncFromValid(effective_min_fam_size, effective_min_fam_per_strand_cov)){
				allele_eval.total_theory.my_eval_families.push_back(EvalFamily(family_it->family_barcode, family_it->strand_key, &read_stack));
				for (vector<Alignment*>::iterator read_it = family_it->valid_family_members.begin(); read_it != family_it->valid_family_members.end(); ++read_it){
					read_stack.push_back(*read_it);
					allele_eval.total_theory.my_eval_families.back().AddNewMember(read_counter);
					++read_counter;
				}
			}
		}

		if (DEBUG > 0){
			cout << endl
					<< "+ Down sample with bi-directional UMT: "<< endl
			        << "  - Down sample " << num_reads_available << " reads to " << downSampleCoverage << ". " << endl
			        << "  - Number of functional families before/after down sampling = "<< num_func_fam<< "." << endl
			        << "  - Total reads after down sampling = " << read_counter << endl;
		}
		return;
	}

	// (Case 2): I can't keep all the reads
	unsigned int num_of_func_fam_after_down_sampling = 0;
	vector<FamInfoForBiDirDownSample> func_families;
	func_families.reserve(num_func_fam);
	for (vector<MolecularFamily>::iterator family_it = my_molecular_families[strand_key].begin(); family_it != my_molecular_families[strand_key].end(); ++family_it){
		if (family_it->SetFuncFromValid(effective_min_fam_size, effective_min_fam_per_strand_cov)){
			if (not family_it->is_valid_family_members_sorted){
				family_it->SortValidFamilyMembers();
			}
			func_families.push_back(FamInfoForBiDirDownSample(&(*family_it), effective_min_fam_size, effective_min_fam_per_strand_cov));
		}
	}
	// Random shuffle func_families to get randomness for the tie situation during sorting.
    random_shuffle(func_families.begin(), func_families.end(), my_rand_schrange);
	// The most important step. Sort func_families according to CompareFuncFamilies
    sort(func_families.begin(), func_families.end(), CompareFuncBiDirFamilies);

    // Step 2.a: Try to get as many functional families as possible
    int read_remaining = (int) downSampleCoverage;
	for (vector<FamInfoForBiDirDownSample>::iterator func_fam_it = func_families.begin(); func_fam_it != func_families.end() and read_remaining > 0; ++func_fam_it){
		// Note that read_remaining may < 0, i.e., I may get more than downSampleCoverage reads because I want to keep one more family.
		if (func_fam_it->min_func_fwd_idx >= 0 and func_fam_it->num_fwd_reads_remaining > 0){
			func_fam_it->num_fwd_reads_remaining -= (unsigned int) (func_fam_it->min_func_fwd_idx + 1);
			read_remaining -= (func_fam_it->min_func_fwd_idx + 1);
		}
		if (func_fam_it->min_func_rev_idx >= 0 and func_fam_it->num_rev_reads_remaining > 0){
			func_fam_it->num_rev_reads_remaining -= (unsigned int) (func_fam_it->min_func_rev_idx + 1);
			read_remaining -= (func_fam_it->min_func_rev_idx + 1);
		}
		++num_of_func_fam_after_down_sampling;
	}

	// Step 2.b
	// I can make every family functional and I still have reads left.
	if (read_remaining > 0){
		// Sort by number of reads remaining.
	    sort(func_families.begin(), func_families.end());
	    // Get as many reads as possible until I don't have reads remaining.
	    while (read_remaining > 0 and (func_families[0].num_fwd_reads_remaining + func_families[0].num_rev_reads_remaining) > 0){
	    	for (vector<FamInfoForBiDirDownSample>::iterator func_fam_it = func_families.begin(); func_fam_it != func_families.end() and read_remaining > 0; ++func_fam_it){
	    		if (func_fam_it->num_fwd_reads_remaining + func_fam_it->num_rev_reads_remaining == 0){
	    			break;
	    		}
	    		if (func_fam_it->num_fwd_reads_remaining == 0){
	    			--func_fam_it->num_rev_reads_remaining;
	    		}else if (func_fam_it->num_rev_reads_remaining == 0){
	    			--func_fam_it->num_fwd_reads_remaining;
	    		}else{
	    			if (func_fam_it->fwd_read_indicies.size() - func_fam_it->num_fwd_reads_remaining > func_fam_it->rev_read_indicies.size() - func_fam_it->num_rev_reads_remaining){
	    				--func_fam_it->num_rev_reads_remaining;
	    			}else{
	    				--func_fam_it->num_fwd_reads_remaining;
	    			}
	    		}
	    		--read_remaining;
	    	}
	    }
	}

	// Step 3
	// Fill in read stack
	allele_eval.total_theory.my_eval_families.reserve(num_of_func_fam_after_down_sampling);
	read_stack.reserve((int) downSampleCoverage - read_remaining);
	for (vector<FamInfoForBiDirDownSample>::iterator func_fam_it = func_families.begin(); func_fam_it != func_families.end(); ++func_fam_it){
		unsigned int num_fwd_reads_in = func_fam_it->fwd_read_indicies.size() - func_fam_it->num_fwd_reads_remaining;
		unsigned int num_rev_reads_in = func_fam_it->rev_read_indicies.size() - func_fam_it->num_rev_reads_remaining;
		if (num_fwd_reads_in + num_rev_reads_in == 0){
			continue;
		}
		allele_eval.total_theory.my_eval_families.push_back(EvalFamily(func_fam_it->ptr_fam->family_barcode, func_fam_it->ptr_fam->strand_key, &read_stack));
		for (int my_strand = 0; my_strand < 2; ++my_strand){
			unsigned int num_reads_in = (my_strand == 0 ? num_fwd_reads_in : num_rev_reads_in);
			vector<int> const * const read_indicies = (my_strand == 0 ? &(func_fam_it->fwd_read_indicies) : &(func_fam_it->rev_read_indicies));
			for (unsigned int idx = 0; idx < num_reads_in; ++idx){
				read_stack.push_back(func_fam_it->ptr_fam->valid_family_members.at(read_indicies->at(idx)));
				allele_eval.total_theory.my_eval_families.back().AddNewMember(read_counter);
				++read_counter;
			}
		}
	}

	if (DEBUG > 0){
		cout << endl
		     << "+ Down sample with bi-directional UMT: "<< endl
		     << "  - Down sample " << num_reads_available << " reads to " << downSampleCoverage << ". " << endl
		     << "  - Number of functional families before down sampling = "<< func_families.size() << "." << endl
		     << "  - Number of functional families after down sampling = "<< num_of_func_fam_after_down_sampling<< "." << endl
		     << "  - Total number of reads after down sampling = " << read_counter << endl;
	}
}

// I apply a strategic downsampling algorithm for molecular tagging using the following rules.
// Rule 0: Only reads in functional families will be evaluated.
// Rule 1: Get as many functional families as possible after down sampling
// Rule 2: If no consensus read, prefer to pick up "rich" families.
// Rule 3: For consensus reads, maximize the family size after downsampling.
// I only pick up the reads from my_molecular_families[strand_key]
// num_reads_available: total number of reads in the functional families (from valid_family_members) on the strand specified by strand_key
// num_func_fam: total number of functional families (from valid_family_members) on the strand specified by strand_key
void EnsembleEval::DoDownSamplingUniDirMolTag(const ExtendParameters &parameters, unsigned int effective_min_fam_size, vector< vector<MolecularFamily> > &my_molecular_families,
			                            unsigned int num_reads_available, unsigned int num_func_fam, int strand_key)
{
    assert(strand_key > 0);    // This function is for uni-directional UMT.
	MyRandSchrange my_rand_schrange(parameters.my_controls.RandSeed); 	// The random number generator that we use to guarantee reproducibility.
    unsigned int read_counter = 0;  // Number of reads on read stack
    unsigned int downSampleCoverage = (unsigned int) parameters.my_controls.downSampleCoverage;

	read_stack.clear();  // reset the stack
	allele_eval.total_theory.my_eval_families.clear();

	// (Case 1): I can keep all the reads in all functional families :D
	if (num_reads_available <= downSampleCoverage){
		allele_eval.total_theory.my_eval_families.reserve(num_func_fam);
		read_stack.reserve(num_reads_available);

		for (vector<MolecularFamily>::iterator family_it = my_molecular_families[strand_key].begin();
				family_it != my_molecular_families[strand_key].end(); ++family_it){
			if (family_it->SetFuncFromValid(effective_min_fam_size)){
				allele_eval.total_theory.my_eval_families.push_back(EvalFamily(family_it->family_barcode, family_it->strand_key, &read_stack));
				for (vector<Alignment*>::iterator read_it = family_it->valid_family_members.begin(); read_it != family_it->valid_family_members.end(); ++read_it){
					read_stack.push_back(*read_it);
					allele_eval.total_theory.my_eval_families.back().AddNewMember(read_counter);
					++read_counter;
				}
			}
		}

		if (DEBUG > 0){
			cout << endl
				 << "+ Down sample with uni-directional UMT on the " << (strand_key == 1? "FWD" : "REV") << " strand:" << endl
			     << "  - Down sample " << num_reads_available << " reads to " << downSampleCoverage << ". "
   		         << "  - Number of functional families before/after down sampling = "<< num_func_fam<< "." << endl
			     << "  - Total reads after down sampling = " << read_counter << endl;
		}
		return;
	}

	// (Case 2): I can't preserve all reads but I can preserve all functional families.
	// Step 1: Find and sort all available func families
	unsigned int reads_remaining = downSampleCoverage;
	unsigned int num_of_func_fam_after_down_sampling = 0;
	vector<FamInfoForDownSample> func_families;
	func_families.reserve(num_func_fam);
	for (vector<MolecularFamily>::iterator family_it = my_molecular_families[strand_key].begin(); family_it != my_molecular_families[strand_key].end(); ++family_it){
		if (family_it->SetFuncFromValid(effective_min_fam_size)){
			func_families.push_back(FamInfoForDownSample(&(*family_it)));
			// Always make sure valid_family_members is sorted
			if (not family_it->is_valid_family_members_sorted){
				family_it->SortValidFamilyMembers();
			}
		}
	}
	// Random shuffle func_families to get randomness for the tie situation during sorting.
    random_shuffle(func_families.begin(), func_families.end(), my_rand_schrange);
	// The most important step. Sort func_families according to CompareFuncFamilies
    sort(func_families.begin(), func_families.end(), CompareFuncFamilies);

	// Step 2:
	// In each family, pick valid_family_members[0] if the read can make the family functional. I call such a read "super read".
	// Again, valid_family_members must be sorted, so is func_families.
	vector<FamInfoForDownSample>::iterator first_poor_fam_it = func_families.end();  // A family is "poor" if it doesn't have such a "super read". first_poor_fam_it is the first poor family.
	for (vector<FamInfoForDownSample>::iterator func_fam_it = func_families.begin(); func_fam_it != func_families.end(); ++func_fam_it){
		if (reads_remaining == 0){
			break;
		}
		if (func_fam_it->ptr_fam->valid_family_members[0]->read_count >= (int) effective_min_fam_size){
			++num_of_func_fam_after_down_sampling;
			--(func_fam_it->num_reads_remaining);
			--reads_remaining;
		}
		else{
			first_poor_fam_it = func_fam_it;
			break;
		}
	}

	// Step 3:
	// Try to let as many poor families be functional as possible, if I still have reads remaining.
	vector<FamInfoForDownSample>::iterator last_poor_fam = func_families.end();
	if (reads_remaining > 0){
		for (vector<FamInfoForDownSample>::iterator func_fam_it = first_poor_fam_it; func_fam_it != func_families.end(); ++func_fam_it){
			if (reads_remaining < effective_min_fam_size){ // The simple criterion is not the best implementation because I may lose some func fam. But it should be acceptable.
				last_poor_fam = func_fam_it;
				break;
			}
			int down_sampled_fam_size = 0;
			for (vector<Alignment*>::iterator read_it = func_fam_it->ptr_fam->valid_family_members.begin(); read_it != func_fam_it->ptr_fam->valid_family_members.end(); ++read_it){
				if (reads_remaining == 0 or down_sampled_fam_size >= (int) effective_min_fam_size){
					break;
				}
				down_sampled_fam_size += ((*read_it)->read_count);
				--(func_fam_it->num_reads_remaining);
				--reads_remaining;
			}
			if (down_sampled_fam_size >= (int) effective_min_fam_size){
				++num_of_func_fam_after_down_sampling;
			}
			else{ // shouldn't happen
				last_poor_fam = func_fam_it;
				break;
			}
		}
	}

	// Step 4:
	// Pick one read in one family until I don't have read remaining.
	// This step should not affect the number of functional families after down sampling.
	if (reads_remaining > 0){
	    sort(func_families.begin(), last_poor_fam);  // Sort by the num_reads_remaining (more num_reads_remaining first). Otherwise it can be super slow in the worst case.
	}
	while (reads_remaining > 0 and func_families[0].num_reads_remaining > 0){ // Although the criterion (func_families[0].num_reads_remaining > 0) should be redundant, it's always safe to check this here.
		for (vector<FamInfoForDownSample>::iterator func_fam_it = func_families.begin(); func_fam_it != last_poor_fam; ++func_fam_it){
			if (reads_remaining == 0 or func_fam_it->num_reads_remaining == 0){
				break;
			}
			if (func_fam_it->num_reads_remaining > 0){
				--(func_fam_it->num_reads_remaining);
				--reads_remaining;
			}
		}
	}

	// Step 5:
	// Fill the reads into read_stack
	allele_eval.total_theory.my_eval_families.reserve(num_of_func_fam_after_down_sampling);
	read_stack.reserve(downSampleCoverage);

	for (vector<FamInfoForDownSample>::iterator func_fam_it = func_families.begin(); func_fam_it != func_families.end(); ++func_fam_it){
		int num_reads_picked = (int) (func_fam_it->ptr_fam->valid_family_members.size()) - func_fam_it->num_reads_remaining;
		if (num_reads_picked == 0){
			continue;
		}
		allele_eval.total_theory.my_eval_families.push_back(EvalFamily(func_fam_it->ptr_fam->family_barcode, func_fam_it->ptr_fam->strand_key, &read_stack));
		allele_eval.total_theory.my_eval_families.back().all_family_members.reserve(num_reads_picked);

		// For fairness, I randomly pick up the reads with the same read_count.
		if (func_fam_it->num_reads_remaining > 0){
			vector<Alignment*>::iterator first_read_it_with_smallest_read_count;
			vector<Alignment*>::iterator next_read_it_not_with_smallest_read_count;

			if (func_fam_it->ptr_fam->valid_family_members[0]->read_count == func_fam_it->ptr_fam->valid_family_members.back()->read_count){
				// All reads have equal read_count. Of course I need to randomly pick up num_reads_picked reads.
				first_read_it_with_smallest_read_count = func_fam_it->ptr_fam->valid_family_members.begin();
				next_read_it_not_with_smallest_read_count = func_fam_it->ptr_fam->valid_family_members.end();
			}
			else{
				// The reads in the family have variable read_count.
				// If more than one read have smallest_read_count, I need to random shuffle these reads and then do down sampling.
				vector<Alignment*>::iterator read_it_with_smallest_read_count = func_fam_it->ptr_fam->valid_family_members.begin() + (num_reads_picked - 1);
				int smallest_read_count = (*read_it_with_smallest_read_count)->read_count;
				// Find the first read after read_it_with_smallest_read_count whose read_count != smallest_read_count
				for (next_read_it_not_with_smallest_read_count = read_it_with_smallest_read_count + 1; next_read_it_not_with_smallest_read_count != func_fam_it->ptr_fam->valid_family_members.end(); ++next_read_it_not_with_smallest_read_count){
					if ((*next_read_it_not_with_smallest_read_count)->read_count != smallest_read_count)
						break;
				}
				// Find the last read before read_it_with_smallest_read_count whose read_count != smallest_read_count
				if ( (*(func_fam_it->ptr_fam->valid_family_members.begin()))->read_count == smallest_read_count){
					first_read_it_with_smallest_read_count = func_fam_it->ptr_fam->valid_family_members.begin();
				}
				else{
					for (first_read_it_with_smallest_read_count = read_it_with_smallest_read_count - 1; first_read_it_with_smallest_read_count != func_fam_it->ptr_fam->valid_family_members.begin(); --first_read_it_with_smallest_read_count){
						if ((*first_read_it_with_smallest_read_count)->read_count != smallest_read_count){
							break;
						}
					}
					++first_read_it_with_smallest_read_count;
				}
			}
			if (first_read_it_with_smallest_read_count != (next_read_it_not_with_smallest_read_count - 1)){
				// Partially random shuffle the reads in the family.
			    random_shuffle(first_read_it_with_smallest_read_count, next_read_it_not_with_smallest_read_count, my_rand_schrange);
			}
		}
		// Pickup the first num_reads_picked reads from valid_family_members.
		for (int i_read = 0; i_read < num_reads_picked; ++i_read){
			// Add the read into read_stack
			read_stack.push_back(func_fam_it->ptr_fam->valid_family_members[i_read]);
			// Add the read in to the family
			allele_eval.total_theory.my_eval_families.back().AddNewMember(read_counter);
			++read_counter;
		}
	}

	if (DEBUG > 0){
		cout << endl
		     << "+ Down sample with uni-directional UMT on the " << (strand_key == 1? "FWD" : "REV") << " strand:" << endl
		     << "  - Down sample " << num_reads_available << " reads to " << downSampleCoverage << ". " << endl
		     << "  - Number of functional families before down sampling = "<< num_func_fam<< "." << endl
		     << "  - Number of functional families after down sampling = "<< num_of_func_fam_after_down_sampling<< "." << endl
		     << "  - Total number of reads after down sampling = " << read_counter << endl;
	}
}

// Currently only take the reads on one strand
void EnsembleEval::StackUpOneVariantMolTag(const ExtendParameters &parameters, vector< vector<MolecularFamily> > &my_molecular_families, int sample_index)
{
	unsigned int effective_min_fam_size = allele_eval.total_theory.effective_min_family_size;
	unsigned int effective_min_fam_per_strand_cov = allele_eval.total_theory.effective_min_fam_per_strand_cov;

	vector<unsigned int> num_func_fam_by_strand = {0, 0, 0};
	vector<unsigned int> num_reads_available_by_strand = {0, 0, 0}; // Here, one consensus read counts one read!
	vector<unsigned int> num_reads_conuts_available_by_strand = {0, 0, 0}; // Here, one consensus read counts by its read counts!

	assert(allele_eval.total_theory.effective_min_family_size > 0);


	// For the current molecular barcoding scheme (bcprimer), the reads in each amplicom should be on on strand only.
	// However, we sometimes get families on both strands, primarily due to false priming.
	// Here I pick the strand that has more functional families
	// strand_index = 0, 1, 2 indicatae bi-directional, uni-directional FWD, uni-directional REV families.
	int best_strand_index = -1;
	int best_func_fam_cov = -1;

	for (unsigned int i_strand = 0; i_strand < my_molecular_families.size(); ++i_strand){
		for (vector< MolecularFamily>::iterator family_it = my_molecular_families[i_strand].begin();
				family_it != my_molecular_families[i_strand].end(); ++family_it){
			// Skip the family if it is not functional
			if (not family_it->SetFuncFromAll(effective_min_fam_size, effective_min_fam_per_strand_cov)){
				continue;
			}
			family_it->ResetValidFamilyMembers();
			if (not family_it->is_all_family_members_sorted){
				family_it->SortAllFamilyMembers(); // In case I haven't done this before.
			}

			// Apply more filtering criteria to filter out some reads in all_family_members
			// The reads in valid_family_members are available for downsampling and get into read stack
			for (vector<Alignment*>::iterator member_it = family_it->all_family_members.begin(); member_it != family_it->all_family_members.end(); ++member_it){
				// Although it has been done previously, do it again to make sure everything is right.
				if ((*member_it)->filtered){
					continue;
				}
				// Although it has been done previously, do it again to make sure everything is right.
				if ((*member_it)->sample_index != sample_index) {
                  continue;
			    }

				// Notes for TS-17069:
				// However, TS-17069 doesn't affect the UMT runs in 5.10 since trim-ampliseq-primers is turned off.
				// So the logic for the fix of TS-17069 is not applied in TS 5.10.1 for UMT (though I should, and also handle the consensus alignment)

			    // Check global conditions to stop reading in more alignments
				if ((*member_it)->original_position > multiallele_window_start
						or (*member_it)->alignment.Position > multiallele_window_start
						or (*member_it)->alignment.GetEndPosition() < multiallele_window_end){
					continue;
				}

				// family_members_temp stores the reads which are not filtered out here
				family_it->valid_family_members.push_back((*member_it));
			}
			family_it->is_valid_family_members_sorted = true; // valid_family_members is of course sorted as well since all_family_members is sorted.
			family_it->CountFamSizeFromValid();
			// Determine the functionality from valid_family_members
			if (family_it->SetFuncFromValid(effective_min_fam_size, effective_min_fam_per_strand_cov)){
				// Count how many reads and functional families available for down sampling
				num_reads_available_by_strand[i_strand] += family_it->valid_family_members.size();
				num_reads_conuts_available_by_strand[i_strand] += family_it->GetValidFamSize();
				++num_func_fam_by_strand[i_strand];
			}
		}
		if ((int) num_func_fam_by_strand[i_strand] > best_func_fam_cov){
			best_func_fam_cov = (int) num_func_fam_by_strand[i_strand];
			best_strand_index = (int) i_strand;
		}
	}

	if (DEBUG > 0){
		vector<string> strand_text = {"BIDIR", "FWD", "REV"};
		cout << endl << "+ Stack up one variant with molecular tagging" << endl
			 << "  - Effective-min-fam-size = " << effective_min_fam_size << endl
			 << "  - Effective-min-fam-per-strand-cov = " << effective_min_fam_per_strand_cov << endl;
		for (unsigned int i_strand = 0; i_strand < num_func_fam_by_strand.size(); ++i_strand){
			 cout << "  - Number of functional families identified on the "<< strand_text[i_strand] <<" strand = " << num_func_fam_by_strand[i_strand] << endl;
		}
		cout << "  - Use the families on the " << strand_text[best_strand_index] <<" strand only." << endl;
		cout << "  - Calculating family size histogram (Zero family size means all family members are filtered out.)..." << endl;
		for (unsigned int i_strand = 0; i_strand < my_molecular_families.size(); ++i_strand){
			 map<int, unsigned int> fam_size_hist;
			for (vector< MolecularFamily>::iterator family_it = my_molecular_families[i_strand].begin(); family_it != my_molecular_families[i_strand].end(); ++family_it){
				int fam_size = family_it->GetFuncFromAll()? family_it->GetValidFamSize() : family_it->GetFamSize();
				++fam_size_hist[fam_size];
			}
			cout << "  - "<< strand_text[i_strand] << " strand: fam_size_hist = [";
			for (map<int, unsigned int>::iterator hist_it = fam_size_hist.begin(); hist_it != fam_size_hist.end(); ++hist_it){
				cout << "(" << hist_it->first << ", " << hist_it->second <<"), ";
			}
			cout << "]" << endl;
		}
	}

	// Do down-sampling
	if (best_strand_index == 0){
		DoDownSamplingBiDirMolTag(parameters, effective_min_fam_size, effective_min_fam_per_strand_cov, my_molecular_families, num_reads_available_by_strand[best_strand_index], num_func_fam_by_strand[best_strand_index], best_strand_index);
	}else{
		DoDownSamplingUniDirMolTag(parameters, effective_min_fam_size, my_molecular_families, num_reads_available_by_strand[best_strand_index], num_func_fam_by_strand[best_strand_index], best_strand_index);
	}
}

// ------------------------------------------------------------

void EnsembleEval::FilterAllAlleles(const ControlCallAndFilters& my_controls, const vector<VariantSpecificParams>& variant_specific_params) {
  if (seq_context.context_detected) {
    for (unsigned int i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
      allele_identity_vector[i_allele].DetectCasesToForceNoCall(seq_context, my_controls, variant_specific_params[i_allele]);
    }
  }
}


BasicFilters const * ServeBasicFilterByType(const AlleleIdentity &variant_identity, const ControlCallAndFilters &my_controls){
	// The old logic for serving min-allele-freq based on allele type. Maybe deprecated in the future.
	if (not my_controls.use_fd_param){
		if (variant_identity.status.isHotSpot and (not my_controls.hotspots_as_de_novo))
			return &(my_controls.filter_hotspot);
		if (variant_identity.ActAsSNP())
			return &(my_controls.filter_snp);
		if (variant_identity.ActAsMNP())
			return &(my_controls.filter_mnp);
		if (variant_identity.ActAsHPIndel())
			return &(my_controls.filter_hp_indel);
		return &(my_controls.filter_snp);
	}

	// The new logic for serving min-allele-freq based on fd
	if (variant_identity.status.isHotSpotAllele and (not my_controls.hotspots_as_de_novo))
		return &(my_controls.filter_hotspot);
	if (variant_identity.fd_level_vs_ref == 0)
		return &(my_controls.filter_fd_0);
	if (variant_identity.fd_level_vs_ref == 1)
		return &(my_controls.filter_fd_5);
	if (variant_identity.fd_level_vs_ref == 2)
		return &(my_controls.filter_fd_10);
	// If indefinate, use the fd-0-min-allele-freq (usually the most conservative one)
	return &(my_controls.filter_fd_0);
}

float FreqThresholdByType(const AlleleIdentity& variant_identity,
		const ControlCallAndFilters& my_controls,
		const VariantSpecificParams& variant_specific_params)
{
	// Override has the top prioirty
	if (variant_specific_params.min_allele_freq_override)
	    return variant_specific_params.min_allele_freq;

	return ServeBasicFilterByType(variant_identity, my_controls)->min_allele_freq;
}


void EnsembleEval::GatherInfoForOfflineFiltering(const ControlCallAndFilters &my_controls, int best_allele_index){
	variant->info["PARAM"].clear();
	variant->info["FDPARAM"].clear();
	variant->info["BAP"].clear();
	variant->info["BAI"].clear();
	variant->info["FDBAP"].clear();
	variant->info["AAHPINDEL"].clear();
	variant->info["ISHPINDEL"].clear();


	for (unsigned int i_alt = 0; i_alt < allele_identity_vector.size(); ++i_alt){
		variant->info["AAHPINDEL"].push_back((allele_identity_vector[i_alt].ActAsHPIndel()? "1" : "0"));
		variant->info["ISHPINDEL"].push_back((allele_identity_vector[i_alt].status.isHPIndel? "1" : "0"));
	}
	ControlCallAndFilters my_controls_temp = my_controls;
	for (int use_fd = 0; use_fd < 2; ++use_fd){
		my_controls_temp.use_fd_param = (use_fd == 1);
		for (unsigned int i_alt = 0; i_alt < allele_identity_vector.size(); ++i_alt){
			BasicFilters const *my_basic_filter = ServeBasicFilterByType(allele_identity_vector[i_alt], my_controls_temp);
			string param_type;
			if (my_basic_filter == &(my_controls_temp.filter_fd_0)){
				param_type = "fd_0";
			}else if (my_basic_filter == &(my_controls_temp.filter_fd_5)){
				param_type = "fd_5";
			}else if (my_basic_filter == &(my_controls_temp.filter_fd_10)){
				param_type = "fd_10";
			}else if (my_basic_filter == &(my_controls_temp.filter_snp)){
				param_type = "snp";
			}else if (my_basic_filter == &(my_controls_temp.filter_mnp)){
				param_type = "mnp";
			}else if (my_basic_filter == &(my_controls_temp.filter_hp_indel)){
				param_type = "indel";
			}else if (my_basic_filter == &(my_controls_temp.filter_hotspot)){
				param_type = "hotspot";
			}else{
				param_type = ".";
			}
			variant->info[(my_controls_temp.use_fd_param? "FDPARAM" : "PARAM")].push_back(param_type);
		}
	}

	// The Best Alt allele Index. Allele X = Alt X - 1 => (X - 1)
	variant->info["BAI"].push_back(convertToString(best_allele_index));
	// Best Allele Pair
	variant->info["BAP"].push_back(convertToString(diploid_choice[0]));
	variant->info["BAP"].push_back(convertToString(diploid_choice[1]));
	// FD between Best Allele Pair
	string fd_bpa_string;
	int fd_bpa = global_flow_disruptive_matrix[diploid_choice[0]][diploid_choice[1]];
	if (fd_bpa == 0){
		fd_bpa_string = "0";
	}else if (fd_bpa == 1){
		fd_bpa_string = "5";
	}else if (fd_bpa == 2){
		fd_bpa_string = "10";
	}else{
		fd_bpa_string = "-1";
	}
	variant->info["FDBAP"].push_back(fd_bpa_string);
}
