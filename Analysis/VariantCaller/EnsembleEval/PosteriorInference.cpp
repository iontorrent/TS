/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PosteriorInference.h"

ScanSpace::ScanSpace(){
  scan_done = false;
  freq_pair.assign(2,0);
  // reference and hypothesis
  freq_pair[1]=1;
  freq_pair[0]=0;
  freq_pair_weight = 1.0f; // everything together
  max_ll = -999999999.0f; // anything is better than this
  max_index = 0;
  min_detail_level_for_fast_scan = 2500;
  // member variables for fast scan

  is_scanned_.clear();
  coarse_freq_resolution_ = 0.01f;
  num_of_fibonacci_blocks = 3; // Assume the maximums for Hom, Het, Hom
  fine_log_posterior_cutoff_gap_ = -log(0.00001f);
  fine_freq_search_increment_ = 0.02f;
  min_fine_log_posterior_gap_ = 0.01f;
  fine_scan_penalty_order_ = 1.5f;
  max_log_posterior_scanned_ = -999999999999.0f;  // anything is better than this
  argmax_log_posterior_scanned_ = 0;
}

FreqMaster::FreqMaster(){
  data_reliability = 1.0f-0.001f;
  max_hyp_freq.resize(2);
  prior_frequency_weight.resize(2);
  prior_frequency_weight.assign(2,0.0f); // additional prior weight here
  germline_prior_strength = 0.0f;
  germline_log_prior_normalization = 0.0f;
}

bool FreqMaster::Compare(vector <float> &original, int numreads, float threshold){
  float delta = 0.0f;
  for (unsigned int i_hyp=0; i_hyp<max_hyp_freq.size(); i_hyp++)
    delta += abs((original[i_hyp]-max_hyp_freq[i_hyp]));
  if (delta*numreads>threshold)
    return(false);
  else
    return(true);
}

PosteriorInference::PosteriorInference() {

  params_ll = 0.0f;

  vector<float> local_freq_start;
  local_freq_start.assign(2,0.5f);
  clustering.SetHypFreq(local_freq_start);
}

void FreqMaster::SetHypFreq(const vector<float> &local_freq){
    max_hyp_freq = local_freq;
 // max_hyp_freq.at(0) = local_freq;
 // max_hyp_freq.at(1) = 1.0f-max_hyp_freq.at(0);
}

// needs to match multialleles
void FreqMaster::SetPriorStrength(const vector<float> &local_freq){
  prior_frequency_weight = local_freq;
  //prior_frequency_weight.at(0) = local_freq;
  //prior_frequency_weight.at(1) = 1-local_freq;
  for (unsigned int i_hyp=0; i_hyp<prior_frequency_weight.size(); i_hyp++){
    prior_frequency_weight[i_hyp] = germline_prior_strength * prior_frequency_weight[i_hyp];
  }
  // set prior normalization constant used in evaluation of log-likelihood
  // this is a dirichlet (beta for 2 alleles)
  // note: on 0 prior strength we should see parameter alpha+1=1.0 for the gamma parameter, i.e. uniformly distributed
  double n_hyp = 1.0*prior_frequency_weight.size(); // always at least 2
  // this is n! for uniform distribution over n_hyp
  float total_weight = 0.0f;
  for (unsigned int i_hyp=0; i_hyp<prior_frequency_weight.size(); i_hyp++){
    total_weight += prior_frequency_weight[i_hyp];
  }
  germline_log_prior_normalization = 1.0*lgamma(total_weight+n_hyp);
  for (unsigned int i_hyp=0; i_hyp<prior_frequency_weight.size(); i_hyp++){
    germline_log_prior_normalization -= lgamma(prior_frequency_weight[i_hyp]+1.0);
  }
}

float ScanSpace::FindMaxFrequency(){
  float cur_best = log_posterior_by_frequency[0];
  float local_freq = 0.0f;
  int local_index = 0;
  for (unsigned int i_eval = 0; i_eval < log_posterior_by_frequency.size(); i_eval++) {
    if (log_posterior_by_frequency[i_eval] > cur_best) {
      local_freq = eval_at_frequency[i_eval];
      cur_best = log_posterior_by_frequency[i_eval];
      local_index = i_eval;
    }
  }
  max_ll = cur_best; // maximum value found
  max_index = local_index;
  return(local_freq);
}

void PosteriorInference::FindMaxFrequency(bool update_frequency) {
  float local_freq = ref_vs_all.FindMaxFrequency();
  // obsolete
  if (update_frequency) {
    //cout << "do not use this path: maxfreq" << endl;
    vector<float> local_freq_start = clustering.max_hyp_freq;
    ref_vs_all.UpdatePairedFrequency(local_freq_start, clustering, local_freq);
    clustering.SetHypFreq(local_freq_start);
  }
}

// assume tmp_freq has been set except for the two being checked
void ScanSpace::UpdatePairedFrequency(vector <float > &tmp_freq, FreqMaster &base_clustering, float local_freq){
  freq_pair_weight = base_clustering.max_hyp_freq[freq_pair[0]]+base_clustering.max_hyp_freq[freq_pair[1]];
  tmp_freq[freq_pair[0]] = freq_pair_weight*local_freq;
  tmp_freq[freq_pair[1]] = freq_pair_weight*(1.0f-local_freq);
}

// try one vs same relative proportions of all others
void FreqMaster::UpdateFrequencyAgainstOne(vector<float> &tmp_freq, float local_freq, int source_state){
  // in case something sops up the frequency
  // make sure we're normalized
  float freq_all_weight = 0.0f;
  for (unsigned int i_eval=0; i_eval<max_hyp_freq.size(); i_eval++)
      freq_all_weight += max_hyp_freq[i_eval];
  // source-state at local_freq
  tmp_freq[source_state] = freq_all_weight * local_freq;
  // divide up remaining 1.0f-local-freq into the other modes
  // by relative frequency of other modes
  float remaining_freq = (1.0f-local_freq)*freq_all_weight;
  float safety_zero = 0.000001f;  // if we have truly 0 frequency in all other nodes, balance out, otherwise be happy
  float remaining_weight = (freq_all_weight-max_hyp_freq[source_state]) + (max_hyp_freq.size()-1.0f)*safety_zero;
  // everything else balanced from the remainder.
   for (unsigned int i_eval=0; i_eval<max_hyp_freq.size(); i_eval++)
    if(i_eval!=(unsigned int)source_state){
      // X vs Y,Z => (1-f)*(X+Y+Z)*(Y/(Y+Z)), *(Z/(Y+Z)), with safety factor
      float fraction_remaining = (max_hyp_freq[i_eval]+safety_zero)/remaining_weight;
      tmp_freq[i_eval] =  remaining_freq * fraction_remaining;
    }
}

float ScanSpace::LogDefiniteIntegral(float _alpha, float _beta) {
  float alpha = _alpha;
  float beta = _beta;
  // guard rails
  if (alpha < 0.0f)
    alpha = 0.0f;
  if (alpha > 1.0f)
    alpha = 1.0f;
  if (beta < 0.0f)
    beta = 0.0f;
  if (beta > 1.0f)
    beta = 1.0f;
  if (alpha > beta)
    alpha = beta;

  int n_plus_one = log_posterior_by_frequency.size();
  int n_reads = n_plus_one - 1;

  float alpha_n = alpha * n_reads;
  float beta_n = beta * n_reads;

  int alpha_low = floor(alpha_n);
  int alpha_hi = ceil(alpha_n);

  int beta_low = floor(beta_n);
  int beta_hi = ceil(beta_n);

  // get relevant scale for this integral
  float local_max_ll = log_posterior_by_frequency[alpha_low];
  for (int i_eval = alpha_low; i_eval <= beta_hi; i_eval++) {
    if (local_max_ll < log_posterior_by_frequency[i_eval]) {
      local_max_ll = log_posterior_by_frequency[i_eval];
    }
  }
  // now I know my local exponential scale
  // I can compute my end-points
  float delta_alpha = alpha_n - alpha_low;
  float alpha_interpolate = exp(log_posterior_by_frequency[alpha_low] - local_max_ll) * (1.0 - delta_alpha) + exp(log_posterior_by_frequency[alpha_hi] - local_max_ll) * delta_alpha;
  float delta_beta  = beta_n - beta_low;
  float beta_interpolate = exp(log_posterior_by_frequency[beta_low] - local_max_ll) * (1.0 - delta_beta) + exp(log_posterior_by_frequency[beta_hi] - local_max_ll) * delta_beta;

  // trapezoidal rule
  float integral_sum = 0.0f;
  float distance_to_next;
  float cur_point_val;
  float cur_area;
  float old_point_val;
  // if there is a middle segment
  if (alpha_hi < beta_hi) {
    // to next integer point
    distance_to_next = alpha_hi - alpha_n;
    cur_point_val = exp(log_posterior_by_frequency[alpha_hi] - local_max_ll);
    cur_area = 0.5f * distance_to_next * (alpha_interpolate + cur_point_val);
    old_point_val = cur_point_val;
    integral_sum += cur_area; // distance to starting integer
    // interior segments
    for (int i_eval = alpha_hi + 1; i_eval <= beta_low; i_eval++) {
      cur_point_val = exp(log_posterior_by_frequency[i_eval] - local_max_ll);
      cur_area = 0.5 * (old_point_val + cur_point_val);
      integral_sum += cur_area;
      old_point_val = cur_point_val;
    }
    // final segments
    distance_to_next = beta_n - beta_low;
    cur_area = 0.5f * distance_to_next * (old_point_val + beta_interpolate);
    integral_sum += cur_area;
  }
  else {
    integral_sum = 0.5f * (beta_n - alpha_n) * (alpha_interpolate + beta_interpolate);
  }
  // now rescale back on log-scale
  float integral_log = log(integral_sum) + local_max_ll;

  if (isnan(integral_log)) {
    // trap!
    integral_log = -999999999.0f + local_max_ll;
    cout << "Warning: generated nan when integrating" << _alpha << " " << _beta << " " << n_reads << " " << local_max_ll << endl;
  }

  return(integral_log);
}


void FibInterval(vector<unsigned int> &samples, int eval_start, int detail_level) {
  int i_interval = 1;
  int i_interval_old = 0;
  int i_interval_new = 1;
  samples.resize(0);
  for (unsigned int i_eval = eval_start + 1; i_eval < (unsigned int) detail_level;) {
    samples.push_back(i_eval);
    i_eval += i_interval;
    i_interval_new = i_interval + i_interval_old;
    i_interval_old = i_interval;
    i_interval = i_interval_new;
  }
  samples.push_back((unsigned int)detail_level);
  // step down
  i_interval = 1;
  i_interval_old = 0;
  i_interval_new = 1;
  for (unsigned int i_eval = eval_start; i_eval > 0;) {
    samples.push_back(i_eval);
    i_eval -= i_interval;
    i_interval_new = i_interval + i_interval_old;
    i_interval_old = i_interval;
    i_interval = i_interval_new;
  }
  samples.push_back(0);
  std::sort(samples.begin(), samples.end());
}

unsigned int ScanSpace::ResizeToMatch(ShortStack &total_theory, unsigned max_detail_level ){
    unsigned int detail_level = total_theory.DetailLevel();
    if(max_detail_level>0) detail_level = (detail_level < max_detail_level) ? (detail_level+1) : (max_detail_level+1);
      float fdetail_level = (float) detail_level;
  log_posterior_by_frequency.resize(detail_level + 1);
  eval_at_frequency.resize(detail_level + 1);
  for (unsigned int i_eval = 0; i_eval < eval_at_frequency.size(); i_eval++) {
    eval_at_frequency[i_eval] = (float)i_eval / fdetail_level;
    } 
  return(detail_level);
}




// approximately unimodal under most pictures of the world
// scan at maximum and build up a picture of the likelihood using log(n)*constant measurements, interpolate to linearize
/*
void PosteriorInference::InterpolateFrequencyScan(ShortStack &total_theory, bool update_frequency, int strand_key) {
  
  unsigned int detail_level = ResizeToMatch(total_theory);
  
  float fdetail_level = (float) detail_level;

  UpdateMaxFreqFromResponsibility(total_theory, strand_key);

  float start_ratio = max_hyp_freq.at(freq_pair.at(0))/(max_hyp_freq.at(freq_pair.at(1))+max_hyp_freq.at(freq_pair.at(0))+0.001f); // catch divide by zero if crazy
  int eval_start = (int)(start_ratio * detail_level); // balancing the frequencies of 0 vs 1
  vector<unsigned int> samples;
  FibInterval(samples, eval_start, detail_level);

  vector<float> hyp_freq = max_hyp_freq;

  unsigned int i_last = 0;
  eval_at_frequency.at(i_last) = (float)i_last / fdetail_level;
  UpdatePairedFrequency(hyp_freq, eval_at_frequency.at(i_last));
  log_posterior_by_frequency.at(i_last) = total_theory.PosteriorFrequencyLogLikelihood(hyp_freq, prior_frequency_weight, germline_log_prior_normalization, data_reliability, strand_key);
  int bottom = log_posterior_by_frequency.at(i_last);
  int top = bottom;
  for (unsigned int i_dx = 1; i_dx < samples.size(); i_dx++) {
    unsigned int i_eval = samples.at(i_dx);
    eval_at_frequency.at(i_eval) = (float)i_eval / fdetail_level;
    UpdatePairedFrequency(hyp_freq, eval_at_frequency.at(i_eval));
    log_posterior_by_frequency.at(i_eval) = total_theory.PosteriorFrequencyLogLikelihood(hyp_freq,prior_frequency_weight, germline_log_prior_normalization, data_reliability, strand_key);
    top = log_posterior_by_frequency.at(i_eval);
    for (unsigned int i_mid = i_last + 1; i_mid < i_eval; i_mid++) {
      int delta_low = i_mid - i_last;
      int delta_hi = i_eval - i_last;
      eval_at_frequency.at(i_mid) = (float)i_mid / fdetail_level;
      log_posterior_by_frequency.at(i_mid) = (top * delta_low + bottom * delta_hi) / (delta_low + delta_hi);
    }
    bottom = top;
    i_last = i_eval;

  }
  FindMaxFrequency(update_frequency);
  scan_done = true;
}*/

void ScanSpace::DoPosteriorFrequencyScan(ShortStack &total_theory, FreqMaster &base_clustering, bool update_frequency, int strand_key, bool scan_ref, int max_detail_level) {
//    cout << "ScanningFrequency" << endl;
//    posterior frequency inference given current data/likelihood pairing
    unsigned int detail_level = ResizeToMatch(total_theory, (unsigned) max_detail_level);  // now fills in frequency

    is_scanned_.clear(); // Very important step!
    // Set the pointers for DoPosteriorFrequencyScanOneHypFreq_
    ptr_total_theory_ = &total_theory;
    ptr_base_clustering_ = &base_clustering;
    ptr_strand_key_ = &strand_key;
    ptr_scan_ref_ = &scan_ref;

    if(detail_level < min_detail_level_for_fast_scan or detail_level < (unsigned int)(1.0f / coarse_freq_resolution_)){
    	// Do full scan
        for (unsigned int i_eval = 0; i_eval < eval_at_frequency.size(); ++i_eval){
	        DoPosteriorFrequencyScanOneHypFreq_(i_eval);
        }
        FindMaxFrequency();
    }
    else{
	    DoFastScan_();
	    max_ll = max_log_posterior_scanned_;
	    max_index = (int) argmax_log_posterior_scanned_;
    }

    // if doing monomorphic eval, set frequency to begin with and don't update
    //FindMaxFrequency(update_frequency);
    //   log_posterior now contains all frequency information inferred from the data
    scan_done = true;

    // clear the pointers for DoPosteriorFrequencyScanOneHypFreq_
    ptr_total_theory_ = NULL;
    ptr_base_clustering_ = NULL;
    ptr_strand_key_ = NULL;
    ptr_scan_ref_ = NULL;
}

void ScanSpace::DoPosteriorFrequencyScanOneHypFreq_(unsigned int i_eval){
	if(not is_scanned_.empty()){
		if(is_scanned_[i_eval])
			return; // scan has been done at i_eval
	}

    vector<float> hyp_freq = ptr_base_clustering_->max_hyp_freq;
    // should scan genotypes only for dual
    if (!*ptr_scan_ref_){
	    UpdatePairedFrequency(hyp_freq, *ptr_base_clustering_, eval_at_frequency[i_eval]);
    }
	else{
	    ptr_base_clustering_->UpdateFrequencyAgainstOne(hyp_freq, eval_at_frequency[i_eval], 0);
	}
    log_posterior_by_frequency[i_eval] = ptr_total_theory_->PosteriorFrequencyLogLikelihood(hyp_freq, ptr_base_clustering_->prior_frequency_weight, ptr_base_clustering_->germline_log_prior_normalization, ptr_base_clustering_->data_reliability, *ptr_strand_key_);

	if(not is_scanned_.empty()){
		// mark as scanned
        is_scanned_[i_eval] = true;
	}
}



void PosteriorInference::UpdateMaxFreqFromResponsibility(ShortStack &total_theory, int strand_key) {
  // skip time consuming scan and use responsibilities as cluster entry
  total_theory.MultiFrequencyFromResponsibility(clustering.max_hyp_freq, clustering.prior_frequency_weight, strand_key);

  ref_vs_all.max_ll = total_theory.PosteriorFrequencyLogLikelihood(clustering.max_hyp_freq, clustering.prior_frequency_weight,clustering.germline_log_prior_normalization, clustering.data_reliability, strand_key);
  ref_vs_all.scan_done = false; // didn't come from scan
}


/*
// initialize the ensemble 
void PosteriorInference::StartAtNull(ShortStack &total_theory, bool update_frequency) {
  DoPosteriorFrequencyScan(total_theory, update_frequency, ALL_STRAND_KEY, false);
  total_theory.UpdateResponsibility(max_hyp_freq, data_reliability);
}*/



// do a hard classification as though the reads were independent
// i.e. look more like the data in the BAM file
void PosteriorInference::StartAtHardClassify(ShortStack &total_theory, bool update_frequency, const vector<float> &start_frequency) {
  // just to allocate
  ref_vs_all.ResizeToMatch(total_theory);
  if (update_frequency) {
    clustering.SetHypFreq(start_frequency);
    clustering.SetPriorStrength(start_frequency);
    ref_vs_all.max_ll = total_theory.PosteriorFrequencyLogLikelihood(clustering.max_hyp_freq, clustering.prior_frequency_weight,clustering.germline_log_prior_normalization, clustering.data_reliability, ALL_STRAND_KEY);
  } 

  total_theory.UpdateResponsibility(clustering.max_hyp_freq, clustering.data_reliability);
}

void PosteriorInference::QuickUpdateStep(ShortStack &total_theory){
    UpdateMaxFreqFromResponsibility(total_theory, ALL_STRAND_KEY);
    total_theory.UpdateResponsibility(clustering.max_hyp_freq, clustering.data_reliability); // update cluster responsibilities
}
/*
void PosteriorInference::DetailedUpdateStep(ShortStack &total_theory, bool update_frequency){
    DoPosteriorFrequencyScan(total_theory, update_frequency, ALL_STRAND_KEY, false); // update max frequency using new likelihoods -> estimate overall max likelihood
    total_theory.UpdateResponsibility(max_hyp_freq, data_reliability); // update cluster responsibilities
}*/


// functions for fast scan

// Note that i_1 can't equal i_2
void ScanSpace::LinearInterpolation_(unsigned int i_1, unsigned int i_2, unsigned int i_intp){
	if(i_intp == i_1){
		log_posterior_by_frequency[i_intp] = log_posterior_by_frequency[i_1];
	    return;
	}
	if(i_intp == i_2){
		log_posterior_by_frequency[i_intp] = log_posterior_by_frequency[i_2];
	    return;
	}
	log_posterior_by_frequency[i_intp] = log_posterior_by_frequency[i_1] + (log_posterior_by_frequency[i_2] - log_posterior_by_frequency[i_1]) * float(i_intp - i_1) / float(i_2 - i_1);
}

void ScanSpace::DoInterpolation_(){
    unsigned int scanned_idx_left = 0;
    unsigned int scanned_idx_right = 0;

    // make sure we scan the first and last
    DoPosteriorFrequencyScanOneHypFreq_(0);
    DoPosteriorFrequencyScanOneHypFreq_(eval_at_frequency.size() - 1);

    for(unsigned int i_eval = 0; i_eval < eval_at_frequency.size(); ++i_eval){
    	if(is_scanned_[i_eval]){
    		scanned_idx_left = i_eval;
    		continue;
    	}
    	while(scanned_idx_right <= i_eval or (not is_scanned_[scanned_idx_right])){
    		++scanned_idx_right;
    	}
    	LinearInterpolation_(scanned_idx_right, scanned_idx_left, i_eval);
    }
}


// Fibonacci search for finding the "local" maximum of log-posterior in the closed frequency interval [eval_at_frequency[i_left], eval_at_frequency[i_right]]
unsigned int ScanSpace::FibonacciSearchMax_(unsigned int i_left, unsigned int i_right){
    float golden_ratio_ = 1.6180339887498949f;
	unsigned int i_middle = (unsigned int) (float((float)i_right + golden_ratio_ * (float)i_left) / (1.0f + golden_ratio_));

    assert(i_left <= i_middle and i_middle <= i_right);
    return FibonacciSearchMax_(i_left, i_right, i_middle);
}

unsigned int ScanSpace::FibonacciSearchMax_(unsigned int i_left, unsigned int i_right, unsigned int i_middle){
	// Always make sure we scan these points
	DoPosteriorFrequencyScanOneHypFreq_(i_left);
	DoPosteriorFrequencyScanOneHypFreq_(i_right);
	DoPosteriorFrequencyScanOneHypFreq_(i_middle);

	// Stop criterion for the recursion
	if(i_right - i_left < 3){
		float local_max_app = max(log_posterior_by_frequency[i_left], max(log_posterior_by_frequency[i_middle], log_posterior_by_frequency[i_right]));
		if(log_posterior_by_frequency[i_left] == local_max_app)
			return i_left;
		if(log_posterior_by_frequency[i_right] == local_max_app)
			return i_right;
		if(log_posterior_by_frequency[i_middle] == local_max_app)
			return i_middle;
	}
	// Now we have i_left <= i_middle <= i_right since i_right - i_left >= 3.
	// But we need to make sure i_middle != i_left and i_middle != i_right.
	// Otherwise the recursion will never stop.
    if(i_left == i_middle){
        ++i_middle;
        DoPosteriorFrequencyScanOneHypFreq_(i_middle);
    }
    else if(i_right == i_middle){
        --i_middle;
        DoPosteriorFrequencyScanOneHypFreq_(i_middle);
    }

    // probe index
	unsigned int i_probe = i_left + i_right - i_middle; // Here is where the name "Fibonacci" come from.
	DoPosteriorFrequencyScanOneHypFreq_(i_probe);

    if(i_probe >= i_middle){
        if(log_posterior_by_frequency[i_probe] < log_posterior_by_frequency[i_middle])
            return FibonacciSearchMax_(i_left, i_probe, i_middle);
        else
            return FibonacciSearchMax_(i_middle, i_right, i_probe);
    }
    else{
        if(log_posterior_by_frequency[i_probe] > log_posterior_by_frequency[i_middle])
            return FibonacciSearchMax_(i_left, i_middle, i_probe);
        else
            return FibonacciSearchMax_(i_probe, i_right, i_middle);
    }
}

void ScanSpace::DoFineScan_(unsigned int i_left, unsigned int i_right, unsigned int i_middle){
	float penalty = 0.0f;
	float left_max = 0.0f;
	float right_max = 0.0f;

	// Always make sure we scan these points
	DoPosteriorFrequencyScanOneHypFreq_(i_left);
	DoPosteriorFrequencyScanOneHypFreq_(i_right);
	DoPosteriorFrequencyScanOneHypFreq_(i_middle);

	left_max = max(log_posterior_by_frequency[i_left], log_posterior_by_frequency[i_middle]);
	right_max = max(log_posterior_by_frequency[i_right], log_posterior_by_frequency[i_middle]);
	max_log_posterior_scanned_ = max(max_log_posterior_scanned_, max(left_max, right_max));

	if(i_middle - i_left > 1){
		// The penalty is used to obtain finer resolution around max_log_posterior_scanned_
		penalty = exp(- fine_scan_penalty_order_ * (max_log_posterior_scanned_ - left_max)); // 0 < penalty <= 1
		// if (the gap between the two points) * penalty > min_fine_log_posterior_gap_, we scan the middle of the two points
		if (abs(log_posterior_by_frequency[i_middle] - log_posterior_by_frequency[i_left]) * penalty > min_fine_log_posterior_gap_)
			DoFineScan_(i_left, i_middle,  (i_left + i_middle) / 2);
	}
	if(i_right - i_middle > 1){
		penalty = exp(- fine_scan_penalty_order_ * (max_log_posterior_scanned_ - right_max));
		if (abs(log_posterior_by_frequency[i_middle] - log_posterior_by_frequency[i_right]) * penalty > min_fine_log_posterior_gap_)
			DoFineScan_(i_middle, i_right,  (i_right + i_middle) / 2);
	}

    return;
}

// The complexity of full scan is O(N^2) where N = detail_level = (# of reads)
// Tvc can be extremely slow when detail_level is large.
// Lowering detail_level gives the complexity of scan to be O(detail_level * (# of reads)).
// However, it causes loss of resolution in log-posterior and degrades the inference results.
// Note that what tvc wants to infer from log_posterior_by_frequency are as follows:
// (a. For the EM algorithm) The maximum of log_posterior_by_frequency
// (b. For PASS/NOCALL, QUAL, GT, GQ) The definite integral of posterior_by_frequency (i.e., convert log_posterior_by_frequency to linear scale and normalize it)
// DoFastScan_() computes an approximate log_posterior_by_frequency that provides good results for both (a) and (b) without scanning all frequencies.
// In particular, the number of frequencies scanned = O(log(N)), which results the complexity of DoFastScan_() = O(N*log(N)).
// (Note 1): The underlying assumption of log_posterior_by_frequency for DoFastScan_() is that log_posterior_by_frequency is "unimodal" (at least in large scale).
// (Note 2): DoFastScan_() is a "blind" scan algorithm that has no information about log_posterior_by_frequency priorly.
// @TODO: Instead of the "blind" approach, the fast scan algorithm can be improved by taking max_hyp_freq into account.
void ScanSpace::DoFastScan_(){
	unsigned int num_scanned_freq = 0;

	// Step 0): Initialization
	unsigned int detail_level = eval_at_frequency.size() - 1;
	is_scanned_.assign(detail_level + 1, false);
	max_log_posterior_scanned_ = -999999999999.0f;  // anything is better than this
	argmax_log_posterior_scanned_ = 0;

	if(debug_){
    	cout<<"<<Fast scan start>>"<< endl;
    	cout<<"    detail_level = "<< detail_level<< endl;
    	cout<<"    # of reads = "<<ptr_total_theory_->my_hypotheses.size()<<endl;
	}


	// Step 1): Do coarse scan
	float croase_max_log_posterior = 0.0f;
	unsigned int croase_argmax_log_posterior = 0;
	unsigned int num_coarse_scan = (unsigned int) (1.0f / coarse_freq_resolution_) + 1; // also scan the last index
	unsigned int coarse_scan_spacing = detail_level / (num_coarse_scan - 1);
	vector<unsigned int> coarse_scan_indices;
	coarse_scan_indices.assign(num_coarse_scan, 0);

	// Scan the last index
	coarse_scan_indices[coarse_scan_indices.size() - 1] = detail_level;
	DoPosteriorFrequencyScanOneHypFreq_(detail_level);
	croase_max_log_posterior = log_posterior_by_frequency[detail_level];
	croase_argmax_log_posterior = coarse_scan_indices.size() - 1;

	for(unsigned int i = 0; i < num_coarse_scan - 1; ++i){
		unsigned int i_eval =  i * coarse_scan_spacing;
		coarse_scan_indices[i] = i_eval;
	    DoPosteriorFrequencyScanOneHypFreq_(i_eval);
	    if(log_posterior_by_frequency[i_eval] > croase_max_log_posterior){
	    	croase_argmax_log_posterior = i;
	    	croase_max_log_posterior = log_posterior_by_frequency[i_eval];
	    }
	}

	if(debug_){
    	unsigned int num_scanned_freq_tmp = 0;
    	for(unsigned int i_eval = 0; i_eval < is_scanned_.size(); ++i_eval){
    		num_scanned_freq_tmp += (unsigned int) is_scanned_[i_eval];
    	}
    	cout<<"    # of coarse freq scanned = "<< num_scanned_freq_tmp - num_scanned_freq<<"  (coarse_freq_resolution_= "<< coarse_freq_resolution_<< ")"<<endl;
    	num_scanned_freq = num_scanned_freq_tmp;
	}

	// Step 2): Find the maximum of log_posterior_by_frequency
	unsigned int block_size = (detail_level / num_of_fibonacci_blocks);
	for(unsigned int i = 0; i < num_of_fibonacci_blocks; ++i){
		unsigned int i_left = i * block_size;
		unsigned int i_right = (i == num_of_fibonacci_blocks - 1)? detail_level : (i_left + block_size);
		// Fibonacci search for the maximum in the block
		unsigned int i_eval = FibonacciSearchMax_(i_left, i_right);
		if(log_posterior_by_frequency[i_eval] > max_log_posterior_scanned_){
			argmax_log_posterior_scanned_ = i_eval;
			max_log_posterior_scanned_ = log_posterior_by_frequency[argmax_log_posterior_scanned_];
		}
	}
	// In case Fibonacci search returns a local maximum which is less than croase_max_log_posterior
	if(max_log_posterior_scanned_ < croase_max_log_posterior){
		// Do Fibonacci search around coarse_scan_indices[croase_argmax_log_posterior]
		unsigned int i_left = croase_argmax_log_posterior == 0? coarse_scan_indices[0] : coarse_scan_indices[croase_argmax_log_posterior - 1];
		unsigned int i_right = croase_argmax_log_posterior == coarse_scan_indices.size() - 1? coarse_scan_indices[coarse_scan_indices.size() - 1] : coarse_scan_indices[croase_argmax_log_posterior + 1];
		unsigned int i_eval = FibonacciSearchMax_(i_left, i_right);
		if(log_posterior_by_frequency[i_eval] < croase_max_log_posterior){
			// croase_max_log_posterior still beats the search results, although it shouldn't happen.
			max_log_posterior_scanned_ = croase_max_log_posterior;
			argmax_log_posterior_scanned_ = coarse_scan_indices[croase_argmax_log_posterior];
		}
		else{
			argmax_log_posterior_scanned_ = i_eval;
			max_log_posterior_scanned_ = log_posterior_by_frequency[argmax_log_posterior_scanned_];
		}
	}

	// Step 3): Determine the interval for fine scan
	// For (b. For PASS/NOCALL, QUAL, GT, GQ) The definite integral of posterior_by_frequency (i.e., convert log_posterior_by_frequency to linear scale and normalize it)),
	// what I really care about is the main mass of the posterior probability function.
	// The log-posterior(f) that is dominated by max_log_posterior_scanned_ is not important at all.
	// By letting fine_log_posterior_cutoff_gap_ = -log(r), I say that posterior posterior(f) is neglectable if posterior(f) / posterior(f_max) < r.
	// I start from the frequency at max_log_posterior_scanned_, and then increase/decrease by fine_freq_search_increment_ until I obtain
	// log_posterior(f) - max_log_posterior_scanned_ > fine_log_posterior_cutoff_gap_.
	// Then I do fine scan within the interval [fine_cutoff_i_left, fine_cutoff_i_right]
	int index_increament = max((int)((float) detail_level * fine_freq_search_increment_), 1);
	int fine_cutoff_i_left = (int) argmax_log_posterior_scanned_;
    while(fine_cutoff_i_left >= 0){
    	DoPosteriorFrequencyScanOneHypFreq_((unsigned int) fine_cutoff_i_left);
    	if(log_posterior_by_frequency[fine_cutoff_i_left] > max_log_posterior_scanned_){
    		argmax_log_posterior_scanned_ = (unsigned int) fine_cutoff_i_left;
			max_log_posterior_scanned_ = log_posterior_by_frequency[argmax_log_posterior_scanned_];
    	}
        if(max_log_posterior_scanned_ - log_posterior_by_frequency[fine_cutoff_i_left] > fine_log_posterior_cutoff_gap_)
            break;

        fine_cutoff_i_left -= index_increament;
    }
    fine_cutoff_i_left = max(fine_cutoff_i_left, 0);

	unsigned int fine_cutoff_i_right = argmax_log_posterior_scanned_;
    while(fine_cutoff_i_right <= detail_level){
    	DoPosteriorFrequencyScanOneHypFreq_(fine_cutoff_i_right);
    	if(log_posterior_by_frequency[fine_cutoff_i_right] > max_log_posterior_scanned_){
    		argmax_log_posterior_scanned_ = fine_cutoff_i_right;
			max_log_posterior_scanned_ = log_posterior_by_frequency[argmax_log_posterior_scanned_];
    	}
        if(max_log_posterior_scanned_ - log_posterior_by_frequency[fine_cutoff_i_right] > fine_log_posterior_cutoff_gap_)
            break;

        fine_cutoff_i_right += index_increament;
    }
    fine_cutoff_i_right = min(fine_cutoff_i_right, detail_level);

	if(debug_){
    	unsigned int num_scanned_freq_tmp = 0;
    	for(unsigned int i_eval = 0; i_eval < is_scanned_.size(); ++i_eval){
    		num_scanned_freq_tmp += (unsigned int) is_scanned_[i_eval];
    	}
    	cout<<"    # of freq scanned for peak finding = "<< num_scanned_freq_tmp - num_scanned_freq<<endl;
    	num_scanned_freq = num_scanned_freq_tmp;
	}


    // Step 4):
    // Do fine scan within the index interval [fine_cutoff_i_left, fine_cutoff_i_right]
    // Basically, I scan until all the adjacent scanned indices i_1, i_2 such that (log_posterior_by_frequency[i_1] - log_posterior_by_frequency[i_2]) < min_fine_log_posterior_gap_
    // There is a penalty that allows to scan less points if their log_posterior is much less than max_log_posterior_scanned_
    DoFineScan_((unsigned int) fine_cutoff_i_left, fine_cutoff_i_right, argmax_log_posterior_scanned_);

	if(debug_){
    	unsigned int num_scanned_freq_tmp = 0;
    	for(unsigned int i_eval = 0; i_eval < is_scanned_.size(); ++i_eval){
    		num_scanned_freq_tmp += (unsigned int) is_scanned_[i_eval];
    	}
    	cout<<"    # of fine freq scanned = "<< num_scanned_freq_tmp - num_scanned_freq<<endl;
    	num_scanned_freq = num_scanned_freq_tmp;
	}

    // Step 5):
    DoInterpolation_();

    // output debug message if needed
    if(debug_){
    	unsigned int num_scanned = 0;
    	for(unsigned int i_eval = 0; i_eval < is_scanned_.size(); ++i_eval){
    		num_scanned += (unsigned int) is_scanned_[i_eval];
    	}
    	cout<<"    Total # of freq scanned = "<< num_scanned<< endl;
    	cout<<"    max(log_posterior_by_frequency) = "<<max_log_posterior_scanned_<<" at "<< "eval_at_frequency["<< argmax_log_posterior_scanned_ << "] = "<< eval_at_frequency[argmax_log_posterior_scanned_]<< endl;
    	cout<<"    log_posterior_by_frequency[fine_cutoff_i_left] = "<< log_posterior_by_frequency[fine_cutoff_i_left] <<", fine_cutoff_i_left = "<< fine_cutoff_i_left<<", eval_at_frequency[fine_cutoff_i_left] = "<<  eval_at_frequency[fine_cutoff_i_left]<<endl;
    	cout<<"    log_posterior_by_frequency[fine_cutoff_i_right] = "<< log_posterior_by_frequency[fine_cutoff_i_right] <<", fine_cutoff_i_right = "<< fine_cutoff_i_right<<", eval_at_frequency[fine_cutoff_i_right] = "<<  eval_at_frequency[fine_cutoff_i_right]<<endl;
    	cout<<"<<Fast scan done>>"<< endl;
    }
    is_scanned_.clear();
}
