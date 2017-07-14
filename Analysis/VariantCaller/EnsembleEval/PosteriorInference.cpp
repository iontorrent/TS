/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PosteriorInference.h"

ScanSpace::ScanSpace(){
  scan_pair_done = false;
  scan_ref_done = false;
  freq_pair.assign(2,0);
  // reference and hypothesis
  freq_pair[1]=1;
  freq_pair[0]=0;
  freq_pair_weight = 1.0f; // everything together
  max_ll = -999999999.0f; // anything is better than this
  max_index = 0;
  min_detail_level_for_fast_scan = 2500;
  max_detail_level = 0;
  // private variables for fast scan
  is_scanned_.clear();
  max_log_posterior_scanned_ = -999999999999.0f;  // anything is better than this
  argmax_log_posterior_scanned_ = 0;

  scan_more_frequencies_ = {0.0001f, 0.0005f, 0.001f, 0.005f}; // prevent singularity at edge frequencies

  DEBUG = 0;
}

FreqMaster::FreqMaster(){
  outlier_prob = 0.001f;
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
  DEBUG = 0;
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
  double integral_sum = 0.0; // use double to calculate cumsum to enhance numerical stability.
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
    integral_sum += (double) cur_area; // distance to starting integer
    // interior segments
    for (int i_eval = alpha_hi + 1; i_eval <= beta_low; i_eval++) {
      cur_point_val = exp(log_posterior_by_frequency[i_eval] - local_max_ll);
      cur_area = 0.5f * (old_point_val + cur_point_val);
      integral_sum += (double) cur_area;
      old_point_val = cur_point_val;
    }
    // final segments
    distance_to_next = beta_n - beta_low;
    cur_area = 0.5f * distance_to_next * (old_point_val + beta_interpolate);
    integral_sum += (double) cur_area;
  }
  else {
    integral_sum = (double) (0.5f * (beta_n - alpha_n) * (alpha_interpolate + beta_interpolate));
  }
  // now rescale back on log-scale
  float integral_log = log((float) integral_sum) + local_max_ll;

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

unsigned int ScanSpace::ResizeToMatch(ShortStack &total_theory, unsigned int max_detail_level){
  unsigned int detail_level = total_theory.DetailLevel();
  if (max_detail_level > 0 and detail_level > max_detail_level){
    detail_level = max_detail_level;
  }
  float fdetail_level = (float) detail_level;
  log_posterior_by_frequency.resize(detail_level + 1);
  eval_at_frequency.resize(detail_level + 1);
  for (unsigned int i_eval = 0; i_eval < eval_at_frequency.size(); i_eval++) {
    eval_at_frequency[i_eval] = (float) i_eval / fdetail_level;
  }
  return detail_level;
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

void ScanSpace::DoFullScan_(){
	if(DEBUG > 1){
		cout << "      - Do full scan: scan all " << eval_at_frequency.size() << " frequencies."<< endl;
	}
	max_log_posterior_scanned_ = -999999999999.0f;  // anything is better than this
	argmax_log_posterior_scanned_ = 0;

	for (unsigned int i_eval = 0; i_eval < eval_at_frequency.size(); ++i_eval){
        DoPosteriorFrequencyScanOneHypFreq_(i_eval);
        UpdateMaxPosteior_(i_eval);
    }
    max_ll = max_log_posterior_scanned_;
    max_index = (int) argmax_log_posterior_scanned_;
}

void ScanSpace::DoPosteriorFrequencyScan(ShortStack &total_theory, FreqMaster &base_clustering, bool update_frequency, int strand_key, bool scan_ref) {
//    cout << "ScanningFrequency" << endl;
//    posterior frequency inference given current data/likelihood pairing
    unsigned long t0 = clock();
	unsigned int detail_level = ResizeToMatch(total_theory, max_detail_level);  // now fills in frequency

    is_scanned_.resize(0); // (Important): Always make sure is_scanned_ is empty.
    // Set the pointers for DoPosteriorFrequencyScanOneHypFreq_
    ptr_total_theory_ = &total_theory;
    ptr_base_clustering_ = &base_clustering;
    ptr_strand_key_ = &strand_key;
    ptr_scan_ref_ = &scan_ref;

    if (DEBUG > 1){
        cout << "    + Do posterior frequency scan:" << endl;
        cout << "      - Baseline allele_freq = " << PrintIteratorToString(base_clustering.max_hyp_freq.begin(), base_clustering.max_hyp_freq.end()) << endl;
        if (scan_ref) {
        	cout << "      - Scan type: ref vs. (all alt) => f = allele_freq[0]/allele_freq[1:]" << endl;
        }
        else {
        	cout << "      - Scan type: (allele "<< freq_pair[0] <<") vs. (allele "<< freq_pair[1]
				 << ") => f = allele_freq[" <<  freq_pair[0] << "]/(allele_freq[" << freq_pair[0] << "]+allele_freq[" << freq_pair[1] << "])" << endl;
        }
    }

    if(detail_level < min_detail_level_for_fast_scan or detail_level < (unsigned int)(1.0f / kCoarseFreqResolution_)){
        DoFullScan_();
    }
    else{
	    DoFastScan_();
    }
    // if doing monomorphic eval, set frequency to begin with and don't update
    //FindMaxFrequency(update_frequency);
    //   log_posterior now contains all frequency information inferred from the data
    if(scan_ref){
    	scan_ref_done = true;
    }else{
    	scan_pair_done = true;
    }

    // clear the pointers for DoPosteriorFrequencyScanOneHypFreq_
    ptr_total_theory_ = NULL;
    ptr_base_clustering_ = NULL;
    ptr_strand_key_ = NULL;
    ptr_scan_ref_ = NULL;

    if (DEBUG > 1){
        cout << "    + Posterior frequency scan done. Processing time = " << (double) (clock() - t0) / 1E6 << " sec." << endl;
        float f_resolution_debug = 0.05f;
        unsigned int num_freq_debug = (unsigned int) (1.0f / f_resolution_debug) + 1;
        if (num_freq_debug >= log_posterior_by_frequency.size()){
            cout << "      - Scan results: (f, log-posterior(f)) = ";
            for (unsigned int i_eval = 0; i_eval < log_posterior_by_frequency.size(); ++i_eval){
                cout << "(" << eval_at_frequency[i_eval] << ", " <<  log_posterior_by_frequency[i_eval] << "), ";
            }
        }
        else{
            cout << "      - Sampled scan results: (f, log-posterior(f)) = ";
            for(unsigned int i = 0; i < num_freq_debug; ++i){
            	unsigned int i_eval = (i == (num_freq_debug - 1))? eval_at_frequency.size() - 1: (unsigned int) (double(i * (log_posterior_by_frequency.size() - 1)) / double(num_freq_debug - 1));
                cout << "(" << eval_at_frequency[i_eval] << ", " <<  log_posterior_by_frequency[i_eval] << "), ";
            }
        }
        cout << endl;
        cout << "      - max(log-posterior(f)) = " << max_ll << " @ f = "<< eval_at_frequency[max_index] << endl;
    }
}

void ScanSpace::DoPosteriorFrequencyScanOneHypFreq_(unsigned int i_eval){
	if(not is_scanned_.empty()){
		if(is_scanned_[i_eval]){
			return; // scan has been done at i_eval
		}
		// I am going to scan at i_eval.
		is_scanned_[i_eval] = true;
	}

    vector<float> hyp_freq = ptr_base_clustering_->max_hyp_freq;
    // should scan genotypes only for dual
    if (!*ptr_scan_ref_){
	    UpdatePairedFrequency(hyp_freq, *ptr_base_clustering_, eval_at_frequency[i_eval]);
    }
	else{
	    ptr_base_clustering_->UpdateFrequencyAgainstOne(hyp_freq, eval_at_frequency[i_eval], 0);
	}
    log_posterior_by_frequency[i_eval] = ptr_total_theory_->PosteriorFrequencyLogLikelihood(hyp_freq, ptr_base_clustering_->prior_frequency_weight, ptr_base_clustering_->germline_log_prior_normalization, ptr_base_clustering_->outlier_prob, *ptr_strand_key_);
}



void PosteriorInference::UpdateMaxFreqFromResponsibility(ShortStack &total_theory, int strand_key) {
  // skip time consuming scan and use responsibilities as cluster entry
  total_theory.MultiFrequencyFromResponsibility(clustering.max_hyp_freq, clustering.prior_frequency_weight, strand_key);

  ref_vs_all.max_ll = total_theory.PosteriorFrequencyLogLikelihood(clustering.max_hyp_freq, clustering.prior_frequency_weight,clustering.germline_log_prior_normalization, clustering.outlier_prob, strand_key);
  ref_vs_all.scan_ref_done = false; // didn't come from scan
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
    ref_vs_all.max_ll = total_theory.PosteriorFrequencyLogLikelihood(clustering.max_hyp_freq, clustering.prior_frequency_weight,clustering.germline_log_prior_normalization, clustering.outlier_prob, ALL_STRAND_KEY);
  } 

  total_theory.UpdateResponsibility(clustering.max_hyp_freq, clustering.outlier_prob);
}

void PosteriorInference::QuickUpdateStep(ShortStack &total_theory){
    UpdateMaxFreqFromResponsibility(total_theory, ALL_STRAND_KEY);
    total_theory.UpdateResponsibility(clustering.max_hyp_freq, clustering.outlier_prob); // update cluster responsibilities

    if(DEBUG > 1){
    	cout << "    + allele_freq updated from responsibility "<< endl
    		 << "      - allele_freq = " << PrintIteratorToString(clustering.max_hyp_freq.begin(), clustering.max_hyp_freq.end()) << endl
    	     << "      - ref_vs_all.max_ll = " << ReturnJustLL() << endl;
    }
}
/*
void PosteriorInference::DetailedUpdateStep(ShortStack &total_theory, bool update_frequency){
    DoPosteriorFrequencyScan(total_theory, update_frequency, ALL_STRAND_KEY, false); // update max frequency using new likelihoods -> estimate overall max likelihood
    total_theory.UpdateResponsibility(max_hyp_freq, data_reliability); // update cluster responsibilities
}*/

void ScanSpace::DoInterpolation_(){
    unsigned int scanned_idx_left = 0;
    unsigned int scanned_idx_right = 0;

    // Always make sure we scan the first and last
    DoPosteriorFrequencyScanOneHypFreq_(0);
    DoPosteriorFrequencyScanOneHypFreq_(eval_at_frequency.size() - 1);

    while(scanned_idx_left < eval_at_frequency.size() - 1){
    	scanned_idx_right = scanned_idx_left + 1;
    	while(not is_scanned_[scanned_idx_right]){
    		++scanned_idx_right;
    	}
    	float delta_idx = (float) (scanned_idx_right - scanned_idx_left);
		for (unsigned int i_eval = scanned_idx_left + 1; i_eval < scanned_idx_right; ++i_eval){
			float alpha = (float) (i_eval - scanned_idx_left) / delta_idx;
			// Do linear interpolation here.
			// My experiments show that doing interpolation in log-domain has better result.
			// Although quadratic interpolation (i.e., assume posterior is Gaussian) has lower interpolation error, the linear interpolation is good enough.
			log_posterior_by_frequency[i_eval] = alpha * log_posterior_by_frequency[scanned_idx_right] + (1.0f - alpha) * log_posterior_by_frequency[scanned_idx_left];
		}
    	scanned_idx_left = scanned_idx_right;
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
	UpdateMaxPosteior_(i_left);
	UpdateMaxPosteior_(i_right);
	UpdateMaxPosteior_(i_middle);

	left_max = max(log_posterior_by_frequency[i_left], log_posterior_by_frequency[i_middle]);
	right_max = max(log_posterior_by_frequency[i_right], log_posterior_by_frequency[i_middle]);

	if(i_middle - i_left > 1){
		// The penalty is used to obtain finer resolution around max_log_posterior_scanned_
		penalty = exp(- kFineScanPenaltyOrder_ * (max_log_posterior_scanned_ - left_max)); // 0 < penalty <= 1
		// if (the gap between the two points) * penalty > min_fine_log_posterior_gap_, we scan the middle of the two points
		if (abs(log_posterior_by_frequency[i_middle] - log_posterior_by_frequency[i_left]) * penalty > kMinFineLogPosteriorGap_)
			DoFineScan_(i_left, i_middle,  (i_left + i_middle) / 2);
	}
	if(i_right - i_middle > 1){
		penalty = exp(- kFineScanPenaltyOrder_ * (max_log_posterior_scanned_ - right_max));
		if (abs(log_posterior_by_frequency[i_middle] - log_posterior_by_frequency[i_right]) * penalty > kMinFineLogPosteriorGap_)
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
// I will scan three neighborhood of frequencies:
// (a): The neighborhood around the peak of posterior_by_frequency
// (b): The neighborhoods around min_allele_freq and 1 - min_allele_freq
// (c): The neighborhoods around f = 0.0 and f = 1.0 (In case singularity in the edge cases)
// In particular, the number of frequencies scanned = O(log(N)), which results the complexity of DoFastScan_() = O(N*log(N)).
// (Note 1): The underlying assumption of log_posterior_by_frequency for DoFastScan_() is that log_posterior_by_frequency is "unimodal" (at least in large scale).
// (Note 2): DoFastScan_() is a "blind" scan algorithm that has no information about log_posterior_by_frequency priorly.
// (Note 3): The interpolation error should be neglectable in liner scale, while the error may show up in the log-domain (i.e., in "large" QUAL).
void ScanSpace::DoFastScan_(){
	unsigned int num_scanned_freq = 0;

	// Step 0): Initialization
	unsigned int num_coarse_scan = (unsigned int) (1.0f / kCoarseFreqResolution_) + 1; // also scan the last index
	// Is it necessarily to do fast scan?
	if( 0.9f * (float) eval_at_frequency.size() <= (float) num_coarse_scan){
		DoFullScan_();
		return;
	}

	unsigned int detail_level = eval_at_frequency.size() - 1;
	is_scanned_.assign(detail_level + 1, false);
	max_log_posterior_scanned_ = -999999999999.0f;  // anything is better than this
	argmax_log_posterior_scanned_ = 0;

	if(DEBUG > 1){
		if(ptr_total_theory_->GetIsMolecularTag()){
			cout << "      - Do fast scan (detail_level = "<< detail_level << ", number of functional families = "<< ptr_total_theory_->GetNumFuncFamilies() << ")" << endl;
		}
		else{
			cout << "      - Do fast scan (detail_level = "<< detail_level << ", number of valid reads = "<< ptr_total_theory_->valid_indexes.size() << ")" << endl;
		}
	}

	// Step 1): Do coarse scan
	float croase_max_log_posterior = 0.0f;
	unsigned int croase_argmax_log_posterior = 0;
	vector<unsigned int> coarse_scan_indices;
	coarse_scan_indices.assign(num_coarse_scan, 0);

	// Scan the last index
	coarse_scan_indices[coarse_scan_indices.size() - 1] = detail_level;
	DoPosteriorFrequencyScanOneHypFreq_(detail_level);
	croase_max_log_posterior = log_posterior_by_frequency[detail_level];
	croase_argmax_log_posterior = coarse_scan_indices.size() - 1;

	for(unsigned int i = 0; i < num_coarse_scan - 1; ++i){
		unsigned int i_eval =  (unsigned int) (double(i * (detail_level)) / double(num_coarse_scan - 1));
		coarse_scan_indices[i] = i_eval;
	    DoPosteriorFrequencyScanOneHypFreq_(i_eval);
	    if(log_posterior_by_frequency[i_eval] > croase_max_log_posterior){
	    	croase_argmax_log_posterior = i;
	    	croase_max_log_posterior = log_posterior_by_frequency[i_eval];
	    }
	}

	// Step 2): Find the maximum of log_posterior_by_frequency
	for(unsigned int i = 0; i < kNumOfFibonacciBlocks; ++i){
		unsigned int i_left = (unsigned int) (double(i * detail_level) / double(kNumOfFibonacciBlocks));
		unsigned int i_right = (i == kNumOfFibonacciBlocks - 1)? detail_level : (unsigned int) (double((i + 1) * detail_level) / double(kNumOfFibonacciBlocks));
		// Fibonacci search for the maximum in the block
		unsigned int i_eval = FibonacciSearchMax_(i_left, i_right);
		UpdateMaxPosteior_(i_eval);
	}
	// In case Fibonacci search returns a local maximum which is less than croase_max_log_posterior
	if(max_log_posterior_scanned_ < croase_max_log_posterior){
		// Do Fibonacci search around coarse_scan_indices[croase_argmax_log_posterior]
		unsigned int i_left = croase_argmax_log_posterior == 0? coarse_scan_indices[0] : coarse_scan_indices[croase_argmax_log_posterior - 1];
		unsigned int i_right = croase_argmax_log_posterior == coarse_scan_indices.size() - 1? coarse_scan_indices[coarse_scan_indices.size() - 1] : coarse_scan_indices[croase_argmax_log_posterior + 1];
		unsigned int i_eval = FibonacciSearchMax_(i_left, i_right);
		UpdateMaxPosteior_(croase_argmax_log_posterior); // in case croase_max_log_posterior still beats the search results, although it shouldn't happen.
		UpdateMaxPosteior_(i_eval);
	}

	if(DEBUG > 1){
		cout << "      - Fibonacci search found max(log-posterior(f)) = "<< max_log_posterior_scanned_<< " @ f = eval_at_frequency["<< argmax_log_posterior_scanned_ << "] = " << eval_at_frequency[argmax_log_posterior_scanned_] << endl;
	}

    // Step 3):
    // Scan the frequencies at scan_more_frequencies to enhance the accuracy.
    // The entries in scan_more_frequencies are obtained from: a) edge frequencies, b) min-allele-freq
    for (unsigned int i_freq = 0; i_freq < scan_more_frequencies_.size(); ++i_freq){
		unsigned int i_eval = (unsigned int) ( (double) detail_level * (double) scan_more_frequencies_[i_freq]);
	    DoPosteriorFrequencyScanOneHypFreq_(i_eval); // scan freq
	    UpdateMaxPosteior_(i_eval);
	    DoPosteriorFrequencyScanOneHypFreq_(detail_level - i_eval);  // scan 1.0 - freq
	    UpdateMaxPosteior_(detail_level - i_eval);
    }
    // Scan more edge frequencies to prevent singularity
    unsigned int scan_more_edge_index = (eval_at_frequency.size() < 8)? eval_at_frequency.size() : 8;
    for (unsigned int i_eval = 0; i_eval < scan_more_edge_index; ++i_eval){
	    DoPosteriorFrequencyScanOneHypFreq_(i_eval);
	    UpdateMaxPosteior_(i_eval);
	    DoPosteriorFrequencyScanOneHypFreq_(detail_level - i_eval);
	    UpdateMaxPosteior_(detail_level - i_eval);
    }

	// Step 4): Determine the interval for fine scan
	// For (b. For PASS/NOCALL, QUAL, GT, GQ) The definite integral of posterior_by_frequency (i.e., convert log_posterior_by_frequency to linear scale and normalize it)),
	// what I really care about is the main mass of the posterior probability function.
	// The log-posterior(f) that is dominated by max_log_posterior_scanned_ is not important at all.
	// By letting fine_log_posterior_cutoff_gap_ = -log(r), I say that posterior posterior(f) is neglectable if posterior(f) / posterior(f_max) < r.
	// I start from the frequency at max_log_posterior_scanned_, and then increase/decrease by fine_freq_search_increment_ until I obtain
	// log_posterior(f) - max_log_posterior_scanned_ > fine_log_posterior_cutoff_gap_.
	// Then I do fine scan within the interval [fine_cutoff_i_left, fine_cutoff_i_right]
	int index_increament = max((int)((float) detail_level * kFineFreqSearchIncrement_), 1);
	int fine_cutoff_i_left = (int) argmax_log_posterior_scanned_;
    while(fine_cutoff_i_left >= 0){
    	DoPosteriorFrequencyScanOneHypFreq_((unsigned int) fine_cutoff_i_left);
    	UpdateMaxPosteior_((unsigned int) fine_cutoff_i_left);
        if(max_log_posterior_scanned_ - log_posterior_by_frequency[fine_cutoff_i_left] > kFineLogPosteriorCutoffGap_)
            break;

        fine_cutoff_i_left -= index_increament;
    }
    fine_cutoff_i_left = max(fine_cutoff_i_left, 0);

	unsigned int fine_cutoff_i_right = argmax_log_posterior_scanned_;
    while(fine_cutoff_i_right <= detail_level){
    	DoPosteriorFrequencyScanOneHypFreq_(fine_cutoff_i_right);
    	UpdateMaxPosteior_((unsigned int) fine_cutoff_i_right);
        if(max_log_posterior_scanned_ - log_posterior_by_frequency[fine_cutoff_i_right] > kFineLogPosteriorCutoffGap_)
            break;

        fine_cutoff_i_right += index_increament;
    }
    fine_cutoff_i_right = min(fine_cutoff_i_right, detail_level);

	if(DEBUG > 1){
		cout << "      - Do fine scan in the frequency interval ["<< eval_at_frequency[fine_cutoff_i_left] << ", " <<eval_at_frequency[fine_cutoff_i_right] <<"]"
			 << ", where log-posterior(f="<< eval_at_frequency[fine_cutoff_i_left] << ") = "<< log_posterior_by_frequency[fine_cutoff_i_left]
			 << ", log-posterior(f="<< eval_at_frequency[fine_cutoff_i_right] << ") = "<< log_posterior_by_frequency[fine_cutoff_i_right] << endl;
	}

    // Step 5):
    // Do fine scan within the index interval [fine_cutoff_i_left, fine_cutoff_i_right]
    // Basically, I scan until all the adjacent scanned indices i_1, i_2 such that (log_posterior_by_frequency[i_1] - log_posterior_by_frequency[i_2]) < min_fine_log_posterior_gap_
    // There is a penalty that allows to scan less points if their log_posterior is much less than max_log_posterior_scanned_
    DoFineScan_((unsigned int) fine_cutoff_i_left, fine_cutoff_i_right, argmax_log_posterior_scanned_);

    // Step 6):
    // Do interpolation
    DoInterpolation_();

    // output debug message if needed
    if (DEBUG > 1){
    	unsigned int num_scanned = 0;
    	for(unsigned int i_eval = 0; i_eval < is_scanned_.size(); ++i_eval){
    		num_scanned += (unsigned int) is_scanned_[i_eval];
    	}
    	cout<< "      - Fast scan done. Number of frequencies scanned = " << num_scanned << endl;
    }
    if (DEBUG > 2){
    	cout << "        + Fast scan results:" << endl;
    	cout << "          - eval_at_frequency = linspace(0, 1, "<< eval_at_frequency.size() << ")"<< endl;
    	cout << "          - scanned_index = [";
    	for(unsigned int i_eval = 0; i_eval < is_scanned_.size(); ++i_eval)
    	    if(is_scanned_[i_eval]) {cout << i_eval << ", ";}
    	cout << "]" << endl;
    	cout << "          - scanned_log_posterior_by_frequency = [";
    	for(unsigned int i_eval = 0; i_eval < is_scanned_.size(); ++i_eval)
    	    if(is_scanned_[i_eval]) {cout << log_posterior_by_frequency[i_eval] << ", ";}
    	cout << "]" << endl;
    }
    is_scanned_.resize(0);
    max_ll = max_log_posterior_scanned_;
    max_index = (int) argmax_log_posterior_scanned_;
}

void ScanSpace::UpdateMaxPosteior_(unsigned int i_eval){
	if (log_posterior_by_frequency[i_eval] > max_log_posterior_scanned_){
		argmax_log_posterior_scanned_ = i_eval;
		max_log_posterior_scanned_ = log_posterior_by_frequency[argmax_log_posterior_scanned_];
	}
}


void ScanSpace::SetTargetMinAlleleFreq(const ExtendParameters& my_param, const vector<VariantSpecificParams>& variant_specific_params){
	vector<float> all_min_allele_freq;
	if (my_param.program_flow.is_multi_min_allele_freq){
		all_min_allele_freq = my_param.program_flow.multi_min_allele_freq;
	}

	all_min_allele_freq.reserve(all_min_allele_freq.size() + 4);
	if (my_param.my_controls.use_fd_param){
		all_min_allele_freq = {my_param.my_controls.filter_fd_0.min_allele_freq, my_param.my_controls.filter_fd_5.min_allele_freq, my_param.my_controls.filter_fd_10.min_allele_freq};
	}else{
		all_min_allele_freq = {my_param.my_controls.filter_snp.min_allele_freq, my_param.my_controls.filter_mnp.min_allele_freq, my_param.my_controls.filter_hp_indel.min_allele_freq};
	}
	if (not my_param.my_controls.hotspots_as_de_novo){
		all_min_allele_freq.push_back(my_param.my_controls.filter_hotspot.min_allele_freq);
	}

	for (unsigned int i_allele = 0; i_allele < variant_specific_params.size(); ++i_allele){
		if (variant_specific_params[i_allele].min_allele_freq_override){
			all_min_allele_freq.push_back(variant_specific_params[i_allele].min_allele_freq);
		}
	}

	int num_steps = 4;
	for (vector<float>::iterator f_it = all_min_allele_freq.begin(); f_it != all_min_allele_freq.end(); ++f_it){
		scan_more_frequencies_.push_back(*f_it);
		float maf_plus = *f_it + kCoarseFreqResolution_;
		maf_plus = (maf_plus > 1.0f) ? 1.0f : maf_plus;
		for (int i_step = 0; i_step < num_steps; ++i_step){
			maf_plus = (*f_it + maf_plus) * 0.5f;
			scan_more_frequencies_.push_back(min(maf_plus, 1.0f)); // always make sure not scan a frequency > 1.0f
		}
		float maf_minus = *f_it - kCoarseFreqResolution_;
		maf_minus = (maf_minus < 0.0f) ? 0.0f : maf_minus;
		for (int i_step = 0; i_step < num_steps; ++i_step){
			maf_minus = (*f_it + maf_minus) * 0.5f;
			scan_more_frequencies_.push_back(max(maf_minus, 0.0f)); // always make sure not scan a negative frequency
		}
	}
}

