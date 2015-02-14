/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PosteriorInference.h"

ScanSpace::ScanSpace(){
  scan_done = false;
  freq_pair.assign(2,0);
  // reference and hypothesis
  freq_pair[1]=1;
  freq_pair[0]=0;
  freq_pair_weight = 1.0f; // everything together
  max_ll = -999999999.0f; // anyting is better than this
  max_index = 0;
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
    unsigned int detail_level = total_theory.my_hypotheses.size();
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
//cout << "ScanningFrequency" << endl;
// posterior frequency inference given current data/likelihood pairing
  unsigned int detail_level = ResizeToMatch(total_theory, (unsigned) max_detail_level);  // now fills in frequency
  // local scan size 2
  vector<float> hyp_freq = base_clustering.max_hyp_freq;
  // should scan genotypes only for dual
  for (unsigned int i_eval = 0; i_eval < eval_at_frequency.size(); i_eval++) {
  
    if (!scan_ref)
      UpdatePairedFrequency(hyp_freq,base_clustering, eval_at_frequency[i_eval]);
    else
      base_clustering.UpdateFrequencyAgainstOne(hyp_freq, eval_at_frequency[i_eval],0);

    log_posterior_by_frequency[i_eval] = total_theory.PosteriorFrequencyLogLikelihood(hyp_freq, base_clustering.prior_frequency_weight,base_clustering.germline_log_prior_normalization, base_clustering.data_reliability, strand_key);
  }
  // if doing monomorphic eval, set frequency to begin with and don't update
  //FindMaxFrequency(update_frequency);
  FindMaxFrequency();
  scan_done = true;
// log_posterior now contains all frequency information inferred from the data
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
