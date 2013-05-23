/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PosteriorInference.h"



PosteriorInference::PosteriorInference() {
  max_freq = 0.5f;  // sure, go ahead and let there be variants
  max_ll = -999999999.0f; // anyting is better than this
  max_index = 0;
  params_ll = 0.0f;
  scan_done = false;
  data_reliability = 1.0f-0.001f;
}


void PosteriorInference::FindMaxFrequency(bool update_frequency) {
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
  if (update_frequency) {
    max_freq = local_freq;
  }
}

float PosteriorInference::LogDefiniteIntegral(float _alpha, float _beta) {
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


void FibInterval(vector<unsigned int> &samples, int eval_start, int num_reads) {
  int i_interval = 1;
  int i_interval_old = 0;
  int i_interval_new = 1;
  samples.resize(0);
  for (unsigned int i_eval = eval_start + 1; i_eval < (unsigned int) num_reads;) {
    samples.push_back(i_eval);
    i_eval += i_interval;
    i_interval_new = i_interval + i_interval_old;
    i_interval_old = i_interval;
    i_interval = i_interval_new;
  }
  samples.push_back((unsigned int)num_reads);
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

unsigned int PosteriorInference::ResizeToMatch(ShortStack &total_theory){
    unsigned int num_reads = total_theory.my_hypotheses.size();
  log_posterior_by_frequency.resize(num_reads + 1);
  eval_at_frequency.resize(num_reads + 1);
  return(num_reads);
}


// approximately unimodal under most pictures of the world
// scan at maximum and build up a picture of the likelihood using log(n)*constant measurements, interpolate to linearize
void PosteriorInference::InterpolateFrequencyScan(ShortStack &total_theory, bool update_frequency, int strand_key) {
  
  unsigned int num_reads = ResizeToMatch(total_theory);
  
  float fnum_reads = (float) num_reads;

  UpdateMaxFreqFromResponsibility(total_theory, strand_key);

  int eval_start = (int)(max_freq * num_reads);
  vector<unsigned int> samples;
  FibInterval(samples, eval_start, num_reads);
  unsigned int i_last = 0;
  eval_at_frequency[i_last] = (float)i_last / fnum_reads;
  log_posterior_by_frequency[i_last] = total_theory.PosteriorFrequencyLogLikelihood(eval_at_frequency[i_last], data_reliability, strand_key);
  int bottom = log_posterior_by_frequency[i_last];
  int top = bottom;
  for (unsigned int i_dx = 1; i_dx < samples.size(); i_dx++) {
    unsigned int i_eval = samples[i_dx];
    eval_at_frequency[i_eval] = (float)i_eval / fnum_reads;
    log_posterior_by_frequency[i_eval] = total_theory.PosteriorFrequencyLogLikelihood(eval_at_frequency[i_eval],data_reliability, strand_key);
    top = log_posterior_by_frequency[i_eval];
    for (unsigned int i_mid = i_last + 1; i_mid < i_eval; i_mid++) {
      int delta_low = i_mid - i_last;
      int delta_hi = i_eval - i_last;
      eval_at_frequency[i_mid] = (float)i_mid / fnum_reads;
      log_posterior_by_frequency[i_mid] = (top * delta_low + bottom * delta_hi) / (delta_low + delta_hi);
    }
    bottom = top;
    i_last = i_eval;

  }
  FindMaxFrequency(update_frequency);
  scan_done = true;

};

void PosteriorInference::DoPosteriorFrequencyScan(ShortStack &total_theory, bool update_frequency, int strand_key) {
//cout << "ScanningFrequency" << endl;
// posterior frequency inference given current data/likelihood pairing
  unsigned int num_reads = ResizeToMatch(total_theory);

  float fnum_reads = (float) num_reads;
  for (unsigned int i_eval = 0; i_eval < eval_at_frequency.size(); i_eval++) {
    eval_at_frequency[i_eval] = (float)i_eval / fnum_reads;
    log_posterior_by_frequency[i_eval] = total_theory.PosteriorFrequencyLogLikelihood(eval_at_frequency[i_eval], data_reliability, strand_key);
  }
  // if doing monomorphic eval, set frequency to begin with and don't update
  FindMaxFrequency(update_frequency);
  scan_done = true;
// log_posterior now contains all frequency information inferred from the data
}

void PosteriorInference::UpdateMaxFreqFromResponsibility(ShortStack &total_theory, int strand_key) {
  // skip time consuming scan and use responsibilities as cluster entry
  float max_freq = total_theory.FrequencyFromResponsibility(strand_key);
  max_ll = total_theory.PosteriorFrequencyLogLikelihood(max_freq, data_reliability, strand_key);
  scan_done = false; // didn't come from scan
}


// initialize the ensemble 
void PosteriorInference::StartAtNull(ShortStack &total_theory, bool update_frequency) {
  DoPosteriorFrequencyScan(total_theory, update_frequency, ALL_STRAND_KEY);
  total_theory.UpdateResponsibility(max_freq, data_reliability);
}

// do a hard classification as though the reads were independent
// i.e. look more like the data in the BAM file
void PosteriorInference::StartAtHardClassify(ShortStack &total_theory, bool update_frequency, float start_frequency) {
  // just to allocate
  ResizeToMatch(total_theory);
  if (update_frequency) {
    max_freq = start_frequency;
    max_ll = total_theory.PosteriorFrequencyLogLikelihood(max_freq, data_reliability, ALL_STRAND_KEY);
  } 
  total_theory.UpdateResponsibility(max_freq, data_reliability);
}

void PosteriorInference::QuickUpdateStep(ShortStack &total_theory){
     UpdateMaxFreqFromResponsibility(total_theory, ALL_STRAND_KEY);
    total_theory.UpdateResponsibility(max_freq, data_reliability); // update cluster responsibilities
}

void PosteriorInference::DetailedUpdateStep(ShortStack &total_theory, bool update_frequency){
    DoPosteriorFrequencyScan(total_theory, update_frequency, ALL_STRAND_KEY); // update max frequency using new likelihoods -> estimate overall max likelihood
    total_theory.UpdateResponsibility(max_freq, data_reliability); // update cluster responsibilities
}