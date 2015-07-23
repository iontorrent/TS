/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "EvaluateKey.h"
#include "IonH5Eigen.h"
#include "ZeromerMatDiff.h"
#include <malloc.h>
#include <Eigen/Dense>
#include <Eigen/LU>

using namespace Eigen;
#define NUM_STATS_FIELDS 9
#define SNR_IDX 0
#define MAD_IDX 1
#define OK_IDX 2
#define TRACE_SD_IDX 3
#define PROJ_RES_IDX 4
#define PEAK_IDX 5
#define TAUE_IDX 6
#define TAUB_IDX 7
#define MEAN_SIG_IDX 8
#define MIN_CRAZY_PEAK_VAL 0
#define MAX_CRAZY_PEAK_VAL 2000
#define MIN_INTEGRATION_WINDOW 15
#define DEFAULT_INTEGRATION_WINDOW 20
#define MIN_SAMPLES_FOR_STATS .05
#define MIN_SNR_FOR_STATS 8
#define MIN_PEAK_FOR_STATS 35
#define MAX_PEAK_FOR_STATS 1000
#define MAX_MAD_FOR_STATS 12
#define MIN_PEAK_SEARCH_END 15
#define MAX_PEAK_WINDOW 2

EvaluateKey::EvaluateKey() {
  Init();
}

void EvaluateKey::Init() {
  m_debug = false;
  m_trace_data = m_ref_data = m_zeromer_est = m_shifted_ref = NULL;
  m_avg_0mer = m_sd_0mer = m_avg_1mer = m_sd_1mer = NULL;
  m_num_wells = m_num_well_flows = m_num_frames = m_num_flows = 0;
  m_integration_start = m_integration_end = m_peak_start = m_peak_end = 0;
  m_min_integration = 0;
  m_flow_avg_1mer = NULL;
  m_flow_count_1mer = NULL;
  m_doing_darkmatter = m_use_projection = m_peak_signal_frames = m_integration_width = m_normalize_1mers = false;
  fill(m_flow_key_avg.begin(), m_flow_key_avg.end(), 0.0f);
  m_flow_key_avg.resize(0);
}

void EvaluateKey::Alloc(size_t num_well_flows, size_t num_frames, size_t num_flows, int num_wells) {
  size_t total_size = num_well_flows * num_frames;
  if (m_num_well_flows != num_well_flows || m_num_frames != num_frames) {
    if (m_trace_data != NULL) {
      Cleanup();
    }
    m_num_wells = num_wells;
    m_num_well_flows = num_well_flows;
    m_num_frames = num_frames;
    m_num_flows = num_flows;
    m_trace_data = (float *) memalign(32, sizeof(float) * total_size);
    m_ref_data = (float *) memalign(32, sizeof(float) * total_size);
    m_zeromer_est = (float *) memalign(32, sizeof(float) * total_size);
    m_shifted_ref = (float *) memalign(32, sizeof(float) * total_size);
    m_avg_0mer = (float *) memalign(32, sizeof(float) * num_frames);
    m_sd_0mer = (float *) memalign(32, sizeof(float) * num_frames);
    m_avg_1mer = (float *) memalign(32, sizeof(float) * num_frames);
    m_sd_1mer = (float *) memalign(32, sizeof(float) * num_frames);
    m_flow_avg_1mer = (float *) memalign(32, sizeof(float) * num_frames * num_flows);
    m_flow_count_1mer = (int *) memalign(32, sizeof(int) * num_flows);
    // these are small, just for sanity initialize them
    memset(m_avg_0mer,0,sizeof(float) * num_frames);
    memset(m_sd_0mer,0,sizeof(float) * num_frames);
    memset(m_avg_1mer,0,sizeof(float) * num_frames);
    memset(m_sd_1mer,0,sizeof(float) * num_frames);
    memset(m_flow_avg_1mer, 0, sizeof(float) * num_frames * num_flows);
    memset(m_flow_count_1mer, 0, sizeof(int) * num_flows);
  }
}

void EvaluateKey::Cleanup() {
    if (m_trace_data != NULL) {
      free(m_trace_data);
      free(m_ref_data);
      free(m_zeromer_est);
      free(m_shifted_ref);
      free(m_avg_0mer);
      free(m_sd_0mer);
      free(m_avg_1mer);
      free(m_sd_1mer);
      free(m_flow_avg_1mer);
      free(m_flow_count_1mer);
      Init();
    }
  }

void EvaluateKey::SetSizes(int row_start, int row_end, 
                           int col_start, int col_end,
                           int flow_start, int flow_end,
                           int frame_start, int frame_end) {
  size_t num_wells = (row_end - row_start) * (col_end - col_start);
  size_t num_well_flows = num_wells * (flow_end - flow_start);
  size_t num_frames = frame_end - frame_start;
  size_t num_flows = flow_end - flow_start;
  Alloc( num_well_flows, num_frames, num_flows, num_wells);
  m_peak_start = m_integration_start = 0;
  m_min_integration = min((size_t)DEFAULT_INTEGRATION_WINDOW, num_frames);
  m_integration_end = min(m_num_frames, (size_t)DEFAULT_INTEGRATION_WINDOW);
  m_peak_end = min(m_num_frames, (size_t)MIN_PEAK_SEARCH_END);
  // these are small, just for sanity initialize them
  memset(m_avg_0mer,0,sizeof(float) * num_frames);
  memset(m_sd_0mer,0,sizeof(float) * num_frames);
  memset(m_avg_1mer,0,sizeof(float) * num_frames);
  memset(m_sd_1mer,0,sizeof(float) * num_frames);
}

void EvaluateKey::SetUpMatrices(TraceStoreCol &trace_store, 
                                int col_stride, int flow_stride,
                                int row_start, int row_end, int col_start, int col_end,
                                int flow_start, int flow_end,
                                int frame_start, int frame_end,
                                float *trace_data,
                                float *ref_data) {
  int row_size = row_end - row_start;
  int col_size = col_end - col_start;
  int flow_size = flow_end - flow_start;
  int frame_size = frame_end - frame_start;
  size_t local_flow_stride = row_size * col_size;
  size_t total_rows = local_flow_stride * flow_size;

  std::vector<float> trace(trace_store.GetNumFrames());
  std::vector<float> ref_trace(trace_store.GetNumFrames());

  for (int flow_ix = flow_start; flow_ix < flow_end; flow_ix++) {
    for (int row_ix = row_start; row_ix < row_end; row_ix++) {
      for (int col_ix = col_start; col_ix < col_end; col_ix++) {
        int well_ix =  row_ix * col_stride + col_ix;
        int local_well_ix = (flow_ix - flow_start) * local_flow_stride + (row_ix - row_start) * col_size + (col_ix - col_start);
        trace_store.GetReferenceTrace(well_ix, flow_ix, &ref_trace[0]);
        for (int frame_ix = frame_start; frame_ix < frame_end; frame_ix++) {
          ref_data[local_well_ix + (frame_ix - frame_start) * total_rows] = ref_trace[frame_ix];
        }
      }
      for (int frame_ix = frame_start; frame_ix < frame_end; frame_ix++) {
        int well_ix = row_ix * col_stride + col_start;
        int local_well_ix = (row_ix - row_start) * col_size;
        int16_t *__restrict store_start = trace_store.GetMemPtr() + flow_ix * trace_store.mFrameStride + frame_ix * trace_store.mFlowFrameStride + well_ix;
        int16_t *__restrict store_end = store_start + col_size;
        float *__restrict out_start = trace_data + (frame_ix - frame_start) * total_rows + (flow_ix-flow_start) * local_flow_stride + local_well_ix;
        while(store_start != store_end) {
          *out_start++ = *store_start++;
        }
      }
    }
  }
}


void EvaluateKey::FitTauB(KeySeq &key, const float *time, float taue_est, int frame_start, int frame_end, float *__restrict taub) {
  if (m_flow_order.size() >= m_num_flows) {
    ZeromerMatDiff::FitTauBNuc(&key.zeroFlows[0], key.zeroFlows.size(),
                               m_trace_data, m_shifted_ref,
                               &m_flow_order[0],
                               m_num_wells, m_num_flows, m_num_well_flows,
                               m_num_frames, taue_est, taub);
  }
  else {
    ZeromerMatDiff::FitTauB(&key.zeroFlows[0], key.zeroFlows.size(),
                            m_trace_data, m_shifted_ref,
                            m_num_wells, m_num_flows, m_num_well_flows,
                            m_num_frames, taue_est, taub);
  }
}

void EvaluateKey::PredictZeromersVec(const float *time, float taue_est, float * __restrict taub) {
  ZeromerMatDiff::PredictZeromersSignal(time, m_num_frames,
                                        m_trace_data, m_shifted_ref, m_zeromer_est,
                                        m_num_wells, m_num_flows, m_num_well_flows,
                                        taue_est, taub, "");
}

void EvaluateKey::ScoreKeySignals(KeySeq &key, float *__restrict key_signal_ptr, 
                                  int integration_start, int integration_end,
                                  int peak_start, int peak_end,
                                  float *onemer_norm,
                                  float *__restrict key_results_ptr, 
                                  int num_result_cols, float *__restrict taub, float taue_est) {
  Map<MatrixXf, Aligned> key_signal(key_signal_ptr, m_num_well_flows, m_num_frames);
  Map<MatrixXf, Aligned> key_results(key_results_ptr, m_num_wells, num_result_cols);

  // Calculate basics per well for integral, mad, max for each well/flow
  VectorXf key_integrals(m_num_well_flows), key_mad(m_num_well_flows), key_max(m_num_well_flows);
  key_integrals.setZero();
  key_mad.setZero();
  key_max.setZero();
  for (int frame_ix = peak_start; frame_ix < peak_end; frame_ix++) {
    float *__restrict key_sig_start = key_signal.data() + frame_ix * m_num_well_flows;
    //    float *__restrict one_forward = key_signal.data() + min(frame_ix+1,(int)m_num_frames-1) * m_num_well_flows;
    float *__restrict one_backward = key_signal.data() + max(frame_ix-1,0) * m_num_well_flows;
    float *__restrict key_sig_end = key_sig_start + m_num_well_flows;
    float *__restrict peak = key_max.data();
    while (key_sig_start != key_sig_end) {
      float value = (*key_sig_start + *one_backward) / 2;
      *peak = max(*peak, value);
      peak++;
      one_backward++;
      key_sig_start++;
    }
  }

  for (int frame_ix = integration_start; frame_ix < integration_end; frame_ix++) {
    key_integrals.array() += key_signal.col(frame_ix).array();
    key_mad.array() += key_signal.col(frame_ix).array().abs();
  }

  // Our summary statistics for onemers and zeromers.
  VectorXf onemer_mean(m_num_wells), onemer_m2(m_num_wells), zeromer_mean(m_num_wells), zeromer_m2(m_num_wells);
  VectorXf all_mean(m_num_wells), all_m2(m_num_wells);

  all_mean.setZero();
  all_m2.setZero();
  onemer_mean.setZero();
  onemer_m2.setZero();
  zeromer_mean.setZero();
  zeromer_m2.setZero();

  int n_onemer = 0;
  // loop through all the onemers and calculate the mean and sd
  for (size_t i = 0; i < key.onemerFlows.size(); i++) {
    n_onemer++;
    int flow_ix = key.onemerFlows[i];
    float norm_factor = 1.0f;
    if (onemer_norm != NULL) { 
      norm_factor = onemer_norm[i];
    }
    size_t offset = m_num_wells * flow_ix;
    float *__restrict integral_start = key_integrals.data() + offset;
    float *__restrict integral_end = integral_start + m_num_wells;
    float *__restrict max_key = key_max.data() + offset;
    float *__restrict mean = onemer_mean.data();
    float *__restrict m2 = onemer_m2.data();
    float *__restrict mean_peak = key_results.col(PEAK_IDX).data();
    float delta;
    while (integral_start != integral_end) {
      *integral_start = *integral_start * norm_factor;
      delta = (*integral_start) - *mean;
      *mean += delta/n_onemer;
      *m2 += delta * (*integral_start - *mean);
      *mean_peak++ += *max_key++;
      integral_start++;
      mean++;
      m2++;
    }
  }

  int n_flow = 0;
  for (size_t flow_ix = 0; flow_ix < key.usableKeyFlows; flow_ix++) {
    n_flow++;
    size_t offset = m_num_wells * flow_ix;
    float *__restrict integral_start = key_integrals.data() + offset;
    float *__restrict integral_end = integral_start + m_num_wells;
    float *__restrict mean = all_mean.data();
    float *__restrict m2 = all_m2.data();
    float delta;
    while (integral_start != integral_end) {
      delta = *integral_start - *mean;
      *mean += delta/n_flow;
      *m2 += delta * (*integral_start - *mean);
      integral_start++;
      mean++;
      m2++;
    }
  }


  int num_col = m_num_frames;
  int n_zeromer = 0;
  // loop through all the zeromers and calculate the mean and sd
  for (size_t i = 0; i < key.zeroFlows.size(); i++) {
    n_zeromer++;
    int flow_ix = key.zeroFlows[i];
    size_t offset = m_num_wells * flow_ix;
    float *__restrict integral_start = key_integrals.data() + offset;
    float *__restrict integral_end = integral_start + m_num_wells;
    float *__restrict mad = key_mad.data() + offset;
    float *__restrict mean = zeromer_mean.data();
    float *__restrict m2 = zeromer_m2.data();
    float *__restrict mean_mad = key_results.col(MAD_IDX).data();

    float delta;
    while (integral_start != integral_end) {
      delta = *integral_start - *mean;
      *mean += delta/n_zeromer;
      *m2 += delta * (*integral_start - *mean);
      *mean_mad++ += *mad++ / num_col;
      integral_start++;
      mean++;
      m2++;
    }
  }
    
  /* combine all the stats into final summaries. */
  float *__restrict snr_start = key_results.col(SNR_IDX).data();
  float *__restrict snr_end = snr_start + m_num_wells;
  float *__restrict o_mean = onemer_mean.data();
  float *__restrict o_m2 = onemer_m2.data();
  float *__restrict z_mean = zeromer_mean.data();
  float *__restrict z_m2 = zeromer_m2.data();
  float *__restrict a_m2 = all_m2.data();
  float *__restrict ok = key_results.col(OK_IDX).data();
  float *__restrict trace_sd = key_results.col(TRACE_SD_IDX).data();
  float *__restrict taub_out = key_results.col(TAUB_IDX).data();
  float *__restrict taue_out = key_results.col(TAUE_IDX).data();
  float *__restrict mean_mad = key_results.col(MAD_IDX).data();
  float *__restrict mean_peak = key_results.col(PEAK_IDX).data();
  float *__restrict onemer_sig = key_results.col(MEAN_SIG_IDX).data();
  while(snr_start != snr_end) {
    float sig = *o_mean++ - *z_mean++;
    float o_sd = sqrt(*o_m2++/n_onemer);
    float z_sd = sqrt(*z_m2++/n_zeromer);
    *onemer_sig++ = sig;
    *snr_start = sig/((o_sd +z_sd)/2.0f);
    *ok = isfinite(*snr_start) ? 1 : 0;
    *trace_sd++ = sqrt(*a_m2++ / n_flow);
    //    *trace_sd++ = sig;
    *taub_out++ = *taub++;
    *taue_out++ = taue_est;
    ok++;
    snr_start++;
    *mean_peak /= n_onemer;
    *mean_mad /= n_zeromer;
    mean_peak++;
    mean_mad++;
  }
}

void PickBestKey(std::vector<Eigen::MatrixXf> &key_results, std::vector<KeySeq> &keys, int local_index, int global_index, KeyFit &fit) {
  fit.keyIndex = -1;
  for (size_t key_ix = 0; key_ix < keys.size(); key_ix++) {
    Eigen::MatrixXf &results = key_results[key_ix];
    float snr = results.coeff(local_index, SNR_IDX);
    float peak = results.coeff(local_index, PEAK_IDX);
    if (key_ix == 0 || ((snr > fit.snr && snr > keys[key_ix].minSnr))) { // || (peak > keys[key_ix].good_enough_peak && snr > keys[key_ix].good_enough_snr))) {
      fit.snr = results.coeff(local_index, SNR_IDX);
      fit.mad = results.coeff(local_index, MAD_IDX);
      fit.ok = results.coeff(local_index, OK_IDX);
      fit.peakSig = results.coeff(local_index, PEAK_IDX);
      fit.sd = results.coeff(local_index, TRACE_SD_IDX);
      fit.tauE = results.coeff(local_index, TAUE_IDX);
      fit.tauB = results.coeff(local_index, TAUB_IDX);
      fit.onemerAvg = results.coeff(local_index, MEAN_SIG_IDX);
    }
    if (isfinite(snr) && (snr >= fit.snr && snr >= keys[key_ix].minSnr && peak >= keys[key_ix].minPeak)) { // || (peak > keys[key_ix].good_enough_peak && snr > keys[key_ix].good_enough_snr))) {
      fit.keyIndex = key_ix;
      fit.snr = snr;
    }
  }
}

void CalculateIntegrationWindow(float *avg_1mer, float *sd_1mer, float *avg_0mer, float *sd_0mer, int n_frames,
                                int min_integration, size_t &integration_start, size_t &integration_end) {
  integration_start = 0;
  integration_end = min_integration;
  while (integration_end < (size_t)n_frames) {
    float signal = avg_1mer[integration_end] - avg_0mer[integration_end];
    float noise_0mer = sd_0mer[integration_end]; // @todo cws - should we have a multiple on noise or avg with 1mer noise?
    float noise_1mer = sd_1mer[integration_end]; // @todo cws - should we have a multiple on noise or avg with 1mer noise?
    if (signal < (noise_0mer + noise_1mer) || (integration_end + 1) == (size_t) n_frames) {
      break;
    }
    integration_end++;
  }
  //  fprintf(stdout, "Finishing on frame %d with signal %.2f and noise %.2f 1mer noise %.2f 0mer signal %.2f\n", integration_end, avg_1mer[integration_end] - avg_0mer[integration_end], sd_0mer[integration_end], sd_1mer[integration_end], avg_0mer[integration_end]);
}

// @todo - cws don't use our soft filtred wells for this
int CalculateIncorporationStats(std::vector<Eigen::MatrixXf> &key_results,  
                                std::vector<Eigen::MatrixXf> &key_signals,
                                std::vector<KeySeq> &keys, 
                                int num_wells, int num_flows, int num_well_flows, int num_frames,
                                float min_snr, float min_peak,
                                float *avg_0mer, float *sd_0mer, float *avg_1mer, 
                                float *sd_1mer,
                                float *flow_avg_1mer, int *flow_count_1mer) {
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> flow_avg(flow_avg_1mer, num_flows, num_frames);
  VectorXf flow_onemer_mean(num_frames), onemer_mean(num_frames), onemer_m2(num_frames), zeromer_mean(num_frames), zeromer_m2(num_frames);
  VectorXi flow_onemer_counts(num_frames), onemer_counts(num_frames), zeromer_counts(num_frames);
  VectorXi well_counts(num_wells);
  memset(flow_count_1mer, 0, sizeof(int) * num_flows);
  onemer_mean.setZero();
  onemer_m2.setZero();
  zeromer_mean.setZero();
  zeromer_m2.setZero();
  onemer_counts.setZero();
  zeromer_counts.setZero();
  flow_avg.setZero();
  well_counts.setZero();
  int too_low_snr = 0, too_low_peak = 0, too_high_peak = 0, too_mad = 0;
  for (int flow_ix = 0; flow_ix < num_flows; flow_ix++) {
    flow_onemer_mean.setZero();
    flow_onemer_counts.setZero();
    for (int frame_ix = 0; frame_ix < num_frames; frame_ix++) {
      for (size_t key_ix = 0; key_ix < keys.size(); key_ix++) {
        float *m2 = NULL;
        float *mean = NULL;
        int *count = NULL;
        float *flow_mean = NULL;
        int *flow_count = NULL;
        KeySeq &key = keys[key_ix];
        if (key.flows[flow_ix] == 0) {
          mean = &zeromer_mean[frame_ix];
          m2 = &zeromer_m2[frame_ix];
          count = &zeromer_counts[frame_ix];
        }
        else if (key.flows[flow_ix] == 1) {
          mean = &onemer_mean[frame_ix];
          m2 = &onemer_m2[frame_ix];
          count = &onemer_counts[frame_ix];
          flow_mean = &flow_onemer_mean[frame_ix];
          flow_count = &flow_onemer_counts[frame_ix];
        }
        else {
          continue;
        }
        MatrixXf &results = key_results[key_ix];
        float *__restrict signal_start = key_signals[key_ix].data() + frame_ix * num_well_flows + num_wells * flow_ix;
        float *__restrict signal_end = signal_start + num_wells;
        float *__restrict snr = results.col(SNR_IDX).data();
        float *__restrict peak = results.col(PEAK_IDX).data();
        float *__restrict mad = results.col(MAD_IDX).data();
        int *__restrict well_counts_start =well_counts.data();
        float delta;
        while (signal_start != signal_end) {
          if (*snr >= key.minSnr ) {
            if (*snr >= min_snr && *peak >= min_peak && *peak < MAX_PEAK_FOR_STATS && *mad < MAX_MAD_FOR_STATS) {
              if (flow_mean != NULL) {
                *flow_mean += *signal_start;
                *flow_count += 1;
              }
              if (frame_ix == 0 && flow_ix == 0) {
                (*well_counts_start)++;
              }
              (*count)++;
              delta = *signal_start - *mean;
              *mean += delta / *count;
              *m2 += delta * (*signal_start - *mean);
            }
          }
          well_counts_start++;
          mad++;
          signal_start++;
          snr++;
          peak++;
        }
      }
    }
    for (int frame_ix = 0; frame_ix < num_frames; frame_ix++) {
      flow_avg(flow_ix, frame_ix) = flow_onemer_counts.coeff(frame_ix) > 0 ? flow_onemer_mean.coeff(frame_ix)/flow_onemer_counts.coeff(frame_ix) : 0.0f;
    }
    flow_count_1mer[flow_ix] = flow_onemer_counts[0];
  }
  //  fprintf(stdout, "Filters for summary stats: %d low_snr %d low_peak %d high_peak %d mad\n",too_low_snr , too_low_peak, too_high_peak, too_mad);
  for (int frame_ix = 0; frame_ix < num_frames; frame_ix++) {
    avg_0mer[frame_ix] = zeromer_mean[frame_ix];
    sd_0mer[frame_ix] = sqrt(zeromer_m2[frame_ix]/zeromer_counts[frame_ix]);
    avg_1mer[frame_ix] = onemer_mean[frame_ix];
    sd_1mer[frame_ix] = sqrt(onemer_m2[frame_ix]/onemer_counts[frame_ix]);
  }
  // let the calling function set thresholds for using resulting statistics
  return well_counts.sum();
}

void EvaluateKey::Calculate1MerNormalization(size_t num_flows, size_t num_frames,
                                             std::vector<int> &onemer_flows,
                                             float *flow_1mer_avg, int *flow_1mer_count,
                                             float *norm_factors,
                                             size_t integration_start, size_t integration_end) {

  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> flow_avg(flow_1mer_avg, num_flows, num_frames);
  std::fill(norm_factors, norm_factors + onemer_flows.size(), 1.0f);
  int onemer_count = onemer_flows.size();
  float integration_sum[onemer_count];
  float mean_integration = 0;
  int threshold = MIN_SAMPLES_FOR_STATS * m_num_wells;
  int seen = 0;
  std::fill(integration_sum, integration_sum + onemer_count, 0);
  for (int i = 0; i < onemer_count; i++) {
    if (flow_1mer_count[onemer_flows[i]] < threshold) {
      continue;
    }
    for (int frame_ix = integration_start; frame_ix < (int)integration_end; frame_ix++) {
      integration_sum[i] += flow_avg(onemer_flows[i], frame_ix);
    }
  }
  
  for (int i = 0; i < onemer_count; i++) {
    mean_integration += integration_sum[i];
  }
  
  mean_integration = mean_integration / onemer_count;
  

  if (mean_integration > 0) {
    for (int i = 0; i < onemer_count; i++) {
      norm_factors[i] = mean_integration / integration_sum[i];
    }
  }
}

void EvaluateKey::FindBestKey(int row_start, int row_end, int col_start, int col_end,
                              int frame_start, int frame_end,
                              int flow_stride, int col_stride, 
                              const float *time, std::vector<KeySeq> &keys, float taue_est, 
                              float shift, char *bad_wells,
                              std::vector<KeyFit> &key_fits) {
  Eigen::Matrix<float, Dynamic, Dynamic, AutoAlign | ColMajor> zeromer_buffer;
  // allocate some scratch buffers for our intermediate results
  std::vector<Eigen::MatrixXf> key_results(keys.size());
  std::vector<Eigen::MatrixXf> key_signals(keys.size());
  std::vector<Eigen::VectorXf> key_taub(keys.size());
  std::vector<std::vector<float> > key_norm_signals(keys.size());
  for (size_t key_ix = 0; key_ix < keys.size(); key_ix++) {
    // 8 for snr, mad, ok, traceSd, projResid, peakSig, taue, taub
    key_results[key_ix].resize(m_num_wells, NUM_STATS_FIELDS);
    key_results[key_ix].setZero(); // clean out.
    key_signals[key_ix].resize(m_num_well_flows, m_num_frames); // this will be initialized every time.
    key_taub[key_ix].resize(m_num_well_flows);
    
  }
  // for each key calculate our estimated signals m_num_well_flows, a no-op currently as we don't shift reference
  ZeromerMatDiff::ShiftReference(m_num_frames, m_num_well_flows, shift, m_ref_data, m_shifted_ref);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> trace_data(m_trace_data, m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> zeromer_est(m_zeromer_est, m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> shifted_ref(m_shifted_ref, m_num_well_flows, m_num_frames);
  for (size_t key_ix = 0; key_ix < keys.size(); key_ix++) {
    // Fit taub for each well assuming that key 0mers are 0mers
    FitTauB(keys[key_ix], time, taue_est, frame_start, frame_end, key_taub[key_ix].data());
    // Predict the zeromers
    PredictZeromersVec(time, taue_est, key_taub[key_ix].data());
    // Calculate our "signal" after subtracting estimated zeromer
    key_signals[key_ix] = trace_data - zeromer_est;
    // Normalization default to 1.0 (no normalization)
    key_norm_signals[key_ix].resize(keys[key_ix].onemerFlows.size());
    fill(key_norm_signals[key_ix].begin(), key_norm_signals[key_ix].end(), 1.0f);
    // Gather summary stats about each key fit
    ScoreKeySignals(keys[key_ix], key_signals[key_ix].data(), 
                    m_integration_start, m_integration_end,
                    m_peak_start, m_peak_end,
                    &key_norm_signals[key_ix][0],
                    key_results[key_ix].data(), key_results[key_ix].cols(), 
                    key_taub[key_ix].data(), taue_est);

  }

  // If we're doing any modifications to fitting try rescoring with new modifications
  if (m_doing_darkmatter || m_use_projection || m_peak_signal_frames || m_integration_width) {
    int num_samples = CalculateIncorporationStats(key_results, key_signals, keys,
                                                  m_num_wells, m_num_flows, m_num_well_flows, m_num_frames,
                                                  MIN_SNR_FOR_STATS, MIN_PEAK_FOR_STATS,
                                                  m_avg_0mer, m_sd_0mer, m_avg_1mer, m_sd_1mer, m_flow_avg_1mer,
                                                  m_flow_count_1mer);
    //    fprintf(stdout, "Num good samples: %d\n", num_samples);
    if (num_samples >= MIN_SAMPLES_FOR_STATS * m_num_wells) {
      Eigen::MatrixXf proj_signal;
      if (m_integration_width) {
        CalculateIntegrationWindow(m_avg_1mer, m_sd_1mer, m_avg_0mer, m_sd_0mer, m_num_frames, 
                                   MIN_INTEGRATION_WINDOW,m_integration_start, m_integration_end);
      }
      for (size_t key_ix = 0; key_ix < keys.size(); key_ix++) {
        key_results[key_ix].setZero(); // reset anything learned last time...
        
        if (m_doing_darkmatter) {
          for (int i = 0; i < key_signals[key_ix].cols(); i++) {
            key_signals[key_ix].col(i).array() -= m_avg_0mer[i];
          }
        }
        if (m_use_projection) {
          Eigen::VectorXf proj_1mer(m_num_frames);
          std::copy(m_avg_1mer, m_avg_1mer + m_num_frames, proj_1mer.data());
          proj_1mer.normalize();
          Eigen::VectorXf coefficients(key_signals[key_ix].rows());
          coefficients.setZero();
          // Should be faster to do this at the column level than row by row
          for (int i = 0; i < key_signals[key_ix].cols(); i++) {
            coefficients.array() += key_signals[key_ix].col(i).array() * proj_1mer[i];
          }
          // Replace signal with coefficient projection of 1mer
          proj_signal.resize(key_signals[key_ix].rows(),key_signals[key_ix].cols());
          for (int i = 0; i < key_signals[key_ix].cols(); i++) {
            //key_signals[key_ix].col(i).array() = coefficients.array() * proj_1mer[i];
            proj_signal.col(i).array() = coefficients.array() * proj_1mer[i];
          }
        }
        if (m_peak_signal_frames) {
          int max_frame = 0;
          float max_value = 0;
          for (int i = 0; i < (int)m_num_frames; i++) {
            if (max_value < m_avg_1mer[i]) {
              max_value = m_avg_1mer[i];
              max_frame = i;
            }
          }
          m_peak_start = max(max_frame - MAX_PEAK_WINDOW,0);
          m_peak_end = min((size_t)(max_frame+ MAX_PEAK_WINDOW),m_num_frames);
        }
        if (m_normalize_1mers) {
          float * norm_1mer = &key_norm_signals[key_ix][0];
          Calculate1MerNormalization(m_num_flows, m_num_frames, 
                                     keys[key_ix].onemerFlows,
                                     m_flow_avg_1mer, m_flow_count_1mer,
                                     norm_1mer,
                                     m_integration_start, 
                                     m_integration_end);
        }          
        Eigen::MatrixXf *signal = &key_signals[key_ix];
        if (m_use_projection) {
          signal = &proj_signal;
        }
        // rescore the key signals after all modifications
        ScoreKeySignals(keys[key_ix], signal->data(),
                        m_integration_start, m_integration_end,
                        m_peak_start, m_peak_end,
                        &(key_norm_signals[key_ix][0]),
                        key_results[key_ix].data(), key_results[key_ix].cols(), 
                        key_taub[key_ix].data(), taue_est);
      }
    }
  }

  // For each well pick the best key... 
  int col_size = col_end - col_start;
  int sum_flow_frame_stride = m_num_frames * m_num_flows;
  std::vector<float> key_flow_avg_sum(keys.size() * sum_flow_frame_stride, 0);
  m_key_counts.resize(keys.size(), 0);
  std::fill(m_key_counts.begin(), m_key_counts.end(), 0);
  m_flow_key_avg.resize(keys.size() * m_num_flows * m_num_frames);
  std::fill(m_flow_key_avg.begin(), m_flow_key_avg.end(), 0.0f);
  Eigen::Map<Eigen::VectorXf, Eigen::Aligned> dark_matter(m_avg_0mer, m_num_frames);
  for (int row_ix = row_start; row_ix < row_end; row_ix++) {
    for (int col_ix = col_start; col_ix < col_end; col_ix++) {
      int global_index = row_ix * col_stride + col_ix;
      int local_index = (row_ix - row_start) * col_size + (col_ix - col_start);
      key_fits[global_index].wellIdx = global_index;
      // @todo cws copy in the best signal here as well, with library or best key if not called though?
      PickBestKey(key_results, keys, local_index, global_index, key_fits[global_index]);
      // store the preferred zeromer and stats for chosen key if we're doing a dump of data
      int key_ix = key_fits[global_index].keyIndex;
      bool no_nan = true;
      if (key_ix >= 0) {
        for (size_t frame_ix = 0; frame_ix < m_num_frames; frame_ix++) {
          for (size_t flow_ix = 0; flow_ix < m_num_flows; flow_ix++) {
            size_t z_ix = flow_ix * m_num_wells + local_index;
            if (!isfinite(key_signals[key_ix].coeff(z_ix, frame_ix))) {
              no_nan = false;
            }
          }
        }
      }
      if (no_nan) {
        if (key_ix >= 0 && bad_wells[global_index] == 0 && 
            key_results[key_ix].coeff(local_index, PEAK_IDX) > MIN_CRAZY_PEAK_VAL &&
            key_results[key_ix].coeff(local_index, PEAK_IDX) < MAX_CRAZY_PEAK_VAL) {
          m_key_counts[key_ix]++;
          int sum_index = key_ix * sum_flow_frame_stride;
          for (size_t frame_ix = 0; frame_ix < m_num_frames; frame_ix++) {
            for (size_t flow_ix = 0; flow_ix < m_num_flows; flow_ix++) {
              size_t z_ix = flow_ix * m_num_wells + local_index;
              key_flow_avg_sum[sum_index + flow_ix * m_num_frames + frame_ix] += key_signals[key_ix].coeff(z_ix, frame_ix);
            }
          }
        }
      }
      if (m_debug) {
        key_ix = max(0,key_ix);
        for (size_t flow_ix = 0; flow_ix < m_num_flows; flow_ix++) {
          size_t z_ix = flow_ix * m_num_wells + local_index;
          zeromer_est.row(z_ix).array()  = trace_data.row(z_ix).array() - key_signals[key_ix].row(z_ix).array();
          zeromer_est.row(z_ix).array() -= dark_matter.array(); // set
        }
        // for (size_t frame_ix = 0; frame_ix < m_num_frames; frame_ix++) {
        //   zeromer_est.col(frame_ix).array() = zeromer_est.col(frame_ix).array() + m_avg_0mer[frame_ix]; // add back the dark matter, should be zeros if not using
        // }
      }
    }
  }
  for (size_t key_ix = 0; key_ix < keys.size(); key_ix++) {
    if (m_key_counts[key_ix] > 0) {
      int sum_index = key_ix * sum_flow_frame_stride;
      int sum_index_end = sum_index + sum_flow_frame_stride;
      for (int i = sum_index; i < sum_index_end; i++) {
        m_flow_key_avg[i] = key_flow_avg_sum[i] / m_key_counts[key_ix];
      }
    }
  }                                                        
}
