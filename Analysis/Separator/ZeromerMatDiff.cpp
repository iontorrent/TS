/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Eigen/Dense>
#include <Eigen/LU>
#include "ZeromerMatDiff.h"
#include "H5Eigen.h"
#include "SampleStats.h"
#include "SampleQuantiles.h"

void ZeromerMatDiff::ShiftReference(int n_frames, size_t n_flow_wells, float shift,
                                    float *orig_data, float *shifted_data) {
  if (shift == 0.0f) {
    memcpy(shifted_data, orig_data, sizeof(float) * n_flow_wells * n_frames);
    return;
  }
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> orig(orig_data, n_flow_wells, n_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> shifted(shifted_data, n_flow_wells, n_frames);
  for (int frame_ix = 0; frame_ix < n_frames; frame_ix++) {
    int start_frame = floor(frame_ix + shift);
    int end_frame = ceil(frame_ix + shift);
    //    if (shift < 0) { std::swap(start_frame, end_frame); }
    if (start_frame >= 0 && end_frame < n_frames) { 
      //interpolate...
      float mult = shift - floor(shift);
      shifted.col(frame_ix).array() = (orig.col(end_frame).array() - orig.col(start_frame).array()) * mult;
      shifted.col(frame_ix).array() += orig.col(start_frame).array();
    }
    else {
      // extrapolate backwards
      if (shift < 0) {
        // calculate slope
        shifted.col(frame_ix).array() = (orig.col(1).array() - orig.col(0).array());
        shifted.col(frame_ix).array() = orig.col(0).array() + shifted.col(frame_ix).array() * (frame_ix + shift);
      }
      // extrapolate forwards
      else if (shift > 0) {
        shifted.col(frame_ix).array() = (orig.col(n_frames - 1).array() - orig.col(n_frames - 2).array());
        shifted.col(frame_ix).array() = orig.col(n_frames - 1).array() + shifted.col(frame_ix).array() * (frame_ix + shift - (n_frames - 1));
      }
      else {
        assert(0);
      }
    }
  }
}

void ZeromerMatDiff::PredictZeromersSignal(const float *time, int n_frames,
                                           float *trace, float *ref, float *zeromer,
                                           size_t n_wells, size_t n_flows, 
                                           size_t n_flow_wells, 
                                           float taue_est, float *__restrict taub,
                                           const std::string &h5_dump) {
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> trace_data(trace, n_flow_wells, n_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> ref_data(ref, n_flow_wells, n_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> zeromer_est(zeromer, n_flow_wells, n_frames);
  zeromer_est.setZero();
  Eigen::Map<Eigen::VectorXf, Eigen::Aligned> taub_v(taub, n_flow_wells);
  Eigen::VectorXf cdelta(n_flow_wells);
  cdelta.setZero();
  for (int f_ix = 1; f_ix < trace_data.cols(); f_ix++) {
    float dtime = time[f_ix] - time[f_ix -1];
    zeromer_est.col(f_ix).array() = ref_data.col(f_ix).array() * (taue_est + dtime);
    zeromer_est.col(f_ix) = zeromer_est.col(f_ix) + cdelta;
    zeromer_est.col(f_ix).array() = zeromer_est.col(f_ix).array() / (taub_v.array() + dtime);
    cdelta.array() += ref_data.col(f_ix).array() - zeromer_est.col(f_ix).array();
  }
  if (!h5_dump.empty()) {
    H5File h5(h5_dump);
    h5.Open(true);
    Eigen::MatrixXf trace_data_tmp = trace_data;
    Eigen::MatrixXf ref_data_tmp = ref_data;
    Eigen::MatrixXf zeromer_est_tmp = zeromer_est;
    H5Eigen::WriteMatrix(h5, "/traces", trace_data_tmp);
    H5Eigen::WriteMatrix(h5, "/ref", ref_data_tmp);
    H5Eigen::WriteMatrix(h5, "/shifted_ref", ref_data_tmp);
    H5Eigen::WriteMatrix(h5, "/zeromer", zeromer_est_tmp);
    h5.Close();
  }
  //  zeromer_est = trace_data - zeromer_est;
}

void ZeromerMatDiff::ZeromerMadError(const int *zero_flows, size_t n_zero_flows, 
                                     float *signal_data, 
                                     size_t n_wells, size_t n_flows, 
                                     size_t n_flow_wells, size_t n_frames, 
                                     float &mad) {
  // Calculate basics per well for integral, mad, max for each well/flow
  SampleStats<double> mad_mean;
  for (size_t z_ix = 0; z_ix < n_zero_flows; z_ix++) {
    size_t flow_ix = zero_flows[z_ix];
    for (size_t col_ix = 0; col_ix < n_frames; col_ix++) {
      float *__restrict signal_start = signal_data + n_wells * flow_ix + n_flow_wells * col_ix;
      float *__restrict signal_end = signal_start + n_wells;
      while (signal_start != signal_end) {
        mad_mean.AddValue(fabs(*signal_start));
        signal_start++;
      }
    }
  }
  mad = mad_mean.GetMean();
}


void ZeromerMatDiff::ZeromerSumSqErrorTrim(const int *zero_flows, size_t n_zero_flows, 
                                           const char *bad_wells,
                                           float *signal_data, float *predict_data, 
                                           size_t n_wells, size_t n_flows, 
                                           size_t n_flow_wells, size_t n_frames, 
                                           double &ssq) {
  double ssq_sum = 0;
  size_t bad_count = 0;
  float max_value = 1000.0f;
  for (size_t z_ix = 0; z_ix < n_zero_flows; z_ix++) {
    size_t flow_ix = zero_flows[z_ix];
    for (size_t col_ix = 0; col_ix < n_frames; col_ix++) {
      float *__restrict signal_start = signal_data + n_wells * flow_ix + n_flow_wells * col_ix;
      float *__restrict signal_end = signal_start + n_wells;
      float *__restrict predict_start = predict_data + n_wells * flow_ix + n_flow_wells * col_ix;
      const char *__restrict bad_start = bad_wells;
      while (signal_start != signal_end) {
        double value = *signal_start - *predict_start; //*signal_start * *signal_start;
        if (*bad_start == 0 && isfinite(value) && fabs(value) <= max_value) {
          ssq_sum += value * value;
        }
        else if (*bad_start == 0) {
          *predict_start = *signal_start + max_value;
          ssq_sum += max_value;
          bad_count++;
        }
        // levmar has it's own version of residual calculation so have to sub in the nan values
        if (!isfinite(value) || isnan(value) || value > max_value) {
          *predict_start = *signal_start + max_value;
        }
        bad_start++;
        signal_start++;
        predict_start++;
      }
    }
  }
  ssq = ssq_sum;
}

void ZeromerMatDiff::FitTauB(const int *zero_flows, size_t n_zero_flows, 
                             const float *trace_data, const float *ref_data, 
                             size_t n_wells, size_t n_flows, size_t n_flow_wells,
                             size_t n_frames, float taue_est, float *__restrict taub) {
  Eigen::Map<Eigen::VectorXf, Eigen::Aligned> taub_v(taub, n_flow_wells);
  Eigen::VectorXf vec_sum_x2(n_wells), vec_sum_xy(n_wells), 
    vec_previous(n_wells), taub_sum(n_wells);
  taub_sum.setZero();  
  taub_v.setZero();
  for (size_t flow_ix = 0; flow_ix < n_zero_flows; flow_ix++) {
    int z_ix = zero_flows[flow_ix];

    vec_sum_x2.setZero();
    vec_sum_xy.setZero();
    vec_previous.setZero();
    for (size_t frame_ix = 0; frame_ix < n_frames; frame_ix++) {
      size_t offset = frame_ix * n_flow_wells + n_wells * z_ix;
      const float * __restrict trace_ptr_start = trace_data + offset;
      const float * __restrict trace_ptr_end = trace_ptr_start + n_wells;
      const float * __restrict ref_ptr = ref_data + offset;
      float * __restrict previous_ptr = vec_previous.data();
      float * __restrict xx_ptr = vec_sum_x2.data();
      float * __restrict xy_ptr = vec_sum_xy.data();
      while (trace_ptr_start != trace_ptr_end) {
        float diff = *ref_ptr - *trace_ptr_start;
        float taues = *ref_ptr * taue_est;
        float y = *previous_ptr + diff + taues;
        *previous_ptr += diff;
        float x = *trace_ptr_start;
        *xx_ptr += x * x;
        *xy_ptr += x * y;

        trace_ptr_start++;
        ref_ptr++;
        previous_ptr++;
        xx_ptr++;
        xy_ptr++;
      }
    }
    float *__restrict tau_b_start = taub_sum.data();
    float *__restrict tau_b_end = tau_b_start + n_wells;
    float *__restrict xx_ptr = vec_sum_x2.data();
    float *__restrict xy_ptr = vec_sum_xy.data();
    while (tau_b_start != tau_b_end) {
      *tau_b_start++ += *xy_ptr++ / *xx_ptr++;
    }
  }

  /* for now same taub estimate per nuc, just copy for each flow. */
  for (size_t flow_ix = 0; flow_ix < n_flows; flow_ix++) {
    float *__restrict tau_b_start = taub + flow_ix * n_wells;
    float *__restrict tau_b_end = tau_b_start + n_wells;
    float *__restrict tau_b_sum_start = taub_sum.data();
    while (tau_b_start != tau_b_end) {
      *tau_b_start++ = *tau_b_sum_start++ / n_zero_flows;
    }
  }

}

void ZeromerMatDiff::FitTauBNuc(const int *zero_flows, size_t n_zero_flows, 
                                const float *trace_data, const float *ref_data, 
                                int *nuc_flows,
                                size_t n_wells, size_t n_flows, size_t n_flow_wells,
                                size_t n_frames, float taue_est, float *__restrict taub) {
  float nuc_weight_mult = .3;
  float combo_weight_mult = .7;
     
  Eigen::Map<Eigen::VectorXf, Eigen::Aligned> taub_v(taub, n_flow_wells);
  Eigen::VectorXf vec_sum_x2(n_wells), vec_sum_xy(n_wells), 
    vec_previous(n_wells), taub_sum(n_wells);
  Eigen::MatrixXf taub_nuc(n_wells, 4);
  int nuc_counts[4] = {0,0,0,0};
  taub_nuc.setZero();
  taub_v.setZero();
  taub_sum.setZero();    
  for (size_t flow_ix = 0; flow_ix < n_zero_flows; flow_ix++) {
    int z_ix = zero_flows[flow_ix];
    vec_sum_x2.setZero();
    vec_sum_xy.setZero();
    vec_previous.setZero();
    for (size_t frame_ix = 0; frame_ix < n_frames; frame_ix++) {
      size_t offset = frame_ix * n_flow_wells + n_wells * z_ix;
      const float * __restrict trace_ptr_start = trace_data + offset;
      const float * __restrict trace_ptr_end = trace_ptr_start + n_wells;
      const float * __restrict ref_ptr = ref_data + offset;
      float * __restrict previous_ptr = vec_previous.data();
      float * __restrict xx_ptr = vec_sum_x2.data();
      float * __restrict xy_ptr = vec_sum_xy.data();
      while (trace_ptr_start != trace_ptr_end) {
        float diff = *ref_ptr - *trace_ptr_start;
        float taues = *ref_ptr * taue_est;
        float y = *previous_ptr + diff + taues;
        *previous_ptr += diff;
        float x = *trace_ptr_start;
        *xx_ptr += x * x;
        *xy_ptr += x * y;

        trace_ptr_start++;
        ref_ptr++;
        previous_ptr++;
        xx_ptr++;
        xy_ptr++;
      }
    }
    int nuc_ix = nuc_flows[z_ix];
    nuc_counts[nuc_ix]++;
    float *__restrict tau_b_nuc_start = taub_nuc.col(nuc_ix).data();
    float *__restrict tau_b_start = taub_sum.data();
    float *__restrict tau_b_end = tau_b_start + n_wells;
    float *__restrict xx_ptr = vec_sum_x2.data();
    float *__restrict xy_ptr = vec_sum_xy.data();
    while (tau_b_start != tau_b_end) {
      float value = *xy_ptr++ / *xx_ptr++;
      if (!isfinite(value)) {
        value = 0;
      }
      *tau_b_start++ += value;
      *tau_b_nuc_start++ += value;
    }
  }

  /* for now same taub estimate per nuc, just copy for each flow. */
  for (size_t flow_ix = 0; flow_ix < n_flows; flow_ix++) {
    int nuc_ix = nuc_flows[flow_ix];
    float *__restrict tau_b_start = taub + flow_ix * n_wells;
    float *__restrict tau_b_end = tau_b_start + n_wells;
    float *__restrict tau_b_sum_start = taub_sum.data();
    float *__restrict tau_b_nuc_sum_start = taub_nuc.col(nuc_ix).data();
    // little awkward as we mult weight by number of observances but then divide out for average but more readable to keep calc
    float nuc_div = nuc_counts[nuc_ix];
    float nuc_weight = nuc_counts[nuc_ix] * nuc_weight_mult;
    float nuc_mult = nuc_div > 0.0f ? nuc_weight / nuc_div : 0.0f;
    float total_weight = nuc_weight + combo_weight_mult;
    float combo_mult = combo_weight_mult/n_zero_flows;
    while (tau_b_start != tau_b_end) {
      // weighted average of nuc specific and overall for this well taub
      *tau_b_start = ((*tau_b_sum_start * combo_mult) + (*tau_b_nuc_sum_start * nuc_mult)) / total_weight;
      tau_b_start++;
      tau_b_sum_start++;
      tau_b_nuc_sum_start++;
    }
  }
}
