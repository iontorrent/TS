/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Eigen/Dense>
#include <Eigen/LU>
#include "ZeromerMatDiff.h"
#include "IonH5Eigen.h"
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
      shifted.col(frame_ix).array() = (orig.col(end_frame).array() - orig.col(start_frame).array()) * mult +
    		                           orig.col(start_frame).array();
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
  // column based operations vectorized by eigen for speed
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
  float max_value_sq = max_value*max_value;
  for (size_t z_ix = 0; z_ix < n_zero_flows; z_ix++) {
    size_t flow_ix = zero_flows[z_ix];
    for (size_t col_ix = 0; col_ix < n_frames; col_ix++) {
      float *__restrict signal_start = signal_data + n_wells * flow_ix + n_flow_wells * col_ix;
      float *__restrict predict_start = predict_data + n_wells * flow_ix + n_flow_wells * col_ix;
      const char *__restrict bad_start = bad_wells;
      for(size_t trc=0;trc<n_wells;trc++) {
        float value = signal_start[trc] - predict_start[trc]; //*signal_start * *signal_start;
        value=value*value;


        if (isfinite(value) && value <= max_value_sq) {
        	if(bad_start[trc] == 0)
        		ssq_sum += value;
        }
        else{
          predict_start[trc] = signal_start[trc] + max_value;
          if(bad_start[trc] == 0){
			  ssq_sum += max_value_sq;
			  bad_count++;
          }
        }
      }
    }
  }
  ssq = ssq_sum;
}

void ZeromerMatDiff::FitTauB(const int *zero_flows, size_t n_zero_flows, 
                             const float *trace_data, const float *ref_data, 
                             size_t n_wells, size_t n_flows, size_t n_flow_wells,
                             size_t n_frames, float taue_est, float *__restrict taub) {
  float vec_sum_xx   [n_wells] __attribute__ ((aligned (64)));
  float vec_sum_xy   [n_wells] __attribute__ ((aligned (64)));
  float vec_previous [n_wells] __attribute__ ((aligned (64)));
  float taub_sum     [n_wells] __attribute__ ((aligned (64)));

  memset(taub_sum,0,sizeof(taub_sum));
  for (size_t flow_ix = 0; flow_ix < n_zero_flows; flow_ix++) {
    int z_ix = zero_flows[flow_ix];

    memset(vec_sum_xx,0,sizeof(vec_sum_xx));
    memset(vec_sum_xy,0,sizeof(vec_sum_xy));
    memset(vec_previous,0,sizeof(vec_previous));
    for (size_t frame_ix = 0; frame_ix < n_frames; frame_ix++) {
      size_t offset = frame_ix * n_flow_wells + n_wells * z_ix;
      const float * __restrict trace_ptr = trace_data + offset;
      const float * __restrict ref_ptr = ref_data + offset;
      for(size_t trc=0;trc<n_wells;trc++) {
        float diff = ref_ptr[trc] - trace_ptr[trc];
        float taues = ref_ptr[trc] * taue_est;
        float y = vec_previous[trc] + diff + taues;
        vec_previous[trc] += diff;
        float x = trace_ptr[trc];
        vec_sum_xx[trc] += x * x;
        vec_sum_xy[trc] += x * y;
      }
    }
    for(size_t trc=0;trc<n_wells;trc++) {
    	taub_sum[trc] += vec_sum_xy[trc] / vec_sum_xx[trc];
    }
  }

  /* for now same taub estimate per nuc, just copy for each flow. */
  for (size_t flow_ix = 0; flow_ix < n_flows; flow_ix++) {
    float *__restrict tau_b_start = taub + flow_ix * n_wells;
    for(size_t trc=0;trc<n_wells;trc++) {
      tau_b_start[trc] = taub_sum[trc] / n_zero_flows;
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
  float vec_sum_xx[n_wells] __attribute__ ((aligned (64)));
  float vec_sum_xy[n_wells] __attribute__ ((aligned (64)));
  float vec_previous[n_wells] __attribute__ ((aligned (64)));
  float taub_sum[n_wells] __attribute__ ((aligned (64)));
  float taub_nuc[4][n_wells] __attribute__ ((aligned (64)));
  int nuc_counts[4] = {0,0,0,0};

  memset(taub_nuc,0,sizeof(taub_nuc));
  memset(taub_sum,0,sizeof(taub_sum));
  for (size_t flow_ix = 0; flow_ix < n_zero_flows; flow_ix++) {
    int z_ix = zero_flows[flow_ix];
    memset(vec_sum_xx,0,sizeof(vec_sum_xx));
    memset(vec_sum_xy,0,sizeof(vec_sum_xy));
    memset(vec_previous,0,sizeof(vec_previous));
    for (size_t frame_ix = 0; frame_ix < n_frames; frame_ix++) {
      size_t offset = frame_ix * n_flow_wells + n_wells * z_ix;
      const float * trace_ptr = trace_data + offset;
      const float * ref_ptr = ref_data + offset;
      for(size_t trc=0;trc<n_wells;trc++) {
        float diff = ref_ptr[trc] - trace_ptr[trc];
        float taues = ref_ptr[trc] * taue_est;
        float y = vec_previous[trc] + diff + taues;
        vec_previous[trc] += diff;
        float x = trace_ptr[trc];
        vec_sum_xx[trc] += x * x;
        vec_sum_xy[trc] += x * y;
      }
    }
    int nuc_ix = nuc_flows[z_ix];
    nuc_counts[nuc_ix]++;
    float *tau_b_nuc = taub_nuc[nuc_ix];
    for(size_t trc=0;trc<n_wells;trc++) {
      float value = vec_sum_xy[trc] / vec_sum_xx[trc];
      if (!isfinite(value)) {
        value = 0;
      }
      taub_sum[trc] += value;
      tau_b_nuc[trc] += value;
    }
  }

  /* for now same taub estimate per nuc, just copy for each flow. */
  for (size_t flow_ix = 0; flow_ix < n_flows; flow_ix++) {
    int nuc_ix = nuc_flows[flow_ix];
    float *tau_b = taub + flow_ix * n_wells;
    float *tau_b_nuc_sum = taub_nuc[nuc_ix];
    // little awkward as we mult weight by number of observances but then divide out for average but more readable to keep calc
    float nuc_div = nuc_counts[nuc_ix];
    float nuc_weight = nuc_counts[nuc_ix] * nuc_weight_mult;
    float nuc_mult = nuc_div > 0.0f ? nuc_weight / nuc_div : 0.0f;
    float total_weight = nuc_weight + combo_weight_mult;
    float combo_mult = combo_weight_mult/n_zero_flows;
    for(size_t trc=0;trc<n_wells;trc++) {
      // weighted average of nuc specific and overall for this well taub
      tau_b[trc] = ((taub_sum[trc] * combo_mult) + (tau_b_nuc_sum[trc] * nuc_mult)) / total_weight;
    }
  }
}
