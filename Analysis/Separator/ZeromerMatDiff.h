/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ZEROMERMATDIFF_H
#define ZEROMERMATDIFF_H

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include "TraceStoreCol.h"

class ZeromerMatDiff {
 public:

  ZeromerMatDiff() {
    Init();
  }

  ~ZeromerMatDiff() {
    Cleanup();
  }

  void Init() {
    m_bad_wells = NULL;
    m_trace_data = m_ref_data = m_zeromer_est = m_taub = m_shifted_ref = NULL;
    m_num_frames = m_total_size = m_num_wells = m_num_well_flows = m_num_flows = 0;
  }

  void Alloc(size_t total_size, size_t total_rows, size_t num_wells, size_t num_flows, int num_frames ) {
    if (total_size != m_total_size) {
      if (m_trace_data != NULL) {
        Cleanup();
      }
      m_trace_data = (float *) memalign(32, sizeof(float) * total_size);
      m_ref_data = (float *) memalign(32, sizeof(float) * total_size);
      m_zeromer_est = (float *) memalign(32, sizeof(float) * total_size);
      m_taub = (float *) memalign(32, sizeof(float) * total_rows);
      m_shifted_ref = (float *) memalign(32, sizeof(float) * total_size);
      m_bad_wells = (char *) memalign(32, sizeof(char) * num_wells);
      m_total_size = total_size;
      m_num_well_flows = total_rows;
      m_num_wells = num_wells;
      m_num_flows = num_flows;
      m_num_frames = num_frames;
    }
  }

  void Cleanup() {
    if (m_trace_data != NULL) {
      free(m_trace_data);
      free(m_ref_data);
      free(m_zeromer_est);
      free(m_taub);
      free(m_shifted_ref);
      free(m_bad_wells);
      Init();
    }
  }
  
  void SolveTauE(double *param, double *residuals);

  static void ShiftReference(int n_frames, size_t n_flow_wells, float shift,
                             float *orig, float *shifted);

  float *GetTraceData() { return m_trace_data; }
  float *GetRefData() { return m_ref_data; }
  float *GetZeromerData() {return m_zeromer_est; }

  static void FitTauB(const int *zero_flows, size_t n_zero_flows, 
                      const float *trace_data, const float *ref_data, 
                      size_t n_wells, size_t n_flows, size_t n_flow_wells,
                      size_t n_frames, float taue_est, float *__restrict taub);


  static void FitTauBNuc(const int *zero_flows, size_t n_zero_flows, 
                        const float *trace_data, const float *ref_data, 
                        int *nuc_flows,
                        size_t n_wells, size_t n_flows, size_t n_flow_wells,
                        size_t n_frames, float taue_est, float *__restrict taub);

  static void PredictZeromersSignal(const float *time, int n_frames,
                                    float *trace, float *ref, float *zeromer,
                                    size_t n_wells, size_t n_flows, 
                                    size_t n_flow_wells, 
                                    float taue_est, float *__restrict taub,
                                    const std::string &h5_dump);

  static void ZeromerSumSqErrorTrim(const int *zero_flows, size_t n_zero_flows, 
                                    const char *bad_wells,
                                    float *signal_data, float *predict_data, 
                                    size_t n_wells, size_t n_flows, 
                                    size_t n_flow_wells, size_t n_frames, 
                                    double &ssq);

  static void ZeromerMadError(const int *zero_flows, size_t n_zero_flows, 
                              float *signal_data, 
                              size_t n_wells, size_t n_flows, 
                              size_t n_flow_wells, size_t n_frames, 
                              float &mad);

  void SetUpMatricesClean(TraceStoreCol &trace_store, 
                          const char *bad_wells,
                          const float *time, int row_step_sample,
                          int col_step_sample,
                          int col_stride, int flow_stride,
                          int row_start, int row_end, int col_start, int col_end,
                          //                         int flow_start, int flow_end,
                          int *zero_flows, int num_zeromer_flows,
                          int frame_start, int frame_end) {
    m_row_start = row_start;
    m_row_end = row_end;
    m_col_start = col_start;
    m_col_end = col_end;
    int row_size = ceil((row_end - row_start) / (float)row_step_sample);
    int col_size = ceil((col_end - col_start) / (float)col_step_sample);
    int flow_size = num_zeromer_flows;
    size_t frame_size = frame_end - frame_start;

    // setup time
    m_time.resize(frame_size);
    for (int frame_ix = frame_start; frame_ix < frame_end; frame_ix++) {
      m_time[frame_ix-frame_start] = time[frame_ix];
    }
    // How many good wells are there
    int good_wells = 0;
    for (int row_ix = row_start; row_ix < row_end; row_ix+= row_step_sample) {
      for (int col_ix = col_start; col_ix < col_end; col_ix+= col_step_sample) {
        int well_ix = row_ix * col_stride + col_ix;
        if (bad_wells[well_ix] == 0) {
          good_wells++;
        }
      }
    }
    size_t local_flow_stride = good_wells;
    size_t total_rows = local_flow_stride * flow_size;
    size_t total_size = total_rows * frame_size;
    
    // Allocate enough space
    Alloc(total_size, total_rows, good_wells, flow_size, frame_size);
    // copy in the data
    std::vector<float> trace(trace_store.GetNumFrames());
    std::vector<float> ref_trace(trace_store.GetNumFrames());


    // Copy the bad well data, this is redudant now as they should all be 0
    int local_index = 0;
    for (int row_ix = row_start; row_ix < row_end; row_ix+= row_step_sample) {
      const char *__restrict bad_well_start = bad_wells + row_ix * col_stride + col_start;
      const char *__restrict bad_well_end = bad_well_start + col_end - col_start;
      char *__restrict bad_out = m_bad_wells + local_index;
      while (bad_well_start < bad_well_end) {
        if (*bad_well_start == 0) {
          *bad_out = *bad_well_start;
          bad_out++;
          local_index++;
        }
        bad_well_start += col_step_sample;
      }
    }

    // copy the traces and reference
    for (int flow_ix = 0; flow_ix < num_zeromer_flows; flow_ix++) {
      int z_ix = zero_flows[flow_ix];
      int local_well_ix = 0;
      for (int row_ix = row_start; row_ix < row_end; row_ix+= row_step_sample) {
        for (int col_ix = col_start; col_ix < col_end; col_ix+=col_step_sample) {
          int well_ix = row_ix * col_stride + col_ix;
          if (bad_wells[well_ix] == 0) {
            trace_store.GetTrace(well_ix, z_ix, trace.begin());
            trace_store.GetReferenceTrace(well_ix, z_ix, &ref_trace[0]);
            /* if (col_ix + col_step_sample >= col_end) { */
            /*   cout << "here we go..." << endl; */
            /* } */
            //            int flow_offset = (row_ix - row_start)/row_step_sample * col_size + (col_ix - col_start)/col_step_sample;
            //            int local_well_ix = (z_ix - flow_start) * local_flow_stride + flow_offset;
            for (int frame_ix = frame_start; frame_ix < frame_end; frame_ix++) {
              m_trace_data[local_well_ix + (frame_ix - frame_start) * total_rows] = trace[frame_ix];
              m_ref_data[local_well_ix + (frame_ix - frame_start) * total_rows] = ref_trace[frame_ix];
            }
            local_well_ix++;
          }
        }
      }
    }
  }

  void SetUpMatrices(TraceStoreCol &trace_store, 
                     const char *bad_wells,
                     const float *time, int row_step_sample,
                     int col_step_sample,
                     int col_stride, int flow_stride,
                     int row_start, int row_end, int col_start, int col_end,
                     int flow_start, int flow_end,
                     int frame_start, int frame_end) {
    int row_size = ceil((row_end - row_start) / (float)row_step_sample);
    int col_size = ceil((col_end - col_start) / (float)col_step_sample);
    int flow_size = flow_end - flow_start;
    size_t frame_size = frame_end - frame_start;
    size_t local_flow_stride = row_size * col_size;
    size_t total_rows = local_flow_stride * flow_size;
    size_t total_size = total_rows * frame_size;
    // setup time
    m_time.resize(frame_size);
    for (int frame_ix = frame_start; frame_ix < frame_end; frame_ix++) {
      m_time[frame_ix-frame_start] = time[frame_ix];
    }
    // Allocate enough space
    Alloc(total_size, total_rows, col_size * row_size, flow_size, frame_size);
    // copy in the data
    std::vector<float> trace(trace_store.GetNumFrames());
    std::vector<float> ref_trace(trace_store.GetNumFrames());
    
    // Copy the bad well data
    for (int row_ix = row_start; row_ix < row_end; row_ix+= row_step_sample) {
      const char *__restrict bad_well_start = bad_wells + row_ix * col_stride + col_start;
      const char *__restrict bad_well_end = bad_well_start + col_end - col_start;
      char *__restrict bad_out = m_bad_wells + (row_ix - row_start)/row_step_sample * col_size;
      while (bad_well_start < bad_well_end) {
        *bad_out = *bad_well_start;
        bad_out++;
        bad_well_start += col_step_sample;
      }
    }
    
    // copy the traces and reference
    for (int flow_ix = flow_start; flow_ix < flow_end; flow_ix++) {
      for (int row_ix = row_start; row_ix < row_end; row_ix+= row_step_sample) {
          for (int col_ix = col_start; col_ix < col_end; col_ix+=col_step_sample) {
            int well_ix = row_ix * col_stride + col_ix;
            trace_store.GetTrace(well_ix, flow_ix, trace.begin());
            trace_store.GetReferenceTrace(well_ix, flow_ix, &ref_trace[0]);
            /* if (col_ix + col_step_sample >= col_end) { */
            /*   cout << "here we go..." << endl; */
            /* } */
            int flow_offset = (row_ix - row_start)/row_step_sample * col_size + (col_ix - col_start)/col_step_sample;
            int local_well_ix = (flow_ix - flow_start) * local_flow_stride + flow_offset;
            for (int frame_ix = frame_start; frame_ix < frame_end; frame_ix++) {
              m_trace_data[local_well_ix + (frame_ix - frame_start) * total_rows] = trace[frame_ix];
              m_ref_data[local_well_ix + (frame_ix - frame_start) * total_rows] = ref_trace[frame_ix];
            }
          }
      }
    }
  }

  int m_row_start, m_row_end, m_col_start, m_col_end;
  size_t m_num_wells, m_num_well_flows, m_num_flows, m_total_size, m_num_frames;
  float *m_trace_data, *m_ref_data, *m_zeromer_est, *m_taub, *m_shifted_ref;
  char *m_bad_wells;
  std::string m_h5_dump;
  std::vector<float> m_time;
};

#endif // ZEROMERMATDIFF_H
