/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BFMETRIC_H
#define BFMETRIC_H

#include <malloc.h>
#include "Mask.h"

#include "ImageNNAvg.h"
#include "Image.h"
#include "Vecs.h"

class BfMetric {

 public:

  BfMetric() { 
    m_gain_min_mult = 1.0;
    m_num_rows = m_num_cols = m_num_frames = 0;
    m_trace_min = NULL;
    m_trace_min_frame = NULL;
    m_bf_metric = NULL;
    m_norm_bf_metric = NULL;
  }

  ~BfMetric() { Cleanup(); }

  void Init(int n_rows, int n_cols, int n_frames) {
    m_num_rows = n_rows;
    m_num_cols = n_cols;
    m_num_frames = n_frames;
    m_trace_min = (float *) memalign(VEC8F_SIZE_B, sizeof(float) * (n_cols * n_rows));
    memset(m_trace_min, 0, sizeof(float) * (n_cols * n_rows));
    m_trace_min_frame = (int *) memalign(VEC8F_SIZE_B, sizeof(int) * (n_cols * n_rows));
    memset(m_trace_min_frame, 0, sizeof(int) * (n_cols * n_rows));
    m_bf_metric = (int *) memalign(VEC8F_SIZE_B, sizeof(int) * n_rows * n_cols);
    memset(m_bf_metric, 0, sizeof(int) * (n_cols * n_rows));
    m_norm_bf_metric = (float *) memalign(VEC8F_SIZE_B, sizeof(float) * n_rows *n_cols);
    memset(m_norm_bf_metric, 0, sizeof(float) * n_rows *n_cols);
  }

  void Cleanup() {
    m_num_rows = m_num_cols = m_num_frames = 0;
    m_nn_avg.Cleanup(); 
    m_diff_sum_avg.Cleanup(); 
    m_diff_sum_sq_avg.Cleanup(); 
    FREEZ(&m_trace_min);
    FREEZ(&m_trace_min_frame);
    FREEZ(&m_bf_metric);
    FREEZ(&m_norm_bf_metric);
  }

  void CalcBfSdMetric(const short *__restrict image, 
                      Mask *mask,
                      const char *__restrict bad_wells,
                      const float *__restrict t0,
                      const std::string &dbg_name,
                      int t0_window_start,
                      int t0_window_end,
                      int row_step,
                      int col_step,
                      int num_row_neighbors,
                      int num_col_neighbors);

  void CalcBfSdMetricReduce(const short *__restrict image, 
                            Mask *mask,
                            const char *__restrict bad_wells,
                            const float *__restrict t0,
                            const std::string &dbg_name,
                            int t0_window_start,
                            int t0_window_end,
                            int row_step,
                            int col_step,
                            int num_row_neighbors,
                            int num_col_neighbors,
                            int frame_window);

  void NormalizeMetric(const char *bad_wells,
                       int row_step, int col_step,
                       int num_row_neighbors, 
                       int num_col_neighbors);
   
  void NormalizeMetricRef(const char *ref_wells, const char *bad_wells,
                          int row_clip, int col_clip,
                          int num_row_neighbors, 
                          int num_col_neighbors);

  int GetBfMetric(int well_index) { return m_bf_metric[well_index]; }
  float GetNormBfMetric(int well_index) { return m_norm_bf_metric[well_index]; }

  const int *GetBfMetrcPtr() { return m_bf_metric; }
  const int *GetSdFrame() { return m_trace_min_frame; }
  int GetSdFrame(int well_index) { return m_trace_min_frame[well_index]; }
  void SetGainMinMult(int mult) { m_gain_min_mult = mult; }

 private:
  int m_num_rows, m_num_cols, m_num_frames;
  int m_gain_min_mult;
  float *m_trace_min;
  int *m_trace_min_frame;
  int * m_bf_metric;
  float *m_norm_bf_metric;
  ImageNNAvg m_nn_avg;
  ImageNNAvg m_diff_sum_avg;
  ImageNNAvg m_diff_sum_sq_avg;
  ION_DISABLE_COPY_ASSIGN(BfMetric)
};

#endif // BFMETRIC_H
