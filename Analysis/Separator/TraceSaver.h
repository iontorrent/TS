/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACESAVER_H
#define TRACESAVER_H

#include <stdlib.h>
#include <malloc.h>
#include "EvaluateKey.h"
#include "H5File.h"

class TraceSaver {
public:  
  TraceSaver() {
    Init();
  }

  ~TraceSaver() {
    Cleanup();
  }

  void Init() {
    m_trace_data = m_ref_data = m_zeromer_est = m_shifted_ref = m_dark_matter = m_stats = NULL;
    m_num_cols = m_num_wells = m_num_well_flows = m_num_frames = m_num_flows = m_num_stats = 0;
  }

  void Cleanup() {
    if (m_trace_data != NULL) {
      free(m_trace_data);
      free(m_ref_data);
      free(m_zeromer_est);
      free(m_shifted_ref);
      free(m_dark_matter);
      free(m_stats);
    }
    Init();
  }

  void Alloc(size_t num_cols, size_t num_wells, size_t num_frames, size_t num_flows, size_t num_stats) {
    Cleanup();
    m_num_cols = num_cols;
    m_num_wells = num_wells;
    m_num_frames = num_frames;
    m_num_flows = num_flows;
    m_num_stats = num_stats;
    m_num_well_flows = m_num_wells * m_num_flows;
    size_t total_size = m_num_wells * m_num_frames * m_num_flows;
    m_trace_data = (float *) memalign(32, sizeof(float) * total_size);
    m_ref_data = (float *) memalign(32, sizeof(float) * total_size);
    m_zeromer_est = (float *) memalign(32, sizeof(float) * total_size);
    m_shifted_ref = (float *) memalign(32, sizeof(float) * total_size);
    m_dark_matter = (float *) memalign(32, sizeof(float) * total_size);
    m_stats = (float *) memalign(32, sizeof(float) * total_size);
  }

  void StoreResults(int row_start, int row_end, int col_start, int col_end,
                    int flow_start, int flow_end, EvaluateKey &evaluator);

  void WriteResults(H5File &h5);

  size_t m_num_wells, m_num_well_flows, m_num_frames, m_num_flows, m_num_cols, m_num_stats;
  float *m_trace_data, *m_ref_data, *m_zeromer_est, *m_shifted_ref, *m_dark_matter, *m_stats;

};

#endif // TRACESAVER_H
