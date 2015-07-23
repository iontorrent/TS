/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "TraceSaver.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include "IonH5Eigen.h"

void TraceSaver::StoreResults(int row_start, int row_end, int col_start, int col_end,
                              int flow_start, int flow_end, EvaluateKey &evaluator) {
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> all_trace(m_trace_data, m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> all_ref(m_ref_data, m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> all_zeromer_est(m_zeromer_est, m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> all_shifted_ref(m_shifted_ref, m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> all_dark_matter(m_dark_matter, m_num_well_flows, m_num_frames);

  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> loc_trace(evaluator.m_trace_data, evaluator.m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> loc_ref(evaluator.m_ref_data, evaluator.m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> loc_zeromer_est(evaluator.m_zeromer_est, evaluator.m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> loc_shifted_ref(evaluator.m_shifted_ref, evaluator.m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> loc_dark_matter(evaluator.m_avg_0mer, 1, m_num_frames);
    
  int count = 0;
  for (int flow_ix = flow_start; flow_ix < flow_end; flow_ix++) {
    for (int row_ix = row_start; row_ix < row_end; row_ix++) {
      for (int col_ix = col_start; col_ix < col_end; col_ix++) {
        int index = flow_ix * m_num_wells + row_ix * m_num_cols + col_ix;
        all_trace.row(index).array() = loc_trace.row(count).array();
        all_ref.row(index).array() = loc_ref.row(count).array();
        all_zeromer_est.row(index).array() = loc_zeromer_est.row(count).array();
        all_shifted_ref.row(index).array() = loc_shifted_ref.row(count).array();
        all_dark_matter.row(index).array() = loc_dark_matter.row(0).array();
        count++;
      }
    } 
  }
}


void TraceSaver::WriteResults(H5File &h5) {
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> all_trace(m_trace_data, m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> all_ref(m_ref_data, m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> all_zeromer_est(m_zeromer_est, m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> all_shifted_ref(m_shifted_ref, m_num_well_flows, m_num_frames);
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> all_dark_matter(m_dark_matter, m_num_well_flows, m_num_frames);
  Eigen::MatrixXf tmp = all_trace;
  H5Eigen::WriteMatrix(h5, "/traces", tmp);
  tmp = all_ref;
  H5Eigen::WriteMatrix(h5, "/ref", tmp);
  tmp = all_shifted_ref;
  H5Eigen::WriteMatrix(h5, "/shiftedref", tmp);
  tmp = all_zeromer_est;
  H5Eigen::WriteMatrix(h5, "/zeromer", tmp);
  tmp = all_dark_matter;
  H5Eigen::WriteMatrix(h5, "/darkmatter", tmp);
}
