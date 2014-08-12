/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TAUEFITTER_H
#define TAUEFITTER_H

#include "LevMarFitterV2.h"
#include "ZeromerMatDiff.h"

struct FitTauEParams {
  float taue;
  float ref_shift;
  float converged;
};

class TauEFitter : public LevMarFitterV2 {

 public:
  
  TauEFitter(size_t n_values, float *signal_values, ZeromerMatDiff *zmd) {
    m_zmd = zmd;
    Initialize(1, n_values, signal_values);
  }
  
  // optionally set maximum value for parameters
  void SetParamMax(FitTauEParams _max_params) {
    m_max_params = _max_params;
    LevMarFitterV2::SetParamMax((float *)&m_max_params);
  }

  // optionally set minimum value for parameters
  void SetParamMin(FitTauEParams  _min_params) {
    m_min_params = _min_params;
    LevMarFitterV2::SetParamMin((float *)&m_min_params);
  }

  void SetInitialParam(FitTauEParams start_param) {
    m_params = start_param;
  }

  FitTauEParams GetParam() {
    return m_params;
  }

  // entry point for grid search
  void GridSearch(int steps,float *y) {
    LevMarFitterV2::GridSearch(steps,y,(float *)(&m_params));
  }

  // evaluates the function using the values in params
  virtual void Evaluate(float *y) {
    Evaluate(y,(float *)(&m_params));
  }
  
  virtual void Evaluate(float *y, float *params) {
    /* m_zmd->ShiftReference(m_zmd->m_num_frames, m_zmd->m_num_well_flows, */
    /*                       params[1], m_zmd->m_ref_data, m_zmd->m_shifted_ref); */
    memcpy(m_zmd->m_shifted_ref, m_zmd->m_ref_data, sizeof(float) * m_zmd->m_total_size);
    int flows[m_zmd->m_num_flows];
    for (size_t i = 0; i < m_zmd->m_num_flows; i++) { flows[i] = i; }
    m_zmd->FitTauB(&flows[0], m_zmd->m_num_flows,
                   m_zmd->m_trace_data, m_zmd->m_shifted_ref,
                   m_zmd->m_num_wells, m_zmd->m_num_flows, m_zmd->m_num_well_flows,
                   m_zmd->m_total_size / m_zmd->m_num_well_flows,
                   params[0], m_zmd->m_taub);
    m_zmd->PredictZeromersSignal(&m_zmd->m_time[0], m_zmd->m_num_frames,
                                 m_zmd->m_trace_data, m_zmd->m_shifted_ref, m_zmd->m_zeromer_est,
                                 m_zmd->m_num_wells, m_zmd->m_num_flows,
                                 m_zmd->m_num_well_flows, 
                                 params[0], m_zmd->m_taub,
                                 m_zmd->m_h5_dump);
    double residual = 0.0;
    m_zmd->ZeromerSumSqErrorTrim(&flows[0], m_zmd->m_num_flows,
                                 m_zmd->m_bad_wells,
                                 m_zmd->m_trace_data, m_zmd->m_zeromer_est,
                                 m_zmd->m_num_wells, m_zmd->m_num_flows,
                                 m_zmd->m_num_well_flows, m_zmd->m_num_frames,
                                 residual);
    memcpy(y,m_zmd->m_zeromer_est,sizeof(float) * m_zmd->m_total_size);
    double sumx = 0.0;
    double diffx = 0.0;
    for (size_t i = 0; i < m_zmd->m_total_size; i++) {
      float val =  m_zmd->m_trace_data[i] - m_zmd->m_zeromer_est[i];
      diffx += val * val;
      sumx += m_zmd->m_zeromer_est[i];
    }
    /* fprintf(stdout, "Region: %d %d TauE %.10f Shift %.10f - Residual %.10f %.2f %.2f\n",  */
    /*         m_zmd->m_row_start, m_zmd->m_col_start, params[0], params[1], residual, */
    /*         diffx/m_zmd->m_total_size, sumx/m_zmd->m_total_size); */
  }

  virtual void EvaluateNoShift(float *y, float *params) {
    int flows[m_zmd->m_num_flows];
    for (size_t i = 0; i < m_zmd->m_num_flows; i++) { flows[i] = i; }
    m_zmd->FitTauB(&flows[0], m_zmd->m_num_flows,
                   m_zmd->m_trace_data, m_zmd->m_ref_data,
                   m_zmd->m_num_wells, m_zmd->m_num_flows, m_zmd->m_num_well_flows,
                   m_zmd->m_total_size / m_zmd->m_num_well_flows,
                   params[0], m_zmd->m_taub);
    m_zmd->PredictZeromersSignal(&m_zmd->m_time[0], m_zmd->m_num_frames,
                                 m_zmd->m_trace_data, m_zmd->m_ref_data, m_zmd->m_zeromer_est,
                                 m_zmd->m_num_wells, m_zmd->m_num_flows,
                                 m_zmd->m_num_well_flows, 
                                 params[0], m_zmd->m_taub,
                                 m_zmd->m_h5_dump);
    double residual = 0.0;
    m_zmd->ZeromerSumSqErrorTrim(&flows[0], m_zmd->m_num_flows,
                                 m_zmd->m_bad_wells,
                                 m_zmd->m_trace_data, m_zmd->m_zeromer_est,
                                 m_zmd->m_num_wells, m_zmd->m_num_flows,
                                 m_zmd->m_num_well_flows, m_zmd->m_num_frames,
                                 residual);
    memcpy(y,m_zmd->m_zeromer_est,sizeof(float) * m_zmd->m_total_size);
  }

  // entry point for fitting
  virtual int Fit(bool gauss_newton, int max_iter,float *y)  {
    return(LevMarFitterV2::Fit(gauss_newton, max_iter,y,(float *)(&m_params)));
  }

  // the starting point and end point of the fit
  FitTauEParams m_params,m_min_params,m_max_params;
  ZeromerMatDiff *m_zmd;

};


#endif // TAUEFITTER_H
