/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef EXPTAILFITTER_H
#define EXPTAILFITTER_H

#include "TimeCompression.h"
#include "BeadParams.h"
#include "RegionParams.h"
#include "FlowBuffer.h"
#include <armadillo>

class ExpTailFitter {
public:

   arma::mat lhs,rhs,vals;

   ExpTailFitter()
   {
      lhs = arma::zeros<arma::mat>(2,2);
      rhs = arma::zeros<arma::mat>(2,1);
      vals = arma::zeros<arma::mat>(2,1);
   }

   // corrects all flows in the block of incorporation traces for a single bead using the exponential-tail-fit
   // method of extracting the dc-shift term
   // traces are corrected in-place
   float CorrectTraces(float *incorporation_traces,float *bkg_traces,BeadParams *p,reg_params *rp,FlowBufferInfo *my_flow,TimeCompression &time_c, int flow_block_size, int flow_block_start);
   void FitTauAdj(float *incorporation_traces,float *bkg_traces,BeadParams *p,reg_params *rp,FlowBufferInfo *my_flow,TimeCompression &time_c, int flow_block_size, int flow_block_start);
   void  AdjustBackground(float *incorporation_traces,float *bkg_traces,BeadParams *p,reg_params *rp,FlowBufferInfo *my_flow,TimeCompression &time_c, int flow_block_size, int flow_block_start);
   bool ComputeAverageValidTrace(float *avg_trace, float *incorporation_traces,BeadParams *p, int npts, float low_A, float hi_A, int flow_block_size);
   // fits a subset of the point of the provided incorporation trace to and exponential-decay + dc-offset term.
   // returns the dc offset term, scaled by the average amplitude of the empty-well background trace over the same time interval
   // selects the subset of points to fit based on the estimated HP length in the flow
   float generic_exp_tail_fit(float *trc, float *bkg_trace, TimeCompression& time_c, float tau, float mer_guess, float t_mid_nuc, bool debug = false);

   // fits the specified points to an exponential-decay + dc offset and returns the dc-offset term
   float fit_exp_tail_to_data(float *trc, int start_pt, int end_pt, std::vector<float> &ftimes, float tau, float *exp_amp = NULL, float *mse_err = NULL, bool debug = false);


private:

};

// convolves the kernel (normalized) with the input data
void smooth_kern(float *out, float *in, float *kern, int dist, int npts);

#endif // EXPTAILFITTER_H

