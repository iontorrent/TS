/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef EXPTAILDECAYFIT_H
#define EXPTAILDECAYFIT_H

#include "SignalProcessingMasterFitter.h"

#include "TimeCompression.h"
#include "BeadParams.h"
#include "RegionParams.h"
#include "FlowBuffer.h"
#include <armadillo>

// this is a specialized fitter
// to construct a corrector to the buffering (taub) per bead
// using the first 20 flows (1-3 mers)
// needs an approximate estimate of amplitude
class ExpTailDecayFit{
public:
  SignalProcessingMasterFitter &bkg; // reference to source class for now
  ExpTailDecayFit (SignalProcessingMasterFitter &_bkg);

  void AdjustBufferingEveryBead(int flow_block_size, int flow_block_start);
  void AdjustBufferingOneBead(int ibd, int flow_block_size, int flow_block_start);

  void FitTauAdj(float *incorporation_traces,float *bkg_traces,BeadParams *p,reg_params *rp,FlowBufferInfo *my_flow,TimeCompression &time_c, int flow_block_size, int flow_block_start);
  bool ComputeAverageValidTrace(float *avg_trace, float *incorporation_traces,BeadParams *p, int npts, float low_A, float hi_A, int flow_block_size);
};

#endif // EXPTAILDECAYFIT_H
