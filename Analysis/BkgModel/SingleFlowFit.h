/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SINGLEFLOWFIT_H
#define SINGLEFLOWFIT_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "BkgMagicDefines.h"
#include "BkgFitMatrixPacker.h"  // this is a bad way of making sure LAPACK is here
#include "MathOptim.h"
#include "DiffEqModel.h"
#include "MultiFlowModel.h"
#include "Utils.h"
#include "BeadParams.h"
#include "BeadTracker.h"
#include "RegionParams.h"
#include "RegionTracker.h"
#include "EmphasisVector.h"
// this is a bit kludgey, but to make things easy, the single-flow lev mar fitter
// uses struct bead_params and struct reg_params...so this include needs to be below
// those definitions
#include "BkgModSingleFlowFit.h"
#include "BkgModSingleFlowFitKrate.h"


class single_flow_optimizer
{
  public:
    // object used to fit a single well's data in a single flow
    BkgModSingleFlowFit *oneFlowFit;
    BkgModSingleFlowFitKrate *oneFlowFitKrate;
    // keep these around to avoid allocating when I need to talk to the above objects
    BkgModSingleFlowFitParams pbound;
    BkgModSingleFlowFitKrateParams pboundkrate;
    
    float decision_threshold[NUMFB];
    
    single_flow_optimizer();
    ~single_flow_optimizer();
    void SetUpperLimitAmplFit(float AmplLim,float krateLim, float dmultLim);
    void SetLowerLimitAmplFit(float AmplLim,float krateLim, float dmultLim);
    void AllocLevMar(TimeCompression &time_c, PoissonCDFApproxMemo *math_poiss, float damp_kmult);
    void Delete();
    void FitKrateOneFlow(int fnum, float *evect, 
                                            bead_params *p, float *signal_corrected, float *nucRise, 
                                            int *i_start, flow_buffer_info &my_flow, TimeCompression &time_c, 
                                            EmphasisClass &emphasis_data,RegionTracker &my_regions);
    void FitThisOneFlow(int fnum, float *evect, bead_params *p,  float *signal_corrected, float *nucRise, int *i_start,
                                           flow_buffer_info &my_flow, TimeCompression &time_c, EmphasisClass &emphasis_data,RegionTracker &my_regions);
    void FitOneFlow(int fnum, float *evect, bead_params *p,  float *signal_corrected, float *nucRise, int *i_start,
                                           flow_buffer_info &my_flow, TimeCompression &time_c, EmphasisClass &emphasis_data,RegionTracker &my_regions);
                                           void FillDecisionThreshold(float *nuc_threshold, int *my_nucs);
};



#endif // SINGLEFLOWFIT_H