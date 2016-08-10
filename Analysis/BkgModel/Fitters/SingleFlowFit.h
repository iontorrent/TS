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
// uses struct BeadParams and struct reg_params...so this include needs to be below
// those definitions

#define AMPLITUDE 0
#define KMULT 1

#include "BkgModSingleFlowFit.h"



class single_flow_optimizer
{
  public:
    //PoissonCDFApproxMemo *math_poiss;
    // object used to fit a single well's data in a single flow
    BkgModSingleFlowFit *oneFlowFit;

    BkgModSingleFlowFit *oneFlowFitKrate;

    // extra decision boundaries to try to trap slow incorporations
     float kmult_at_bottom;
     float fit_error_too_high; //@TODO: this does not scale with signal!!!!
     float final_minimum_kmult;
     //float extend_emphasis_amount;

    // keep these around to avoid allocating when I need to talk to the above objects
    float local_max_param[2];
    float local_min_param[2];
    float val_param[2];
    float pmax_param[2];
    float pmin_param[2];
    
    float decision_threshold;
    bool always_slow;

    bool use_fval_cache;

    bool gauss_newton_fit;
    int cur_hits;
    
    single_flow_optimizer();
    ~single_flow_optimizer();
    void SetUpperLimitAmplFit(float AmplLim,float krateLim);
    void SetLowerLimitAmplFit(float AmplLim,float krateLim);
    void AllocLevMar(TimeCompression &time_c, PoissonCDFApproxMemo *math_poiss,  float kmult_low_limit, float kmult_hi_limit, float AmplLowerLimit);
    void SetupOneParameter(TimeCompression &time_c, PoissonCDFApproxMemo *_math_poiss);
    void SetupTwoParameter(TimeCompression &time_c, PoissonCDFApproxMemo *_math_poiss);

    void Delete();

    // picks from the below options
    int FitOneFlow (int fnum, float *evect, BeadParams *p,  error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
                    int flow_block_start,  EmphasisClass &emphasis_data,RegionTracker &my_regions);
    // my different optimizers
    int FitStandardPath (int fnum, float *evect, BeadParams *p,  error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
        int flow_block_start,  EmphasisClass &emphasis_data,RegionTracker &my_regions);
    int FitKrateOneFlow(int fnum, float *evect, BeadParams *p, error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
                         int flow_block_start,  EmphasisClass &emphasis_data,RegionTracker &my_regions);
    int FitThisOneFlow(int fnum, float *evect, BeadParams *p,  error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
                        int flow_block_start,  EmphasisClass &emphasis_data,RegionTracker &my_regions);
                                    
    void FillDecisionThreshold(float nuc_threshold);
    int SpecialReFitSlowIncorporations(int fnum, float *evect, BeadParams *p, float *signal_corrected, EmphasisClass &emphasis_data);
    void SpecialStartChooseKmult(int fnum, BeadParams *p, error_track *err_t, float *signal_corrected);
    // initialize relevant concerns
    void BringUpOptimizer(BkgModSingleFlowFit *OneFit, int fnum, float *evect, BeadParams *p,  int NucID, float *lnucRise, int l_i_start,
                                                 int flow_block_start, RegionTracker &my_regions);
    void ReturnTrackedData(BkgModSingleFlowFit *OneFit,int fnum,  BeadParams *p, error_track *err_t, float *signal_corrected, float *signal_predicted, EmphasisClass &emphasis_data);


    void ResetStandardLimits();

    void SetUpEmphasisForLevMarOptimizer(EmphasisClass* emphasis_data)
    {
      if (oneFlowFit)
        oneFlowFit->SetUpEmphasis(emphasis_data);      
      if (oneFlowFitKrate)
        oneFlowFitKrate->SetUpEmphasis(emphasis_data);      
    }
    void SendLimitsToOptimizer(BkgModSingleFlowFit *OneFit);

};



#endif // SINGLEFLOWFIT_H
