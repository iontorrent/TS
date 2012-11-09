/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SingleFlowFit.h"


single_flow_optimizer::single_flow_optimizer()
{
    // this can't get allocated until after we know how many data points we will
    // be processing
    oneFlowFit = NULL;
    oneFlowFitKrate = NULL;
    ProjectionFit = NULL;
    AltFit = NULL;

    for (int i=0; i<2; i++)
    {
        max_param[i] = min_param[i] = val_param[i] = 0.0f;
    }

    decision_threshold = 0.0f; // always use variable rate
    var_kmult_only = false;
    use_projection_search_ampl_fit = false;
    cur_hits = 0;
    fit_alt = false;
    use_fval_cache = false; // really?  should this be true or does it break lev-mar?

    retry_limit = 0;
}

void single_flow_optimizer::Delete()
{
    if (AltFit!=NULL) delete AltFit;
    if (oneFlowFitKrate!=NULL) delete oneFlowFitKrate;
    if (oneFlowFit!=NULL) delete oneFlowFit;
    if (ProjectionFit!=NULL) delete ProjectionFit;
}

single_flow_optimizer::~single_flow_optimizer()
{
    Delete();
}

void single_flow_optimizer::FillDecisionThreshold (float nuc_threshold)
{
    decision_threshold = nuc_threshold;
}

void single_flow_optimizer::SetLowerLimitAmplFit (float AmplLim,float krateLim)
{
    // not all may apply

    min_param[AMPLITUDE] = AmplLim;
    min_param[KMULT] = krateLim;

    oneFlowFit->SetParamMin (&min_param[0]);
    oneFlowFitKrate->SetParamMin (&min_param[0]);
}


void single_flow_optimizer::SetUpperLimitAmplFit (float AmplLim,float krateLim)
{
    // not all may apply

    max_param[AMPLITUDE] = AmplLim;
    max_param[KMULT] = krateLim;

    oneFlowFit->SetParamMax (&max_param[0]);
    oneFlowFitKrate->SetParamMax (&max_param[0]);
}

void single_flow_optimizer::AllocLevMar (TimeCompression &time_c, PoissonCDFApproxMemo *_math_poiss, float damp_kmult, bool _var_kmult_only, float kmult_low_limit, float kmult_hi_limit, float AmplLowerLimit)
{
    //@TODO:  All these fitters share the curried calc_trace object, so we can do something useful by sharing it
    math_poiss = _math_poiss;
    // one parameter fit
    oneFlowFit = new BkgModSingleFlowFit (time_c.npts(),&time_c.frameNumber[0],&time_c.deltaFrame[0],&time_c.deltaFrameSeconds[0],math_poiss,1);
    // two parameter fit
    oneFlowFitKrate = new BkgModSingleFlowFit (time_c.npts(),&time_c.frameNumber[0],&time_c.deltaFrame[0],&time_c.deltaFrameSeconds[0], math_poiss, 2);

    // set up levmar state before going off on a wild adventure
    oneFlowFit->SetLambdaThreshold (10.0);
    oneFlowFitKrate->SetLambdaThreshold (1.0);

    SetLowerLimitAmplFit (AmplLowerLimit,kmult_low_limit);
    SetUpperLimitAmplFit (MAX_HPLEN-1,kmult_hi_limit);

    float my_prior[2] = {0.0, 1.0}; // amplitude, kmult
    float my_damper[2] = {0.0,damp_kmult*time_c.npts()}; // must make a difference to change kmult
    oneFlowFitKrate->SetPrior (&my_prior[0]);
    oneFlowFitKrate->SetDampers (&my_damper[0]);
    var_kmult_only = _var_kmult_only;

    // boot up the new projection fit >which is not an instance of LevMar<
    ProjectionFit = new ProjectionSearchOneFlow (time_c.npts(), &time_c.deltaFrame[0], &time_c.deltaFrameSeconds[0], math_poiss);
    ProjectionFit->max_paramA = MAX_HPLEN-1;
    ProjectionFit->min_paramA = AmplLowerLimit;

    AltFit = new AlternatingDirectionOneFlow (time_c.npts(), &time_c.deltaFrame[0], &time_c.deltaFrameSeconds[0], math_poiss);
}

void single_flow_optimizer::FitKrateOneFlow (int fnum, float *evect, bead_params *p, error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
        flow_buffer_info &my_flow, TimeCompression &time_c, EmphasisClass &emphasis_data,RegionTracker &my_regions)
{

    oneFlowFitKrate->SetWeightVector (evect);
    oneFlowFitKrate->SetLambdaStart (1E-20);
    oneFlowFitKrate->calc_trace.SetWellRegionParams (p,&my_regions.rp,fnum,
            NucID,my_flow.buff_flow[fnum],
            l_i_start,lnucRise);

    oneFlowFitKrate->SetFvalCacheEnable (use_fval_cache);
    // evaluate the fancier model
    oneFlowFitKrate->InitParams();

// float start_guess = p->Ampl[fnum];
    oneFlowFitKrate->calc_trace.ResetEval();
    oneFlowFitKrate->Fit (NUMSINGLEFLOWITER,signal_corrected);
    cur_hits = oneFlowFitKrate->calc_trace.GetEvalCount();
    p->Ampl[fnum] = oneFlowFitKrate->ReturnNthParam (AMPLITUDE);
    p->kmult[fnum] = oneFlowFitKrate->ReturnNthParam (KMULT);


    // store output for later
    oneFlowFitKrate->SetWeightVector (emphasis_data.EmphasisVectorByHomopolymer[emphasis_data.numEv-1]);
    err_t->mean_residual_error[fnum] = sqrt (oneFlowFitKrate->GetMeanSquaredError (signal_corrected,use_fval_cache)); // backwards compatibility
    oneFlowFitKrate->ReturnPredicted(signal_predicted, use_fval_cache);
}


void single_flow_optimizer::FitThisOneFlow (int fnum, float *evect, bead_params *p,  error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
        flow_buffer_info &my_flow, TimeCompression &time_c, EmphasisClass &emphasis_data,RegionTracker &my_regions)
{
    oneFlowFit->SetWeightVector (evect);
    // SetWellParams should leave data invariant
    oneFlowFit->SetLambdaStart (1E-20);
    oneFlowFit->calc_trace.SetWellRegionParams (p,&my_regions.rp,fnum,
            NucID,my_flow.buff_flow[fnum],
            l_i_start,lnucRise);

    oneFlowFit->SetFvalCacheEnable (use_fval_cache);
    oneFlowFit->InitParams();
    oneFlowFit->calc_trace.ResetEval();
    oneFlowFit->Fit (NUMSINGLEFLOWITER,signal_corrected); // Not enough evidence to warrant krate fitting to this flow, do the simple thing.
    cur_hits = oneFlowFit->calc_trace.GetEvalCount();
    p->Ampl[fnum] = oneFlowFit->ReturnNthParam (AMPLITUDE);

    // re-calculate residual based on a the highest hp weighting vector (which is the most flat)
    oneFlowFit->SetWeightVector (emphasis_data.EmphasisVectorByHomopolymer[emphasis_data.numEv-1]);
    err_t->mean_residual_error[fnum] = sqrt (oneFlowFit->GetMeanSquaredError (signal_corrected,use_fval_cache)); // backwards compatibility
    oneFlowFit->ReturnPredicted(signal_predicted, use_fval_cache);

}

void single_flow_optimizer::FitProjection (int fnum, float *evect, bead_params *p,  error_track *err_t, float *signal_corrected, int NucID, float *lnucRise, int l_i_start,
        flow_buffer_info &my_flow, TimeCompression &time_c, EmphasisClass &emphasis_data,RegionTracker &my_regions)
{
    ProjectionFit->SetWeightVector (evect);
    // SetWellParams should leave data invariant

    ProjectionFit->calc_trace.SetWellRegionParams (p,&my_regions.rp,fnum,
            NucID,my_flow.buff_flow[fnum],
            l_i_start,lnucRise);

    ProjectionFit->SetFvalCacheEnable (use_fval_cache);


    ProjectionFit->ProjectionSearch (signal_corrected);

    p->Ampl[fnum] = ProjectionFit->paramA;

    // re-calculate residual based on a the highest hp weighting vector (which is the most flat)
    ProjectionFit->SetWeightVector (emphasis_data.EmphasisVectorByHomopolymer[emphasis_data.numEv-1]);
    err_t->mean_residual_error[fnum] = sqrt (ProjectionFit->GetMeanSquaredError (signal_corrected,use_fval_cache)); // backwards compatibility

}

void single_flow_optimizer::FitAlt (int fnum, float *evect, bead_params *p,  error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
                                    flow_buffer_info &my_flow, TimeCompression &time_c, EmphasisClass &emphasis_data,RegionTracker &my_regions, bool krate_fit)
{
    AltFit->SetWeightVector (evect);
    // SetWellParams should leave data invariant

    AltFit->calc_trace.SetWellRegionParams (p,&my_regions.rp,fnum,
                                            NucID,my_flow.buff_flow[fnum],
                                            l_i_start,lnucRise);

    AltFit->calc_trace.ResetEval();
    AltFit->Defaults(); // reset to defaults
    AltFit->SetMin(min_param);
    AltFit->SetMax(max_param);
    
    // start out in a good location
    // crude initial starting location has no role
    AltFit->paramA[AMPLITUDE] = AltFit->calc_trace.GuessAmplitude(signal_corrected);
    
    //@TODO: local decision for emphasis vector
    AltFit->emphasis_ptr = &emphasis_data;
    AltFit->DynamicEmphasis(AltFit->paramA[AMPLITUDE]); // evect now has no role
    
    AltFit->AlternatingDirection (signal_corrected, krate_fit); // krate_fit has no role
    cur_hits = AltFit->calc_trace.GetEvalCount();
    

// fix direct accesses of variables at some point
    p->Ampl[fnum] = AltFit->paramA[AMPLITUDE];
    p->kmult[fnum] = AltFit->paramA[KMULT];
    

    // re-calculate residual based on a the highest hp weighting vector (which is the most flat)
    AltFit->SetWeightVector (emphasis_data.EmphasisVectorByHomopolymer[emphasis_data.numEv-1]);
    float local_residual = sqrt (AltFit->GetMeanSquaredError (signal_corrected,true));
    err_t->mean_residual_error[fnum] = local_residual; // backwards compatibility
    AltFit->ReturnPredicted(signal_predicted,true);
    //let's try a diagnostic here
    if (p->trace_ndx<(-1))
      printf("ALTFIT: %d %d %d %f %f %f\n", p->trace_ndx, AltFit->calc_trace.flow, cur_hits, p->Ampl[fnum],p->kmult[fnum], local_residual);

}


void single_flow_optimizer::FitStandardPath (int fnum, float *evect, bead_params *p,  error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
        flow_buffer_info &my_flow, TimeCompression &time_c, EmphasisClass &emphasis_data,RegionTracker &my_regions)
{
    float A_start = p->Ampl[fnum];
    bool done = false;
    int retry = 0;

    while (!done && (retry <= retry_limit))
    {
        if (retry > 0)
            emphasis_data.CustomEmphasis (evect, p->Ampl[fnum]);

        bool krate_fit = ( ( (p->Copies*p->Ampl[fnum]) > decision_threshold)); // this may not be the best way of deciding this
        krate_fit = krate_fit || var_kmult_only;


        if (krate_fit)
        {
            FitKrateOneFlow (fnum,evect,p,err_t, signal_corrected,signal_predicted, NucID, lnucRise, l_i_start,my_flow,time_c,emphasis_data,my_regions);
        }
        else
        {
            //  if (use_projection_search_ampl_fit)
            //    FitProjection (fnum,evect,p, err_t, signal_corrected,NucID, lnucRise, l_i_start,my_flow,time_c,emphasis_data,my_regions);
            //  else
            FitThisOneFlow (fnum,evect,p, err_t, signal_corrected,signal_predicted, NucID, lnucRise, l_i_start,my_flow,time_c,emphasis_data,my_regions);
        }


        float delta = A_start - p->Ampl[fnum];

        if (fabs(delta) > 0.1)
            A_start = p->Ampl[fnum];
        else
            done = true;

        retry++;
    }
}

void single_flow_optimizer::FitOneFlow (int fnum, float *evect, bead_params *p,  error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
                                        flow_buffer_info &my_flow, TimeCompression &time_c, EmphasisClass &emphasis_data,RegionTracker &my_regions)
{
    if (!fit_alt)
    {
        FitStandardPath(fnum,evect,p,err_t,signal_corrected, signal_predicted,NucID, lnucRise,l_i_start,my_flow,time_c, emphasis_data,my_regions);
    }
    else
    {
        bool krate_fit = ( ( (p->Copies*p->Ampl[fnum]) > decision_threshold)); // this may not be the best way of deciding this
        krate_fit = krate_fit || var_kmult_only;
        FitAlt (fnum,evect,p, err_t, signal_corrected,signal_predicted, NucID, lnucRise, l_i_start,my_flow,time_c,emphasis_data,my_regions,krate_fit);
    }

}
