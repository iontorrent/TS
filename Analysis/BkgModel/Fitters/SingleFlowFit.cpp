/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SingleFlowFit.h"

#define FITKRATE 1
#define FITAMPONLY 0
#define FITSLOW 2


single_flow_optimizer::single_flow_optimizer()
{
  // this can't get allocated until after we know how many data points we will
  // be processing
  oneFlowFit = NULL;
  oneFlowFitKrate = NULL;

  for (int i=0; i<2; i++)
  {
    local_max_param[i] = local_min_param[i] = pmax_param[i] = pmin_param[i] = val_param[i] = 0.0f;
  }

  decision_threshold = 0.0f; // always use variable rate

  cur_hits = 0;

  use_fval_cache = false; // really?  should this be true or does it break lev-mar?

  gauss_newton_fit = false;
  always_slow = false;
}

void single_flow_optimizer::Delete()
{
  if (oneFlowFitKrate!=NULL) delete oneFlowFitKrate;
  if (oneFlowFit!=NULL) delete oneFlowFit;
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
  local_min_param[AMPLITUDE] = AmplLim;
  local_min_param[KMULT] = krateLim;
  pmin_param[AMPLITUDE] = AmplLim;
  pmin_param[KMULT] = krateLim;

}



void single_flow_optimizer::SetUpperLimitAmplFit (float AmplLim,float krateLim)
{
  local_max_param[AMPLITUDE] = AmplLim;
  local_max_param[KMULT] = krateLim;
  pmax_param[AMPLITUDE] = AmplLim;
  pmax_param[KMULT] = krateLim;
}


void single_flow_optimizer::SendLimitsToOptimizer(BkgModSingleFlowFit *OneFit){
  OneFit->SetParamMin(&local_min_param[0]);
  OneFit->SetParamMax(&local_max_param[0]);
}

void single_flow_optimizer::ResetStandardLimits(){
  SetLowerLimitAmplFit (pmin_param[0],pmin_param[1]);
  SetUpperLimitAmplFit (pmax_param[0],pmax_param[1]);
  SendLimitsToOptimizer(oneFlowFit);
  SendLimitsToOptimizer(oneFlowFitKrate);
}

void single_flow_optimizer::SetupOneParameter(TimeCompression &time_c, PoissonCDFApproxMemo *_math_poiss)
{
  oneFlowFit = new BkgModSingleFlowFit (time_c.npts(),&time_c.frameNumber[0],&time_c.deltaFrame[0],&time_c.deltaFrameSeconds[0],_math_poiss,1);
  oneFlowFit->SetLambdaThreshold (10.0);
  SendLimitsToOptimizer(oneFlowFit);
}

void single_flow_optimizer::SetupTwoParameter(TimeCompression &time_c, PoissonCDFApproxMemo *_math_poiss)
{
  // two parameter fit
  oneFlowFitKrate = new BkgModSingleFlowFit (time_c.npts(),&time_c.frameNumber[0],&time_c.deltaFrame[0],&time_c.deltaFrameSeconds[0], _math_poiss, 2);
  oneFlowFitKrate->SetLambdaThreshold (1.0);
  SendLimitsToOptimizer(oneFlowFitKrate);

  // in case of slow incorporation
  // detect kmult at lower boundary
  // large fit error
  // then allow slowdown
  // and change emphasis to emphasize later flows (using amplkitude, the only method to extend the ad-hoc vectors
  kmult_at_bottom = 0.01f;
  fit_error_too_high = 20.0f; //@TODO: this does not scale with signal!!!!
  final_minimum_kmult = 0.3f;
  //extend_emphasis_amount = 2.0f;  // extend emphasis not currently used
}

void single_flow_optimizer::AllocLevMar (TimeCompression &time_c, PoissonCDFApproxMemo *_math_poiss,  float kmult_low_limit, float kmult_hi_limit, float AmplLowerLimit)
{
  //@TODO:  All these fitters share the curried calc_trace object, so we can do something useful by sharing it
  //math_poiss = _math_poiss;
  SetLowerLimitAmplFit (AmplLowerLimit,kmult_low_limit);
  SetUpperLimitAmplFit (LAST_POISSON_TABLE_COL-0.5f,kmult_hi_limit); // set this lower, so that derivatives using positive steps still work.  No-one expects 23-mers to be accurate, so 22.5 ok
  // one parameter fit
  SetupOneParameter(time_c, _math_poiss);
  SetupTwoParameter(time_c, _math_poiss);
}

void single_flow_optimizer::BringUpOptimizer(BkgModSingleFlowFit *OneFit, int fnum, float *evect, BeadParams *p,  int NucID, float *lnucRise, int l_i_start,
                                             int flow_block_start, RegionTracker &my_regions){
  OneFit->SetWeightVector (evect);
  OneFit->SetLambdaStart (1E-20);
  OneFit->calc_trace.SetWellRegionParams (p,&my_regions.rp,fnum,
                                          NucID, flow_block_start + fnum,
                                          l_i_start,lnucRise);
  OneFit->SetFvalCacheEnable (use_fval_cache);

  // fills in starting guesses from the bead-param pointer(which is the worst possible way to init?
  OneFit->InitParams();

  OneFit->calc_trace.ResetEval();

}

int single_flow_optimizer::SpecialReFitSlowIncorporations(int fnum, float *evect, BeadParams *p, float *signal_corrected, EmphasisClass &emphasis_data)
{
  int local_fit_type = FITKRATE;
  if(fabs( oneFlowFitKrate->ReturnNthParam(KMULT)-local_min_param[KMULT])<kmult_at_bottom){
    float errx=sqrt (oneFlowFitKrate->GetMeanSquaredError (signal_corrected,use_fval_cache));
    if (errx>fit_error_too_high){ //if the fit error is  high it could be a slow incorporation

      local_max_param[KMULT] =local_min_param[KMULT];
      // note by assumption
      local_min_param[KMULT]=final_minimum_kmult;
      SendLimitsToOptimizer(oneFlowFitKrate);
      // current behavior is to start at last position of optimizer
      // do not reset amplitude/kmult
      // do not extend emphasis vector (weights not updated)
      //p->kmult[fnum] = final_minimum_kmult;
      //emphasis_data.CustomEmphasis (evect, oneFlowFitKrate->ReturnNthParam(AMPLITUDE)+extend_emphasis_amount);
      int max_fit_iter = gauss_newton_fit ? NUMSINGLEFLOWITER_GAUSSNEWTON : NUMSINGLEFLOWITER_LEVMAR;
      oneFlowFitKrate->Fit (gauss_newton_fit, max_fit_iter, signal_corrected);
      local_fit_type = FITSLOW;
    }
  }
  return(local_fit_type);
}

void single_flow_optimizer::SpecialStartChooseKmult(int fnum, BeadParams *p, error_track *err_t, float *signal_corrected){
  // spend extra evaluation making sure we don't mess up curve fitting by starting in a poor location
  oneFlowFitKrate->SetNthParam(1.0f, KMULT);
  float eval_one = oneFlowFitKrate->GetMeanSquaredError(signal_corrected, false); // force evaluation at default kmult=1
  oneFlowFitKrate->SetNthParam(local_min_param[KMULT],KMULT);
  float eval_two = oneFlowFitKrate->GetMeanSquaredError(signal_corrected, false);  // evaluate at lower bound
  // start whichever one is better off
  if ((eval_two<eval_one) or (always_slow)){
    oneFlowFitKrate->SetNthParam(local_min_param[KMULT],KMULT);
    p->kmult[fnum] = local_min_param[KMULT]; //? always start low, despite A being chosen with kmult=1
    err_t->initkmult[fnum] = p->kmult[fnum]; // note have to store the revised starting value here, as we actually do over-ride kmult =1
  } else{
    oneFlowFitKrate->SetNthParam(1.0f, KMULT);  // if error not less start at expected value
    p->kmult[fnum] = 1.0f; // don't start low unless it fits better
    err_t->initkmult[fnum] = p->kmult[fnum]; // note have to store the revised starting value here
  }
}


int single_flow_optimizer::FitKrateOneFlow(int fnum, float *evect, BeadParams *p, error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
                                           int flow_block_start, EmphasisClass &emphasis_data,RegionTracker &my_regions)
{

  BringUpOptimizer(oneFlowFitKrate,fnum,evect,p,NucID,lnucRise,l_i_start,flow_block_start, my_regions);
  SpecialStartChooseKmult(fnum, p, err_t, signal_corrected);

  int max_fit_iter = gauss_newton_fit ? NUMSINGLEFLOWITER_GAUSSNEWTON : NUMSINGLEFLOWITER_LEVMAR;
  oneFlowFitKrate->Fit (gauss_newton_fit, max_fit_iter, signal_corrected);

  int local_fit_type = SpecialReFitSlowIncorporations(fnum, evect, p, signal_corrected, emphasis_data);

  ReturnTrackedData(oneFlowFitKrate, fnum, p, err_t, signal_corrected, signal_predicted, emphasis_data);
  return(local_fit_type);
}


int single_flow_optimizer::FitThisOneFlow (int fnum, float *evect, BeadParams *p,  error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
                                           int flow_block_start, EmphasisClass &emphasis_data,RegionTracker &my_regions)
{
  // modify kmult =1 here guarantee
  p->kmult[fnum] = 1.0f; // might start not at kmult=1 even although that is what we are fitting
  BringUpOptimizer(oneFlowFit,fnum,evect,p,NucID,lnucRise,l_i_start,flow_block_start, my_regions);

  int max_fit_iter = gauss_newton_fit ? NUMSINGLEFLOWITER_GAUSSNEWTON : NUMSINGLEFLOWITER_LEVMAR;
  oneFlowFit->Fit (gauss_newton_fit, max_fit_iter, signal_corrected); // Not enough evidence to warrant krate fitting to this flow, do the simple thing.

  ReturnTrackedData(oneFlowFit, fnum, p, err_t, signal_corrected, signal_predicted, emphasis_data);
  return(FITAMPONLY);
}

// common to all fitters is the return of useful information
void single_flow_optimizer::ReturnTrackedData(BkgModSingleFlowFit *OneFit,int fnum,  BeadParams *p, error_track *err_t, float *signal_corrected, float *signal_predicted, EmphasisClass &emphasis_data){
  p->Ampl[fnum] = OneFit->ReturnNthParam (AMPLITUDE);
  if (OneFit->GetnParams()>1)
    p->kmult[fnum] = OneFit->ReturnNthParam(KMULT);
  else
    p->kmult[fnum] = 1.0f; // by definition, if we're not fitting this, it must be kmult=1


  err_t->tauB[fnum] = OneFit->calc_trace.tauB;   // save tauB for output to trace.h5
  err_t->etbR[fnum] = OneFit->calc_trace.etbR;
  err_t->converged[fnum] = OneFit->IsConverged();


  if (!OneFit->IsConverged()){
    // force predict using amplitude/kmult to refresh cache
    // although we don't seem to have this problem, so if we find it we can force

  }  // re-calculate residual based on a the highest hp weighting vector (which is the most flat)
  OneFit->SetWeightVector (emphasis_data.EmphasisVectorByHomopolymer[emphasis_data.numEv-1]);
  err_t->mean_residual_error[fnum] = sqrt (OneFit->GetMeanSquaredError (signal_corrected,use_fval_cache)); // backwards compatibility
  OneFit->ReturnPredicted(signal_predicted, use_fval_cache);
}




int single_flow_optimizer::FitStandardPath (int fnum, float *evect, BeadParams *p,  error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
                                            int flow_block_start,  EmphasisClass &emphasis_data,RegionTracker &my_regions)
{

  int fitType = FITAMPONLY;
  ResetStandardLimits();

  // keep track of our starting point in case the fitter does something crazy from here
  err_t->initA[fnum] = p->Ampl[fnum];
  err_t->initkmult[fnum] = p->kmult[fnum];  // this is usually expected to be 1.0, but might be something else...

  emphasis_data.CustomEmphasis (evect, p->Ampl[fnum]);

  bool krate_fit = ( ( (p->Copies*p->Ampl[fnum]) > decision_threshold)); // this may not be the best way of deciding this
  //krate_fit = krate_fit ; // avoid self assign


  if (krate_fit)
  {
    fitType=FitKrateOneFlow (fnum,evect,p,err_t, signal_corrected,signal_predicted, NucID, lnucRise, l_i_start,flow_block_start,emphasis_data,my_regions);
  }
  else
  {
    fitType=FitThisOneFlow (fnum,evect,p, err_t, signal_corrected,signal_predicted, NucID, lnucRise, l_i_start,flow_block_start,emphasis_data,my_regions);
  }


  return (fitType);
}

int single_flow_optimizer::FitOneFlow (int fnum, float *evect, BeadParams *p,  error_track *err_t, float *signal_corrected, float *signal_predicted, int NucID, float *lnucRise, int l_i_start,
                                       int flow_block_start,EmphasisClass &emphasis_data,RegionTracker &my_regions)
{
  int fitType = 0;

  fitType = FitStandardPath(fnum,evect,p,err_t,signal_corrected, signal_predicted,NucID, lnucRise,l_i_start,flow_block_start, emphasis_data,my_regions);

  return (fitType);
}
