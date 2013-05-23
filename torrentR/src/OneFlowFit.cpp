/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <string>
#include <iostream>
#include <Rcpp.h>
#include "DiffEqModel.h"
#include "DNTPRiseModel.h"

#define AMPLITUDE 0
#define KMULT 1

#include "TraceCurry.h"
#include "BkgModSingleFlowFit.h"



using namespace std;

// duplicates the single flow "tuning" fit 
// that we do after the regional parameters and well parameters are nailed down.
RcppExport SEXP SingleFlowFitR(
    SEXP R_observed,
    SEXP R_nuc_rise, SEXP R_sub_steps, 
    SEXP R_deltaFrame, SEXP R_my_start, 
    SEXP R_Astart, SEXP R_KmultStart,
    SEXP R_C, SEXP R_Amplitude, SEXP R_copies, SEXP R_krate, SEXP R_kmax, SEXP R_diffusion, SEXP R_sens, SEXP R_tauB) 
{
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    vector<float> observed = Rcpp::as< vector<float> > (R_observed);
    vector<float> nuc_rise = Rcpp::as< vector<float> > (R_nuc_rise);
    int sub_steps = Rcpp::as<int> (R_sub_steps);
    vector<float> delta_frame = Rcpp::as< vector<float> > (R_deltaFrame);
    int my_start_index = Rcpp::as<int> (R_my_start);
    float max_concentration = Rcpp::as<float> (R_C);
    float amplitude = Rcpp::as<float> (R_Amplitude);
    float copies = Rcpp::as<float> (R_copies);
    float krate = Rcpp::as<float> (R_krate);
    float kmax = Rcpp::as<float> (R_kmax);
    float diffusion = Rcpp::as<float> (R_diffusion);
    float sens = Rcpp::as<float> (R_sens);
    float tauB = Rcpp::as<float> (R_tauB);
    float Astart = Rcpp::as<float> (R_Astart);
    float Kstart = Rcpp::as<float> (R_KmultStart);

    int my_frame_len = delta_frame.size();

    float *old_nuc_rise, *old_delta_frame, *tmp_delta_frame;
    old_nuc_rise = new float [nuc_rise.size()];
    old_delta_frame = new float [my_frame_len];
    tmp_delta_frame = new float [my_frame_len];

    float frames_per_second = 15.0f;

    for (int i=0; i<my_frame_len; i++){
      old_delta_frame[i] = delta_frame[i]/frames_per_second; // keep interface transparent for R as this is now in seconds
      tmp_delta_frame[i] = delta_frame[i]; // dumb, but that's the way the code caches
    }
    // may have more sub-steps than the output
    int nuc_len = nuc_rise.size();
    for (int i=0; i<nuc_len; i++){
      old_nuc_rise[i] = nuc_rise[i];
    }

    float *old_observed;
    old_observed = new float [observed.size()];
    int o_size = observed.size();
    for (int i=0; i<o_size; i++)
      old_observed[i] = observed[i];

    float *old_vb_out;


    // output in frames synchronized
    old_vb_out = new float [my_frame_len];

  

    PoissonCDFApproxMemo    *math_poiss = new PoissonCDFApproxMemo;
    math_poiss->Allocate(MAX_POISSON_TABLE_COL,MAX_POISSON_TABLE_ROW,POISSON_TABLE_STEP);
    math_poiss->GenerateValues();
    

    float gain = 1.0f;
    if (max_concentration<0) printf("useless");
    // calculate cumulative hydrogens from amplitude (hp mixture), copies on bead, krate, kmax, diffusion
    // and of course the rate at which nuc is available above the well

// generate object needed
    BkgModSingleFlowFit *oneFlowFitKrate;
//@TODO: frameNumber dummy variable?
float *frameNumber = &tmp_delta_frame[0];

    oneFlowFitKrate = new BkgModSingleFlowFit (my_frame_len,frameNumber,tmp_delta_frame,old_delta_frame, math_poiss, 2); 
    oneFlowFitKrate->SetLambdaThreshold(1.0);
    float max_param[2];
    float min_param[2];
    min_param[0] = 0.001;
    min_param[1] = 0.65;
    max_param[0] = 10.0;
    max_param[1] = 1.8;
    oneFlowFitKrate->SetParamMax(max_param);
    oneFlowFitKrate->SetParamMin(min_param);
    
  //oneFlowFitKrate->SetWeightVector (evect);
  oneFlowFitKrate->SetLambdaStart (1E-20);

  oneFlowFitKrate->calc_trace.SetContextParams(my_start_index, old_nuc_rise, sub_steps, max_concentration,
          copies, krate, kmax, diffusion, sens, gain, tauB);

  oneFlowFitKrate->SetFvalCacheEnable(true);
  // evaluate the fancier model
  // specify starting point somehow
 
  oneFlowFitKrate->SetNthParam(Astart,AMPLITUDE);
  oneFlowFitKrate->SetNthParam(Kstart,KMULT);
// do our fit
  int iter = oneFlowFitKrate->Fit (NUMSINGLEFLOWITER,old_observed);

// return the data
  oneFlowFitKrate->Evaluate(old_vb_out);
  float Ampl =  oneFlowFitKrate->ReturnNthParam(AMPLITUDE);
  float Kmult = oneFlowFitKrate->ReturnNthParam(KMULT);

    delete oneFlowFitKrate;
    delete math_poiss;

    vector<double> my_vb_out;

    for (int i=0; i<my_frame_len; i++)
    {
      my_vb_out.push_back(old_vb_out[i]);

    }

    ret = Rcpp::List::create(Rcpp::Named("RedTrace") = my_vb_out,
                             Rcpp::Named("Ampl")     = Ampl,
                             Rcpp::Named("x")        = amplitude,
                             Rcpp::Named("Kmult")    = Kmult,
                             Rcpp::Named("Iter")     = iter);

    delete[] old_vb_out;


    delete[] old_nuc_rise;
    delete[] old_delta_frame;
    delete[] tmp_delta_frame;
    delete[] old_observed;
    

  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


