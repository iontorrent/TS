/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <string>
#include <iostream>
#include <Rcpp.h>
#include "RegionParams.h"

using namespace std;

//compute the change in parameters over time
//use the annoying "hyperparameter" models that bkg model uses
// exporting the models not because they are complicated
// but to make sure the code is >consistent<
// NucModifyRatio
// etbR
// RatioDrift
// flow = number of flow
RcppExport SEXP AdjustEmptyToBeadRatioForFlowR(SEXP R_etbR, SEXP R_NucModifyRatio, SEXP R_RatioDrift, SEXP R_flow, SEXP R_fit_taue) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    float etbR = Rcpp::as<float>(R_etbR);
    float NucModifyRatio = Rcpp::as<float>(R_NucModifyRatio);
    float RatioDrift = Rcpp::as<float>(R_RatioDrift);
    int flow = Rcpp::as<int>(R_flow);
    bool fit_taue = Rcpp::as<bool> (R_fit_taue);
    float out_val;
    if (!fit_taue){
      // if_use_obsolete_etbR_equation==true, therefore Copy(=1.0) and phi(=0.6) are not used
        float local_phi = 0.6;
        float local_Copy = 1.0;
        float local_Ampl = 0.0;
       
        out_val = xAdjustEmptyToBeadRatioForFlow(etbR,local_Ampl, local_Copy, local_phi, NucModifyRatio, RatioDrift, flow, true);
      }
    else
      out_val = xAdjustEmptyToBeadRatioForFlowWithAdjR(etbR,NucModifyRatio,RatioDrift,flow);

    ret = Rcpp::List::create(Rcpp::Named("etbR") = out_val);

  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
//xComputeTauBfromEmptyUsingRegionLinearModel(float tau_R_m,float tau_R_o, float etbR);
RcppExport SEXP ComputeTauBfromEmptyUsingRegionLinearModelR(SEXP R_etbR, SEXP R_tau_R_m, SEXP R_tau_R_o) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    float etbR = Rcpp::as<float>(R_etbR);
    float tau_R_m = Rcpp::as<float>(R_tau_R_m);
    float tau_R_o = Rcpp::as<float>(R_tau_R_o);
    
    float out_val = xComputeTauBfromEmptyUsingRegionLinearModel(tau_R_m, tau_R_o, etbR,4,65);

    ret = Rcpp::List::create(Rcpp::Named("tauB") = out_val);

  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


RcppExport SEXP ComputeTauBfromEmptyUsingRegionLinearModelUsingTauER(SEXP R_etbR, SEXP R_tauE) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    float etbR = Rcpp::as<float>(R_etbR);
    float tauE = Rcpp::as<float>(R_tauE);
    
    float out_val = xComputeTauBfromEmptyUsingRegionLinearModelWithAdjR(tauE,etbR,4,65);
    
    ret = Rcpp::List::create(Rcpp::Named("tauB") = out_val);
    
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}

