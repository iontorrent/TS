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
RcppExport SEXP AdjustEmptyToBeadRatioForFlowR(SEXP R_etbR, SEXP R_NucModifyRatio, SEXP R_RatioDrift, SEXP R_flow) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    float etbR = Rcpp::as<float>(R_etbR);
    float NucModifyRatio = Rcpp::as<float>(R_NucModifyRatio);
    float RatioDrift = Rcpp::as<float>(R_RatioDrift);
    int flow = Rcpp::as<int>(R_flow);

    float out_val = xAdjustEmptyToBeadRatioForFlow(etbR,NucModifyRatio,RatioDrift,flow);

      RcppResultSet rs;
      rs.add("etbR",      out_val);
      ret = rs.getReturnList();

  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
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
    
    float out_val = xComputeTauBfromEmptyUsingRegionLinearModel(tau_R_m, tau_R_o, etbR);

      RcppResultSet rs;
      rs.add("tauB",      out_val);
      ret = rs.getReturnList();

  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}

