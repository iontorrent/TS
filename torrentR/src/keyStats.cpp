/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include "CafieSolver.h"

RcppExport SEXP keyStats(SEXP measured_in, SEXP keyFlow_in) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    Rcpp::NumericMatrix measured_temp(measured_in);
    int nWell         = measured_temp.rows();
    int nMeasuredFlow = measured_temp.cols();
    Rcpp::NumericVector keyFlow_temp(keyFlow_in);
    int nKeyFlow      = keyFlow_temp.size();
  
    if(nKeyFlow > nMeasuredFlow) {
      std::string exception = "Number of measured flows less than length of key\n";
      exceptionMesg = strdup(exception.c_str());
    } else {
      double *key_1_med = new double[nWell];
      double *key_0_med = new double[nWell];
      double *key_1_sd  = new double[nWell];
      double *key_0_sd  = new double[nWell];
      double *key_sig   = new double[nWell];
      double *key_sd    = new double[nWell];
      double *key_snr   = new double[nWell];
      double *measured  = new double[nMeasuredFlow];
      int *keyFlow      = new int[keyFlow_temp.size()];
      for(int i=0; i<nKeyFlow; i++)
        keyFlow[i] = keyFlow_temp(i);

      for(int well=0; well<nWell; well++) {
        for(int flow=0; flow<nMeasuredFlow; flow++)
          measured[flow] = measured_temp(well,flow);
        key_snr[well] = KeySNR(measured,keyFlow,nKeyFlow,key_0_med+well,key_0_sd+well,key_1_med+well,key_1_sd+well,key_sig+well,key_sd+well);
      }

      Rcpp::NumericVector key_1_med_ret(nWell);
      Rcpp::NumericVector key_1_sd_ret(nWell);
      Rcpp::NumericVector key_0_med_ret(nWell);
      Rcpp::NumericVector key_0_sd_ret(nWell);
      Rcpp::NumericVector key_sig_ret(nWell);
      Rcpp::NumericVector key_sd_ret(nWell);
      Rcpp::NumericVector key_snr_ret(nWell);
      for(int well=0; well<nWell; well++) {
        key_1_med_ret(well) = key_1_med[well];
        key_1_sd_ret(well)  = key_1_sd[well];
        key_0_med_ret(well) = key_0_med[well];
        key_0_sd_ret(well)  = key_0_sd[well];
        key_sig_ret(well)   = key_sig[well];
        key_sd_ret(well)    = key_sd[well];
        key_snr_ret(well)   = key_snr[well];
      }
      ret = Rcpp::List::create(Rcpp::Named("key_1_med") = key_1_med_ret,
                               Rcpp::Named("key_0_med") = key_0_med_ret,
                               Rcpp::Named("key_1_sd")  = key_1_sd_ret,
                               Rcpp::Named("key_0_sd")  = key_0_sd_ret,
                               Rcpp::Named("key_sig")   = key_sig_ret,
                               Rcpp::Named("key_sd")    = key_sd_ret,
                               Rcpp::Named("key_snr")   = key_snr_ret);

      delete [] measured;
      delete [] keyFlow;
      delete [] key_1_med;
      delete [] key_0_med;
      delete [] key_1_sd;
      delete [] key_0_sd;
      delete [] key_sig;
      delete [] key_sd;
      delete [] key_snr;
    }
  } catch(std::exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
