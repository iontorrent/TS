/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <Rcpp.h>
#include "CafieSolver.h"

using namespace std;

RcppExport SEXP SimulateCAFIE(SEXP Rseq, SEXP RflowOrder, SEXP Rcf, SEXP Rie, SEXP Rdr, SEXP Rnflows, SEXP RhpSignal, SEXP RsigMult)
{
  SEXP ret = R_NilValue;
  //char *exceptionMesg = NULL;

  try {
    const char* seq     = Rcpp::as<const char*>(Rseq);
    const char* order   = Rcpp::as<const char*>(RflowOrder);
    double      cf      = Rcpp::as<double>(Rcf);
    double      ie      = Rcpp::as<double>(Rie);
    double      dr      = Rcpp::as<double>(Rdr);
    int         nflows  = Rcpp::as<int>(Rnflows);
    double      sigMult = Rcpp::as<double>(RsigMult);
    Rcpp::NumericVector tempHpSignal(RhpSignal);
    int nHpSignal = tempHpSignal.size();
  
    if(nHpSignal != MAX_MER) {
      stringstream temp;
      temp << MAX_MER;
      std::string exception = "hpSignal must be of length " + temp.str();
      cerr << exception << endl;
      //exceptionMesg = strdup(exception.c_str());
    } else {
      double *hpSignal = new double[nHpSignal];
      for(int i=0; i<nHpSignal; i++)
      	hpSignal[i] = tempHpSignal(i);
      
      double* predicted  = new double[nflows];
       
      CafieSolver solver;
      solver.SimulateCAFIE(predicted, seq, order, cf, ie, dr, nflows, hpSignal, nHpSignal, sigMult);

      Rcpp::NumericVector out_signal(nflows);
      for(int i=0; i<nflows; i++)
        out_signal(i) = predicted[i];

      ret = Rcpp::List::create(Rcpp::Named("sig") = out_signal);

      delete [] hpSignal;

    }
  } catch(std::exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  
  return ret;
}


