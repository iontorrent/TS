/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <string>
#include <vector>
#include <Rcpp.h>
#include "PhaseSim.h"

using namespace std;

RcppExport SEXP phaseSimulator(SEXP Rseq, SEXP RflowCycle, SEXP RnucConc, SEXP Rcf, SEXP Rie, SEXP Rdr, SEXP RhpScale, SEXP RnFlow, SEXP RmaxAdv, SEXP RdroopType, SEXP RextraTaps)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    string seq             = Rcpp::as<string>(Rseq);
    string flowCycle       = Rcpp::as<string>(RflowCycle);
    unsigned int nFlow     = (unsigned int) Rcpp::as<int>(RnFlow);
    unsigned int maxAdv    = (unsigned int) Rcpp::as<int>(RmaxAdv);
    unsigned int extraTaps = (unsigned int) Rcpp::as<int>(RextraTaps);
    string drType          = Rcpp::as<string>(RdroopType);
    Rcpp::NumericMatrix cc(RnucConc);
    Rcpp::NumericMatrix cf(Rcf);
    Rcpp::NumericMatrix ie(Rie);
    Rcpp::NumericMatrix dr(Rdr);
    Rcpp::NumericVector hpScale(RhpScale);
  
    DroopType droopType;
    bool badDroopType = false;
    if(drType == "ONLY_WHEN_INCORPORATING") {
      droopType = ONLY_WHEN_INCORPORATING;
    } else if(drType == "EVERY_FLOW") {
      droopType = EVERY_FLOW;
    } else {
      badDroopType = true;
    }
  
    if(badDroopType) {
      string exception = "bad droop type supplied\n";
      exceptionMesg = strdup(exception.c_str());
    } else if(cc.rows() != (int) N_NUCLEOTIDES) {
      string exception = "concentration matrix should have 4 rows\n";
      exceptionMesg = strdup(exception.c_str());
    } else if(cc.cols() != (int) N_NUCLEOTIDES) {
      string exception = "concentration matrix should have 4 columns\n";
      exceptionMesg = strdup(exception.c_str());
    } else {
      weight_vec_t cfMod(cf.ncol());
      for(int i=0; i<cf.ncol(); i++)
        cfMod[i] = cf(0,i);
      weight_vec_t ieMod(ie.ncol());
      for(int i=0; i<ie.ncol(); i++)
        ieMod[i] = ie(0,i);
      weight_vec_t drMod(dr.ncol());
      for(int i=0; i<dr.ncol(); i++)
        drMod[i] = dr(0,i);
      weight_vec_t hpScaleMod(hpScale.size());
      for(int i=0; i<hpScale.size(); i++)
        hpScaleMod[i] = hpScale(i);
      
      weight_vec_t signal;
      vector<weight_vec_t> hpWeight;
      weight_vec_t droopWeight;
      bool returnIntermediates=true;

      vector<weight_vec_t> conc(cc.rows());
      for(int iRow=0; iRow < cc.rows(); iRow++) {
        conc[iRow].resize(cc.cols());
        for(unsigned int iCol=0; iCol < N_NUCLEOTIDES; iCol++)
          conc[iRow][iCol] = cc(iRow,iCol);
      }

      PhaseSim p;
      p.setExtraTaps(extraTaps);
      p.setHpScale(hpScaleMod);
      p.simulate(flowCycle, seq, conc, cfMod, ieMod, drMod, nFlow, signal, hpWeight, droopWeight, returnIntermediates, droopType, maxAdv);
  
      Rcpp::NumericVector out_signal(nFlow);
      for(unsigned int i=0; i<nFlow; i++)
        out_signal(i) = signal[i];


      if(returnIntermediates) {
        unsigned int nHP = hpWeight[0].size();
        Rcpp::NumericMatrix out_hpWeight(nFlow,nHP);
        Rcpp::NumericVector out_droopWeight(nFlow);
        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++) {
          out_droopWeight(iFlow) = droopWeight[iFlow];
          for(unsigned int iHP=0; iHP < nHP; iHP++) {
            out_hpWeight(iFlow,iHP)    = hpWeight[iFlow][iHP];
          }
        }
        ret = Rcpp::List::create(Rcpp::Named("sig") = out_signal,
                                 Rcpp::Named("hpWeight") = out_hpWeight,
                                 Rcpp::Named("droopWeight") = out_droopWeight);
      } else {
        ret = Rcpp::List::create(Rcpp::Named("sig") = out_signal);
      }
   
    }
  } catch(exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
