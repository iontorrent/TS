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
    RcppMatrix<double> cc(RnucConc);
    RcppVector<double> cf(Rcf);
    RcppVector<double> ie(Rie);
    RcppVector<double> dr(Rdr);
    RcppVector<double> hpScale(RhpScale);
  
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
      exceptionMesg = copyMessageToR(exception.c_str());
    } else if(cc.rows() != (int) N_NUCLEOTIDES) {
      string exception = "concentration matrix should have 4 rows\n";
      exceptionMesg = copyMessageToR(exception.c_str());
    } else if(cc.cols() != (int) N_NUCLEOTIDES) {
      string exception = "concentration matrix should have 4 columns\n";
      exceptionMesg = copyMessageToR(exception.c_str());
    } else {
      weight_vec_t cfMod(cf.size());
      for(int i=0; i<cf.size(); i++)
        cfMod[i] = cf(i);
      weight_vec_t ieMod(ie.size());
      for(int i=0; i<ie.size(); i++)
        ieMod[i] = ie(i);
      weight_vec_t drMod(dr.size());
      for(int i=0; i<dr.size(); i++)
        drMod[i] = dr(i);
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
  
      RcppVector<double> out_signal(nFlow);
      for(unsigned int i=0; i<nFlow; i++)
        out_signal(i) = signal[i];

      RcppResultSet rs;
      rs.add("sig",        out_signal);

      if(returnIntermediates) {
        unsigned int nHP = hpWeight[0].size();
        RcppMatrix<double> out_hpWeight(nFlow,nHP);
        RcppVector<double> out_droopWeight(nFlow);
        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++) {
          out_droopWeight(iFlow) = droopWeight[iFlow];
          for(unsigned int iHP=0; iHP < nHP; iHP++) {
            out_hpWeight(iFlow,iHP)    = hpWeight[iFlow][iHP];
          }
        }
        rs.add("hpWeight",        out_hpWeight);
        rs.add("droopWeight",     out_droopWeight);
      }
      ret = rs.getReturnList();
    }
  } catch(exception& ex) {
    exceptionMesg = copyMessageToR(ex.what());
  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
