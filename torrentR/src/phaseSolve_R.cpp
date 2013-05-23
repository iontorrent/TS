/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>
#include "PhaseSolve.h"

RcppExport SEXP phaseSolve(SEXP Rsignal, SEXP RflowCycle, SEXP RnucConc, SEXP Rcf, SEXP Rie, SEXP Rdr, SEXP RhpScale, SEXP RdroopType, SEXP RmaxAdv, SEXP RnIterations, SEXP RresidualScale, SEXP RresidualScaleMinFlow, SEXP RresidualScaleMaxFlow, SEXP RextraTaps, SEXP RdebugBaseCall)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    Rcpp::NumericMatrix signal(Rsignal);
    string flowCycle         = Rcpp::as<string>(RflowCycle);
    Rcpp::NumericMatrix cc(RnucConc);
    Rcpp::NumericVector cf(Rcf);
    Rcpp::NumericVector ie(Rie);
    Rcpp::NumericVector dr(Rdr);
    Rcpp::NumericVector hpScale(RhpScale);
    string drType                     = Rcpp::as<string>(RdroopType);
    unsigned int maxAdv               = (unsigned int) Rcpp::as<int>(RmaxAdv);
    unsigned int nIterations          = (unsigned int) Rcpp::as<int>(RnIterations);
    unsigned int extraTaps            = (unsigned int) Rcpp::as<int>(RextraTaps);
    bool residualScale                = Rcpp::as<bool>(RresidualScale);
    int residualScaleMinFlow          = Rcpp::as<int>(RresidualScaleMinFlow);
    int residualScaleMaxFlow          = Rcpp::as<int>(RresidualScaleMaxFlow);
    bool debugBaseCall                = Rcpp::as<bool>(RdebugBaseCall);
  
    unsigned int nFlow = signal.cols();
    unsigned int nRead = signal.rows();

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
      std::string exception = "bad droop type supplied\n";
      exceptionMesg = strdup(exception.c_str());
    } else if(cc.rows() != (int) N_NUCLEOTIDES) {
      std::string exception = "concentration matrix should have 4 rows\n";
      exceptionMesg = strdup(exception.c_str());
    } else if(cc.cols() != (int) N_NUCLEOTIDES) {
      std::string exception = "concentration matrix should have 4 columns\n";
      exceptionMesg = strdup(exception.c_str());
    } else {
      // recast cf, ie, dr, hpScale
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
      
      // recast nuc concentration
      vector<weight_vec_t> ccMod(cc.rows());
      for(int iRow=0; iRow < cc.rows(); iRow++) {
        ccMod[iRow].resize(cc.cols());
        for(unsigned int iCol=0; iCol < N_NUCLEOTIDES; iCol++)
          ccMod[iRow][iCol] = cc(iRow,iCol);
      }

      // Other recasts
      hpLen_t maxAdvMod = (hpLen_t) maxAdv;

      // Prepare objects for holding and passing back results
      Rcpp::NumericMatrix        predicted_out(nRead,nFlow);
      Rcpp::NumericMatrix        residual_out(nRead,nFlow);
      Rcpp::IntegerMatrix        hpFlow_out(nRead,nFlow);
      std::vector< std::string>  seq_out(nRead);
      Rcpp::NumericMatrix        multiplier_out(nRead,1+nIterations);

      // Iterate over all reads
      weight_vec_t sigMod(nFlow);
      string result;
      for(unsigned int iRead=0; iRead < nRead; iRead++) {
        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++)
          sigMod[iFlow] = (weight_t) signal(iRead,iFlow);
        
        PhaseSolve p;
        p.SetResidualScale(residualScale);
        if(residualScaleMinFlow >= 0)
          p.SetResidualScaleMinFlow((unsigned int) residualScaleMinFlow);
        if(residualScaleMaxFlow >= 0)
          p.SetResidualScaleMaxFlow((unsigned int) residualScaleMaxFlow);
        p.setExtraTaps(extraTaps);
        p.setHpScale(hpScaleMod);
        p.setPhaseParam(flowCycle,maxAdvMod,ccMod,cfMod,ieMod,drMod,droopType);
        p.GreedyBaseCall(sigMod, nIterations, debugBaseCall);
        p.getSeq(result);
        seq_out[iRead] = result;
        weight_vec_t & predicted = p.GetPredictedSignal();
        weight_vec_t & residual  = p.GetResidualSignal();
        hpLen_vec_t &  hpFlow    = p.GetPredictedHpFlow();
        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++) {
          predicted_out(iRead,iFlow) = (double) predicted[iFlow];
          residual_out(iRead,iFlow)  = (double) residual[iFlow];
          hpFlow_out(iRead,iFlow)    = (int)    hpFlow[iFlow];
        }
        if(residualScale) {
          weight_vec_t & multiplier  = p.GetMultiplier();
          // We re-order these so the last multiplier comes first.  This is for convenience
          // as it allows us grab the first col of the matrix as the last multiplier applied
          // even if each read ended up taking different numbers of iterations.
          unsigned int i1,i2;
          for(i1=0,i2=multiplier.size()-1; i1 < multiplier.size(); i1++,i2--) {
            multiplier_out(iRead,i1) = (double) multiplier[i2];
          }
          // If the read took fewer than all available iterations, pad with zero
          for(; i1 <= nIterations; i1++) {
            multiplier_out(iRead,i1) = 0;
          }
        }

      }

      // Store results
      std::map<std::string,SEXP> map;
      map["seq"]          = Rcpp::wrap( seq_out );
      map["predicted"]    = Rcpp::wrap( predicted_out );
      map["residual"]     = Rcpp::wrap( residual_out );
      map["hpFlow"]       = Rcpp::wrap( hpFlow_out );
      if(residualScale)
        map["multiplier"] = Rcpp::wrap( multiplier_out );

      ret = Rcpp::wrap( map );
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
