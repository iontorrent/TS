/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>
#include "DPTreephaser.h"

RcppExport SEXP treePhaser(SEXP Rsignal, SEXP RkeyFlow, SEXP RflowCycle, SEXP Rcf, SEXP Rie, SEXP Rdr, SEXP Rbasecaller)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    RcppMatrix<double>   signal(Rsignal);
    RcppVector<int>      keyFlow(RkeyFlow);
    string flowCycle   = Rcpp::as<string>(RflowCycle);
    double cf          = Rcpp::as<double>(Rcf);
    double ie          = Rcpp::as<double>(Rie);
    double dr          = Rcpp::as<double>(Rdr);
    string basecaller  = Rcpp::as<string>(Rbasecaller);
  
    unsigned int nFlow = signal.cols();
    unsigned int nRead = signal.rows();

    if(basecaller != "treephaser-swan" && basecaller != "dp-treephaser" && basecaller != "treephaser-adaptive") {
      std::string exception = "base value for basecaller supplied: " + basecaller;
      exceptionMesg = copyMessageToR(exception.c_str());
    } else if (flowCycle.length() < nFlow) {
      std::string exception = "Flow cycle is shorter than number of flows to solve";
      exceptionMesg = copyMessageToR(exception.c_str());
    } else {

      // Prepare objects for holding and passing back results
      RcppMatrix<double>        predicted_out(nRead,nFlow);
      RcppMatrix<double>        residual_out(nRead,nFlow);
      RcppMatrix<int>           hpFlow_out(nRead,nFlow);
      std::vector< std::string> seq_out(nRead);

      // Set up key flow vector
      int nKeyFlow = keyFlow.size(); 
      vector <int> keyVec(nKeyFlow);
      for(int iFlow=0; iFlow < nKeyFlow; iFlow++)
        keyVec[iFlow] = keyFlow(iFlow);

      // Iterate over all reads
      vector <float> sigVec(nFlow);
      string result;
      for(unsigned int iRead=0; iRead < nRead; iRead++) {
        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++)
          sigVec[iFlow] = (float) signal(iRead,iFlow);
        BasecallerRead read;
        read.SetDataAndKeyNormalize(&(sigVec[0]), (int)nFlow, &(keyVec[0]), nKeyFlow-1);
        DPTreephaser dpTreephaser(flowCycle.c_str(), flowCycle.length(), 8);
        if (basecaller == "dp-treephaser")
          dpTreephaser.SetModelParameters(cf, ie, dr);
        else
          dpTreephaser.SetModelParameters(cf, ie, 0); // Adaptive normalization
          
        // Execute the iterative solving-normalization routine
        if (basecaller == "dp-treephaser")
          dpTreephaser.NormalizeAndSolve4(read, nFlow);
        else if (basecaller == "treephaser-adaptive")
          dpTreephaser.NormalizeAndSolve3(read, nFlow); // Adaptive normalization
        else
          dpTreephaser.NormalizeAndSolve5(read, nFlow); // sliding window adaptive normalization

        read.flowToString(flowCycle,seq_out[iRead]);
        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++) {
          predicted_out(iRead,iFlow) = (double) read.prediction[iFlow];
          residual_out(iRead,iFlow)  = (double) read.normalizedMeasurements[iFlow] - read.prediction[iFlow];
          hpFlow_out(iRead,iFlow)    = (int)    read.solution[iFlow];
        }

        // Store results
        RcppResultSet rs;
        rs.add("seq",        seq_out);
        rs.add("predicted",  predicted_out);
        rs.add("residual",   residual_out);
        rs.add("hpFlow",     hpFlow_out);

        ret = rs.getReturnList();
      }
    }
  } catch(std::exception& ex) {
    exceptionMesg = copyMessageToR(ex.what());
  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
