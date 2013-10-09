/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>
#include "DPTreephaser.h"

RcppExport SEXP treePhaser(SEXP Rsignal, SEXP RkeyFlow, SEXP RflowCycle,
		                   SEXP Rcf, SEXP Rie, SEXP Rdr, SEXP Rbasecaller, SEXP RterminatorChemistry)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    Rcpp::NumericMatrix  signal(Rsignal);
    Rcpp::IntegerVector  keyFlow(RkeyFlow);
    string flowCycle   = Rcpp::as<string>(RflowCycle);
    double cf          = Rcpp::as<double>(Rcf);
    double ie          = Rcpp::as<double>(Rie);
    double dr          = Rcpp::as<double>(Rdr);
    string basecaller  = Rcpp::as<string>(Rbasecaller);
    unsigned int isTerminatorRun     = Rcpp::as<int>(RterminatorChemistry);

    ion::FlowOrder flow_order(flowCycle, flowCycle.length());
    unsigned int nFlow = signal.cols();
    unsigned int nRead = signal.rows();

    if(basecaller != "treephaser-swan" && basecaller != "dp-treephaser" && basecaller != "treephaser-adaptive") {
      std::string exception = "base value for basecaller supplied: " + basecaller;
      exceptionMesg = strdup(exception.c_str());
    } else if (flowCycle.length() < nFlow) {
      std::string exception = "Flow cycle is shorter than number of flows to solve";
      exceptionMesg = strdup(exception.c_str());
    } else {

      // Prepare objects for holding and passing back results
      Rcpp::NumericMatrix        predicted_out(nRead,nFlow);
      Rcpp::NumericMatrix        residual_out(nRead,nFlow);
      std::vector< std::string> seq_out(nRead);

      // Set up key flow vector
      int nKeyFlow = keyFlow.size(); 
      vector <int> keyVec(nKeyFlow);
      for(int iFlow=0; iFlow < nKeyFlow; iFlow++)
        keyVec[iFlow] = keyFlow(iFlow);

      // Iterate over all reads
      vector <float> sigVec(nFlow);
      BasecallerRead read;
      DPTreephaser dpTreephaser(flow_order);
      dpTreephaser.SetTerminatorChemistry((isTerminatorRun>0));

      // In contrast to pipeline, we always use droop here.
      // To have the same behavior of treephaser-swan as in the pipeline, supply dr=0
      dpTreephaser.SetModelParameters(cf, ie, dr);

      // Main loop iterating over reads and solving them
      for(unsigned int iRead=0; iRead < nRead; iRead++) {

        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++)
          sigVec[iFlow] = (float) signal(iRead,iFlow);
        read.SetDataAndKeyNormalize(&(sigVec[0]), (int)nFlow, &(keyVec[0]), nKeyFlow-1);
          
        // Execute the iterative solving-normalization routine
        if (basecaller == "dp-treephaser")
          dpTreephaser.NormalizeAndSolve4(read, nFlow);
        else if (basecaller == "treephaser-adaptive")
          dpTreephaser.NormalizeAndSolve3(read, nFlow); // Adaptive normalization
        else
          dpTreephaser.NormalizeAndSolve5(read, nFlow); // sliding window adaptive normalization

        seq_out[iRead].assign(read.sequence.begin(), read.sequence.end());
        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++) {
          predicted_out(iRead,iFlow) = (double) read.prediction[iFlow];
          residual_out(iRead,iFlow)  = (double) read.normalized_measurements[iFlow] - read.prediction[iFlow];
        }
      }

      // Store results
      ret = Rcpp::List::create(Rcpp::Named("seq")       = seq_out,
                               Rcpp::Named("predicted") = predicted_out,
                               Rcpp::Named("residual")  = residual_out);
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

// ======================================================================
RcppExport SEXP treePhaserSim(SEXP Rsequence, SEXP RflowCycle, SEXP Rcf, SEXP Rie, SEXP Rdr,
		SEXP Rmaxflows, SEXP RgetStates, SEXP RterminatorChemistry)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {

    Rcpp::StringVector   sequences(Rsequence);
    string flowCycle   = Rcpp::as<string>(RflowCycle);
    double cf          = Rcpp::as<double>(Rcf);
    double ie          = Rcpp::as<double>(Rie);
    double dr          = Rcpp::as<double>(Rdr);
    unsigned int max_flows       = Rcpp::as<int>(Rmaxflows);
    unsigned int get_states      = Rcpp::as<int>(RgetStates);
    unsigned int isTerminatorRun = Rcpp::as<int>(RterminatorChemistry);

    ion::FlowOrder flow_order(flowCycle, flowCycle.length());
    unsigned int nFlow = flow_order.num_flows();
    unsigned int nRead = sequences.size();

    // Prepare objects for holding and passing back results
    Rcpp::NumericMatrix       predicted_out(nRead,nFlow);
    vector<vector<float> >    query_states;
    vector<int>               hp_lengths;

    // Iterate over all sequences
    BasecallerRead read;
    DPTreephaser dpTreephaser(flow_order);
    dpTreephaser.SetModelParameters(cf, ie, dr);
    dpTreephaser.SetTerminatorChemistry((isTerminatorRun>0));
    unsigned int max_length = (2*flow_order.num_flows());
    // XXX
    //cout << "Simulate:: Terminator Chemistry Run: " << isTerminatorRun << endl;

    for(unsigned int iRead=0; iRead<nRead; iRead++) {
      string mySequence = Rcpp::as<std::string>(sequences(iRead));
      read.sequence.clear();
      read.sequence.reserve(2*flow_order.num_flows());
      for(unsigned int iBase=0; iBase<mySequence.length() and iBase<max_length; ++iBase){
        read.sequence.push_back(mySequence[iBase]);
      }
      if (nRead == 1 and get_states > 0)
        dpTreephaser.QueryAllStates(read, query_states, hp_lengths, max_flows);
      else
        dpTreephaser.Simulate(read, max_flows);

      for(unsigned int iFlow=0; iFlow<nFlow and iFlow<max_flows; ++iFlow){
		predicted_out(iRead,iFlow) = (double) read.prediction[iFlow];
      }
    }

    // Store results
    if (nRead == 1 and get_states > 0) {
      Rcpp::NumericMatrix        states(hp_lengths.size(), nFlow);
      Rcpp::NumericVector        HPlengths(hp_lengths.size());
      for (unsigned int iHP=0; iHP<hp_lengths.size(); iHP++){
        HPlengths(iHP) = (double)hp_lengths[iHP];
        for (unsigned int iFlow=0; iFlow<nFlow; iFlow++)
          states(iHP, iFlow) = (double)query_states[iHP][iFlow];
      }
      ret = Rcpp::List::create(Rcpp::Named("sig")  = predicted_out,
                               Rcpp::Named("states")  = states,
                               Rcpp::Named("HPlengths")  = HPlengths);
    } else {
      ret = Rcpp::List::create(Rcpp::Named("sig")  = predicted_out);
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
