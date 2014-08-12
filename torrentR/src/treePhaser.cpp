/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>
#include "RecalibrationModel.h"
#include "DPTreephaser.h"


RcppExport SEXP treePhaser(SEXP Rsignal, SEXP RkeyFlow, SEXP RflowCycle,
                           SEXP Rcf, SEXP Rie, SEXP Rdr, SEXP Rbasecaller, SEXP RdiagonalStates,
                           SEXP RmodelFile, SEXP RmodelThreshold, SEXP Rxval, SEXP Ryval)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    Rcpp::NumericMatrix      signal(Rsignal);
    Rcpp::IntegerVector      keyFlow(RkeyFlow);
    string flowCycle       = Rcpp::as<string>(RflowCycle);
    Rcpp::NumericVector      cf_vec(Rcf);
    Rcpp::NumericVector      ie_vec(Rie);
    Rcpp::NumericVector      dr_vec(Rdr);
    string basecaller      = Rcpp::as<string>(Rbasecaller);
    unsigned int diagonalStates = Rcpp::as<int>(RdiagonalStates);

    // Recalibration Variables
    string model_file      = Rcpp::as<string>(RmodelFile);
    int model_threshold    = Rcpp::as<int>(RmodelThreshold);
    Rcpp::IntegerVector      x_values(Rxval);
    Rcpp::IntegerVector      y_values(Ryval);
    RecalibrationModel       recalModel;
    if (model_file.length() > 0) {
      recalModel.InitializeModel(model_file, model_threshold);
    }


    ion::FlowOrder flow_order(flowCycle, flowCycle.length());
    unsigned int nFlow = signal.cols();
    unsigned int nRead = signal.rows();

    if(basecaller != "treephaser-swan" && basecaller != "treephaser-solve" && basecaller != "dp-treephaser" && basecaller != "treephaser-adaptive") {
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
      dpTreephaser.SetStateProgression((diagonalStates>0));

      // In contrast to pipeline, we always use droop here.
      // To have the same behavior of treephaser-swan as in the pipeline, supply dr=0
      bool per_read_phasing = true;
      if (cf_vec.size() == 1) {
        per_read_phasing = false;
        dpTreephaser.SetModelParameters((double)cf_vec(0), (double)ie_vec(0), (double)dr_vec(0));
      }
 
      // Main loop iterating over reads and solving them
      for(unsigned int iRead=0; iRead < nRead; iRead++) {

        // Set phasing parameters for this read
        if (per_read_phasing)
          dpTreephaser.SetModelParameters((double)cf_vec(iRead), (double)ie_vec(iRead), (double)dr_vec(iRead));
        // And load recalibration model
        if (recalModel.is_enabled()) {
          int my_x = (int)x_values(iRead);
          int my_y = (int)y_values(iRead);
          const vector<vector<vector<float> > > * aPtr = 0;
          const vector<vector<vector<float> > > * bPtr = 0;
          aPtr = recalModel.getAs(my_x, my_y);
          bPtr = recalModel.getBs(my_x, my_y);
          if (aPtr == 0 or bPtr == 0) {
            cout << "Error finding a recalibration model for x: " << x_values(iRead) << " y: " << y_values(iRead);
            cout << endl;
          }
          dpTreephaser.SetAsBs(aPtr, bPtr);
        }

        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++)
          sigVec[iFlow] = (float) signal(iRead,iFlow);
        
        // Interface to just solve without any normalization
        if (basecaller == "treephaser-solve") { // Interface to just solve without any normalization
          read.SetData(sigVec, (int)nFlow);
        } 
        else {
          read.SetDataAndKeyNormalize(&(sigVec[0]), (int)nFlow, &(keyVec[0]), nKeyFlow-1);
        }
          
        // Execute the iterative solving-normalization routine
        if (basecaller == "dp-treephaser") {
          dpTreephaser.NormalizeAndSolve_GainNorm(read, nFlow);
        }
        else if (basecaller == "treephaser-solve") {
          dpTreephaser.Solve(read, nFlow);
        }
        else if (basecaller == "treephaser-adaptive") {
          dpTreephaser.NormalizeAndSolve_Adaptive(read, nFlow); // Adaptive normalization
        }
        else {
          dpTreephaser.NormalizeAndSolve_SWnorm(read, nFlow); // sliding window adaptive normalization
        }

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
                              SEXP Rmaxflows, SEXP RgetStates, SEXP RdiagonalStates,
                              SEXP RmodelFile, SEXP RmodelThreshold, SEXP Rxval, SEXP Ryval)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {

    Rcpp::StringVector   sequences(Rsequence);
    string flowCycle   = Rcpp::as<string>(RflowCycle);
    Rcpp::NumericVector      cf_vec(Rcf);
    Rcpp::NumericVector      ie_vec(Rie);
    Rcpp::NumericVector      dr_vec(Rdr);
    unsigned int max_flows       = Rcpp::as<int>(Rmaxflows);
    unsigned int get_states      = Rcpp::as<int>(RgetStates);
    unsigned int diagonalStates = Rcpp::as<int>(RdiagonalStates);

    // Recalibration Variables
    string model_file      = Rcpp::as<string>(RmodelFile);
    int model_threshold    = Rcpp::as<int>(RmodelThreshold);
    Rcpp::IntegerVector      x_values(Rxval);
    Rcpp::IntegerVector      y_values(Ryval);
    RecalibrationModel       recalModel;
    if (model_file.length() > 0) {
      recalModel.InitializeModel(model_file, model_threshold);
    }

    ion::FlowOrder flow_order(flowCycle, flowCycle.length());
    unsigned int nFlow = flow_order.num_flows();
    unsigned int nRead = sequences.size();
    max_flows = min(max_flows, nFlow);

    // Prepare objects for holding and passing back results
    Rcpp::NumericMatrix       predicted_out(nRead,nFlow);
    vector<vector<float> >    query_states;
    vector<int>               hp_lengths;

    // Iterate over all sequences
    BasecallerRead read;
    DPTreephaser dpTreephaser(flow_order);
    bool per_read_phasing = true;
    if (cf_vec.size() == 1) {
      per_read_phasing = false;
      dpTreephaser.SetModelParameters((double)cf_vec(0), (double)ie_vec(0), (double)dr_vec(0));
    }
    dpTreephaser.SetStateProgression((diagonalStates>0));
    unsigned int max_length = (2*flow_order.num_flows());

    for(unsigned int iRead=0; iRead<nRead; iRead++) {

      string mySequence = Rcpp::as<std::string>(sequences(iRead));
      read.sequence.clear();
      read.sequence.reserve(2*flow_order.num_flows());
      for(unsigned int iBase=0; iBase<mySequence.length() and iBase<max_length; ++iBase){
        read.sequence.push_back(mySequence.at(iBase));
      }
      // Set phasing parameters for this read
      if (per_read_phasing)
        dpTreephaser.SetModelParameters((double)cf_vec(iRead), (double)ie_vec(iRead), (double)dr_vec(iRead));

      // If you bothered specifying a recalibration model you probably want its effect on the predictions...
      if (recalModel.is_enabled()) {
        int my_x = (int)x_values(iRead);
        int my_y = (int)y_values(iRead);
        const vector<vector<vector<float> > > * aPtr = 0;
        const vector<vector<vector<float> > > * bPtr = 0;
        aPtr = recalModel.getAs(my_x, my_y);
        bPtr = recalModel.getBs(my_x, my_y);
        if (aPtr == 0 or bPtr == 0) {
          cout << "Error finding a recalibration model for x: " << x_values(iRead) << " y: " << y_values(iRead);
          cout << endl;
        }
        dpTreephaser.SetAsBs(aPtr, bPtr);
      }

      if (nRead == 1 and get_states > 0)
        dpTreephaser.QueryAllStates(read, query_states, hp_lengths, max_flows);
      else
        dpTreephaser.Simulate(read, max_flows);

      for(unsigned int iFlow=0; iFlow<nFlow and iFlow<max_flows; ++iFlow){
		predicted_out(iRead,iFlow) = (double) read.prediction.at(iFlow);
      }
    }

    // Store results
    if (nRead == 1 and get_states > 0) {
      Rcpp::NumericMatrix        states(hp_lengths.size(), nFlow);
      Rcpp::NumericVector        HPlengths(hp_lengths.size());
      for (unsigned int iHP=0; iHP<hp_lengths.size(); iHP++){
        HPlengths(iHP) = (double)hp_lengths[iHP];
        for (unsigned int iFlow=0; iFlow<nFlow; iFlow++)
          states(iHP, iFlow) = (double)query_states.at(iHP).at(iFlow);
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
