/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>
#include "DPTreephaser.h"

//val <- .Call("normalizeRead", signal, prediction, method, windowSize, numSteps, startFlow, endFlow, PACKAGE="torrentR")

RcppExport SEXP normalizeRead(SEXP Rsignal, SEXP Rprediction, SEXP Rmethod, SEXP RwindowSize,
                           SEXP RnumSteps, SEXP RstartFlow, SEXP RendFlow)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {

    Rcpp::NumericMatrix      signal(Rsignal);
    Rcpp::NumericMatrix      prediction(Rsignal);
    string norm_method     = Rcpp::as<string>(Rmethod);
    int window_size        = Rcpp::as<int>(RwindowSize);
    int num_steps          = Rcpp::as<int>(RnumSteps);
    int start_flow         = Rcpp::as<int>(RstartFlow);
    int end_flow           = Rcpp::as<int>(RendFlow);
    // return normalized signal


    // Normalization does not depend on flow order but a flow order is necessary to construct treephaser object.
    int num_flows = signal.cols();
    int num_reads = signal.rows();
    Rcpp::NumericMatrix      normalized_out(num_reads,num_flows);

    vector<float> measurements(num_flows, 0);
    ion::FlowOrder flow_order("TACG", num_flows);
    DPTreephaser treephaser(flow_order);
    BasecallerRead read;

    // set and adjust window size
    if (window_size > 0) {
      if (window_size < 20 or window_size > 60) {
        cout << "Warning: Treephaser only accepts a normalization window size in the interval [20,60] -- Using default value of " << treephaser.kWindowSizeDefault_ << endl;
        window_size = treephaser.kWindowSizeDefault_;
      }
      else
        treephaser.SetNormalizationWindowSize(window_size);
    }
    else
      window_size = treephaser.kWindowSizeDefault_;

    // Adjust start and end point of the normalization (by method)
    start_flow = max(start_flow, 0);
    end_flow = min(end_flow, num_flows);
    if (num_steps==0)
      num_steps = num_flows/window_size;
    num_steps = min(num_steps, num_flows/window_size);

    // Loop over all the reads
    for(int iRead=0; iRead < num_reads; iRead++) {

      // Load read into object
      for (int iFlow=0; iFlow < num_flows; iFlow++)
        measurements.at(iFlow) = (float)signal(iRead, iFlow);
      read.SetData(measurements, num_flows);
      for (int iFlow=0; iFlow < num_flows; iFlow++)
        read.prediction.at(iFlow) = (float)prediction(iRead, iFlow);

      if (norm_method == "adaptive") {
        treephaser.WindowedNormalize(read, num_steps, window_size);
      } else if (norm_method == "gain") {
        treephaser.Normalize(read, start_flow, end_flow);
      } else if (norm_method == "pid") {
        treephaser.PIDNormalize(read, start_flow, end_flow);
      } else {
    	cout << "Unknown Normalization Method. Nothing Happend!" << endl;
      }

      // Store results
      for(int iFlow=0; iFlow < num_flows; iFlow++) {
        normalized_out(iRead,iFlow) = (double) read.normalized_measurements.at(iFlow);
      }
    }

    // Store results
    ret = Rcpp::List::create(Rcpp::Named("method")     = norm_method,
                             Rcpp::Named("normalized") = normalized_out);


  } catch(std::exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }

  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);
  return ret;
}
