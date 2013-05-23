/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>

#include "calcHypothesesDistancesEngine.h"

// ------------------------------------------------------------------------------


RcppExport SEXP calcHypothesesDistances(SEXP Rsignal, SEXP Rcf, SEXP Rie, SEXP Rdr,
		SEXP RflowCycle, SEXP RHypotheses, SEXP RstartFlow, SEXP Rnormalize, SEXP Rverbose)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
	    Rcpp::NumericVector     signal(Rsignal);
	    std::string flowCycle = Rcpp::as<string>(RflowCycle);
	    double cf             = Rcpp::as<double>(Rcf);
	    double ie             = Rcpp::as<double>(Rie);
	    double dr             = Rcpp::as<double>(Rdr);
	    Rcpp::StringVector      Hypotheses(RHypotheses);
	    int startFlow         = Rcpp::as<int>(RstartFlow);
	    int verbose           = Rcpp::as<int>(Rverbose);
	    int normalize         = Rcpp::as<int>(Rnormalize);

	    ion::FlowOrder flow_order(flowCycle, flowCycle.length());
	    int nFlow = signal.size();
	    int nHyp  = Hypotheses.size();

	    // Prepare objects for holding and passing back results
	    Rcpp::NumericVector      DistanceObserved(nHyp);
	    Rcpp::NumericVector      DistanceHypotheses(nHyp-1);
	    Rcpp::NumericMatrix      predicted_out(nHyp,nFlow);
	    Rcpp::NumericMatrix      normalized_out(nHyp,nFlow);

	    // Copy data into c++ data types
	    vector<float> Measurements(nFlow);
	    for (int i=0; i<nFlow; i++)
	        Measurements[i] = signal(i);
	    vector<string> HypVector(nHyp);
	    for (int i=0; i<nHyp; i++)
	    	HypVector[i] = Hypotheses(i);
	    vector<float> DistObserved;
	    DistObserved.assign(nHyp,0);
	    vector<float> DistHypotheses;
	    DistHypotheses.assign(nHyp-1,0);
	    vector<vector<float> > predictions(nHyp);
	    vector<vector<float> > normalized(nHyp);


	    CalculateHypDistances(Measurements, cf, ie, dr, flow_order, HypVector, startFlow,
	    		DistObserved, DistHypotheses, predictions, normalized, normalize, verbose);


	    // Store return values into return structure
	    for (int i=0; i<nHyp; i++){
	    	DistanceObserved(i) = (double)DistObserved[i];
	    	if (i>0)
	    		DistanceHypotheses(i-1) = (double)DistHypotheses[i-1];
	    	for (int iFlow=0; iFlow<nFlow; ++iFlow){
	    		predicted_out(i,iFlow)  = (double) predictions[i][iFlow];
	    		normalized_out(i,iFlow) = (double) normalized[i][iFlow];
	    	}
	    }
        ret = Rcpp::List::create(Rcpp::Named("DistanceObserved")   = DistanceObserved,
                                 Rcpp::Named("DistanceHypotheses") = DistanceHypotheses,
                                 Rcpp::Named("Predictions")        = predicted_out,
                                 Rcpp::Named("Normalized")         = normalized_out);

  } catch(std::exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }

  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


