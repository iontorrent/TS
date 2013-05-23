/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include "RawWells.h"
#include "CafieSolver.h"

RcppExport SEXP correctCafie(
  SEXP measured_in,
  SEXP flowOrder_in,
  SEXP keyFlow_in,
  SEXP nKeyFlow_in,
  SEXP cafEst_in,
  SEXP ieEst_in,
  SEXP droopEst_in
) {

    SEXP rl = R_NilValue; 		// Use this when there is nothing to be returned.
    char *exceptionMesg = NULL;

    try {
	// First do some annoying but necessary type casting on the input parameters.
	// measured & nFlow
	Rcpp::NumericMatrix measured_temp(measured_in);
	int nWell = measured_temp.rows();
	int nFlow = measured_temp.cols();
	// flowOrder
	Rcpp::StringVector  flowOrder_temp(flowOrder_in);
	char *flowOrder = strdup(flowOrder_temp(0));
	int flowOrderLen = strlen(flowOrder);
	// keyFlow
	Rcpp::IntegerVector keyFlow_temp(keyFlow_in);
	int *keyFlow = new int[keyFlow_temp.size()];
	for(int i=0; i<keyFlow_temp.size(); i++) {
	  keyFlow[i] = keyFlow_temp(i);
	}
	// nKeyFlow
	Rcpp::IntegerVector nKeyFlow_temp(nKeyFlow_in);
	int nKeyFlow = nKeyFlow_temp(0);
	// cafEst, ieEst, droopEst
	Rcpp::NumericVector cafEst_temp(cafEst_in);
	double cafEst = cafEst_temp(0);
	Rcpp::NumericVector ieEst_temp(ieEst_in);
	double ieEst = ieEst_temp(0);
	Rcpp::NumericVector droopEst_temp(droopEst_in);
	double droopEst = droopEst_temp(0);
 
	if(flowOrderLen != nFlow) {
	    exceptionMesg = strdup("Flow order and signal should be of same length");
	} else if(nKeyFlow <= 0) {
	    exceptionMesg = strdup("keyFlow must have length > 0");
	} else {
	    double *measured = new double[nFlow];
	    Rcpp::NumericMatrix predicted(nWell,nFlow);
	    Rcpp::NumericMatrix corrected(nWell,nFlow);
	    CafieSolver solver;
	    solver.SetFlowOrder(flowOrder);
	    solver.SetCAFIE(cafEst, ieEst);
	    for(int well=0; well < nWell; well++) {
		// Set up the input signal for the well
		for(int flow=0; flow<nFlow; flow++) {
		    measured[flow] = measured_temp(well,flow);
		}

		// Initialize the sovler object and find the best CAFIE
		solver.SetMeasured(nFlow, measured);
	        solver.Normalize(keyFlow, nKeyFlow, droopEst, false);
		solver.Solve(3, true);

		// Store the predicted & corrected signals
		for(int flow=0; flow<nFlow; flow++) {
		    predicted(well,flow) = solver.GetPredictedResult(flow);
		    corrected(well,flow) = solver.GetCorrectedResult(flow);
		}
		// Store the estimated sequence
		//const double *normalized_ptr = solver.GetMeasured();
	        //const char *seqEstimate_ptr = solver.GetSequence();
	        //int seqEstimateLen = strlen(seqEstimate_ptr);
	    }

	    // Build result set to be returned as a list to R.
        rl = Rcpp::List::create(Rcpp::Named("predicted") = predicted,
                                Rcpp::Named("corrected") = corrected);

	    delete [] measured;
	}

    free(flowOrder);
	delete [] keyFlow;

    } catch(std::exception& ex) {
	forward_exception_to_r(ex);
    } catch(...) {
	::Rf_error("c++ exception (unknown reason)");
    }
    
    if(exceptionMesg != NULL)
	Rf_error(exceptionMesg);

    return rl;
}
