/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include "RawWells.h"
#include "CafieSolver.h"

RcppExport SEXP normalizeIonogram(
  SEXP measured_in,
  SEXP keyFlow_in,
  SEXP nKeyFlow_in,
  SEXP flowOrder_in
) {

    SEXP rl = R_NilValue; 		// Use this when there is nothing to be returned.
    char *exceptionMesg = NULL;

    try {

	// First do some annoying but necessary type casting on the input parameters.
	// measured & nFlow
	Rcpp::NumericMatrix measured_temp(measured_in);
	int nWell = measured_temp.rows();
	int nFlow = measured_temp.cols();
	// keyFlow
	Rcpp::IntegerVector keyFlow_temp(keyFlow_in);
	int *keyFlow = new int[keyFlow_temp.size()];
	for(int i=0; i<keyFlow_temp.size(); i++) {
	  keyFlow[i] = keyFlow_temp(i);
	}
	// nKeyFlow
	Rcpp::IntegerVector nKeyFlow_temp(nKeyFlow_in);
	int nKeyFlow = nKeyFlow_temp(0);
	// flowOrder
	Rcpp::StringVector  flowOrder_temp(flowOrder_in);
	char *flowOrder = strdup(flowOrder_temp(0));
 
	// Do the normalization
	double *measured = new double[nFlow];
	Rcpp::NumericMatrix normalized(nWell,nFlow);
	for(int well=0; well < nWell; well++) {
	    // Set up the input signal for the well
	    for(int flow=0; flow<nFlow; flow++) {
		measured[flow] = measured_temp(well,flow);
	    }
	    // Do the normalization
	    CafieSolver solver;
	    solver.SetFlowOrder(flowOrder);
	    solver.SetCAFIE(0.0, 0.0);
	    solver.SetMeasured(nFlow, measured);
	    solver.Normalize(keyFlow, nKeyFlow, 0, false);
	    // Store the results
	    const double *normalized_ptr = solver.GetMeasured();
	    for(int flow=0; flow<nFlow; flow++) {
		normalized(well,flow) = normalized_ptr[flow];
	    }
	}

	// Build result set to be returned as a list to R.
	rl = Rcpp::List::create(Rcpp::Named("normalized") = normalized);

	// Clear allocated memory
	delete [] measured;
    delete [] keyFlow;
    free(flowOrder);

    } catch(std::exception& ex) {
	forward_exception_to_r(ex);
    } catch(...) {
	::Rf_error("c++ exception (unknown reason)");
    }
    
    if(exceptionMesg != NULL)
	Rf_error(exceptionMesg);

    return rl;
}
