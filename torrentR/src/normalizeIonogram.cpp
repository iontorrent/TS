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
	RcppMatrix<double> measured_temp(measured_in);
	int nWell = measured_temp.rows();
	int nFlow = measured_temp.cols();
	// keyFlow
	RcppVector<int> keyFlow_temp(keyFlow_in);
	int *keyFlow = new int[keyFlow_temp.size()];
	for(int i=0; i<keyFlow_temp.size(); i++) {
	  keyFlow[i] = keyFlow_temp(i);
	}
	// nKeyFlow
	RcppVector<int> nKeyFlow_temp(nKeyFlow_in);
	int nKeyFlow = nKeyFlow_temp(0);
	// flowOrder
	RcppStringVector  flowOrder_temp(flowOrder_in);
	char *flowOrder = strdup(flowOrder_temp(0).c_str());
 
	// Do the normalization
	double *measured = new double[nFlow];
	RcppMatrix<double> normalized(nWell,nFlow);
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
	RcppResultSet rs;
	rs.add("normalized",  normalized);

	// Set the list to be returned to R.
	rl = rs.getReturnList();

	// Clear allocated memory
	delete [] measured;
    delete [] keyFlow;
    free(flowOrder);

    } catch(std::exception& ex) {
	exceptionMesg = copyMessageToR(ex.what());
    } catch(...) {
	exceptionMesg = copyMessageToR("unknown reason");
    }
    
    if(exceptionMesg != NULL)
	Rf_error(exceptionMesg);

    return rl;
}
