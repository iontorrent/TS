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
	RcppMatrix<double> measured_temp(measured_in);
	int nWell = measured_temp.rows();
	int nFlow = measured_temp.cols();
	// flowOrder
	RcppStringVector  flowOrder_temp(flowOrder_in);
	char *flowOrder = strdup(flowOrder_temp(0).c_str());
	int flowOrderLen = strlen(flowOrder);
	// keyFlow
	RcppVector<int> keyFlow_temp(keyFlow_in);
	int *keyFlow = new int[keyFlow_temp.size()];
	for(int i=0; i<keyFlow_temp.size(); i++) {
	  keyFlow[i] = keyFlow_temp(i);
	}
	// nKeyFlow
	RcppVector<int> nKeyFlow_temp(nKeyFlow_in);
	int nKeyFlow = nKeyFlow_temp(0);
	// cafEst, ieEst, droopEst
	RcppVector<double> cafEst_temp(cafEst_in);
	double cafEst = cafEst_temp(0);
	RcppVector<double> ieEst_temp(ieEst_in);
	double ieEst = ieEst_temp(0);
	RcppVector<double> droopEst_temp(droopEst_in);
	double droopEst = droopEst_temp(0);
 
	if(flowOrderLen != nFlow) {
	    exceptionMesg = copyMessageToR("Flow order and signal should be of same length");
	} else if(nKeyFlow <= 0) {
	    exceptionMesg = copyMessageToR("keyFlow must have length > 0");
	} else {
	    double *measured = new double[nFlow];
	    RcppMatrix<double> predicted(nWell,nFlow);
	    RcppMatrix<double> corrected(nWell,nFlow);
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
	    RcppResultSet rs;
	    rs.add("predicted",  predicted);
	    rs.add("corrected",  corrected);

	    // Get the list to be returned to R.
	    rl = rs.getReturnList();

	    delete [] measured;
	}

    free(flowOrder);
	delete [] keyFlow;

    } catch(std::exception& ex) {
	exceptionMesg = copyMessageToR(ex.what());
    } catch(...) {
	exceptionMesg = copyMessageToR("unknown reason");
    }
    
    if(exceptionMesg != NULL)
	Rf_error(exceptionMesg);

    return rl;
}
