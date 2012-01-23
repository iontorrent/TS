/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <Rcpp.h>
#include "RawWells.h"
#include <iostream>
RcppExport SEXP readWells(SEXP wellDir_in, SEXP wellFile_in, SEXP nCol_in, SEXP nRow_in, SEXP x_in, SEXP y_in, SEXP flow_in) {

    SEXP rl = R_NilValue; 		// Use this when there is nothing to be returned.
    char *exceptionMesg = NULL;

    try {

	// Recast input arguments
	// wellDir & wellFile
	RcppStringVector  wellDir_temp(wellDir_in);
	char *wellDir = strdup(wellDir_temp(0).c_str());
	RcppStringVector  wellFile_temp(wellFile_in);
	char *wellFile = strdup(wellFile_temp(0).c_str());
	// nCol and nRow
	RcppVector<int> nCol_temp(nCol_in);
	uint64_t nCol = nCol_temp(0);
	RcppVector<int> nRow_temp(nRow_in);
	uint64_t nRow = nRow_temp(0);
	// x
	RcppVector<int> x_temp(x_in);
	uint64_t nX = x_temp.size();
	RcppVector<int> x(nX);
	int xMin = INT_MAX;
	int xMax = -1;
	int newVal = 0;
	for(uint64_t i=0; i<nX; i++) {
	    newVal = x_temp(i);
	    x(i) = newVal;
	    if(newVal < xMin)
		xMin = newVal;
	    if(newVal > xMax)
		xMax = newVal;
	}
	// y
	RcppVector<int> y_temp(y_in);
	uint64_t nY = y_temp.size();
	RcppVector<int> y(nX);
	int yMin = INT_MAX;
	int yMax = -1;
	for(uint64_t i=0; i<nY; i++) {
	    newVal = y_temp(i);
	    y(i) = newVal;
	    if(newVal < yMin)
		yMin = newVal;
	    if(newVal > yMax)
		yMax = newVal;
	}
	// flow
	RcppVector<int> flow(flow_in);
	uint64_t nFlowRequested = flow.size();
 
 
	// Initiate RawWells object
	RawWells wells(wellDir, wellFile, nRow, nCol);
        wells.SetSubsetToLoad(&x(0), &y(0), nX);
	// Open wellfile and get header data
	wells.OpenForRead();
	uint64_t nFlow = wells.NumFlows();
	// Make sure all requested flows are in range, if a subset was requested
	bool inRange = true;
	for(uint64_t i=0; i<nFlowRequested; i++) {
		if(flow(i) < 0 || flow(i) >= (int)nFlow)
		inRange = false;
	}
        std::string *flowOrder = NULL;
	if(!inRange) {
	    exceptionMesg = copyMessageToR("all requested flows must be in-range");
	} else {
          std::string fo = wells.FlowOrder();
          flowOrder = new std::string();
          for (size_t i = 0; i < nFlow; i++) {
            flowOrder->push_back(fo.at(i % fo.length()));
          }

	    // Pull out the data for requested wells
	    RcppVector<int> rank(nX);
            RcppMatrix<double> signal(nX,(nFlowRequested > 0) ? nFlowRequested : nFlow);
	    for(uint64_t i=0; i<nX; i++) {
              const WellData *w = wells.ReadXY(x(i), y(i));
              rank(i) = w->rank;
              if(nFlowRequested > 0) {
                for(size_t j=0; j<nFlowRequested; j++)
                  signal(i,j) = w->flowValues[flow(j)]; //wellData[flow(j) + (nFlow * (x(i) + (nCol * y(i))))];
              } else {
                for(size_t j=0; j<nFlow; j++)
                  signal(i,j) = w->flowValues[j]; //wellData[j + (nFlow * (x(i) + (nCol * y(i))))];
              }
	    }

	    // Make vector holding the acquired flows
	    RcppVector<int> flow_out((nFlowRequested > 0) ? nFlowRequested : nFlow);
	    if(nFlowRequested > 0) {
		for(uint64_t j=0; j<nFlowRequested; j++)
		    flow_out(j) = flow(j);
	    } else {
		for(uint64_t j=0; j<nFlow; j++)
		    flow_out(j) = j;
	    }

	    // Build result set to be returned as a list to R.
	    RcppResultSet rs;
	    rs.add("nFlow",        (int)nFlow);
	    rs.add("flowOrder",    *flowOrder);
	    rs.add("flow",         flow);
	    rs.add("rank",         rank);
	    rs.add("signal",       signal);

	    // Get the list to be returned to R.
	    rl = rs.getReturnList();
	}

    if(flowOrder) delete flowOrder;

    } catch(std::exception& ex) {
	exceptionMesg = copyMessageToR(ex.what());
    } catch(...) {
	exceptionMesg = copyMessageToR("unknown reason");
    }
    
    if(exceptionMesg != NULL)
	Rf_error(exceptionMesg);

    return rl;
}
