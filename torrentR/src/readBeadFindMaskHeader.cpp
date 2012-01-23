/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include "Mask.h"

RcppExport SEXP readBeadFindMaskHeader(SEXP beadFindFile_in) {

    SEXP rl = R_NilValue; 		// Use this when there is nothing to be returned.
    char *exceptionMesg = NULL;

    try {

	RcppStringVector  beadFindFile_temp(beadFindFile_in);
	char *beadFindFile = strdup(beadFindFile_temp(0).c_str());
 
	FILE *fp = NULL;
	fp = fopen(beadFindFile,"rb");
	if(!fp) {
	    std::string exception = "unable to open beadFindFile " + *beadFindFile;
	    exceptionMesg = copyMessageToR(exception.c_str());
	} else {
	    int32_t nRow = 0;
	    int32_t nCol = 0;
	    if ((fread (&nRow, sizeof(uint32_t), 1, fp )) != 1) {
		// Read number of rows (aka "y" or "height")
		std::string exception = "Problem reading nRow from beadFindFile" + *beadFindFile;
		exceptionMesg = copyMessageToR(exception.c_str());
	    } else if ((fread (&nCol, sizeof(uint32_t), 1, fp )) != 1) {
		// Read number of cols (aka "x" or "width")
		std::string exception = "Problem reading nCol from beadFindFile" + *beadFindFile;
		exceptionMesg = copyMessageToR(exception.c_str());
	    }

	    fclose(fp);

	    // Build result set to be returned as a list to R.
	    RcppResultSet rs;
	    rs.add("nRow",          nRow);
	    rs.add("nCol",          nCol);

	    // Get the list to be returned to R.
	    rl = rs.getReturnList();
	}

    free(beadFindFile);
	
    } catch(std::exception& ex) {
	exceptionMesg = copyMessageToR(ex.what());
    } catch(...) {
	exceptionMesg = copyMessageToR("unknown reason");
    }
    
    if(exceptionMesg != NULL)
	Rf_error(exceptionMesg);

    return rl;
}
