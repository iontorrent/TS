/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include "Mask.h"

RcppExport SEXP readBeadFindMask(SEXP beadFindFile_in, SEXP x_in, SEXP y_in) {

    SEXP rl = R_NilValue; 		// Use this when there is nothing to be returned.
    char *exceptionMesg = NULL;

    try {

	// Recasting of input arguments
	RcppStringVector  beadFindFile_temp(beadFindFile_in);
	std::string *beadFindFile = new std::string(beadFindFile_temp(0));
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
 
	// Ensure lower bounds are positive and less than the upper bounds, then proceed to open the file
	if(nX != nY) {
	    exceptionMesg = copyMessageToR("x and y should be of the same length");
	} else if(nX < 0) {
	    exceptionMesg = copyMessageToR("x and y should be of posiive length");
	} else if(xMin < 0) {
	    exceptionMesg = copyMessageToR("xMin must be positive");
	} else if(yMin < 0) {
	    exceptionMesg = copyMessageToR("yMin must be positive");
	} else if(xMin > xMax) {
	    exceptionMesg = copyMessageToR("xMin must be less than xMax");
	} else if(yMin > yMax) {
	    exceptionMesg = copyMessageToR("yMin must be less than yMax");
	} else {
	    FILE *fp = NULL;
	    fp = fopen(beadFindFile->c_str(),"rb");
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
		} else {
		    // Ensure upper bounds are within range before continuing
		    if(yMax >= nRow) {
			exceptionMesg = copyMessageToR("yMax must be less than the number of rows");
		    } else if(xMax >= nCol) {
			exceptionMesg = copyMessageToR("xMax must be less than the number of cols");
		    } else {
			// Iterate over mask data and store out the encoded Boolean values
			uint16_t mask = 0;
			RcppVector<int> maskEmpty(nX);
			RcppVector<int> maskBead(nX);
			RcppVector<int> maskLive(nX);
			RcppVector<int> maskDud(nX);
			RcppVector<int> maskReference(nX);
			RcppVector<int> maskTF(nX);
			RcppVector<int> maskLib(nX);
			RcppVector<int> maskPinned(nX);
			RcppVector<int> maskIgnore(nX);
			RcppVector<int> maskWashout(nX);
			RcppVector<int> maskExclude(nX);
			RcppVector<int> maskKeypass(nX);
			RcppVector<int> maskFilteredBadKey(nX);
			RcppVector<int> maskFilteredShort(nX);
			RcppVector<int> maskFilteredBadPPF(nX);
			RcppVector<int> maskFilteredBadResidual(nX);
			int64_t headerSkipBytes = 2 * sizeof(uint32_t);
			int64_t dataSkipBytes = sizeof(uint16_t);
			int64_t offset;
			for (uint64_t i=0; i < nX; i++) {
			    int col = x(i); 
			    int row = y(i);
			    offset = headerSkipBytes + (row * nCol + col) * dataSkipBytes;
			    fseek(fp, offset, SEEK_SET);
			    if ((fread (&mask, sizeof(uint16_t), 1, fp)) != 1) {
				std::string exception = "Problem reading mask values from " + *beadFindFile;
				exceptionMesg = copyMessageToR(exception.c_str());
				break;
			    } else {
				maskEmpty(i)                = (mask & MaskEmpty) > 0;
				maskBead(i)                 = (mask & MaskBead) > 0;
				maskLive(i)                 = (mask & MaskLive) > 0;
				maskDud(i)                  = (mask & MaskDud) > 0;
				maskReference(i)            = (mask & MaskReference) > 0;
				maskTF(i)                   = (mask & MaskTF) > 0;
				maskLib(i)                  = (mask & MaskLib) > 0;
				maskPinned(i)               = (mask & MaskPinned) > 0;
				maskIgnore(i)               = (mask & MaskIgnore) > 0;
				maskWashout(i)              = (mask & MaskWashout) > 0;
				maskExclude(i)              = (mask & MaskExclude) > 0;
				maskKeypass(i)              = (mask & MaskKeypass) > 0;
				maskFilteredBadKey(i)       = (mask & MaskFilteredBadKey) > 0;
				maskFilteredShort(i)        = (mask & MaskFilteredShort) > 0;
				maskFilteredBadPPF(i)       = (mask & MaskFilteredBadPPF) > 0;
				maskFilteredBadResidual(i)  = (mask & MaskFilteredBadResidual) > 0;
			    }
			}

			// Check if there are any non-zero values for Keypass and Exclude.  If
			// not then it is likely that this is a legacy bfmask.bin file and we won't return
			// these fields.
			bool haveExcludeMask=0;
			for (uint64_t i=0; i < nX; i++) {
			    if(maskExclude(i)) {
				haveExcludeMask=1;
				break;
			    }
			}
			bool haveKeypassMask=0;
			for (uint64_t i=0; i < nX; i++) {
			    if(maskKeypass(i)) {
				haveKeypassMask=1;
				break;
			    }
			}

			// Check if there are any non-zero values for Keypass and Exclude.  If
			// not then it is likely that this is a legacy bfmask.bin file and we won't return
			// these fields.
			bool haveFiltered=0;
			for (uint64_t i=0; i < nX; i++) {
			    if(maskFilteredBadKey(i) || maskFilteredShort(i) || maskFilteredBadPPF(i) || maskFilteredBadResidual(i)) {
				haveFiltered=1;
				break;
			    }
			}

			// Build result set to be returned as a list to R.
			RcppResultSet rs;
			rs.add("beadFindMaskFile", *beadFindFile);
			rs.add("nCol",             nCol);
			rs.add("nRow",             nRow);
			rs.add("col",              x);
			rs.add("row",              y);
			rs.add("maskEmpty",        maskEmpty);
			rs.add("maskBead",         maskBead);
			rs.add("maskLive",         maskLive);
			rs.add("maskDud",          maskDud);
			rs.add("maskReference",    maskReference);
			rs.add("maskTF",           maskTF);
			rs.add("maskLib",          maskLib);
			rs.add("maskPinned",       maskPinned);
			rs.add("maskIgnore",       maskIgnore);
			rs.add("maskWashout",      maskWashout);
			if(haveExcludeMask)
			    rs.add("maskExclude",      maskExclude);
			if(haveKeypassMask)
			    rs.add("maskKeypass",      maskKeypass);
			if(haveFiltered) {
			    rs.add("maskFilteredBadKey",       maskFilteredBadKey);
			    rs.add("maskFilteredShort",        maskFilteredShort);
			    rs.add("maskFilteredBadPPF",       maskFilteredBadPPF);
			    rs.add("maskFilteredBadResidual",  maskFilteredBadResidual);
			}


			// Get the list to be returned to R.
			rl = rs.getReturnList();
		    }
		}

		fclose(fp);
	    }
	}

	delete beadFindFile;
	
    } catch(std::exception& ex) {
	exceptionMesg = copyMessageToR(ex.what());
    } catch(...) {
	exceptionMesg = copyMessageToR("unknown reason");
    }
    
    if(exceptionMesg != NULL)
	Rf_error(exceptionMesg);

    return rl;
}
