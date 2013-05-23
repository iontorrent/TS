/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include "Mask.h"

RcppExport SEXP readBeadFindMask(SEXP beadFindFile_in, SEXP x_in, SEXP y_in) {

    SEXP rl = R_NilValue; 		// Use this when there is nothing to be returned.
    char *exceptionMesg = NULL;

    try {

	// Recasting of input arguments
	Rcpp::StringVector  beadFindFile_temp(beadFindFile_in);
	std::string *beadFindFile = new std::string(beadFindFile_temp(0));
	// x
	Rcpp::IntegerVector x_temp(x_in);
	uint64_t nX = x_temp.size();
	Rcpp::IntegerVector x(nX);
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
	Rcpp::IntegerVector y_temp(y_in);
	uint64_t nY = y_temp.size();
	Rcpp::IntegerVector y(nX);
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
	    exceptionMesg = strdup("x and y should be of the same length");
	} else if(nX < 0) {
	    exceptionMesg = strdup("x and y should be of posiive length");
	} else if(xMin < 0) {
	    exceptionMesg = strdup("xMin must be positive");
	} else if(yMin < 0) {
	    exceptionMesg = strdup("yMin must be positive");
	} else if(xMin > xMax) {
	    exceptionMesg = strdup("xMin must be less than xMax");
	} else if(yMin > yMax) {
	    exceptionMesg = strdup("yMin must be less than yMax");
	} else {
	    FILE *fp = NULL;
	    fp = fopen(beadFindFile->c_str(),"rb");
	    if(!fp) {
		std::string exception = "unable to open beadFindFile " + *beadFindFile;
		exceptionMesg = strdup(exception.c_str());
	    } else {
		int32_t nRow = 0;
		int32_t nCol = 0;
    
		if ((fread (&nRow, sizeof(uint32_t), 1, fp )) != 1) {
		    // Read number of rows (aka "y" or "height")
		    std::string exception = "Problem reading nRow from beadFindFile" + *beadFindFile;
		    exceptionMesg = strdup(exception.c_str());
		} else if ((fread (&nCol, sizeof(uint32_t), 1, fp )) != 1) {
		    // Read number of cols (aka "x" or "width")
		    std::string exception = "Problem reading nCol from beadFindFile" + *beadFindFile;
		    exceptionMesg = strdup(exception.c_str());
		} else {
		    // Ensure upper bounds are within range before continuing
		    if(yMax >= nRow) {
			exceptionMesg = strdup("yMax must be less than the number of rows");
		    } else if(xMax >= nCol) {
			exceptionMesg = strdup("xMax must be less than the number of cols");
		    } else {
			// Iterate over mask data and store out the encoded Boolean values
			uint16_t mask = 0;
			Rcpp::IntegerVector maskEmpty(nX);
			Rcpp::IntegerVector maskBead(nX);
			Rcpp::IntegerVector maskLive(nX);
			Rcpp::IntegerVector maskDud(nX);
			Rcpp::IntegerVector maskReference(nX);
			Rcpp::IntegerVector maskTF(nX);
			Rcpp::IntegerVector maskLib(nX);
			Rcpp::IntegerVector maskPinned(nX);
			Rcpp::IntegerVector maskIgnore(nX);
			Rcpp::IntegerVector maskWashout(nX);
			Rcpp::IntegerVector maskExclude(nX);
			Rcpp::IntegerVector maskKeypass(nX);
			Rcpp::IntegerVector maskFilteredBadKey(nX);
			Rcpp::IntegerVector maskFilteredShort(nX);
			Rcpp::IntegerVector maskFilteredBadPPF(nX);
			Rcpp::IntegerVector maskFilteredBadResidual(nX);
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
				exceptionMesg = strdup(exception.c_str());
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

			// Check if there are any non-zero values for Keypass and Exclude and filtered data.
			// If not then it is likely that this is a legacy bfmask.bin file and we won't return
			// these fields.
			//bool haveFiltered=0;
			//for (uint64_t i=0; i < nX; i++) {
			//    if(maskKeypass(i) || maskExclude(i) || maskFilteredBadKey(i) || maskFilteredShort(i) ||
            //       maskFilteredBadPPF(i) || maskFilteredBadResidual(i)) {
			//	haveFiltered=1;
			//	break;
			//    }
			//}

			// Build result set to be returned as a list to R.
            std::map<std::string,SEXP> map;
            map["beadFindMaskFile"] = Rcpp::wrap( *beadFindFile );
            map["nCol"]             = Rcpp::wrap( nCol );
            map["nRow"]             = Rcpp::wrap( nRow );
            map["col"]              = Rcpp::wrap( x );
            map["row"]              = Rcpp::wrap( y );
            map["maskEmpty"]        = Rcpp::wrap( maskEmpty );
            map["maskBead"]         = Rcpp::wrap( maskBead );
            map["maskLive"]         = Rcpp::wrap( maskLive );
            map["maskDud"]          = Rcpp::wrap( maskDud );
            map["maskReference"]    = Rcpp::wrap( maskReference );
            map["maskTF"]           = Rcpp::wrap( maskTF );
            map["maskLib"]          = Rcpp::wrap( maskLib );
            map["maskPinned"]       = Rcpp::wrap( maskPinned );
            map["maskIgnore"]       = Rcpp::wrap( maskIgnore );
            map["maskWashout"]      = Rcpp::wrap( maskWashout );
			//if(haveFiltered) {
              map["maskExclude"]              = Rcpp::wrap( maskExclude );
              map["maskKeypass"]              = Rcpp::wrap( maskKeypass );
              map["maskFilteredBadKey"]       = Rcpp::wrap( maskFilteredBadKey );
              map["maskFilteredShort"]        = Rcpp::wrap( maskFilteredShort );
              map["maskFilteredBadPPF"]       = Rcpp::wrap( maskFilteredBadPPF );
              map["maskFilteredBadResidual"]  = Rcpp::wrap( maskFilteredBadResidual );
            //}


			// Get the list to be returned to R.
            rl = Rcpp::wrap( map ) ;
		    }
		}

		fclose(fp);
	    }
	}

	delete beadFindFile;
	
    } catch(std::exception& ex) {
	forward_exception_to_r(ex);
    } catch(...) {
	::Rf_error("c++ exception (unknown reason)");
    }
    
    if(exceptionMesg != NULL)
	Rf_error(exceptionMesg);

    return rl;
}
