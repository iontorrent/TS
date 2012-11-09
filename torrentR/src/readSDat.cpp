/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <string>
#include <iostream>
#include <Rcpp.h>
#include <algorithm>
#include <iterator>
#include "Image.h"
#include "SynchDatSerialize.h"
#include "H5File.h"

using namespace std;

RcppExport SEXP R_readSDat(
			 SEXP RsdatFile
			 ) {

  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {

    char *sdatFile  = (char *)Rcpp::as< const char * >(RsdatFile);

    // Outputs
    unsigned int nCol   = 0;
    unsigned int nRow   = 0;
    vector<unsigned int> colOut,rowOut;

    TraceChunkSerializer serializer;
    SynchDat sdat;
    serializer.Read(sdatFile, sdat);
    // find max of mDepth
    GridMesh<TraceChunk> &dataMesh = sdat.mChunks;
    vector<size_t> depth(dataMesh.mBins.size());
    for(size_t bIx =0; bIx < dataMesh.mBins.size(); bIx++) {
      TraceChunk chunk = dataMesh.mBins[bIx];
      depth.push_back(chunk.mDepth);
    }
    vector<size_t>::const_iterator it;
    it = max_element(depth.begin(),depth.end());
    size_t nFrame = (size_t) *it;
    size_t nWell = dataMesh.mCol * dataMesh.mRow;
    RcppMatrix<double> signal(nWell,nFrame);
    size_t count = 0;
    for(size_t bIx =0; bIx < dataMesh.mBins.size(); bIx++) {
      TraceChunk &chunk = dataMesh.mBins[bIx];
      size_t rowEnd = chunk.mRowStart + chunk.mHeight;
      size_t colEnd = chunk.mColStart + chunk.mWidth;
      size_t frameEnd = chunk.mFrameStart + chunk.mDepth;
      for (size_t irow = chunk.mRowStart; irow < rowEnd; irow++) {
	for (size_t icol = chunk.mColStart; icol < colEnd; icol++) {
	  size_t fr = 0;
	  for (size_t frame = chunk.mFrameStart; frame < frameEnd; frame++) {
	    signal(count,fr++) = (double)chunk.At(irow, icol, frame);
	  }
	  count++;
	  colOut.push_back(icol);
	  rowOut.push_back(irow);
	}
      }
    }
    
    nCol = colOut.size();
    nRow = rowOut.size();
    // Recast colOut and rowOut from int to unsigned int because Rcpp has no unsigned int
    vector<int> colOutInt,rowOutInt;
    colOutInt.resize(colOut.size());
    for(unsigned int i=0; i<colOutInt.size(); i++)
      colOutInt[i] = (int) colOut[i];
    rowOutInt.resize(rowOut.size());
    for(unsigned int i=0; i<rowOutInt.size(); i++)
      rowOutInt[i] = (int) rowOut[i];
  
    RcppResultSet rs;
    rs.add("sdatFile",       sdatFile);
    rs.add("nCol",          (int) nCol);
    rs.add("nRow",          (int) nRow);
    rs.add("nFrame",        (int) nFrame);
    rs.add("col",           colOutInt);
    rs.add("row",           rowOutInt);
    rs.add("signal",        signal);
    ret = rs.getReturnList();
  }
 catch(exception& ex) {
  exceptionMesg = copyMessageToR(ex.what());
 } catch(...) {
  exceptionMesg = copyMessageToR("unknown reason");
 }
    
if(exceptionMesg != NULL)
  Rf_error(exceptionMesg);

return ret;
}
