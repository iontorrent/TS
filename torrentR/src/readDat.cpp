/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <string>
#include <iostream>
#include <Rcpp.h>
#include "Image.h"

using namespace std;

RcppExport SEXP readDat(
  SEXP RdatFile,
  SEXP Rcol,
  SEXP Rrow,
  SEXP RminCol,
  SEXP RmaxCol,
  SEXP RminRow,
  SEXP RmaxRow,
  SEXP RreturnSignal,
  SEXP RreturnWellMean,
  SEXP RreturnWellSD,
  SEXP RreturnWellLag,
  SEXP Runcompress,
  SEXP RdoNormalize,
  SEXP RnormStart,
  SEXP RnormEnd,
  SEXP RXTCorrect,
  SEXP RchipType,
  SEXP RbaselineMinTime,
  SEXP RbaselineMaxTime,
  SEXP RloadMinTime,
  SEXP RloadMaxTime
) {

  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  std::map<std::string,SEXP> map;
  char *exceptionMesg = NULL;
  Rcpp::IntegerVector* sigInt = NULL;

  try {

    vector<string> datFile  = Rcpp::as< vector<string> >(RdatFile);
    vector<int> colInt      = Rcpp::as< vector<int> >(Rcol);
    vector<int> rowInt      = Rcpp::as< vector<int> >(Rrow);
    int   minCol            = Rcpp::as<int>(RminCol);
    int   maxCol            = Rcpp::as<int>(RmaxCol);
    int   minRow            = Rcpp::as<int>(RminRow);
    int   maxRow            = Rcpp::as<int>(RmaxRow);
    bool  returnSignal      = Rcpp::as<bool>(RreturnSignal);
    bool  returnWellMean    = Rcpp::as<bool>(RreturnWellMean);
    bool  returnWellSD      = Rcpp::as<bool>(RreturnWellSD);
    bool  returnWellLag     = Rcpp::as<bool>(RreturnWellLag);
    bool  uncompress        = Rcpp::as<bool>(Runcompress);
    bool  doNormalize       = Rcpp::as<bool>(RdoNormalize);
    int   normStart         = Rcpp::as<int>(RnormStart);
    int   normEnd           = Rcpp::as<int>(RnormEnd);
    bool  XTCorrect         = Rcpp::as<bool>(RXTCorrect);
    string chipType         = Rcpp::as<string>(RchipType);
    double baselineMinTime  = Rcpp::as<double>(RbaselineMinTime);
    double baselineMaxTime  = Rcpp::as<double>(RbaselineMaxTime);
    double loadMinTime      = Rcpp::as<double>(RloadMinTime);
    double loadMaxTime      = Rcpp::as<double>(RloadMaxTime);

    // Outputs
    unsigned int nCol   = 0;
    unsigned int nRow   = 0;
    unsigned int nFrame = 0;
    vector<unsigned int> colOut,rowOut;
    vector< vector<double> > frameStart,frameEnd;
    vector< vector< vector<short> > > signal;
    vector< vector<short> > mean,sd,lag;
    vector<int> colOutInt,rowOutInt;

    // Recast int to unsigned int because Rcpp has no unsigned int
    vector<unsigned int> col,row;
    col.resize(colInt.size());
    for(unsigned int i=0; i<col.size(); i++)
      col[i] = (unsigned int) colInt[i];
    row.resize(rowInt.size());
    for(unsigned int i=0; i<row.size(); i++)
      row[i] = (unsigned int) rowInt[i];

    Image i;
		std::cout << "Loading slice." << std::endl;
    if(!i.LoadSlice(
      // Inputs
      datFile,
      col,
      row,
      minCol,
      maxCol,
      minRow,
      maxRow,
      returnSignal,
      returnWellMean,
      returnWellSD,
      returnWellLag,
      uncompress,
      doNormalize,
      normStart,
      normEnd,
      XTCorrect,
      chipType,
      baselineMinTime,
      baselineMaxTime,
      loadMinTime,
      loadMaxTime,
      // Outputs
      nCol,
      nRow,
      colOut,
      rowOut,
      nFrame,
      frameStart,
      frameEnd,
      signal,
      mean,
      sd,
      lag
    )) {
      string exception = "Problem reading dat data\n";
      exceptionMesg = strdup(exception.c_str());
    } else {
      int nDat = datFile.size();
      int nWell = colOut.size();

      sigInt = new Rcpp::IntegerVector(((size_t)nDat*(size_t)nWell*(size_t)nFrame));
      Rcpp::NumericMatrix sigMean(nDat,nWell);
      Rcpp::NumericMatrix sigSD(nDat,nWell);
      Rcpp::NumericMatrix sigLag(nDat,nWell);
      Rcpp::IntegerVector colOutInt(nWell);
      Rcpp::IntegerVector rowOutInt(nWell);
      Rcpp::NumericVector frameStartOut(nDat*nFrame);
      Rcpp::NumericVector frameEndOut(nDat*nFrame);

      for(int i=0; i<nWell; i++) {
        colOutInt(i) = (int) colOut[i];
        rowOutInt(i) = (int) rowOut[i];
      }

      int cnt = 0;
      for(int iDat=0; iDat < nDat; iDat++) {
        unsigned int frameOffset=iDat*nFrame;
        for(unsigned int iFrame=0; iFrame<nFrame; iFrame++) {
          frameStartOut(frameOffset+iFrame) = frameStart[iDat][iFrame];
          frameEndOut(frameOffset+iFrame)   = frameEnd[iDat][iFrame];
        }
        if(returnSignal) {
          for(unsigned int iFrame=0; iFrame < signal[iDat][0].size(); iFrame++) {
            for(unsigned int iWell=0; iWell < signal[iDat].size(); iWell++) {
              (*sigInt)[cnt++]=signal[iDat][iWell][iFrame];
            }
          }
        }
        if(returnWellMean || returnWellSD || returnWellLag) {
          for(int iWell=0; iWell < nWell; iWell++) {
            if(returnWellMean)
              sigMean(iDat,iWell) = mean[iDat][iWell];
            if(returnWellSD)
              sigSD(iDat,iWell) = sd[iDat][iWell];
            if(returnWellLag)
              sigLag(iDat,iWell) = lag[iDat][iWell];
          }
        }
      }

      map["datFile"]    = Rcpp::wrap( datFile );
      map["nCol"]       = Rcpp::wrap( (int) nCol );
      map["nRow"]       = Rcpp::wrap( (int) nRow );
      map["nFrame"]     = Rcpp::wrap( (int) nFrame );
      map["nFlow"]      = Rcpp::wrap( nDat );
      map["col"]        = Rcpp::wrap( colOutInt );
      map["row"]        = Rcpp::wrap( rowOutInt );
      map["frameStart"] = Rcpp::wrap( frameStartOut );
      map["frameEnd"]   = Rcpp::wrap( frameEndOut );
      if(returnSignal){
        map["signal"]   = *sigInt;
      }
      if(returnWellMean)
        map["wellMean"] = Rcpp::wrap( sigMean );
      if(returnWellSD)
        map["wellSD"]   = Rcpp::wrap( sigSD );
      if(returnWellLag)
        map["wellLag"]  =  Rcpp::wrap( sigLag );

      ret = Rcpp::wrap( map );

    }
  } catch(exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }

  delete sigInt;

  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
