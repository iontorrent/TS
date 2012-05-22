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
  char *exceptionMesg = NULL;

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
      exceptionMesg = copyMessageToR(exception.c_str());
    } else {
      int nDat = datFile.size();
      int nWell = colOut.size();

      // Recast colOut and rowOut from int to unsigned int because Rcpp has no unsigned int
      vector<int> colOutInt,rowOutInt;
      colOutInt.resize(colOut.size());
      for(unsigned int i=0; i<colOutInt.size(); i++)
        colOutInt[i] = (int) colOut[i];
      rowOutInt.resize(rowOut.size());
      for(unsigned int i=0; i<rowOutInt.size(); i++)
        rowOutInt[i] = (int) rowOut[i];
  
      // Recast vectors of vectors to arrays.
      vector<int> sigInt;
      vector<double> sigMean,sigSD;
      vector<double> sigLag;
      if(returnSignal) {
        sigInt.reserve( ((size_t)nDat*(size_t)nWell*(size_t)nFrame));
			}
      if(returnWellMean) {
        sigMean.reserve((size_t)nDat*(size_t)nWell); 
			}
      if(returnWellSD) {
        sigSD.reserve((size_t)nDat*(size_t)nWell);
			}
      if (returnWellLag){
        sigLag.reserve((size_t)nDat*(size_t)nWell);
      }
      for(int iDat=0; iDat < nDat; iDat++) {
        if(returnSignal) {
          for(unsigned int iFrame=0; iFrame < signal[iDat][0].size(); iFrame++) {
            for(unsigned int iWell=0; iWell < signal[iDat].size(); iWell++) {
              sigInt.push_back(signal[iDat][iWell][iFrame]);
            }
          }
        }
        if(returnWellMean || returnWellSD) {
          for(int iWell=0; iWell < nWell; iWell++) {
            if(returnWellMean)
              sigMean.push_back(mean[iDat][iWell]);
            if(returnWellSD)
              sigSD.push_back(sd[iDat][iWell]);
          }
        }
        if (returnWellLag){
          for(int iWell=0; iWell<nWell; iWell++){
            sigLag.push_back(lag[iDat][iWell]);
          }
        }
      }
      RcppResultSet rs;
      rs.add("datFile",       datFile);
      rs.add("nCol",          (int) nCol);
      rs.add("nRow",          (int) nRow);
      rs.add("nFrame",        (int) nFrame);
      rs.add("nFlow",         nDat);
      rs.add("col",           colOutInt);
      rs.add("row",           rowOutInt);
      rs.add("frameStart",    frameStart);
      rs.add("frameEnd",      frameEnd);
      if(returnSignal){
        rs.add("signal",        sigInt);
			}
      if(returnWellMean)
        rs.add("wellMean",      sigMean);
      if(returnWellSD)
        rs.add("wellSD",        sigSD);
      if(returnWellLag)
        rs.add("wellLag",       sigLag);
      ret = rs.getReturnList();
    }
  } catch(exception& ex) {
    exceptionMesg = copyMessageToR(ex.what());
  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
