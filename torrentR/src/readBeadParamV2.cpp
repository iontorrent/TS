/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <vector>
#include <iostream>
#include <Rcpp.h>
#include <hdf5.h>
#include <H5File.h>
#include <SampleStats.h>
#include <SampleQuantiles.h>
#include <cstdlib>

using namespace std;
// col, row, and flow are zero-based in this function, info read out include both borders.
//RcppExport SEXP readBeadParamR(SEXP RbeadParamFile,SEXP RminCol, SEXP RmaxCol, SEXP RminRow, SEXP RmaxRow,SEXP RminFlow, SEXP RmaxFlow, SEXP RNUMFB) {
RcppExport SEXP readBeadParamRV2(SEXP RbeadParamFile,SEXP RminCol, SEXP RmaxCol, SEXP RminRow, SEXP RmaxRow,SEXP RminFlow, SEXP RmaxFlow) {
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    //vector<string> beadParamFile = Rcpp::as< vector<string> > (RbeadParamFile);
    //Rcpp::StringVector tBeadParamFile(RbeadParamFile);
    string beadParamFile = Rcpp::as< string > (RbeadParamFile);
    int minCol = Rcpp::as< int > (RminCol);
    int maxCol = Rcpp::as< int > (RmaxCol);
    int minRow = Rcpp::as< int > (RminRow);
    int maxRow = Rcpp::as< int > (RmaxRow);
    int minFlow = Rcpp::as< int > (RminFlow);
    int maxFlow = Rcpp::as< int > (RmaxFlow);
    //int NUMFB = Rcpp::as< int > (RNUMFB);
    // Outputs

    unsigned int nCol = maxCol - minCol+1;
    unsigned int nRow = maxRow - minRow+1;
    unsigned int nFlow = maxFlow - minFlow+1;
    vector<unsigned int> colOut,rowOut,flowOut;
    vector< double > resErr;
    
    colOut.reserve(nCol * nRow * nFlow);
    rowOut.reserve(nCol * nRow * nFlow);

    H5File beadParam;
    beadParam.SetFile(beadParamFile);
    //beadParam.Open();
    beadParam.OpenForReading();
    H5DataSet *resErrDS = beadParam.OpenDataSet("/bead/residual_error");
    //H5DataSet *avgBlkErrDS = beadParam.OpenDataSet("/bead/average_error_by_block");                                                                     
    H5DataSet *ampDS = beadParam.OpenDataSet("bead/amplitude");
    H5DataSet *krateMulDS = beadParam.OpenDataSet("bead/kmult");
    H5DataSet *beadInitDS = beadParam.OpenDataSet("bead/bead_base_parameters");
    H5DataSet *beadDCDS = beadParam.OpenDataSet("bead/trace_dc_offset");

    float *tresErr;
    //float *tavgBlkErr;                                                                                                                                  
    float *tamp;
    float *tkrateMul;
    float *tbeadInit;
    float *tbeadDC;
    int nParams = 5;

    tresErr = (float *) malloc ( sizeof(float) * nCol * nRow * nFlow);
    //tavgBlkErr = (float *) malloc ( sizeof(float) * nCol * nRow * nFlow);                                                                               
    tamp = (float *) malloc ( sizeof(float) * nCol * nRow * nFlow);
    tkrateMul = (float *) malloc ( sizeof(float) * nCol * nRow * nFlow);
    tbeadInit = (float *) malloc ( sizeof(float) * nCol * nRow * nParams);
    tbeadDC = (float *) malloc ( sizeof(float) * nCol * nRow * nFlow);

    size_t starts[3];
    size_t ends[3];
    starts[0] = minCol;
    starts[1] = minRow;
    starts[2] = minFlow;
    ends[0] = maxCol+1;
    ends[1] = maxRow+1;
    ends[2] = maxFlow+1;
    resErrDS->ReadRangeData(starts, ends, sizeof(tresErr),tresErr);
    ampDS->ReadRangeData(starts, ends, sizeof(tamp),tamp);
    krateMulDS->ReadRangeData(starts, ends, sizeof(tkrateMul),tkrateMul);
    beadDCDS->ReadRangeData(starts, ends, sizeof(tbeadDC),tbeadDC);

    starts[2] = 0;
    ends[2] = nParams;
    beadInitDS->ReadRangeData(starts,ends,sizeof(tbeadInit),tbeadInit);
    beadParam.Close();

    Rcpp::NumericMatrix resErrMat(nRow*nCol,nFlow);
    Rcpp::NumericMatrix ampMat(nRow*nCol,nFlow);
    Rcpp::NumericMatrix krateMulMat(nRow*nCol,nFlow);
    Rcpp::NumericMatrix beadInitMat(nRow*nCol,nParams);
    Rcpp::NumericMatrix beadDCMat(nRow*nCol,nFlow);
    vector<int> colOutInt,rowOutInt,flowOutInt;

    int count = 0;
    for(size_t icol=0;icol<nCol;++icol)
      for(size_t irow=0;irow<nRow;++irow) {
	for(size_t iFlow=0;iFlow<nFlow;++iFlow)
	  {
	    ampMat(count,iFlow) = (double) tamp[icol * nRow * nFlow + irow * nFlow + iFlow] ;
	    resErrMat(count,iFlow) = (double) tresErr[icol * nRow * nFlow + irow * nFlow + iFlow] ;
	    krateMulMat(count,iFlow) = (double) tkrateMul[icol * nRow * nFlow + irow * nFlow + iFlow];
	    beadDCMat(count,iFlow) = (double) tbeadDC[icol * nRow * nFlow + irow * nFlow + iFlow];
	  }
	for(size_t ip=0;ip<(size_t)nParams;++ip)
	  beadInitMat(count,ip) = (double) tbeadInit[icol * nRow * nParams + irow * nParams + ip];
	colOutInt.push_back(minCol+icol);
	rowOutInt.push_back(minRow+irow);
	count++;
      }

    for(size_t iFlow=0;iFlow<nFlow;++iFlow)
	flowOutInt.push_back(minFlow+iFlow);

    ret = Rcpp::List::create(Rcpp::Named("beadParamFile")   = beadParamFile,
                             Rcpp::Named("nCol")            = (int)nCol,
                             Rcpp::Named("nRow")            = (int)nRow,
                             Rcpp::Named("minFlow")         = minFlow,
                             Rcpp::Named("maxFlow")         = maxFlow,
                             Rcpp::Named("nFlow")           = (int)nFlow,
                             Rcpp::Named("col")             = colOutInt,
                             Rcpp::Named("row")             = rowOutInt,
                             Rcpp::Named("flow")            = flowOutInt,
                             Rcpp::Named("res_error")       = resErrMat,
                             Rcpp::Named("amplitude")       = ampMat,
                             Rcpp::Named("kmult")           = krateMulMat,
                             Rcpp::Named("bead_init")       = beadInitMat,
                             Rcpp::Named("trace_dc_offset") = beadDCMat);

    free(tresErr);
    free(tamp);
    free(tkrateMul);
    free(tbeadInit);
    free(tbeadDC);
  }
  catch(exception& ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  
  if ( exceptionMesg != NULL)
    Rf_error(exceptionMesg);
  
  return ret;

}
    
