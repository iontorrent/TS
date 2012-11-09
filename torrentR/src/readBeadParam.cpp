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
RcppExport SEXP readBeadParamR(SEXP RbeadParamFile,SEXP RminCol, SEXP RmaxCol, SEXP RminRow, SEXP RmaxRow,SEXP RminFlow, SEXP RmaxFlow) {
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    //vector<string> beadParamFile = Rcpp::as< vector<string> > (RbeadParamFile);
    //RcppStringVector tBeadParamFile(RbeadParamFile);
    string beadParamFile = Rcpp::as< string > (RbeadParamFile);
    int minCol = Rcpp::as< int > (RminCol);
    int maxCol = Rcpp::as< int > (RmaxCol);
    int minRow = Rcpp::as< int > (RminRow);
    int maxRow = Rcpp::as< int > (RmaxRow);
    int minFlow = Rcpp::as< int > (RminFlow);
    int maxFlow = Rcpp::as< int > (RmaxFlow);
    // Outputs

    unsigned int nCol = maxCol - minCol+1;
    unsigned int nRow = maxRow - minRow+1;
    unsigned int nFlow = maxFlow - minFlow+1;
    vector<unsigned int> colOut,rowOut,flowOut;
    vector< double > resErr;
    
    colOut.reserve(nCol * nRow * nFlow);
    rowOut.reserve(nCol * nRow * nFlow);
    // Recast int to unsigned int
    vector<unsigned int> col, row, flow;
    col.resize(nCol);
    for(unsigned int i=0;i<col.size();++i) 
      col[i] = (unsigned int)(i+minCol);
    row.resize(nRow);
    for(unsigned int i=0;i<row.size();++i)
      row[i] = (unsigned int) (i+minRow);
    flow.resize(nFlow);
    for(unsigned int i=0;i<flow.size();++i)
      flow[i] = (unsigned int) (i+minFlow);

    H5File beadParam;
    beadParam.SetFile(beadParamFile);
    beadParam.Open();
    H5DataSet *resErrDS = beadParam.OpenDataSet("/bead/res_error");
    H5DataSet *ampMulDS = beadParam.OpenDataSet("bead/ampl_multiplier");
    H5DataSet *krateMulDS = beadParam.OpenDataSet("bead/k_rate_multiplier");
    H5DataSet *beadInitDS = beadParam.OpenDataSet("bead/bead_init_param");
    H5DataSet *beadDCDS = beadParam.OpenDataSet("bead/fg_bead_DC");
   
    //float  tresErr[nCol * nRow * nFlow];
    float *tresErr;
    float *tamplMul;
    float *tkrateMul;
    float *tbeadInit;
    float *tbeadDC;

    tresErr = (float *) malloc ( sizeof(float) * nCol * nRow * nFlow);
    tamplMul = (float *) malloc ( sizeof(float) * nCol * nRow * nFlow);
    tkrateMul = (float *) malloc ( sizeof(float) * nCol * nRow * nFlow);
    tbeadInit = (float *) malloc ( sizeof(float) * nCol * nRow * 4);
    tbeadDC = (float *) malloc ( sizeof(float) * nCol * nRow * nFlow);

    size_t starts[3];
    size_t ends[3];
    starts[0] = minCol;
    starts[1] = minRow;
    starts[2] = minFlow;
    ends[0] = maxCol;
    ends[1] = maxRow;
    ends[2] = maxFlow;
    resErrDS->ReadRangeData(starts, ends, sizeof(tresErr),tresErr);
    ampMulDS->ReadRangeData(starts, ends, sizeof(tamplMul),tamplMul);
    krateMulDS->ReadRangeData(starts, ends, sizeof(tkrateMul),tkrateMul);
    beadDCDS->ReadRangeData(starts, ends, sizeof(tbeadDC),tbeadDC);

    starts[2] = 0;
    ends[2] = 3;
    beadInitDS->ReadRangeData(starts,ends,sizeof(tbeadInit),tbeadInit);
    beadParam.Close();

    RcppMatrix< double > resErrMat(nRow*nCol,nFlow);
    RcppMatrix< double > amplMulMat(nRow*nCol,nFlow);
    RcppMatrix< double > krateMulMat(nRow*nCol,nFlow);
    RcppMatrix< double > beadInitMat(nRow*nCol,4);
    RcppMatrix< double > beadDCMat(nRow*nCol,nFlow);
    vector<int> colOutInt,rowOutInt,flowOutInt;

    int count = 0;
    for(size_t icol=0;icol<nCol;++icol)
      for(size_t irow=0;irow<nRow;++irow) {
	for(size_t iFlow=0;iFlow<nFlow;++iFlow)
	  {
	    amplMulMat(count,iFlow) = (double) tamplMul[icol * nRow * nFlow + irow * nFlow + iFlow] ;
	    krateMulMat(count,iFlow) = (double) tkrateMul[icol * nRow * nFlow + irow * nFlow + iFlow];
	    beadDCMat(count,iFlow) = (double) tbeadDC[icol * nRow *nFlow + irow * nFlow + iFlow];
	  }
	for(size_t ip=0;ip<4;++ip)
	  beadInitMat(count,ip) = (double) tbeadInit[icol * nRow * ip + irow * 4 + ip];
	colOutInt.push_back(minCol+icol);
	rowOutInt.push_back(minRow+irow);
	count++;
      }

    RcppResultSet rs;
    rs.add("beadParamFile", beadParamFile);
    rs.add("nCol", (int)nCol);
    rs.add("nRow", (int)nRow);
    rs.add("minFlow",minFlow);
    rs.add("maxFlow",maxFlow);
    rs.add("nFlow", (int)nFlow);
    rs.add("col", colOutInt);
    rs.add("row",rowOutInt);
    rs.add("res_error",resErrMat);
    rs.add("ampl_multiplier",amplMulMat);
    rs.add("krate_multiplier",krateMulMat);
    rs.add("bead_init",beadInitMat);
    rs.add("bead_dc",beadDCMat);
    ret = rs.getReturnList();
    free(tresErr);
    free(tamplMul);
    free(tkrateMul);
    free(tbeadInit);
    free(tbeadDC);
  }
  catch(exception& ex) {
    exceptionMesg = copyMessageToR(ex.what());
  } catch (...) {
    exceptionMesg = copyMessageToR("Unknown reason");
  }
  
  if ( exceptionMesg != NULL)
    Rf_error(exceptionMesg);
  
  return ret;

}
    
