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
RcppExport SEXP readFitResidualRV2(SEXP RbeadParamFile, SEXP Rcol, SEXP Rrow, SEXP Rflow, SEXP RminCol, SEXP RmaxCol, SEXP RminRow, SEXP RmaxRow,SEXP RminFlow, SEXP RmaxFlow, SEXP RreturnResErr,SEXP RregXSize, SEXP RregYSize,SEXP RreturnRegStats, SEXP RreturnRegFlowStats) {
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    //vector<string> beadParamFile = Rcpp::as< vector<string> > (RbeadParamFile);
    //RcppStringVector tBeadParamFile(RbeadParamFile);
    string beadParamFile = Rcpp::as< string > (RbeadParamFile);
    vector<int> colInt           = Rcpp::as< vector<int> > (Rcol);
    vector<int> rowInt = Rcpp::as< vector<int> > (Rrow);
    vector<int> flowInt = Rcpp::as< vector<int> > (Rflow);
    int minCol = Rcpp::as< int > (RminCol);
    int maxCol = Rcpp::as< int > (RmaxCol);
    int minRow = Rcpp::as< int > (RminRow);
    int maxRow = Rcpp::as< int > (RmaxRow);
    int minFlow = Rcpp::as< int > (RminFlow);
    int maxFlow = Rcpp::as< int > (RmaxFlow);
    int regXSize = Rcpp::as< int > (RregXSize);
    int regYSize = Rcpp::as< int > (RregYSize);
    bool returnRegStats = Rcpp::as< bool > (RreturnRegStats);
    bool returnRegFlowStats = Rcpp::as< bool > (RreturnRegFlowStats);
    bool returnResErr = Rcpp::as< bool > (RreturnResErr);

    // Outputs

    unsigned int nCol = maxCol - minCol;
    unsigned int nRow = maxRow - minRow ;
    unsigned int nFlow = maxFlow - minFlow;
    vector<unsigned int> colOut,rowOut,flowOut;
    vector< double > resErr;
    
    resErr.reserve(nCol * nRow * nFlow);
    colOut.reserve(nCol * nRow * nFlow);
    rowOut.reserve(nCol * nRow * nFlow);
    // Recast int to unsigned int
    vector<unsigned int> col, row, flow;
    col.resize(colInt.size());
    for(unsigned int i=0;i<col.size();++i) 
      col[i] = (unsigned int) colInt[i];
    row.resize(rowInt.size());
    for(unsigned int i=0;i<row.size();++i)
      row[i] = (unsigned int) rowInt[i];
    flow.resize(flowInt.size());
    for(unsigned int i=0;i<flow.size();++i)
      flow[i] = (unsigned int) flowInt[i];

    H5File beadParam;
    beadParam.SetFile(beadParamFile);
    beadParam.Open();
    H5DataSet *resErrDS = beadParam.OpenDataSet("/bead/residual_error");
    size_t starts[3];
    size_t ends[3];
   
    //float  tresErr[nCol * nRow * nFlow];
    float *tresErr;
    tresErr = (float *) malloc( sizeof(float) * nCol *nRow * nFlow);

    starts[0] = minCol;
    starts[1] = minRow;
    starts[2] = minFlow;
    ends[0] = maxCol;
    ends[1] = maxRow;
    ends[2] = maxFlow;
    // Check if the starts and ends are out of range.
    try {
      resErrDS->ReadRangeData(starts, ends, sizeof(tresErr),tresErr);
    }
    catch(exception& ex) {
      exceptionMesg = copyMessageToR(ex.what());
    }    
    // Compute number of regions
    int nRegRow = nRow/regXSize;
    int nRegCol =  nCol/regYSize;
    int nReg = nRegRow * nRegCol;
    // Compute stats
    vector < int > regRow(nReg,0),regCol(nReg,0);
    //regRow.reserve(nReg);
    //regCol.reserve(nReg);
    
    vector< SampleQuantiles<float> > regStats (nReg, SampleQuantiles<float>(nFlow * nRow * nCol)) ; 
    vector< SampleStats<float>  > regMeanStats (nReg,SampleStats<float>()) ; 
    vector< SampleQuantiles<float>  > regFlowStats (nReg * nFlow,SampleQuantiles<float>(nRow * nCol)) ; 
    vector< SampleStats<float>  > regFlowMeanStats (nReg * nFlow, SampleStats<float>()) ; 

    for(size_t icol=0;icol<nCol;++icol)
      for(size_t irow=0;irow<nRow;++irow)
	for(size_t iflow=0;iflow<nFlow;++iflow)
	  {
	    int regId = (nRegCol)*(int)(irow/regYSize) + (int) icol/regXSize;
	    regCol[regId] = regXSize * ((int)icol/regXSize);
	    regRow[regId] = regYSize * ((int)irow/regYSize);
	    
	    if(tresErr[icol * nRow * nFlow + irow * nFlow +iflow] > 0)
	      {
		colOut.push_back(icol + minCol);
		rowOut.push_back(irow + minRow);
		flowOut.push_back(iflow + minFlow);
		resErr.push_back(tresErr[icol * nRow * nFlow + irow * nFlow + iflow]);
		regStats.at(regId).AddValue(tresErr[icol * nRow * nFlow + irow * nFlow + iflow]);
		regMeanStats.at(regId).AddValue(tresErr[icol * nRow * nFlow + irow * nFlow + iflow]);
		regFlowStats.at(regId * nFlow + iflow).AddValue(tresErr[icol * nRow * nFlow + irow * nFlow + iflow]);
		regFlowMeanStats.at(regId * nFlow + iflow).AddValue(tresErr[icol * nRow * nFlow + irow * nFlow + iflow]);
	      }
	  }
    beadParam.Close();
    
    //RcppMatrix< double > resErrMat(nRow*nCol,nonZeroFlow);    
    vector<int> colOutInt,rowOutInt,flowOutInt;
    colOutInt.resize(colOut.size());
    for(unsigned int i=0;i<colOutInt.size();++i)
      colOutInt[i] = (int) colOut[i];

    rowOutInt.resize(rowOut.size());
    for(unsigned int i=0;i<rowOutInt.size();++i)
      rowOutInt[i] = (int) rowOut[i];

    flowOutInt.resize(flowOut.size());
    for(unsigned int i=0;i<flowOutInt.size();++i)
      flowOutInt[i] = (int) flowOut[i];
    
    vector < float > regMean, regMedian, regSd;
    regMean.reserve(nReg);
    regMedian.reserve(nReg);
    regSd.reserve(nReg);
    
    vector < float > regFlowMean, regFlowMedian, regFlowSd;
    regFlowMean.reserve(nReg * nFlow);
    regFlowMedian.reserve(nReg * nFlow);
    regFlowSd.reserve(nReg * nFlow);
    RcppMatrix<double> mregFlowMean(nReg,nFlow);
    RcppMatrix<double> mregFlowMedian(nReg,nFlow);
    RcppMatrix<double> mregFlowSd(nReg,nFlow);

    for(size_t i=0;i<regStats.size();++i)
      {
	regMean.push_back(regMeanStats.at(i).GetMean());
	regSd.push_back(regMeanStats.at(i).GetSampleVar());
	if(regStats[i].GetNumSeen()>0)
	  regMedian.push_back(regStats[i].GetMedian());
	else
	  regMedian.push_back(0);
      }
    
    for(size_t i=0; i<regFlowStats.size();++i)
      {
	regFlowMean.push_back(regFlowMeanStats.at(i).GetMean());
	regFlowSd.push_back(regFlowMeanStats.at(i).GetSampleVar());
	if(regFlowStats[i].GetNumSeen()>0)
	  regFlowMedian.push_back(regFlowStats[i].GetMedian());
	else
	  regFlowMedian.push_back(0);
      }
    for(int ireg=0;ireg<nReg;ireg++)
      for(unsigned int iflow=0;iflow<nFlow;iflow++)
	{
	  mregFlowMean(ireg,iflow) = (double)regFlowMeanStats.at(nFlow * ireg + iflow).GetMean();
	  mregFlowSd(ireg,iflow) = (double)regFlowMeanStats.at(nFlow * ireg + iflow).GetSampleVar();
	  if(regFlowStats[nFlow * ireg + iflow].GetNumSeen()>0)
	    mregFlowMedian(ireg,iflow) = (double)regFlowStats.at(nFlow * ireg + iflow).GetMedian();
	}
    RcppResultSet rs;
    rs.add("beadParamFile", beadParamFile);
    rs.add("nCol", (int)nCol);
    rs.add("nRow", (int)nRow);
    rs.add("nFlow", (int)nFlow);
    rs.add("nRegCol",nRegCol);
    rs.add("nRegRow",nRegRow);
    rs.add("col", colOutInt);
    rs.add("row",rowOutInt);
    rs.add("flow",flowOutInt);
    rs.add("regRow",regRow);
    rs.add("regCol",regCol);
    if ( returnResErr ) 
      rs.add("res_error",resErr);
    if(returnRegStats)
      {
	vector<double> dregMean(regMean.begin(),regMean.end());
	vector<double> dregMedian(regMedian.begin(),regMedian.end());
	vector<double> dregSd(regSd.begin(),regSd.end());
	rs.add("regMean",dregMean);
	rs.add("regMedian",dregMedian);
	rs.add("regSd",dregSd);
      }
    if(returnRegFlowStats)
      {
	vector<double> dregFlowMean(regFlowMean.begin(),regFlowMean.end());
	vector<double> dregFlowMedian(regFlowMedian.begin(),regFlowMedian.end());
	vector<double> dregFlowSd(regFlowSd.begin(),regFlowSd.end());
	rs.add("regFlowMean",dregFlowMean);
	rs.add("regFlowMedian",dregFlowMedian);
	rs.add("regFlowSd",dregFlowSd);

	rs.add("mregFlowMean",mregFlowMean);
	rs.add("mregFlowMedian",mregFlowMedian);
	rs.add("mregFlowSd",mregFlowSd);
      }
    ret = rs.getReturnList();
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
    
