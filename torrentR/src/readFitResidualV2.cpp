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
RcppExport SEXP readFitResidualRV2(SEXP RbeadParamFile,
				   SEXP Rcol, SEXP Rrow, SEXP Rflow,
				   SEXP RminCol, SEXP RmaxCol,
				   SEXP RminRow, SEXP RmaxRow,
				   SEXP RminFlow, SEXP RmaxFlow, 
				   SEXP RreturnResErr,
				   SEXP RregXSize, SEXP RregYSize,
				   SEXP RreturnRegStats,
				   SEXP RreturnRegFlowStats) {
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    //vector<string> beadParamFile = Rcpp::as< vector<string> > (RbeadParamFile);
    //Rcpp::StringVector tBeadParamFile(RbeadParamFile);
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

    unsigned int nCol = maxCol - minCol + 1;
    unsigned int nRow = maxRow - minRow + 1 ;
    unsigned int nFlow = maxFlow - minFlow + 1;
    vector<unsigned int> colOut,rowOut,flowOut;
    vector< double > resErr;
    
    resErr.reserve(nCol * nRow * nFlow);
    colOut.reserve(nCol * nRow );
    rowOut.reserve(nCol * nRow );
    flowOut.reserve(nFlow);
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
    //beadParam.Open();
    beadParam.OpenForReading();
    H5DataSet *resErrDS = beadParam.OpenDataSet("/bead/residual_error");
    size_t starts[3];
    size_t ends[3];
   
    //float  tresErr[nCol * nRow * nFlow];
    float *tresErr;
    tresErr = (float *) malloc( sizeof(float) * nCol *nRow * nFlow);

    starts[0] = minCol;
    starts[1] = minRow;
    starts[2] = minFlow;
    ends[0] = maxCol+1;
    ends[1] = maxRow+1;
    ends[2] = maxFlow+1;
    // Check if the starts and ends are out of range.
    try {
      resErrDS->ReadRangeData(starts, ends, sizeof(tresErr),tresErr);
    }
    catch(exception& ex) {
      forward_exception_to_r(ex);
    }    
    // Compute number of regions
    int nRegRow = nRow/regXSize + 1 ;
    int nRegCol =  nCol/regYSize + 1;
    int nReg = nRegRow * nRegCol;
    // Compute stats
    vector < int > regRow(nReg,0),regCol(nReg,0);
    //regRow.reserve(nReg);
    //regCol.reserve(nReg);

    Rcpp::NumericMatrix resErrMat(nRow*nCol,nFlow);        
    vector< SampleQuantiles<float> > regStats (nReg, SampleQuantiles<float>(nFlow * nRow * nCol)) ; 
    vector< SampleStats<float>  > regMeanStats (nReg,SampleStats<float>()) ; 
    vector< SampleQuantiles<float>  > regFlowStats (nReg * nFlow,SampleQuantiles<float>(nRow * nCol)) ; 
    vector< SampleStats<float>  > regFlowMeanStats (nReg * nFlow, SampleStats<float>()) ; 
    int count = 0;

    for(size_t iflow=0;iflow<nFlow;++iflow)
      flowOut.push_back(iflow + minFlow);

    for(size_t icol=0;icol<nCol;++icol)
      for(size_t irow=0;irow<nRow;++irow)
	{
	  colOut.push_back(icol + minCol);
	  rowOut.push_back(irow + minRow);
	  int regId = (nRegCol)*(int)(irow/regYSize) + (int) icol/regXSize;
	  regCol[regId] = regXSize * ((int)icol/regXSize);
	  regRow[regId] = regYSize * ((int)irow/regYSize);
	  for(size_t iflow=0;iflow<nFlow;++iflow)
	    {
	      resErrMat(count,iflow) = (double) tresErr[icol*nRow*nFlow + irow*nFlow+iflow];
	      if(tresErr[icol * nRow * nFlow + irow * nFlow +iflow] > 0)
	        {
		  //resErr.push_back(tresErr[icol * nRow * nFlow + irow * nFlow + iflow]);
		  regStats.at(regId).AddValue(tresErr[icol * nRow * nFlow + irow * nFlow + iflow]);
		  regMeanStats.at(regId).AddValue(tresErr[icol * nRow * nFlow + irow * nFlow + iflow]);
		  regFlowStats.at(regId * nFlow + iflow).AddValue(tresErr[icol * nRow * nFlow + irow * nFlow + iflow]);
		  regFlowMeanStats.at(regId * nFlow + iflow).AddValue(tresErr[icol * nRow * nFlow + irow * nFlow + iflow]);
		}
	    }
	  count++;
	}
    beadParam.Close();
    

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
    Rcpp::NumericMatrix mregFlowMean(nReg,nFlow);
    Rcpp::NumericMatrix mregFlowMedian(nReg,nFlow);
    Rcpp::NumericMatrix mregFlowSd(nReg,nFlow);

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
    std::map<std::string,SEXP> map;
    map["beadParamFile"] = Rcpp::wrap( beadParamFile );
    map["nCol"]          = Rcpp::wrap( (int)nCol );
    map["nRow"]          = Rcpp::wrap( (int)nRow );
    map["nFlow"]         = Rcpp::wrap( (int)nFlow );
    map["nRegCol"]       = Rcpp::wrap( nRegCol );
    map["nRegRow"]       = Rcpp::wrap( nRegRow );
    map["col"]           = Rcpp::wrap( colOutInt );
    map["row"]           = Rcpp::wrap( rowOutInt );
    map["flow"]          = Rcpp::wrap( flowOutInt );
    map["regRow"]        = Rcpp::wrap( regRow );
    map["regCol"]        = Rcpp::wrap( regCol );
    if ( returnResErr ) 
      map["res_error"]   = Rcpp::wrap( resErrMat );
    if(returnRegStats)
      {
	vector<double> dregMean(regMean.begin(),regMean.end());
	vector<double> dregMedian(regMedian.begin(),regMedian.end());
	vector<double> dregSd(regSd.begin(),regSd.end());
    map["regMean"]           = Rcpp::wrap( dregMean );
    map["regMedian"]         = Rcpp::wrap( dregMedian );
    map["regSd"]             = Rcpp::wrap( dregSd );
      }
    if(returnRegFlowStats)
      {
	vector<double> dregFlowMean(regFlowMean.begin(),regFlowMean.end());
	vector<double> dregFlowMedian(regFlowMedian.begin(),regFlowMedian.end());
	vector<double> dregFlowSd(regFlowSd.begin(),regFlowSd.end());
    map["regFlowMean"]       = Rcpp::wrap( dregFlowMean );
    map["regFlowMedian"]     = Rcpp::wrap( dregFlowMedian );
    map["regFlowSd"]         = Rcpp::wrap( dregFlowSd );

    map["mregFlowMean"]      = Rcpp::wrap( mregFlowMean );
    map["mregFlowMedian"]    = Rcpp::wrap( mregFlowMedian );
    map["mregFlowSd"]        = Rcpp::wrap( mregFlowSd );
      }
    ret = Rcpp::wrap( map );
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
    
