/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>
#include "DPTreephaserM.h"

RcppExport SEXP multreePhaser(SEXP Rsignal, SEXP RflowLimits, SEXP RflowOrders, SEXP RnumFlows,
		SEXP RphaseParameters, SEXP RkeyFlowMat, SEXP RnumKeyFlows, SEXP Rbasecaller, SEXP Rverbose)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    Rcpp::NumericMatrix   signal(Rsignal);
    Rcpp::IntegerVector   flowLimits(RflowLimits);
    Rcpp::IntegerVector   numFlows(RnumFlows);
	Rcpp::StringVector    flowOrders(RflowOrders);
	Rcpp::IntegerMatrix   keyFlowMatrix(RkeyFlowMat);
	Rcpp::IntegerVector   numKeyFlows(RnumKeyFlows);
	Rcpp::NumericMatrix   PhaseParams(RphaseParameters);
	string basecaller   = Rcpp::as<string>(Rbasecaller);
	int verbose         = Rcpp::as<int>(Rverbose);
	
	unsigned int maxNumFlows = signal.cols();
	unsigned int nRead = signal.rows();
	unsigned int nJointly = PhaseParams.rows();
	unsigned int nSequences = nRead / nJointly;
	//unsigned int maxKeyFlows = keyFlowMatrix.cols();

	// Prepare objects for holding and passing back results
	//Rcpp::NumericMatrix      predicted_out(nRead,nFlow);
	//Rcpp::NumericMatrix      residual_out(nRead,nFlow);
	//Rcpp::IntegerMatrix      hpFlow_out(nRead,nFlow);
	Rcpp::IntegerVector        num_bases_called(nSequences);
	//Rcpp::NumericMatrix      predicted_out(nRead, maxNumFlows);

	std::vector< std::string > seq_out(nSequences);

	// Set up flow order vector and key flows
	vector<ion::FlowOrder>      flow_orders;
	flow_orders.resize(nJointly);
	std::vector< vector<int> > keyFlows;
	keyFlows.resize(nJointly);

    // Set up model parameters
    std::vector<double> cf(nJointly), ie(nJointly), dr;
    dr.assign(nJointly, 0);

	for (unsigned int i=0; i<nJointly; i++) {
		flow_orders[i].SetFlowOrder(Rcpp::as<std::string>(flowOrders(i)), numFlows(i));
		keyFlows[i].resize(numKeyFlows(i));
		cf[i] = PhaseParams(i, 0);
		ie[i] = PhaseParams(i, 1);
		for (unsigned int j=0; j<keyFlows[i].size(); j++) {
			keyFlows[i][j] = keyFlowMatrix(i, j);
		}
	}

	// Iterate over reads
	std::vector<float> signalVals(maxNumFlows);
	BasecallerMultiRead Mread;
	Mread.bases_called = 0;
	Mread.active_until_flow.resize(nJointly);
	Mread.read_vector.resize(nJointly);
	Mread.solution.resize(2*maxNumFlows);

	DPTreephaserM TreephaserM(flow_orders);
	TreephaserM.verbose = verbose;
	TreephaserM.SetPhasingParameters(cf, ie, dr);

	// ----------- For debugging, comment out actual solving
	// Focus on first read only : iSeq<nSequences -> iSeq<1
	int iRead = 0;
	for (unsigned int iSeq=0; iSeq<nSequences; iSeq++) {

	  // Load nJointly reads data into BasecallerMultiRead structure
      for (unsigned int j=0; j<nJointly; j++) {
    	  Mread.active_until_flow[j] = flowLimits(iRead);
    	  if ( Mread.active_until_flow[j]>0) {
    		// Create signal vector
    		  for (int iFlow = 0; iFlow<Mread.active_until_flow[j]; iFlow++)
    			  signalVals[iFlow] = (float) signal(iRead, iFlow);
    		  // Load signal vector into read structure
    		  Mread.read_vector[j].SetDataAndKeyNormalize( &(signalVals[0]),
    		              Mread.active_until_flow[j], &(keyFlows[j][0]), (keyFlows[j].size()-1) );
    	  }
    	  iRead++;
      }
      if (verbose>0) {
          Rprintf("Loaded %d reads into BasecallerMultiRead. key_normalizers:",nJointly);
          for (unsigned int j=0; j<nJointly; j++)
        	  Rprintf(" %f", Mread.read_vector[j].key_normalizer);
          Rprintf("\n");
      }

      // -------------------------
      if (basecaller == "treephaser-adaptive")
    	  TreephaserM.NormalizeAndSolveAN(Mread, maxNumFlows);
      else
    	  TreephaserM.NormalizeAndSolveSWAN(Mread, maxNumFlows);
      // -------------------------

      // Translate output
      string seq_string(Mread.bases_called, 'x');
      for (unsigned int i=0; i<Mread.bases_called; i++) {
    	  seq_string[i] = flow_orders[0].IntToNuc(Mread.solution[i]);
      }

      // store output
      seq_out[iSeq] = seq_string;
      num_bases_called(iSeq) =  Mread.bases_called;
	}
	// -----------------------------

	ret = Rcpp::List::create(Rcpp::Named("seq")    = seq_out,
                             Rcpp::Named("nBases") = num_bases_called);



  } catch(std::exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
