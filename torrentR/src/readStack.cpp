/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include "api/BamReader.h"

#include "../Analysis/file-io/ion_util.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include "BamHelper.h"
#include "ExtendedReadData.h"
#include "StackEngine.h"
#include "calcHypothesesDistancesEngine.h"


using namespace std;

// Rcpp interface code here to isolate complexity


RcppExport SEXP readBamStackAtPosition(SEXP RbamFile, SEXP R_variant_contig, SEXP R_variant_position) {
	SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
	char *exceptionMesg = NULL;

	try {
	char* bamFile         = (char *)Rcpp::as<const char*>(RbamFile);
	string variant_contig((char *)Rcpp::as<const char*>(R_variant_contig));
  unsigned int variant_position = Rcpp::as<int>(R_variant_position);

  StackPlus my_data;
  unsigned int nFlowZM = my_data.GrabStack(bamFile, variant_contig, variant_position);
  cout<< "Grabbed stack " << nFlowZM << "\t" << my_data.read_stack.size() << endl;

// handle all the overhead to actually >return< values that Rcpp requires
    unsigned int n_reads_to_export = my_data.read_stack.size();
  	std::vector< std::string > out_qDNA(n_reads_to_export);
		std::vector< std::string > out_tDNA(n_reads_to_export);
		Rcpp::IntegerVector    out_flowClipLeft(n_reads_to_export);
		Rcpp::IntegerVector    out_col(n_reads_to_export);
		Rcpp::IntegerVector    out_row(n_reads_to_export);

    std::vector< std::string > out_flowOrder(1);

			Rcpp::IntegerVector    out_aligned_flag(n_reads_to_export);
			Rcpp::IntegerVector    out_aligned_pos(n_reads_to_export);

		Rcpp::NumericMatrix out_meas(n_reads_to_export,nFlowZM);
    unsigned int n_phase = 3;
		Rcpp::NumericMatrix out_phase(n_reads_to_export, n_phase);
    for (unsigned int i_read=0; i_read<n_reads_to_export; i_read++)
    {
				out_qDNA[i_read]  = my_data.read_stack[i_read].qDNA;
				out_tDNA[i_read]  = my_data.read_stack[i_read].tDNA;
				out_aligned_flag(i_read)  = my_data.read_stack[i_read].alignment.AlignmentFlag;
				out_aligned_pos(i_read)   = my_data.read_stack[i_read].alignment.Position;
        out_flowClipLeft(i_read)  = my_data.read_stack[i_read].start_flow;
        out_col(i_read) = my_data.read_stack[i_read].col;
        out_row(i_read) = my_data.read_stack[i_read].row;
        // ZM tag
        if (true) {
        unsigned int i_flow=0;
				for(; i_flow<std::min(nFlowZM,(unsigned int)my_data.read_stack[i_read].measurementValue.size()); i_flow++)
					out_meas(i_read,i_flow) = my_data.read_stack[i_read].measurementValue[i_flow];
				while(i_flow<nFlowZM)
					out_meas(i_read,i_flow++) = 0; // which is bad because will lead to biases in extrapolation

        unsigned int i_param=0;
				for(; i_param<std::min(n_phase,(unsigned int)my_data.read_stack[i_read].phase_params.size()); i_param++)
					out_phase(i_read,i_param) = my_data.read_stack[i_read].phase_params[i_param];
				while(i_param<n_phase)
					out_phase(i_read,i_param++) = 0; 
        }
       
    }
    out_flowOrder[0] = my_data.flow_order;
  // set good return
// what do I need for variant calling
  ret = Rcpp::List::create(Rcpp::Named("flowOrder")    = out_flowOrder,
                           Rcpp::Named("qDNA")         = out_qDNA,
                           Rcpp::Named("tDNA")         = out_tDNA,
                           Rcpp::Named("flowClipLeft") = out_flowClipLeft,
                           Rcpp::Named("col")          = out_col,
                           Rcpp::Named("row")          = out_row,
                           Rcpp::Named("measured")     = out_meas,
                           Rcpp::Named("phase")        = out_phase,
                           Rcpp::Named("alignFlag")    = out_aligned_flag,
                           Rcpp::Named("alignPos")     = out_aligned_pos,
                           Rcpp::Named("CoverDepth")   = (int) my_data.read_stack.size(),
                           Rcpp::Named("MagicNumber")  = 11);

} catch(std::exception& ex) {
// safely deal with other cases
		forward_exception_to_r(ex);
	} catch(...) {
		::Rf_error("c++ exception (unknown reason)");
	}

	if(exceptionMesg != NULL)
		Rf_error(exceptionMesg);

	return ret;
}

