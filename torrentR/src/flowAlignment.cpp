/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>
#include "FlowAlignment.h"


RcppExport SEXP flowAlignment(SEXP RtSeq, SEXP RqSeq, SEXP RscaledRes, SEXP RflowOrder, SEXP RstartFlow)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {

    Rcpp::StringVector       targetSeqs(RtSeq);
    Rcpp::StringVector       querySeqs (RqSeq);
    //Rcpp::NumericMatrix      scaledResidual(RscaledRes);
    std::string flow_order = Rcpp::as<std::string>(RflowOrder);
    Rcpp::NumericVector      start_flows(RstartFlow);
    std::string              target_bases, query_bases;
    //std::vector<uint16_t>    scaled_residual(flow_order.length(),0);

    // output values of flow aligner
    std::vector<char>        aln_flow_order;
    std::vector<int>         aligned_qHPs;
    std::vector<int>         aligned_tHPs;
    std::vector<int>         al_flowIdx;
    std::vector<char>        pretty_align;

    // Prepare return values
    Rcpp::StringVector       alnFlowOrder(targetSeqs.length());
    Rcpp::StringVector       prettyAlign (targetSeqs.length());
    Rcpp::NumericVector      alignLength (targetSeqs.length());
    Rcpp::NumericMatrix      alnTargetHPs(targetSeqs.length(), 2*flow_order.length());
    Rcpp::NumericMatrix      alnQueryHPs (targetSeqs.length(), 2*flow_order.length());
    //Rcpp::NumericMatrix      alnScaledResidual(targetSeqs.length(), 2*flow_order.length());


    // Do actual work
    for (unsigned int i_read=0; i_read<(unsigned int)targetSeqs.length(); i_read++){

      int start_flow = start_flows(i_read);
      target_bases = targetSeqs(i_read);
      query_bases  = querySeqs (i_read);
      //for (unsigned int i_flow=0; i_flow<flow_order.length(); i_flow++)
      //  scaled_residual.at(i_flow) = scaledResidual(i_read, i_flow);

      // Return output values to virgin state
      aln_flow_order.clear();
      aligned_qHPs.clear();
      aligned_tHPs.clear();
      al_flowIdx.clear();
      pretty_align.clear();

      bool success = PerformFlowAlignment(target_bases, query_bases, flow_order, start_flow,
    		  aln_flow_order, aligned_qHPs, aligned_tHPs, al_flowIdx, pretty_align);

      // Record output
      if (success) {

        std::string temp_prettyAln(pretty_align.begin(), pretty_align.end());
        prettyAlign(i_read) = temp_prettyAln;

        std::string temp_flowOrder(aln_flow_order.begin(), aln_flow_order.end());
        alnFlowOrder(i_read) = temp_flowOrder;

        alignLength(i_read) = aligned_qHPs.size();
        for (unsigned int i_flow=0; i_flow<(2*flow_order.length()); i_flow++) {
          if (i_flow < aligned_qHPs.size()) {
            alnQueryHPs(i_read, i_flow) = aligned_qHPs.at(i_flow);
            alnTargetHPs(i_read, i_flow) = aligned_tHPs.at(i_flow);
            //alnScaledResidual(i_read, i_flow) = al_scaledRes.at(i_flow);
          } else {
            alnQueryHPs(i_read, i_flow) = 0;
            alnTargetHPs(i_read, i_flow) = 0;
            //alnScaledResidual(i_read, i_flow) = 0;
          }

        }
      } else {
    	  alignLength(i_read) = 0;
    	  alnFlowOrder(i_read) = "ERROR";
    	  prettyAlign (i_read) = "ERROR";
    	  for (unsigned int i_flow=0; i_flow<(2*flow_order.length()); i_flow++) {
    	    alnQueryHPs(i_read, i_flow) = 0;
    	    alnTargetHPs(i_read, i_flow) = 0;
            //alnScaledResidual(i_read, i_flow) = 0;
    	  }
      }
    } // Looped over all reads

    // Store results
    ret = Rcpp::List::create(Rcpp::Named("alnQueryHPs")  = alnQueryHPs,
                             Rcpp::Named("alnTargetHPs") = alnTargetHPs,
                             Rcpp::Named("alnFlowOrder") = alnFlowOrder,
                             Rcpp::Named("prettyAlign")  = prettyAlign);
                             //Rcpp::Named("alnScaledRes") = alnScaledResidual);


  } catch(std::exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }

  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);
  return ret;
}
