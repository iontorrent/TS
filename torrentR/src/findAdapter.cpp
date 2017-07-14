/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>
#include <algorithm>
//#include "LinearCalibrationModel.h" // Screw calibration for now
#include "DPTreephaser.h"

RcppExport SEXP findAdapter(SEXP Rsequence, SEXP Rsignal, SEXP Rphasing, SEXP RscaledResidual,
                            SEXP RflowCycle, SEXP Radapter, SEXP RtrimMethod)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {

    // Process input values
    Rcpp::StringVector       sequences(Rsequence);
    Rcpp::NumericMatrix      signal(Rsignal);
    Rcpp::NumericMatrix      phasing(Rphasing);
    Rcpp::NumericMatrix      scaledResidual(RscaledResidual);
    string flowCycle       = Rcpp::as<string>(RflowCycle);
    string adapter         = Rcpp::as<string>(Radapter);
    int trim_method        = Rcpp::as<int>(RtrimMethod);

    unsigned int nFlow = signal.cols();
    unsigned int nRead = signal.rows();
    ion::FlowOrder flow_order(flowCycle, nFlow);

    // Prepare objects for holding and passing back results
    Rcpp::NumericMatrix        raw_metric_out(nRead,nFlow);
    Rcpp::NumericMatrix        final_metric_out(nRead,nFlow);
    Rcpp::NumericMatrix        scaled_residual_out(nRead,nFlow);
    Rcpp::IntegerMatrix        overlap_out(nRead,nFlow);
    Rcpp::IntegerVector        adapter_position(nRead);

    // Objects we need
    BasecallerRead read;
    DPTreephaser   treephaser(flow_order);
    unsigned int   max_length = (2*flow_order.num_flows());
    vector<int>    base_to_flow(2*nFlow);
    vector<float>  sigVec(nFlow), scaled_residual(nFlow);
    bool per_read_phasing = true;
    if (phasing.nrow() == 1) {
      per_read_phasing = false;
      treephaser.SetModelParameters((double)phasing(0,0), (double)phasing(0,1), (double)phasing(0,2));
    }

    bool create_scaled_residual = false;
    if (( scaledResidual.cols() != signal.cols() ) or ( scaledResidual.rows() != signal.rows() )){
      create_scaled_residual = true;
      cout << "Creating scaled residuals." << endl;
    }


    bool do_normalization =  true;
    cout << "Searching for adapter " << adapter << " using method " << trim_method;
    switch (trim_method){
      case 0: cout << ": flow-align simplified metric" << endl; break;
      case 1: cout << ": flow-align standard metric" << endl; break;
      case 2: cout << ": flow-align standard metric QV" << endl; break;
      case 3: cout << ": predicted signal w. normalization" << endl; break;
      case 4: cout << ": predicted signal NO normalization" << endl; do_normalization=false; break;
      default: cout << ": unknown option!" << endl; exit(1);
    }

    // -------------
    // Main loop iterating over reads and evaluating adapter positions
    for(unsigned int iRead=0; iRead < nRead; iRead++) {

      // Set phasing parameters for this read
      if (per_read_phasing) {
        treephaser.SetModelParameters((double)phasing(iRead,0), (double)phasing(iRead,1), (double)phasing(iRead,2));
      }

      for(unsigned int iFlow=0; iFlow < nFlow; iFlow++)
        sigVec.at(iFlow) = (float) signal(iRead,iFlow);
      read.SetData(sigVec, (int)nFlow);

      string mySequence = Rcpp::as<std::string>(sequences(iRead));
      read.sequence.clear();
      read.sequence.reserve(2*flow_order.num_flows());
      for(unsigned int iBase=0; iBase<mySequence.size() and iBase<max_length; ++iBase){
        read.sequence.push_back(mySequence.at(iBase));
      }

      if (trim_method == 2)
    	  treephaser.ComputeQVmetrics(read);
      else
    	  treephaser.Simulate(read, nFlow, true);

      // Generate base_to_flow just like in BaseCaller
      base_to_flow.clear();
      for (int base = 0, flow = 0; base < (int)read.sequence.size(); ++base) {
          while (flow < (int)nFlow and read.sequence.at(base) != flow_order[flow])
              flow++;
          base_to_flow.push_back(flow);
      }

      // Generate scaled residual
      if (create_scaled_residual) {
        for (unsigned int flow = 0; flow < nFlow; ++flow) {
          //scaled_residual.at(flow) = (read.normalized_measurements.at(flow) - read.prediction.at(flow)) / max( (float)0.01, read.state_inphase.at(flow));
    	  scaled_residual.at(flow) = (read.normalized_measurements.at(flow) - read.prediction.at(flow)) / read.state_inphase.at(flow);
    	  scaled_residual_out(iRead,flow) = scaled_residual.at(flow);
        }
      }
      else {
    	for (unsigned int flow = 0; flow < nFlow; ++flow) {
    	  scaled_residual.at(flow) = scaledResidual(iRead, flow);
    	  scaled_residual_out(iRead,flow) = scaled_residual.at(flow);
    	}
      }



      /*/ XXX
      if (iRead==0){
    	  //cout << "Base to flow: ";
    	  //for (unsigned int base=0; base < base_to_flow.size(); ++base)
    	  //  cout << base_to_flow[base] <<",";
    	  //cout << endl;

    	  cout << "scaled residual: ";
    	  for (unsigned int flow=0; flow < scaled_residual.size(); ++flow)
    		  cout << scaled_residual[flow] <<",";
    	  cout << endl;

      } //*/

      //int  best_start_flow = -1;
      int  best_start_base = -1;
      //int  best_adapter_overlap = -1;

      // ------------------------------------------------------
      // Switch between different classification methods

      if (trim_method <3 ){

        float best_metric = -1e10; // The larger the better
        int   sequence_pos = 0;
        char  adapter_start_base = adapter.at(0);

        for (int adapter_start_flow = 0; adapter_start_flow < flow_order.num_flows(); ++adapter_start_flow) {

          // Only consider start flows that agree with adapter start
          if (flow_order[adapter_start_flow] != adapter_start_base)
            continue;

          while (sequence_pos < (int)read.sequence.size() and base_to_flow.at(sequence_pos) < adapter_start_flow)
            sequence_pos++;
          if (sequence_pos >= (int)read.sequence.size())
            break;
          else if (sequence_pos>0 and read.sequence.at(sequence_pos-1)==adapter_start_base)
            continue; // Make sure we don't get impossible configurations

          // Evaluate this starting position
          int adapter_pos = 0;
          float score_match = 0;
          int score_len_flows = 0;
          int local_sequence_pos = sequence_pos;
          int local_start_base = sequence_pos;

          for (int flow = adapter_start_flow; flow < flow_order.num_flows(); ++flow) {

            int base_delta = 0;
            while (adapter_pos < (int)adapter.length() and adapter.at(adapter_pos) == flow_order[flow]) {
              adapter_pos++;
              base_delta--;
            }
            while (local_sequence_pos < (int)read.sequence.size() and base_to_flow.at(local_sequence_pos) == flow) {
              local_sequence_pos++;
              base_delta++;
            }
            if (flow != adapter_start_flow or base_delta < 0) {
              if (trim_method == 0)
                score_match += base_delta*base_delta;
              else
                score_match += base_delta*base_delta + 2*base_delta*scaled_residual.at(flow) + scaled_residual.at(flow)*scaled_residual.at(flow);
            } else
              local_start_base += base_delta;
            score_len_flows++;

            if (adapter_pos == (int)adapter.length() or local_sequence_pos == (int)read.sequence.size())
              break;
          }

          score_match /= score_len_flows;
          float final_metric = adapter_pos / (float)adapter.length() - score_match; // The higher the better

          // Store output for every evaluated position
          raw_metric_out(iRead, adapter_start_flow) = (double)score_match;
          final_metric_out(iRead, adapter_start_flow) = (double)final_metric;
          overlap_out(iRead, adapter_start_flow) = adapter_pos;

          // Does this adapter alignment match our minimum acceptance criteria? If yes, is it better than other matches seen so far?
          if (adapter_pos < 6)  // Match too short
            continue;
          if (score_match * 2 * adapter.length() > 16)  // Match too dissimilar
            continue;

          if (final_metric > best_metric) {
            best_metric = final_metric;
            //best_start_flow = adapter_start_flow;
            best_start_base = local_start_base;
            //best_adapter_overlap = adapter_pos;
            adapter_position(iRead) = best_start_base;
          }
        }

      }
      // -------------------------------------------------------
      else {
        float best_metric = -0.1; // Inverted to negative value: The larger the better.

        DPTreephaser::TreephaserPath& called_path = treephaser.path(0);   //simulates the main sequence
        DPTreephaser::TreephaserPath& adapter_path = treephaser.path(1);  //branches off to simulate adapter
        treephaser.InitializeState(&called_path);

        for (int adapter_start_base = 0; adapter_start_base < (int)read.sequence.size(); ++adapter_start_base) {

          // Step 1. Consider current position as hypothetical adapter start

          adapter_path.prediction = called_path.prediction;
          int window_start = max(0,called_path.window_start - 8);
          treephaser.AdvanceState(&adapter_path,&called_path, adapter.at(0), flow_order.num_flows());
          int inphase_flow = called_path.flow;
          float state_inphase = called_path.state[inphase_flow];

          int adapter_bases = 0;
          for (int adapter_pos = 1; adapter_pos < (int)adapter.length(); ++adapter_pos) {
            treephaser.AdvanceStateInPlace(&adapter_path, adapter.at(adapter_pos), flow_order.num_flows());
            if (adapter_path.flow < flow_order.num_flows())
              adapter_bases++;
          }

          float xy = 1.0, xy2 = 1.0, yy = 0.0;
          if (do_normalization) {
            xy = xy2 = 0.0;
            for (int metric_flow = window_start; metric_flow < adapter_path.flow; ++metric_flow) {
              xy  += adapter_path.prediction[metric_flow] * read.normalized_measurements[metric_flow];
              xy2 += read.prediction[metric_flow] * read.normalized_measurements[metric_flow];
              yy  += read.normalized_measurements[metric_flow] * read.normalized_measurements[metric_flow];
            }
            if (yy > 0) {
              xy  /= yy;
              xy2 /= yy;
            }
          }

          float metric_num = 0;
          float metric_den = 0;
          for (int metric_flow = window_start; metric_flow < adapter_path.flow; ++metric_flow) {
            float delta_adapter  = read.normalized_measurements[metric_flow]*xy - adapter_path.prediction[metric_flow];
            float delta_sequence = read.normalized_measurements[metric_flow]*xy2 - read.prediction[metric_flow];
            metric_num += delta_adapter*delta_adapter - delta_sequence*delta_sequence;
            metric_den += state_inphase;
          }

          // Changing metric sign to negative so that both algorithms maximize the metric
          float adapter_score = -(metric_num/metric_den + 0.2/adapter_bases);

          raw_metric_out(iRead, inphase_flow) = metric_num;
          final_metric_out(iRead, inphase_flow) = adapter_score;
          overlap_out(iRead, inphase_flow) = adapter_bases;

          if (adapter_score > best_metric) {
            best_metric = adapter_score;
            //best_start_flow = inphase_flow;
            best_start_base = adapter_start_base;
            adapter_position(iRead) = best_start_base;
          }

          // Step 2. Continue to next position
          treephaser.AdvanceStateInPlace(&called_path, read.sequence[adapter_start_base], flow_order.num_flows());
        }
      }

      // -------------------------------------------------------
    }

    // Store results & return them
    ret = Rcpp::List::create(Rcpp::Named("rawMetric")       = raw_metric_out,
                             Rcpp::Named("finalMetric")     = final_metric_out,
                             Rcpp::Named("scaledResidual") = scaled_residual_out,
                             Rcpp::Named("adapterOverlap")  = overlap_out,
                             Rcpp::Named("adapterPosition") = adapter_position);

  }
  catch(std::exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }

  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
