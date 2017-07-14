/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>
#include "LinearCalibrationModel.h"
#include "DPTreephaser.h"


RcppExport SEXP treePhaser(SEXP Rsignal, SEXP RkeyFlow, SEXP RflowCycle,
                           SEXP Rcf, SEXP Rie, SEXP Rdr, SEXP Rbasecaller, SEXP RdiagonalStates,
                           SEXP RmodelFile, SEXP RmodelThreshold, SEXP Rxval, SEXP Ryval)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    Rcpp::NumericMatrix      signal(Rsignal);
    Rcpp::IntegerVector      keyFlow(RkeyFlow);
    string flowCycle       = Rcpp::as<string>(RflowCycle);
    Rcpp::NumericVector      cf_vec(Rcf);
    Rcpp::NumericVector      ie_vec(Rie);
    Rcpp::NumericVector      dr_vec(Rdr);
    string basecaller      = Rcpp::as<string>(Rbasecaller);
    unsigned int diagonalStates = Rcpp::as<int>(RdiagonalStates);

    // Recalibration Variables
    string model_file      = Rcpp::as<string>(RmodelFile);
    int model_threshold    = Rcpp::as<int>(RmodelThreshold);
    Rcpp::IntegerVector      x_values(Rxval);
    Rcpp::IntegerVector      y_values(Ryval);
    LinearCalibrationModel   calibModel;

    // Variably load a json or legacy text file for calibration
    if (model_file.length() > 0) {

      // See if we can load a json file
      calibModel.SetHPthreshold(model_threshold);
      ifstream calibration_file(model_file.c_str(), ifstream::in);

      if (calibration_file.good()) {
        Json::Value temp_calibraiton_file;
        Json::Reader json_reader;
        bool success = json_reader.parse(calibration_file, temp_calibraiton_file, false);

        if (success and temp_calibraiton_file.isMember("LinearModel")){
          calibModel.InitializeModelFromJson(temp_calibraiton_file["LinearModel"]);
        }
      }
      calibration_file.close();

      // If the loading from json was not successful, we assume we have a legacy text file
      if (not calibModel.is_enabled())
        calibModel.InitializeModelFromTxtFile(model_file, model_threshold);

      // And if that also didn't work we print a warning
      if (not calibModel.is_enabled())
        cout << "ERROR initializing calibration model from file " << model_file << endl;

    }


    ion::FlowOrder flow_order(flowCycle, flowCycle.length());
    unsigned int nFlow = signal.cols();
    unsigned int nRead = signal.rows();

    if(basecaller != "treephaser-swan" && basecaller != "treephaser-solve" && basecaller != "dp-treephaser" && basecaller != "treephaser-adaptive") {
      std::string exception = "base value for basecaller supplied: " + basecaller;
      exceptionMesg = strdup(exception.c_str());
    } else if (flowCycle.length() < nFlow) {
      std::string exception = "Flow cycle is shorter than number of flows to solve";
      exceptionMesg = strdup(exception.c_str());
    } else {

      // Prepare objects for holding and passing back results
      Rcpp::NumericMatrix        predicted_out(nRead,nFlow);
      Rcpp::NumericMatrix        residual_out(nRead,nFlow);
      Rcpp::NumericMatrix        norm_additive_out(nRead,nFlow);
      Rcpp::NumericMatrix        norm_multipl_out(nRead,nFlow);
      std::vector< std::string> seq_out(nRead);

      // Set up key flow vector
      int nKeyFlow = keyFlow.size(); 
      vector <int> keyVec(nKeyFlow);
      for(int iFlow=0; iFlow < nKeyFlow; iFlow++)
        keyVec[iFlow] = keyFlow(iFlow);

      // Iterate over all reads
      vector <float> sigVec(nFlow);
      BasecallerRead read;
      DPTreephaser dpTreephaser(flow_order);
      dpTreephaser.SetStateProgression((diagonalStates>0));

      // In contrast to pipeline, we always use droop here.
      // To have the same behavior of treephaser-swan as in the pipeline, supply dr=0
      bool per_read_phasing = true;
      if (cf_vec.size() == 1) {
        per_read_phasing = false;
        dpTreephaser.SetModelParameters((double)cf_vec(0), (double)ie_vec(0), (double)dr_vec(0));
      }
 
      // Main loop iterating over reads and solving them
      for(unsigned int iRead=0; iRead < nRead; iRead++) {

        // Set phasing parameters for this read
        if (per_read_phasing)
          dpTreephaser.SetModelParameters((double)cf_vec(iRead), (double)ie_vec(iRead), (double)dr_vec(iRead));
        // And load recalibration model
        if (calibModel.is_enabled()) {
          int my_x = (int)x_values(iRead);
          int my_y = (int)y_values(iRead);
          const vector<vector<vector<float> > > * aPtr = 0;
          const vector<vector<vector<float> > > * bPtr = 0;
          aPtr = calibModel.getAs(my_x, my_y);
          bPtr = calibModel.getBs(my_x, my_y);
          if (aPtr == 0 or bPtr == 0) {
            cout << "Error finding a recalibration model for x: " << x_values(iRead) << " y: " << y_values(iRead);
            cout << endl;
          }
          dpTreephaser.SetAsBs(aPtr, bPtr);
        }

        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++)
          sigVec[iFlow] = (float) signal(iRead,iFlow);
        
        // Interface to just solve without any normalization
        if (basecaller == "treephaser-solve") { // Interface to just solve without any normalization
          read.SetData(sigVec, (int)nFlow);
        } 
        else {
          read.SetDataAndKeyNormalize(&(sigVec[0]), (int)nFlow, &(keyVec[0]), nKeyFlow-1);
        }
          
        // Execute the iterative solving-normalization routine
        if (basecaller == "dp-treephaser") {
          dpTreephaser.NormalizeAndSolve_GainNorm(read, nFlow);
        }
        else if (basecaller == "treephaser-solve") {
          dpTreephaser.Solve(read, nFlow);
        }
        else if (basecaller == "treephaser-adaptive") {
          dpTreephaser.NormalizeAndSolve_Adaptive(read, nFlow); // Adaptive normalization
        }
        else {
          dpTreephaser.NormalizeAndSolve_SWnorm(read, nFlow); // sliding window adaptive normalization
        }

        seq_out[iRead].assign(read.sequence.begin(), read.sequence.end());
        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++) {
          predicted_out(iRead,iFlow)     = (double) read.prediction[iFlow];
          residual_out(iRead,iFlow)      = (double) read.normalized_measurements[iFlow] - read.prediction[iFlow];
          norm_multipl_out(iRead,iFlow)  = (double) read.multiplicative_correction.at(iFlow);
          norm_additive_out(iRead,iFlow) = (double) read.additive_correction.at(iFlow);
        }
      }

      // Store results
      ret = Rcpp::List::create(Rcpp::Named("seq")       = seq_out,
                               Rcpp::Named("predicted") = predicted_out,
                               Rcpp::Named("residual")  = residual_out,
                               Rcpp::Named("norm_additive") = norm_additive_out,
                               Rcpp::Named("norm_multipl")  = norm_multipl_out);
    }
  } catch(std::exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


// ======================================================================

RcppExport SEXP treePhaserSim(SEXP Rsequence, SEXP RflowCycle, SEXP Rcf, SEXP Rie, SEXP Rdr,
                              SEXP Rmaxflows, SEXP RgetStates, SEXP RdiagonalStates,
                              SEXP RmodelFile, SEXP RmodelThreshold, SEXP Rxval, SEXP Ryval)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {

    Rcpp::StringVector            sequences(Rsequence);
    string flowCycle            = Rcpp::as<string>(RflowCycle);
    Rcpp::NumericMatrix           cf_vec(Rcf);
    Rcpp::NumericMatrix           ie_vec(Rie);
    Rcpp::NumericMatrix           dr_vec(Rdr);
    unsigned int max_flows      = Rcpp::as<int>(Rmaxflows);
    unsigned int get_states     = Rcpp::as<int>(RgetStates);
    unsigned int diagonalStates = Rcpp::as<int>(RdiagonalStates);

    // -----------  Recalibration Variables
    string model_file      = Rcpp::as<string>(RmodelFile);
    int model_threshold    = Rcpp::as<int>(RmodelThreshold);
    Rcpp::IntegerVector      x_values(Rxval);
    Rcpp::IntegerVector      y_values(Ryval);
    LinearCalibrationModel   calibModel;
    if (model_file.length() > 0) {
      calibModel.InitializeModelFromTxtFile(model_file, model_threshold);
    }

    // Variably load a json or legacy text file for calibration
    if (model_file.length() > 0) {

      // See if we can load a json file
      calibModel.SetHPthreshold(model_threshold);
      ifstream calibration_file(model_file.c_str(), ifstream::in);

      if (calibration_file.good()) {
        Json::Value temp_calibraiton_file;
        Json::Reader json_reader;
        bool success = json_reader.parse(calibration_file, temp_calibraiton_file, false);

        if (success and temp_calibraiton_file.isMember("LinearModel")){
          calibModel.InitializeModelFromJson(temp_calibraiton_file["LinearModel"]);
        }
      }
      calibration_file.close();

      // If the loading from json was not successful, we assume we have a legacy text file
      if (not calibModel.is_enabled())
        calibModel.InitializeModelFromTxtFile(model_file, model_threshold);

      // And if that also didn't work we print a warning
      if (not calibModel.is_enabled())
        cout << "ERROR initializing calibration model from file " << model_file << endl;

    }
    // ------------------------------------

    ion::FlowOrder flow_order(flowCycle, flowCycle.length());
    unsigned int nFlow = flow_order.num_flows();
    unsigned int nRead = sequences.size();
    max_flows = min(max_flows, nFlow);

    // Prepare objects for holding and passing back results
    Rcpp::NumericMatrix       predicted_out(nRead,nFlow);
    Rcpp::NumericMatrix       inphase_out(nRead,nFlow);
    vector<vector<float> >    query_states;
    vector<int>               hp_lengths;

    // Iterate over all sequences
    BasecallerRead read;
    DPTreephaser dpTreephaser(flow_order);
    bool per_read_phasing = true;
    if (cf_vec.ncol() == 1) {
      per_read_phasing = false;
      dpTreephaser.SetModelParameters((double)cf_vec(0,0), (double)ie_vec(0,0), (double)dr_vec(0,0));
    }
    dpTreephaser.SetStateProgression((diagonalStates>0));
    unsigned int max_length = (2*flow_order.num_flows());

    for(unsigned int iRead=0; iRead<nRead; iRead++) {

      string mySequence = Rcpp::as<std::string>(sequences(iRead));
      read.sequence.clear();
      read.sequence.reserve(2*flow_order.num_flows());
      for(unsigned int iBase=0; iBase<mySequence.length() and iBase<max_length; ++iBase){
        read.sequence.push_back(mySequence.at(iBase));
      }
      read.state_inphase.assign(nFlow, 1.0);

      // Set phasing parameters for this read
      if (per_read_phasing)
        dpTreephaser.SetModelParameters((double)cf_vec(0,iRead), (double)ie_vec(0,iRead), (double)dr_vec(0,iRead));

      // If you bothered specifying a recalibration model you probably want its effect on the predictions...
      if (calibModel.is_enabled()) {
        int my_x = (int)x_values(iRead);
        int my_y = (int)y_values(iRead);
        const vector<vector<vector<float> > > * aPtr = 0;
        const vector<vector<vector<float> > > * bPtr = 0;
        aPtr = calibModel.getAs(my_x, my_y);
        bPtr = calibModel.getBs(my_x, my_y);
        if (aPtr == 0 or bPtr == 0) {
          cout << "Error finding a recalibration model for x: " << x_values(iRead) << " y: " << y_values(iRead);
          cout << endl;
        }
        dpTreephaser.SetAsBs(aPtr, bPtr);
      }

      if (nRead == 1 and get_states > 0)
        dpTreephaser.QueryAllStates(read, query_states, hp_lengths, max_flows);
      else
        dpTreephaser.Simulate(read, max_flows, true);

      for(unsigned int iFlow=0; iFlow<nFlow and iFlow<max_flows; ++iFlow){
		predicted_out(iRead,iFlow) = (double) read.prediction.at(iFlow);
      }
      if (nRead > 1 and get_states > 0){
        for(unsigned int iFlow=0; iFlow<nFlow and iFlow<max_flows; ++iFlow){
          inphase_out(iRead,iFlow) = (double) read.state_inphase.at(iFlow);
    	}
      }

    }

    // Store results
    if (nRead == 1 and get_states > 0) {
      Rcpp::NumericMatrix        states(hp_lengths.size(), nFlow);
      Rcpp::NumericVector        HPlengths(hp_lengths.size());
      for (unsigned int iHP=0; iHP<hp_lengths.size(); iHP++){
        HPlengths(iHP) = (double)hp_lengths[iHP];
        for (unsigned int iFlow=0; iFlow<nFlow; iFlow++)
          states(iHP, iFlow) = (double)query_states.at(iHP).at(iFlow);
      }
      ret = Rcpp::List::create(Rcpp::Named("sig")       = predicted_out,
                               Rcpp::Named("states")    = states,
                               Rcpp::Named("HPlengths") = HPlengths);
    } else if (get_states > 0){
      ret = Rcpp::List::create(Rcpp::Named("inphase") = inphase_out,
                               Rcpp::Named("sig")     = predicted_out);

    }
    else {
      ret = Rcpp::List::create(Rcpp::Named("sig")  = predicted_out);
    }

  } catch(std::exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }

  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}

// =======================================================================================

RcppExport SEXP DPPhaseSim(SEXP Rsequence, SEXP RflowCycle, SEXP Rcf, SEXP Rie, SEXP Rdr,
                           SEXP Rmaxflows, SEXP RgetStates, SEXP RnucContamination)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {

    Rcpp::StringVector        sequences(Rsequence);
    string flowCycle        = Rcpp::as<string>(RflowCycle);
    Rcpp::NumericMatrix       cf_mat(Rcf);
    Rcpp::NumericMatrix       ie_mat(Rie);
    Rcpp::NumericMatrix       dr_mat(Rdr);
    Rcpp::NumericMatrix       nuc_contamination(RnucContamination);
    unsigned int max_flows  = Rcpp::as<int>(Rmaxflows);
    unsigned int get_states = Rcpp::as<int>(RgetStates);

    ion::FlowOrder flow_order(flowCycle, flowCycle.length());
    unsigned int nFlow = flow_order.num_flows();
    unsigned int nRead = sequences.size();
    max_flows = min(max_flows, nFlow);

    vector<vector<double> >    nuc_availbality;
    nuc_availbality.resize(nuc_contamination.nrow());
    for (unsigned int iFlowNuc=0; iFlowNuc < nuc_availbality.size(); iFlowNuc++){
      nuc_availbality.at(iFlowNuc).resize(nuc_contamination.ncol());
      for (unsigned int iNuc=0; iNuc < nuc_availbality.at(iFlowNuc).size(); iNuc++){
        nuc_availbality.at(iFlowNuc).at(iNuc) = nuc_contamination(iFlowNuc, iNuc);
      }
    }

    // Prepare objects for holding and passing back results
    Rcpp::NumericMatrix       predicted_out(nRead,nFlow);
    Rcpp::StringVector        seq_out(nRead);


    // Set Phasing Model
    DPPhaseSimulator PhaseSimulator(flow_order);
    PhaseSimulator.UpdateNucAvailability(nuc_availbality);

    bool per_read_phasing = true;
    if (cf_mat.nrow() == 1 and cf_mat.ncol() == 1) {
      per_read_phasing = false;
      cout << "DPPhaseSim: Single Phase Parameter set detected." << endl; // XXX
      PhaseSimulator.SetPhasingParameters_Basic((double)cf_mat(0,0), (double)ie_mat(0,0), (double)dr_mat(0,0));
    } else if (cf_mat.nrow() == (int)nFlow and cf_mat.ncol() == (int)nFlow) {
    	cout << "DPPhaseSim: Full Phase Parameter set detected." << endl; // XXX
      per_read_phasing = false;
      vector<vector<double> > cf(nFlow);
      vector<vector<double> > ie(nFlow);
      vector<vector<double> > dr(nFlow);
      for (unsigned int iFlowNuc=0; iFlowNuc < nFlow; iFlowNuc++){
        cf.at(iFlowNuc).resize(nFlow);
        ie.at(iFlowNuc).resize(nFlow);
        dr.at(iFlowNuc).resize(nFlow);
        for (unsigned int iFlow=0; iFlow < nFlow; iFlow++){
          cf.at(iFlowNuc).at(iFlow) = cf_mat(iFlowNuc, iFlow);
          ie.at(iFlowNuc).at(iFlow) = ie_mat(iFlowNuc, iFlow);
          dr.at(iFlowNuc).at(iFlow) = dr_mat(iFlowNuc, iFlow);
        }
      }
      PhaseSimulator.SetPhasingParameters_Full(cf, ie, dr);
    }
    else
      cout << "DPPhaseSim: Per Read Phase Parameter set detected." << endl; //XXX


    // --- Iterate over all sequences

    string         my_sequence, sim_sequence;
    vector<float>  my_prediction;

    for(unsigned int iRead=0; iRead<nRead; iRead++) {

      if (per_read_phasing)
        PhaseSimulator.SetPhasingParameters_Basic((double)cf_mat(0,iRead), (double)ie_mat(0,iRead), (double)dr_mat(0,iRead));

      my_sequence = Rcpp::as<std::string>(sequences(iRead));
      PhaseSimulator.Simulate(my_sequence, my_prediction, max_flows);

      PhaseSimulator.GetSimSequence(sim_sequence); // Simulated sequence might be shorter than input sequence.
      seq_out(iRead) = sim_sequence;
      for(unsigned int iFlow=0; iFlow<nFlow and iFlow<max_flows; ++iFlow) {
		predicted_out(iRead,iFlow) = (double) my_prediction.at(iFlow);
      }
      //cout << "--- DPPhaseSim: Done simulating read "<< iRead << " of " << nRead << endl; // XXX
    }

    // --- Store results
    if (nRead == 1 and get_states > 0) {

      vector<vector<float> >    query_states;
      vector<int>               hp_lengths;
      PhaseSimulator.GetStates(query_states, hp_lengths);

      Rcpp::NumericMatrix       states(hp_lengths.size(), nFlow);
      Rcpp::NumericVector       HPlengths(hp_lengths.size());

      for (unsigned int iHP=0; iHP<hp_lengths.size(); iHP++){
        HPlengths(iHP) = (double)hp_lengths[iHP];
        for (unsigned int iFlow=0; iFlow<nFlow; iFlow++)
          states(iHP, iFlow) = (double)query_states.at(iHP).at(iFlow);
      }

      ret = Rcpp::List::create(Rcpp::Named("sig")       = predicted_out,
                               Rcpp::Named("seq")       = seq_out,
                               Rcpp::Named("states")    = states,
                               Rcpp::Named("HPlengths") = HPlengths);
    } else {
      ret = Rcpp::List::create(Rcpp::Named("sig")  = predicted_out,
                               Rcpp::Named("seq")  = seq_out);
    }

  } catch(std::exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }

  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


// ======================================================================


RcppExport SEXP FitPhasingBurst(SEXP R_signal, SEXP R_flowCycle, SEXP R_read_sequence,
                SEXP R_phasing, SEXP R_burstFlows, SEXP R_maxEvalFlow, SEXP R_maxSimFlow) {

 SEXP ret = R_NilValue;
 char *exceptionMesg = NULL;

 try {

     Rcpp::NumericMatrix  signal(R_signal);
     Rcpp::NumericMatrix  phasing(R_phasing);     // Standard phasing parameters
     string flowCycle   = Rcpp::as<string>(R_flowCycle);
     Rcpp::StringVector   read_sequences(R_read_sequence);
     Rcpp::NumericVector  phasing_burst(R_burstFlows);
     Rcpp::NumericVector  max_eval_flow(R_maxEvalFlow);
     Rcpp::NumericVector  max_sim_flow(R_maxSimFlow);
     int window_size    = 38; // For normalization


     ion::FlowOrder flow_order(flowCycle, flowCycle.length());
     unsigned int num_flows = flow_order.num_flows();
     unsigned int num_reads = read_sequences.size();


     // Containers to store results
     Rcpp::NumericVector null_fit(num_reads);
     Rcpp::NumericMatrix null_prediction(num_reads, num_flows);
     Rcpp::NumericVector best_fit(num_reads);
     Rcpp::NumericVector best_ie_value(num_reads);
     Rcpp::NumericMatrix best_prediction(num_reads, num_flows);


     BasecallerRead bc_read;
     DPTreephaser dpTreephaser(flow_order);
     DPPhaseSimulator PhaseSimulator(flow_order);
     vector<double> cf_vec(num_flows, 0.0);
     vector<double> ie_vec(num_flows, 0.0);
     vector<double> dr_vec(num_flows, 0.0);


     // IE Burst Estimation Loop
     for (unsigned int iRead=0; iRead<num_reads; iRead++) {

       // Set read object
       vector<float> my_signal(num_flows);
       for (unsigned int iFlow=0; iFlow<num_flows; iFlow++)
         my_signal.at(iFlow) = signal(iRead, iFlow);
       bc_read.SetData(my_signal, num_flows);
       string my_sequence = Rcpp::as<std::string>(read_sequences(iRead));

       // Default phasing as baseline
       double my_best_fit, my_best_ie;
       double base_cf  = (double)phasing(iRead, 0);
       double base_ie  = (double)phasing(iRead, 1);
       double base_dr  = (double)phasing(iRead, 2);
       int burst_flow = (int)phasing_burst(iRead);
       vector<float> my_best_prediction;

       cf_vec.assign(num_flows, base_cf);
       dr_vec.assign(num_flows, base_dr);
       int my_max_flow  = min((int)num_flows, (int)max_sim_flow(iRead));
       int my_eval_flow = min(my_max_flow, (int)max_eval_flow(iRead));

       PhaseSimulator.SetBaseSequence(my_sequence);
       PhaseSimulator.SetMaxFlows(my_max_flow);
       PhaseSimulator.SetPhasingParameters_Basic(base_cf, base_ie, base_dr);
       PhaseSimulator.UpdateStates(my_max_flow);
       PhaseSimulator.GetPredictions(bc_read.prediction);
       dpTreephaser.WindowedNormalize(bc_read, (my_eval_flow/window_size), window_size, true);


       my_best_ie = base_ie;
       my_best_prediction = bc_read.prediction;
       my_best_fit = 0;
       for (int iFlow=0; iFlow<my_eval_flow; iFlow++) {
         double residual = bc_read.raw_measurements.at(iFlow) - bc_read.prediction.at(iFlow);
         my_best_fit += residual*residual;
       }
       for (unsigned int iFlow=0; iFlow<num_flows; iFlow++)
         null_prediction(iRead, iFlow) = bc_read.prediction.at(iFlow);
       null_fit(iRead) = my_best_fit;

       // Make sure that there are enough flows to fit a burst
       if (burst_flow < my_eval_flow-10) {
    	 int    num_steps  = 0;
    	 double step_size  = 0.0;
    	 double step_start = 0.0;
    	 double step_end   = 0.0;

         // Brute force phasing burst value estimation using grid search, crude first, then refine
         for (unsigned int iIteration = 0; iIteration<3; iIteration++) {

           switch(iIteration) {
             case 0:
               step_size = 0.05;
               step_end = 0.8;
               break;
             case 1:
               step_end   = (floor(my_best_ie / step_size)*step_size) + step_size;
               step_start = max(0.0, (step_end - 2.0*step_size));
               step_size  = 0.01;
               break;
             default:
               step_end   = (floor(my_best_ie / step_size)*step_size) + step_size;
               step_start = max(0.0, step_end - 2*step_size);
               step_size = step_size / 10;
           }
           num_steps  = 1+ ((step_end - step_start) / step_size);

           for (int iPhase=0; iPhase <= num_steps; iPhase++) {

        	 double try_ie = step_start+(iPhase*step_size);
             ie_vec.assign(num_flows, try_ie);

             PhaseSimulator.SetBasePhasingParameters(burst_flow, cf_vec, ie_vec, dr_vec);
             PhaseSimulator.UpdateStates(my_max_flow);
             PhaseSimulator.GetPredictions(bc_read.prediction);
             dpTreephaser.WindowedNormalize(bc_read, (my_eval_flow/window_size), window_size, true);

             double my_fit = 0.0;
             for (int iFlow=burst_flow+1; iFlow<my_eval_flow; iFlow++) {
               double residual = bc_read.raw_measurements.at(iFlow) - bc_read.prediction.at(iFlow);
               my_fit += residual*residual;
             }
             if (my_fit < my_best_fit) {
               my_best_fit = my_fit;
               my_best_ie  = try_ie;
               my_best_prediction = bc_read.prediction;
             }
           }
         }
       }

       // Set output information for this read
       best_fit(iRead) = my_best_fit;
       best_ie_value(iRead)   = my_best_ie;
       for (unsigned int iFlow=0; iFlow<num_flows; iFlow++)
         best_prediction(iRead, iFlow) = my_best_prediction.at(iFlow);
     }

     ret = Rcpp::List::create(Rcpp::Named("null_fit")        = null_fit,
                              Rcpp::Named("null_prediction") = null_prediction,
                              Rcpp::Named("burst_flow")      = phasing_burst,
                              Rcpp::Named("best_fit")        = best_fit,
                              Rcpp::Named("best_ie_value")   = best_ie_value,
                              Rcpp::Named("best_prediction") = best_prediction);


 } catch(std::exception& ex) {
   forward_exception_to_r(ex);
 } catch(...) {
   ::Rf_error("c++ exception (unknown reason)");
 }

 if(exceptionMesg != NULL)
   Rf_error(exceptionMesg);
 return ret;

}
