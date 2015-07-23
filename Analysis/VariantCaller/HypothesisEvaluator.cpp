/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     HypothesisEvaluator.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "HypothesisEvaluator.h"

// Function to fill in prediceted signal values
int CalculateHypPredictions(
    PersistingThreadObjects  &thread_objects,
    const Alignment          &my_read,
    const InputStructures    &global_context,
    const vector<string>     &Hypotheses,
    const vector<bool>       &same_as_null_hypothesis,
    vector<vector<float> >   &predictions,
    vector<vector<float> >   &normalizedMeasurements,
    int flow_upper_bound) {

    // --- Step 1: Initialize Objects

	if (global_context.DEBUG > 2)
	  cout << "Prediction Generation for read " << my_read.alignment.Name << endl;

    predictions.resize(Hypotheses.size());
    normalizedMeasurements.resize(Hypotheses.size());
    // Careful: num_flows may be smaller than flow_order.num_flows()
    const ion::FlowOrder & flow_order = global_context.flow_order_vector.at(my_read.flow_order_index);
    const int & num_flows = global_context.num_flows_by_run_id.at(my_read.runid);
    int prefix_flow = 0;

    BasecallerRead master_read;
    master_read.SetData(my_read.measurements, flow_order.num_flows());
    InitializeBasecallers(thread_objects, my_read, global_context);

    // --- Step 2: Processing read prefix or solve beginning of the read if desired
    unsigned int prefix_size = 0;
    if (global_context.resolve_clipped_bases or my_read.prefix_flow < 0) {
      prefix_flow = GetStartOfMasterRead(thread_objects, my_read, global_context, Hypotheses, num_flows, master_read);
      prefix_size = master_read.sequence.size();
    }
    else {
      const string & read_prefix = global_context.key_by_read_group.at(my_read.read_group);
      prefix_size = read_prefix.length();
      for (unsigned int i_base=0; i_base < read_prefix.length(); i_base++)
        master_read.sequence.push_back(read_prefix.at(i_base));
      prefix_flow = my_read.prefix_flow;
    }

    // --- Step 3: creating predictions for the individual hypotheses

    // Compute an upper limit of flows to be simulated or solved
    if (global_context.DEBUG > 2)
      cout << "Prediction Generation: determining flow upper bound (flow_order.num_flows()=" << flow_order.num_flows() << ") as the minimum of:"
           << " flow_upper_bound " << flow_upper_bound
           << " measurement_length " << my_read.measurements_length
           << " num_flows " << num_flows << endl;
    flow_upper_bound = min(flow_upper_bound, min(my_read.measurements_length, num_flows));

    vector<BasecallerRead> hypothesesReads(Hypotheses.size());
    int max_last_flow  = 0;

    for (unsigned int i_hyp=0; i_hyp<hypothesesReads.size(); ++i_hyp) {

    	// No need to simulate if a hypothesis is equal to the read as called
    	// We get that info from the splicing module
    	if (same_as_null_hypothesis.at(i_hyp)) {
            predictions[i_hyp] = predictions[0];
            predictions[i_hyp].resize(flow_order.num_flows());
            normalizedMeasurements[i_hyp] = normalizedMeasurements[0];
            normalizedMeasurements[i_hyp].resize(flow_order.num_flows());
        } else {

            hypothesesReads[i_hyp] = master_read;

            // --- add hypothesis sequence to clipped prefix
            unsigned int i_base = 0;
            unsigned int max_bases = 2*(unsigned int)flow_order.num_flows()-prefix_size; // Our maximum allocated memory for the sequence vector
            int i_flow = prefix_flow;

            // Add bases to read object sequence
            // We add one more base beyond 'flow_upper_bound' (if available) to signal Treephaser to not even start the solver
            while (i_base<Hypotheses[i_hyp].length() and i_base<max_bases) {
              IncrementFlow(flow_order, Hypotheses[i_hyp][i_base], i_flow);
              hypothesesReads[i_hyp].sequence.push_back(Hypotheses[i_hyp][i_base]);
              if (i_flow >= flow_upper_bound) {
            	i_flow = flow_upper_bound;
                break;
              }
              i_base++;
            }

            // Find last main incorporating flow of all hypotheses
            max_last_flow = max(max_last_flow, i_flow);

            // Solver simulates beginning of the read and then fills in the remaining clipped bases
            // Above checks on flow_upper_bound and i_flow guarantee that i_flow <= flow_upper_bound <= num_flows
            thread_objects.SolveRead(my_read.flow_order_index, hypothesesReads[i_hyp], min(i_flow,flow_upper_bound), flow_upper_bound);

            // Store predictions and adaptively normalized measurements
            predictions[i_hyp].swap(hypothesesReads[i_hyp].prediction);
            predictions[i_hyp].resize(flow_order.num_flows(), 0);
            normalizedMeasurements[i_hyp].swap(hypothesesReads[i_hyp].normalized_measurements);
            normalizedMeasurements[i_hyp].resize(flow_order.num_flows(), 0);
        }
    }

    // --- verbose ---
    if (global_context.DEBUG>2)
      PredictionGenerationVerbose(Hypotheses, hypothesesReads, my_read, predictions, prefix_size, global_context);

    //return max_last_flow;
    return (max_last_flow);
}

// ----------------------------------------------------------------------

void InitializeBasecallers(PersistingThreadObjects &thread_objects,
                           const Alignment         &my_read,
                           const InputStructures   &global_context) {

  // Set phasing parameters
  thread_objects.SetModelParameters(my_read.flow_order_index, my_read.phase_params);

  // Set up HP recalibration model: hide the recal object behind a mask so we can use the map to select
  thread_objects.DisableRecalibration(my_read.flow_order_index); // Disable use of a previously loaded recalibration model

  if (global_context.do_recal.recal_is_live()) {
    // query recalibration structure using row, column, entity
    // look up entity here: using row, col, runid
    // note: perhaps do this when we first get the read, exploit here
    string found_key = global_context.do_recal.FindKey(my_read.runid, my_read.well_rowcol[1], my_read.well_rowcol[0]);
    MultiAB multi_ab;
    global_context.do_recal.getAB(multi_ab, found_key, my_read.well_rowcol[1], my_read.well_rowcol[0]);
    if (multi_ab.Valid())
      thread_objects.SetAsBs(my_read.flow_order_index, multi_ab.aPtr, multi_ab.bPtr);
  }
}

// ----------------------------------------------------------------------

int GetStartOfMasterRead(PersistingThreadObjects   &thread_objects,
                          const Alignment          &my_read,
                          const InputStructures    &global_context,
                          const vector<string>     &Hypotheses,
                          const int                &nFlows,
                          BasecallerRead           &master_read) {

  // Solve beginning of maybe clipped read
  const ion::FlowOrder & flow_order = global_context.flow_order_vector.at(my_read.flow_order_index);

  int until_flow = min((my_read.start_flow+20), nFlows);
  if (my_read.start_flow > 0)
    thread_objects.SolveRead(my_read.flow_order_index, master_read, 0, until_flow);

  // StartFlow clipped? Get solved HP length at startFlow.
  unsigned int base = 0;
  int flow = 0;
  unsigned int HPlength = 0;
  while (base < master_read.sequence.size()) {
    while (flow < flow_order.num_flows() and flow_order.nuc_at(flow) != master_read.sequence[base]) {
      flow++;
    }
    if (flow > my_read.start_flow or flow == flow_order.num_flows())
      break;
    if (flow == my_read.start_flow)
      HPlength++;
    base++;
  }
  if (global_context.DEBUG>2)
    printf("Solved %d bases until (not incl.) flow %d. HP of height %d at flow %d.\n", base, flow, HPlength, my_read.start_flow);

  // Get HP size at the start of the read as called in Hypotheses[0]
  unsigned int count = 1;
  while (count < Hypotheses[0].length() and Hypotheses[0][count] == Hypotheses[0][0])
    count++;
  if (global_context.DEBUG>2)
    printf("Hypothesis starts with an HP of length %d\n", count);
  // Adjust the length of the prefix and erase extra solved bases
  if (HPlength>count)
    base -= count;
  else
    base -= HPlength;
  master_read.sequence.erase(master_read.sequence.begin()+base, master_read.sequence.end());

  // Get flow of last prefix base
  int prefix_flow = 0;
  for (unsigned int i_base = 0; i_base < master_read.sequence.size(); i_base++)
    IncrementFlow(flow_order, master_read.sequence[i_base], prefix_flow);
  return prefix_flow;
}


// ----------------------------------------------------------------------

void PredictionGenerationVerbose(const vector<string>         &Hypotheses,
                                 const vector<BasecallerRead> &hypothesesReads,
                                 const Alignment              &my_read,
                                 const vector<vector<float> > &predictions,
                                 const int                    &prefix_size,
                                 const InputStructures        &global_context) {

  const int & num_flows = global_context.num_flows_by_run_id.at(my_read.runid);

  printf("Calculating predictions for %d hypotheses starting at flow %d:\n", (int)Hypotheses.size(), my_read.start_flow);
  for (unsigned int iHyp=0; iHyp<Hypotheses.size(); ++iHyp) {
    for (unsigned int iBase=0; iBase<Hypotheses[iHyp].length(); ++iBase)
      printf("%c", Hypotheses[iHyp][iBase]);
    printf("\n");
  }
  printf("5' read prefix: ");
  for (int iBase=0; iBase<prefix_size; ++iBase)
    printf("%c", hypothesesReads[0].sequence[iBase]);
  printf("\n");
  printf("Extended Hypotheses reads to:\n");
  for (unsigned int iHyp=0; iHyp<hypothesesReads.size(); ++iHyp) {
    for (unsigned int iBase=0; iBase<hypothesesReads[iHyp].sequence.size(); ++iBase)
      printf("%c", hypothesesReads[iHyp].sequence[iBase]);
    printf("\n");
  }
  printf("Phasing Parameters, cf: %f ie: %f dr: %f \n Predictions: \n",
          my_read.phase_params[0], my_read.phase_params[1], my_read.phase_params[2]);
  cout << "Flow Order  : ";
  for (int i_flow=0; i_flow<num_flows; i_flow++) {
    cout << global_context.flow_order_vector.at(my_read.flow_order_index).nuc_at(i_flow) << "    ";
  }
  cout << endl << "Flow Index  : ";
  for (int i_flow=0; i_flow<num_flows; i_flow++) {
      cout << i_flow << " ";
      if (i_flow<10)        cout << "   ";
      else if (i_flow<100)  cout << "  ";
      else if (i_flow<1000) cout << " ";
    }

  cout << endl;
  for (unsigned int i_Hyp=0; i_Hyp<hypothesesReads.size(); ++i_Hyp) {
	cout << "Prediction "<< i_Hyp << ": ";
    for (unsigned int i_flow=0; i_flow<predictions[i_Hyp].size(); ++i_flow) {
      printf("%.2f", predictions[i_Hyp][i_flow]);
      if (predictions[i_Hyp][i_flow] < 10)
        cout << " ";
    }
    cout << endl;
  }
}
