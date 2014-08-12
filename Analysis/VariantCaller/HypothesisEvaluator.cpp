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
    vector<vector<float> >   &predictions,
    vector<vector<float> >   &normalizedMeasurements,
    int flow_upper_bound) {

    // --- Step 1: Initialize Objects


    predictions.resize(Hypotheses.size());
    normalizedMeasurements.resize(Hypotheses.size());
    //int nFlows = min(global_context.treePhaserFlowOrder.num_flows(), (int)my_read.measurementValue.size());
    int nFlows = global_context.treePhaserFlowOrder.num_flows();
    int prefix_flow = 0;

    BasecallerRead master_read;
    master_read.SetData(my_read.measurements, global_context.treePhaserFlowOrder.num_flows());
    InitializeBasecallers(thread_objects, my_read, global_context);

    // --- Step 2: Solve beginning of the read
    prefix_flow = GetStartOfMasterRead(thread_objects, my_read, global_context, Hypotheses, nFlows, master_read);
    unsigned int prefix_size = master_read.sequence.size();

    // --- Step 3: creating predictions for the individual hypotheses

    flow_upper_bound = min(nFlows, min(flow_upper_bound,my_read.measurements_length)+50);

    vector<BasecallerRead> hypothesesReads(Hypotheses.size());
    int max_last_flow  = 0;

    for (unsigned int i_hyp=0; i_hyp<hypothesesReads.size(); ++i_hyp) {
        // short circuit if read as called is one of the hypotheses
        if ((i_hyp>0) & (Hypotheses[i_hyp].compare(Hypotheses[0])==0)) {
            predictions[i_hyp] = predictions[0];
            predictions[i_hyp].resize(nFlows);
            normalizedMeasurements[i_hyp] = normalizedMeasurements[0];
            normalizedMeasurements[i_hyp].resize(nFlows);
        } else {
            hypothesesReads[i_hyp] = master_read;

            // --- add hypothesis sequence to clipped prefix
            unsigned int i_base = 0;
            int i_flow = prefix_flow;

            while (i_base<Hypotheses[i_hyp].length() and i_base<(2*(unsigned int)global_context.treePhaserFlowOrder.num_flows()-prefix_size)) {
              IncrementFlow(global_context.treePhaserFlowOrder, Hypotheses[i_hyp][i_base], i_flow);
              if (i_flow >= global_context.treePhaserFlowOrder.num_flows()) {
                if (global_context.DEBUG>2 and i_base < Hypotheses[i_hyp].length()-1)
                  cout << "Prediction Generation: Shortened hypothesis " << i_hyp << " in read" << my_read.alignment.Name << endl;
                break;
              }
              // Add base to sequence only if it fits into flow order
              hypothesesReads[i_hyp].sequence.push_back(Hypotheses[i_hyp][i_base]);
              i_base++;
            }
            // Find last incorporating flow of all hypotheses
            //if (i_flow > max_last_flow)
              max_last_flow = max(max_last_flow, i_flow);
            // ---

            // Solver simulates beginning of the read and then fills in the remaining clipped bases
            // Above check guartantees that i_flow < num_flows()
            if (global_context.use_SSE_basecaller)
                //thread_objects.treephaser_sse.SolveRead(hypothesesReads[i_hyp], i_flow,
                //    min(my_read.measurements_length+50,nFlows)/*nFlows*/);
                thread_objects.treephaser_sse.SolveRead(hypothesesReads[i_hyp], min(i_flow,flow_upper_bound), flow_upper_bound);
            else
                thread_objects.dpTreephaser.Solve(hypothesesReads[i_hyp], nFlows, i_flow);

            // Adaptively normalize each hypothesis (to pot. recalibrated predictions) if desired
            if (global_context.apply_normalization) {
            	int window_size = 50;
                int steps = i_flow / window_size;
                thread_objects.dpTreephaser.WindowedNormalize(hypothesesReads[i_hyp], steps, window_size);
            }

            // Store predictions and adaptively normalized measurements
            predictions[i_hyp].swap(hypothesesReads[i_hyp].prediction);
            predictions[i_hyp].resize(nFlows, 0);
            normalizedMeasurements[i_hyp].swap(hypothesesReads[i_hyp].normalized_measurements);
            normalizedMeasurements[i_hyp].resize(nFlows, 0);
        }
    }

    // --- verbose ---
    if (global_context.DEBUG>2)
      PredictionGenerationVerbose(Hypotheses, hypothesesReads, my_read, predictions, prefix_size, global_context);

    //return max_last_flow;
    return min(max_last_flow,flow_upper_bound);
}

// ----------------------------------------------------------------------

void InitializeBasecallers(PersistingThreadObjects &thread_objects,
                           const Alignment         &my_read,
                           const InputStructures   &global_context) {

  // Set phasing parameters
  if (global_context.use_SSE_basecaller)
    thread_objects.treephaser_sse.SetModelParameters(my_read.phase_params[0], my_read.phase_params[1]);
  else
    thread_objects.dpTreephaser.SetModelParameters(my_read.phase_params[0], my_read.phase_params[1], my_read.phase_params[2]);

  // Set up HP recalibration model: hide the recal object behind a mask so we can use the map to select
  thread_objects.dpTreephaser.DisableRecalibration();   // Disable use of a previously loaded recalibration model
  thread_objects.treephaser_sse.DisableRecalibration();

  if (global_context.do_recal.recal_is_live()) {
    // query recalibration structure using row, column, entity
    // look up entity here: using row, col, runid
    // note: perhaps do this when we first get the read, exploit here
    string found_key = global_context.do_recal.FindKey(my_read.runid, my_read.well_rowcol[1], my_read.well_rowcol[0]);
    MultiAB multi_ab;
    global_context.do_recal.getAB(multi_ab, found_key, my_read.well_rowcol[1], my_read.well_rowcol[0]);
    if (multi_ab.Valid()) {
      thread_objects.dpTreephaser.SetAsBs(multi_ab.aPtr, multi_ab.bPtr);
      thread_objects.treephaser_sse.SetAsBs(multi_ab.aPtr, multi_ab.bPtr);
    }
  }
}

// ----------------------------------------------------------------------

int GetStartOfMasterRead(PersistingThreadObjects  &thread_objects,
                          const Alignment         &my_read,
                          const InputStructures    &global_context,
                          const vector<string>     &Hypotheses,
                          const int                &nFlows,
                          BasecallerRead           &master_read) {

  // Solve beginning of maybe clipped read
  int until_flow = min((my_read.start_flow+20), nFlows);
  if (my_read.start_flow > 0) {
    if (global_context.use_SSE_basecaller)
      thread_objects.treephaser_sse.SolveRead(master_read, 0, until_flow);
    else
      thread_objects.dpTreephaser.Solve(master_read, until_flow, 0);
  }

  // StartFlow clipped? Get solved HP length at startFlow.
  unsigned int base = 0;
  int flow = 0;
  unsigned int HPlength = 0;
  while (base < master_read.sequence.size()) {
    while (flow < global_context.treePhaserFlowOrder.num_flows()
           and global_context.treePhaserFlowOrder.nuc_at(flow) != master_read.sequence[base]) {
      flow++;
    }
    if (flow > my_read.start_flow or flow == global_context.treePhaserFlowOrder.num_flows())
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
    IncrementFlow(global_context.treePhaserFlowOrder, master_read.sequence[i_base], prefix_flow);
  return prefix_flow;
}


// ----------------------------------------------------------------------

void PredictionGenerationVerbose(const vector<string>         &Hypotheses,
                                 const vector<BasecallerRead> &hypothesesReads,
                                 const Alignment              &my_read,
                                 const vector<vector<float> > &predictions,
                                 const int                    &prefix_size,
                                 const InputStructures        &global_context) {

  printf("Calculating predictions for %d hypotheses starting at flow %d:\n", (int)Hypotheses.size(), my_read.start_flow);
  for (unsigned int iHyp=0; iHyp<Hypotheses.size(); ++iHyp) {
    for (unsigned int iBase=0; iBase<Hypotheses[iHyp].length(); ++iBase)
      printf("%c", Hypotheses[iHyp][iBase]);
    printf("\n");
  }
  printf("Solved read prefix: ");
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
  for (unsigned int i_flow=0; i_flow<global_context.flowOrder.length(); i_flow++) {
    cout << global_context.flowOrder[i_flow] << "    ";
  }
  cout << endl << "Flow Index  : ";
  for (unsigned int i_flow=0; i_flow<global_context.flowOrder.length(); i_flow++) {
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
