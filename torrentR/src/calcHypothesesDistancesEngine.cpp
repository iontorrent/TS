/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

// dislike: just for "verbose"
#include <Rcpp.h>

#include "calcHypothesesDistancesEngine.h"

void CalculateHypDistances(const vector<float>& NormalizedMeasurements,
				  const float& cf,
				  const float& ie,
				  const float& droop,
				  const ion::FlowOrder& flow_order,
				  const vector<string>& Hypotheses,
				  const int& startFlow,
				  vector<float>& DistanceObserved,
				  vector<float>& DistanceHypotheses,
				  vector<vector<float> >& predictions,
				  vector<vector<float> >& normalizedMeasurements,
				  int applyNormalization,
				  int verbose)
{
	// Create return data structures
	// Distance of normalized observations to different hypotheses: d(obs,h1), ... , d(obs,hN)
	DistanceObserved.assign(Hypotheses.size(), 0);
	// Distance of hypotheses to first hypothesis: d(h1,h2), ... , d(h1, hN)
	DistanceHypotheses.assign(Hypotheses.size()-1, 0);
	predictions.resize(Hypotheses.size());
	normalizedMeasurements.resize(Hypotheses.size());

	// Loading key normalized values into a read and performing adaptive normalization
	BasecallerRead read;
	read.key_normalizer = 1;
	read.raw_measurements = NormalizedMeasurements;
	read.normalized_measurements = NormalizedMeasurements;
	read.sequence.clear();
	read.sequence.reserve(2*flow_order.num_flows());
	read.prediction.assign(flow_order.num_flows(), 0);
	read.additive_correction.assign(flow_order.num_flows(), 0);
	read.multiplicative_correction.assign(flow_order.num_flows(), 1.0);

	int steps, window_size = 50;
	DPTreephaser dpTreephaser(flow_order);
	dpTreephaser.SetModelParameters(cf, ie, droop);

	// Solve beginning of maybe clipped read
	if (startFlow>0)
		dpTreephaser.Solve(read, (startFlow+20), 0);
	// StartFlow clipped? Get solved HP length at startFlow
    unsigned int base = 0;
    int flow = 0;
    int HPlength = 0;
    while (base<read.sequence.size()){
    	while (flow < flow_order.num_flows() and flow_order.nuc_at(flow) != read.sequence[base])
    		flow++;
    	if (flow > startFlow or flow == flow_order.num_flows())
    		break;
    	if (flow == startFlow)
    		HPlength++;
    	base++;
    }
    if (verbose>0)
      Rprintf("Solved %d bases until (not incl.) flow %d. HP of height %d at flow %d.\n", base, flow, HPlength, startFlow);
    // Get HP size at the start of the reference, i.e., Hypotheses[0]
    int count = 1;
    while (Hypotheses[0][count] == Hypotheses[0][0])
    	count++;
    if (verbose>0)
      Rprintf("Hypothesis starts with an HP of length %d\n", count);
    // Adjust the length of the prefix and erase extra solved bases
    if (HPlength>count)
    	base -= count;
    else
    	base -= HPlength;
    read.sequence.erase(read.sequence.begin()+base, read.sequence.end());
    unsigned int prefix_size = read.sequence.size();

	// creating predictions for the individual hypotheses
	vector<BasecallerRead> hypothesesReads(Hypotheses.size());
	int max_last_flow  = 0;

	for (unsigned int r=0; r<hypothesesReads.size(); ++r) {

		hypothesesReads[r] = read;
		// add hypothesis sequence to prefix
		for (base=0; base<Hypotheses[r].length() and base<(2*(unsigned int)flow_order.num_flows()-prefix_size); base++)
			hypothesesReads[r].sequence.push_back(Hypotheses[r][base]);

		// get last main incorporating flow
		int last_incorporating_flow = 0;
		base = 0;
		flow = 0;
        while (base<hypothesesReads[r].sequence.size() and flow<flow_order.num_flows()){
            while (flow_order.nuc_at(flow) != hypothesesReads[r].sequence[base])
                flow++;
		    last_incorporating_flow = flow;
		    if (last_incorporating_flow > max_last_flow)
		    	max_last_flow = last_incorporating_flow;
		    base++;
		}

		// Simulate sequence
		dpTreephaser.Simulate(hypothesesReads[r], flow_order.num_flows());

		// Adaptively normalize each hypothesis
		if (applyNormalization>0) {
		    steps = last_incorporating_flow / window_size;
		    dpTreephaser.WindowedNormalize(hypothesesReads[r], steps, window_size);
		}

		// Solver simulates beginning of the read and then fills in the remaining clipped bases
		dpTreephaser.Solve(hypothesesReads[r], flow_order.num_flows(), last_incorporating_flow);

		// Store predictions and adaptively normalized measurements
		predictions[r] = hypothesesReads[r].prediction;
		normalizedMeasurements[r] = hypothesesReads[r].normalized_measurements;
	}


	// --- Calculating distances ---
	// Include only flow values in the distance where the predictions differ by more than "threshold"
	float threshold = 0.05;

	// Do not include flows after main inc. flow of lastest hypothesis
	for (int flow=0; flow<(max_last_flow+1); ++flow) {
		bool includeFlow = false;
		for (unsigned int hyp=1; hyp<hypothesesReads.size(); ++hyp)
			if (abs(hypothesesReads[hyp].prediction[flow] - hypothesesReads[0].prediction[flow])>threshold)
				includeFlow = true;

		if (includeFlow) {
			for (unsigned int hyp=0; hyp<hypothesesReads.size(); ++hyp) {
				float residual = hypothesesReads[hyp].normalized_measurements[flow] - hypothesesReads[hyp].prediction[flow];
				DistanceObserved[hyp] += residual * residual;
				if (hyp>0) {
					residual = hypothesesReads[0].prediction[flow] - hypothesesReads[hyp].prediction[flow];
					DistanceHypotheses[hyp-1] += residual * residual;
				}
			}
		}

	}

	// --- verbose ---
	if (verbose>0){
	  Rprintf("Calculating distances between %d hypotheses starting at flow %d:\n", Hypotheses.size(), startFlow);
	  for (unsigned int i=0; i<Hypotheses.size(); ++i){
		for (unsigned int j=0; j<Hypotheses[i].length(); ++j)
			Rprintf("%c", Hypotheses[i][j]);
		Rprintf("\n");
	  }
	  Rprintf("Solved read prefix: ");
	  for (unsigned int j=0; j<prefix_size; ++j)
		Rprintf("%c", read.sequence[j]);
	  Rprintf("\n");
	  Rprintf("Extended Hypotheses reads to:\n");
	  for (unsigned int i=0; i<hypothesesReads.size(); ++i){
		for (unsigned int j=0; j<hypothesesReads[i].sequence.size(); ++j)
		  Rprintf("%c", hypothesesReads[i].sequence[j]);
		Rprintf("\n");
	  }
	  Rprintf("Calculated Distances d2(obs, H_i), d2(H_i, H_0):\n");
	  Rprintf("%f, 0\n", DistanceObserved[0]);
	  for (unsigned int i=1; i<Hypotheses.size(); ++i)
		Rprintf("%f, %f\n", DistanceObserved[i], DistanceHypotheses[i-1]);
    }
    // --------------- */

}

