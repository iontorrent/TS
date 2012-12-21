/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>
#include "DPTreephaser.h"

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

// ------------------------------------------------------------------------------


RcppExport SEXP calcHypothesesDistances(SEXP Rsignal, SEXP Rcf, SEXP Rie, SEXP Rdr,
		SEXP RflowCycle, SEXP RHypotheses, SEXP RstartFlow, SEXP Rnormalize, SEXP Rverbose)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
	    RcppVector<double>      signal(Rsignal);
	    std::string flowCycle = Rcpp::as<string>(RflowCycle);
	    double cf             = Rcpp::as<double>(Rcf);
	    double ie             = Rcpp::as<double>(Rie);
	    double dr             = Rcpp::as<double>(Rdr);
	    RcppStringVector        Hypotheses(RHypotheses);
	    int startFlow         = Rcpp::as<int>(RstartFlow);
	    int verbose           = Rcpp::as<int>(Rverbose);
	    int normalize         = Rcpp::as<int>(Rnormalize);

	    ion::FlowOrder flow_order(flowCycle, flowCycle.length());
	    int nFlow = signal.size();
	    int nHyp  = Hypotheses.size();

	    // Prepare objects for holding and passing back results
	    RcppVector<double>      DistanceObserved(nHyp);
	    RcppVector<double>      DistanceHypotheses(nHyp-1);
	    RcppMatrix<double>      predicted_out(nHyp,nFlow);
	    RcppMatrix<double>      normalized_out(nHyp,nFlow);

	    // Copy data into c++ data types
	    vector<float> Measurements(nFlow);
	    for (int i=0; i<nFlow; i++)
	        Measurements[i] = signal(i);
	    vector<string> HypVector(nHyp);
	    for (int i=0; i<nHyp; i++)
	    	HypVector[i] = Hypotheses(i);
	    vector<float> DistObserved;
	    DistObserved.assign(nHyp,0);
	    vector<float> DistHypotheses;
	    DistHypotheses.assign(nHyp-1,0);
	    vector<vector<float> > predictions(nHyp);
	    vector<vector<float> > normalized(nHyp);


	    CalculateHypDistances(Measurements, cf, ie, dr, flow_order, HypVector, startFlow,
	    		DistObserved, DistHypotheses, predictions, normalized, normalize, verbose);


	    // Store return values into return structure
	    for (int i=0; i<nHyp; i++){
	    	DistanceObserved(i) = (double)DistObserved[i];
	    	if (i>0)
	    		DistanceHypotheses(i-1) = (double)DistHypotheses[i-1];
	    	for (int iFlow=0; iFlow<nFlow; ++iFlow){
	    		predicted_out(i,iFlow)  = (double) predictions[i][iFlow];
	    		normalized_out(i,iFlow) = (double) normalized[i][iFlow];
	    	}
	    }
	    RcppResultSet rs;
	    rs.add("DistanceObserved",        DistanceObserved);
	    rs.add("DistanceHypotheses",      DistanceHypotheses);
	    rs.add("Predictions",             predicted_out);
	    rs.add("Normalized",              normalized_out);
	    ret = rs.getReturnList();

  } catch(std::exception& ex) {
    exceptionMesg = copyMessageToR(ex.what());
  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }

  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


