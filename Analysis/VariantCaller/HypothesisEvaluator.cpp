/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     HypothesisEvaluator.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "HypothesisEvaluator.h"


//@TODO: rationalize all these calls to treephaser to be coherent
// copypaste to get the code booted up - refactor away
int CalculateHypPredictions(const vector<float>& NormalizedMeasurements,
                           const vector<float>& PhaseParameters,
                           const ion::FlowOrder& flow_order,
                           const vector<string>& Hypotheses,
                           const int& startFlow,
                           vector<vector<float> >& predictions,
                           vector<vector<float> >& normalizedMeasurements,
                           int applyNormalization,
                           int verbose) {
  // Create return data structures
  predictions.resize(Hypotheses.size());
  normalizedMeasurements.resize(Hypotheses.size());

  // Step 1: Loading data to a read

  int nFlows = min(flow_order.num_flows(), (int)NormalizedMeasurements.size());

  BasecallerRead read;
  read.key_normalizer = 1;
  read.raw_measurements.reserve(flow_order.num_flows());
  read.raw_measurements = NormalizedMeasurements;

  for (unsigned int iFlow = 0; iFlow < read.raw_measurements.size(); iFlow++)
    if (isnan(read.raw_measurements[iFlow])) {
      cerr << "Warning: Calculate Predictions: NAN in measurements!"<< endl;
      read.raw_measurements[iFlow] = 0;
    }

  read.raw_measurements.resize(flow_order.num_flows(), 0);
  read.normalized_measurements = read.raw_measurements;
  read.sequence.clear();
  read.sequence.reserve(2*flow_order.num_flows());
  read.prediction.assign(flow_order.num_flows(), 0);
  read.additive_correction.assign(flow_order.num_flows(), 0);
  read.multiplicative_correction.assign(flow_order.num_flows(), 1.0);

  // Step 2: Solve beginning of the read

  int steps, window_size = 50;
  DPTreephaser dpTreephaser(flow_order);
  // Do not use droop to be consistent with with the way TreePhaser is used in the Analysis pipeline
  dpTreephaser.SetModelParameters(PhaseParameters.at(0), PhaseParameters.at(1), PhaseParameters.at(2));
  //TreephaserSSE sseTreephaser(flow_order, 50);
  //sseTreephaser.SetModelParameters(PhaseParameters.at(0), PhaseParameters.at(1));

  // Solve beginning of maybe clipped read
  int until_flow = min((startFlow+20), nFlows);
  if (startFlow>0) {
    //sseTreephaser.SolveRead(read, 0, until_flow);
    dpTreephaser.Solve(read, until_flow, 0); // XXX
  }
  // StartFlow clipped? Get solved HP length at startFlow
  unsigned int base = 0;
  int flow = 0;
  int HPlength = 0;
  while (base<read.sequence.size()) {
    while (flow < flow_order.num_flows() and flow_order.nuc_at(flow) != read.sequence[base])
      flow++;
    if (flow > startFlow or flow == flow_order.num_flows())
      break;
    if (flow == startFlow)
      HPlength++;
    base++;
  }
  if (verbose>0)
    printf("Solved %d bases until (not incl.) flow %d. HP of height %d at flow %d.\n", base, flow, HPlength, startFlow);
  // Get HP size at the start of the reference, i.e., Hypotheses[0]
  int count = 1;
  while (Hypotheses[0][count] == Hypotheses[0][0])
    count++;
  if (verbose>0)
    printf("Hypothesis starts with an HP of length %d\n", count);
  // Adjust the length of the prefix and erase extra solved bases
  if (HPlength>count)
    base -= count;
  else
    base -= HPlength;
  read.sequence.erase(read.sequence.begin()+base, read.sequence.end());
  unsigned int prefix_size = read.sequence.size();

  // Step 3: creating predictions for the individual hypotheses
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
    while (base<hypothesesReads[r].sequence.size() and flow<flow_order.num_flows()) {
      while (flow<nFlows and flow_order.nuc_at(flow) != hypothesesReads[r].sequence[base])
        flow++;
      last_incorporating_flow = flow;
      if (last_incorporating_flow > max_last_flow)
        max_last_flow = last_incorporating_flow;
      base++;
    }

    // Simulate sequence
    dpTreephaser.Simulate(hypothesesReads[r], flow_order.num_flows());

    // Adaptively normalize each hypothesis if desired
    if (applyNormalization>0) {
      steps = last_incorporating_flow / window_size;
      dpTreephaser.WindowedNormalize(hypothesesReads[r], steps, window_size);
    }

    // Solver simulates beginning of the read and then fills in the remaining clipped bases
    //sseTreephaser.SolveRead(hypothesesReads[r], last_incorporating_flow, flow_order.num_flows());
    dpTreephaser.Solve(hypothesesReads[r], nFlows, last_incorporating_flow);

    // Store predictions and adaptively normalized measurements
    predictions[r] = hypothesesReads[r].prediction;
    predictions[r].resize(nFlows);
    normalizedMeasurements[r] = hypothesesReads[r].normalized_measurements;
    normalizedMeasurements[r].resize(nFlows);
  }

  // --- verbose ---
  if (verbose>0) {
    printf("Calculating predictions for %d hypotheses starting at flow %d:\n", (int)Hypotheses.size(), startFlow);
    for (unsigned int i=0; i<Hypotheses.size(); ++i) {
      for (unsigned int j=0; j<Hypotheses[i].length(); ++j)
        printf("%c", Hypotheses[i][j]);
      printf("\n");
    }
    printf("Solved read prefix: ");
    for (unsigned int j=0; j<prefix_size; ++j)
      printf("%c", read.sequence[j]);
    printf("\n");
    printf("Extended Hypotheses reads to:\n");
    for (unsigned int i=0; i<hypothesesReads.size(); ++i) {
      for (unsigned int j=0; j<hypothesesReads[i].sequence.size(); ++j)
        printf("%c", hypothesesReads[i].sequence[j]);
      printf("\n");
    }
    printf("Phasing Parameters, cf: %f ie: %f dr: %f \n Predictions: \n",
    		PhaseParameters.at(0), PhaseParameters.at(1), PhaseParameters.at(2));
    for (unsigned int i=0; i<hypothesesReads.size(); ++i) {
      for (unsigned int j=0; j<predictions[i].size(); ++j)
        printf("%f", predictions[i][j]);
      printf("\n");
    }
  }
  // --------------- */
  return(max_last_flow);
}

// -------------------------------------------------------------------------


void HypothesisEvaluator::LoadDataToRead(BasecallerRead &master_read) {
  master_read.key_normalizer = 1;
  master_read.sequence.clear();
  master_read.sequence.reserve(2*treePhaserFlowOrder.num_flows());
  master_read.prediction.assign(treePhaserFlowOrder.num_flows(), 0);
  master_read.additive_correction.assign(treePhaserFlowOrder.num_flows(), 0);
  master_read.multiplicative_correction.assign(treePhaserFlowOrder.num_flows(), 1.0);
  master_read.raw_measurements = measurementValues;

  for (unsigned int iFlow = 0; iFlow < master_read.raw_measurements.size(); iFlow++)
    if (isnan(master_read.raw_measurements[iFlow])) {
      cerr << "Warning: Calculate Distances: NAN in measurements!"<< endl;
      //return -1;
    }
  master_read.raw_measurements.resize(treePhaserFlowOrder.num_flows(), 0);
  master_read.normalized_measurements = master_read.raw_measurements;
}


unsigned int HypothesisEvaluator::SolveBeginningOfRead(DPTreephaser &working_treephaser, BasecallerRead &master_read,
    const vector<string>& Hypotheses, int startFlow) {
  //cout << "Hypothesis sequence: " << Hypotheses[0] << endl;
  // Solve beginning of maybe clipped read
  if (startFlow>0) {
    int until_flow = min((startFlow+20), nFlows);
    working_treephaser.Solve(master_read, until_flow, 0);
  }
  /*cout << "Solved prefix of size " << read.sequence.size() << ": ";
  for (int i=0; i<read.sequence.size(); i++)
   cout << read.sequence[i];
  cout << endl;*/
  // StartFlow clipped? Get solved HP length at startFlow
  unsigned int base = 0;
  int flow = 0;
  int HPlength = 0;
  while (base<master_read.sequence.size()) {
    while (flow < treePhaserFlowOrder.num_flows() and treePhaserFlowOrder.nuc_at(flow) != master_read.sequence[base])
      flow++;
    if (flow > startFlow or flow == treePhaserFlowOrder.num_flows())
      break;
    if (flow == startFlow)
      HPlength++;
    base++;
  }
  // Get HP size at the start of the reference, i.e., Hypotheses[0]
  int count = 1;
  while (Hypotheses[0][count] == Hypotheses[0][0])
    count++;
  // Adjust the length of the base prefix and erase extra solved bases
  if (HPlength>count)
    base -= count;
  else
    base -= HPlength;
  master_read.sequence.erase(master_read.sequence.begin()+base, master_read.sequence.end());
  unsigned int prefix_size = master_read.sequence.size();

  /*cout << "Shortened prefix to size " << prefix_size << " until startFlow" << startFlow << ": ";
  for (int i=0; i<read.sequence.size(); i++)
    cout << read.sequence[i];
  cout << endl;*/
  return(prefix_size);
}

void HypothesisEvaluator::AddHypothesisToPrefix(BasecallerRead &current_read, const string &cur_hypothesis, unsigned int prefix_size) {
  // add hypothesis sequence to prefix
  for (unsigned int base=0; base<cur_hypothesis.length() and base<(2*(unsigned int)treePhaserFlowOrder.num_flows()-prefix_size); base++)
    current_read.sequence.push_back(toupper(cur_hypothesis[base]));
}

void HypothesisEvaluator::EvaluateAllHypotheses(DPTreephaser &working_treephaser, BasecallerRead &master_read,
    vector<BasecallerRead> &hypothesesReadsVector,const vector<string>& Hypotheses, int startFlow, int applyNormalization, int &max_last_flow) {

  unsigned int prefix_size = SolveBeginningOfRead(working_treephaser, master_read, Hypotheses, startFlow);

  //cout << "DistanceCalculator numflows = " << flow_order.num_flows() << " size of hyp = " << Hypotheses.size() << " Start flow = " << startFlow << endl;
  for (unsigned int r = 0; r<hypothesesReadsVector.size(); ++r) {
    hypothesesReadsVector[r] = master_read;
    AddHypothesisToPrefix(hypothesesReadsVector[r], Hypotheses[r], prefix_size);

    int last_incorporating_flow = EvaluateOneHypothesis(working_treephaser, hypothesesReadsVector[r], applyNormalization);
    if (last_incorporating_flow>max_last_flow)
      max_last_flow = last_incorporating_flow;
  }
}

int HypothesisEvaluator::LastIncorporatingFlow(BasecallerRead &current_hypothesis) {
  // get last main incorporating flow
  int last_incorporating_flow = 0;
  unsigned int base = 0;
  int flow = 0;
  while (base<current_hypothesis.sequence.size() and flow<treePhaserFlowOrder.num_flows()) {
    while (flow<nFlows and treePhaserFlowOrder.nuc_at(flow) != current_hypothesis.sequence[base])
      flow++;
    last_incorporating_flow = flow;
    base++;
  }
  /*
  while (base<hypothesesReads[r].sequence.size() and flow<flow_order.num_flows()) {
      while (flow_order.nuc_at(flow) != hypothesesReads[r].sequence[base])
        flow++;
      last_incorporating_flow = flow;
      if (last_incorporating_flow > max_last_flow)
        max_last_flow = last_incorporating_flow;
      base++;
    }
   * */


  //cout << "Simulted up to flow " << last_incorporating_flow << ", sequence of length " << hypothesesReadsVector[r].sequence.size() << endl;
  return(last_incorporating_flow);
}

int HypothesisEvaluator::EvaluateOneHypothesis(DPTreephaser &working_treephaser, BasecallerRead &current_hypothesis, int applyNormalization) {

  int last_incorporating_flow = LastIncorporatingFlow(current_hypothesis);

  // Simulate sequence
  working_treephaser.Simulate(current_hypothesis, nFlows);

  // Adaptively normalize each hypothesis
  if (applyNormalization>0) {
    int window_size= 50;
    int steps = last_incorporating_flow / window_size;
    working_treephaser.WindowedNormalize(current_hypothesis, steps, window_size);
  }

  // Solver simulates beginning of the read and then fills in the remaining clipped bases
  working_treephaser.Solve(current_hypothesis, nFlows, last_incorporating_flow);
  /*cout << "Solved sequence of length: " << hypothesesReadsVector[r].sequence.size() << " ;nFlows = " << nFlows << endl;
  cout << "Total read: ";
  for (int i=0; i<hypothesesReadsVector[r].sequence.size(); i++)
   cout << hypothesesReadsVector[r].sequence[i];
  cout << endl;*/
  return(last_incorporating_flow);
}

bool TestFlowForInclusion(vector<BasecallerRead> &hypothesesReadsVector, int flow, float threshold) {
  bool includeFlow = false;
  for (unsigned int hyp=1; hyp<hypothesesReadsVector.size(); ++hyp)
    if (fabs(hypothesesReadsVector[hyp].prediction[flow] - hypothesesReadsVector[0].prediction[flow])>threshold)
      includeFlow = true;
  return(includeFlow);
}

void UpdateSquaredDistancesAllHypotheses(vector<BasecallerRead> &hypothesesReadsVector, int flow, int DEBUG,
    vector<float>& DistanceObserved,
    vector<float>& DistanceHypotheses) {
  float residual;

  for (unsigned int hyp=0; hyp<hypothesesReadsVector.size(); ++hyp) {

    residual = hypothesesReadsVector[hyp].normalized_measurements[flow] - hypothesesReadsVector[hyp].prediction[flow];
    /* if (DEBUG) {
      cout << " Flow = " << flow << " Hyp = " << hyp << "  measure = " << hypothesesReadsVector[hyp].normalized_measurements[flow]
         << " Prediction = " << hypothesesReadsVector[hyp].prediction[flow] << " residual = " << residual << endl;
    }*/
    DistanceObserved[hyp] += residual * residual;
    if (hyp>0) {
      residual = hypothesesReadsVector[0].prediction[flow] - hypothesesReadsVector[hyp].prediction[flow];
      // this is crazy as we should be putting back in matched locations 1-1
      DistanceHypotheses[hyp-1] += residual * residual;
    }
  }
}

void HypothesisEvaluator::SquaredDistancesAllHypotheses(vector<BasecallerRead> &hypothesesReadsVector, int max_last_flow,
    vector<float>& DistanceObserved,
    vector<float>& DistanceHypotheses) {
  // --- Include only flow values in the squared distance where the
  // predictions differ by more than "threshold"
  float threshold = 0.01;

  int until_flow = min(max_last_flow+1, nFlows);
  for (int flow=0; flow<until_flow; ++flow) {
    if (TestFlowForInclusion(hypothesesReadsVector, flow, threshold)) {
      UpdateSquaredDistancesAllHypotheses(hypothesesReadsVector, flow, this->DEBUG, DistanceObserved,DistanceHypotheses);
    }
  }
}


int HypothesisEvaluator::MatchedFilter(DPTreephaser &working_treephaser, vector<BasecallerRead> &hypothesesReadsVector,int max_last_flow,
                                       int refHpLen, int flowPosition, int startFlow,
                                       vector<float>& DistanceObserved,
                                       vector<float>& DistanceHypotheses) {
  // Matched Filter HP distance is computed here
  vector<float> query_state(nFlows);
  int rHpLen = 0;
  if (flowPosition<startFlow || flowPosition >= nFlows) {
    cout << "Calculate Distances: Unsupported flowPosition! startFlow: " << startFlow << " flowPosition: " << flowPosition << " nFlows: " << nFlows << endl;
    return -1;
  }
  //cout << "Calling Query state " << endl;
  //cout << "Hypothesis = " << hypothesesReadsVector[0] << endl;
  //cout << "flow position = " << flowPosition << endl;
  //cout << "Nflows = " << nFlows << endl;
  //int readSize = hypothesesReadsVector[0].sequence.size();
// for (int i = 0; i < readSize ; i++)
//   cout << "Base = " << hypothesesReadsVector[0].sequence.at(i) << " Measure = " << hypothesesReadsVector[0].normalized_measurements.at(i) << endl;

  working_treephaser.QueryState(hypothesesReadsVector[0], query_state, rHpLen, nFlows, flowPosition);

  if (rHpLen == 0) {
    if (DEBUG) {
      cerr << "Hypothesis evaluator error ReadHpLen = 0 " << endl;
      cerr << "Calling Query state " << endl;
      cerr << "Hypothesis = " << hypothesesReadsVector[0].sequence.size() << endl;
      cerr << "flow position = " << flowPosition << endl;
      cerr << "Nflows = " << nFlows << endl;
      int readSize = hypothesesReadsVector[0].sequence.size();
      for (int i = 0; i < readSize ; i++)
        cerr << " i = " << i << " Base = " << hypothesesReadsVector[0].sequence.at(i) << " Measure = " << hypothesesReadsVector[0].normalized_measurements.at(i) << endl;
    }
    return -1;
  }
  //return -1;

  if (abs(rHpLen-refHpLen) < 3) {
    float filter_num = 0.0f;
    float filter_den = 0.0f;
    if (this->DEBUG)
      cout << "Matched filter details: " << endl;
    for (int flow=0; flow<nFlows; flow++) {
      filter_num += (hypothesesReadsVector[0].normalized_measurements[flow] - hypothesesReadsVector[0].prediction[flow] +
                     ((float)rHpLen*query_state[flow])) * query_state[flow];
      filter_den += query_state[flow] * query_state[flow];

      if ((this->DEBUG) and(query_state[flow] > 0.02 or flow == flowPosition)) {
        cout << "Flow " << flow << " State " << query_state[flow] << " Local delta "
             << ((hypothesesReadsVector[0].normalized_measurements[flow] - hypothesesReadsVector[0].prediction[flow] + ((float)rHpLen*query_state[flow])) * query_state[flow])
             << " Measurements " << hypothesesReadsVector[0].normalized_measurements[flow];
        //printf("Flow %4d  State %1.4f  Local delta %1.4f  Measurements %1.4f");
        //flow, query_state[flow],
        //((read.normalized_measurements[flow] - read.prediction[flow] + ((float)rHpLen*query_state[flow])) * query_state[flow]),
        //read.normalized_measurements[flow]);
        if (flow == flowPosition)
          //printf(" ***\n");
          cout << " ***" << endl;
        else
          //printf("\n");
          cout << endl;
      }
    }
    DistanceObserved[0] = filter_num / filter_den;
    //cout << DistanceObserved[0] << endl;
  } else {
    if (DEBUG)
      cerr << "Wrong rHpLen : " << rHpLen << " " << refHpLen << endl;
    return -1;
  }

  return(0);
}

// --------------------------------------------------------------------
// need to split this for the HP/non-hp case
int HypothesisEvaluator::CalculateDistances(const vector<string>& Hypotheses,
    const int startFlow,
    const int refHpLen,
    const int flowPosition,
    const int applyNormalization,
    vector<float>& DistanceObserved,
    vector<float>& DistanceHypotheses) {

  DPTreephaser working_treephaser(treePhaserFlowOrder);
  working_treephaser.SetModelParameters(treePhaserParams.at(0), treePhaserParams.at(1), treePhaserParams.at(2));

  // Loading signal values into a read
  BasecallerRead master_read;
  LoadDataToRead(master_read);

  vector<BasecallerRead> hypothesesReadsVector(Hypotheses.size());

  int max_last_flow=0;
  EvaluateAllHypotheses(working_treephaser, master_read, hypothesesReadsVector, Hypotheses, startFlow, applyNormalization,max_last_flow);

  // Calculating distances -------------------------------
  // At this point switch between HP case (only one Hypothesis given) and non-HP (#Hypotheses>1)
  //cout << "calculate deltas " << endl;
  //cout << "# hyp = " << Hypotheses.size() << endl;
  // Create return data structures
  // Distance of normalized observations to different hypotheses: d(obs,h1), ... , d(obs,hN)
  DistanceObserved.assign(Hypotheses.size(), 0);
  // Distance of hypotheses to first hypothesis: d(h1,h2), ... , d(h1, hN)
  DistanceHypotheses.assign(Hypotheses.size()-1, 0);

  if (Hypotheses.size()>1) {
    SquaredDistancesAllHypotheses(hypothesesReadsVector, max_last_flow, DistanceObserved,DistanceHypotheses);
  } else {
    return(MatchedFilter(working_treephaser, hypothesesReadsVector,max_last_flow,
                         refHpLen,flowPosition, startFlow,
                         DistanceObserved,
                         DistanceHypotheses));
  }
  return 0;
}




float FindMinDelta(vector<float> &deltaHypotheses) {
  float minDelta = 9999999;

  for (unsigned int i = 0; i < deltaHypotheses.size(); i++) {
    if (deltaHypotheses[i] <= minDelta)
      minDelta = deltaHypotheses[i];
  }
  return(minDelta);
}

float CrudeSumOfSquares(vector<float> &deltaHypotheses, float &minDelta, float &minDistToRead,  int DEBUG) {
  minDistToRead = 999999;
  //distDelta = 0;

  minDelta=FindMinDelta(deltaHypotheses);
  float totalDelta = 0;

  for (unsigned int i = 0; i < deltaHypotheses.size(); i++) {
    float my_distance = (deltaHypotheses[i] - minDelta);
    if (DEBUG)
      cout << "Deltas[ " <<  i << "] = " << my_distance << endl;
    //if (my_distance < 0)
    // cout << "ERROR: Residual for Hypothesis: " << i << " is less than Read Hypothesis - " << deltaHypotheses[i] << " , " << deltaHypotheses[deltaHypotheses.size()-1] << endl;

    totalDelta += my_distance;

    if (i < deltaHypotheses.size()-1) {
      if (my_distance < minDistToRead)
        minDistToRead = my_distance;
    }
  }
  return(totalDelta);
}

// -------------------------------------------------------------------------
int HypothesisEvaluator::calculateCrudeLikelihood(vector<string> &hypotheses, int start_flow,
    float & refLikelihood, float & minDelta, float & minDistToRead, float & distDelta) {

  vector<float> deltaHypotheses;
  vector<float> distHypotheses;
  int retValue = CalculateDistances(hypotheses, start_flow, 0,0,0, deltaHypotheses, distHypotheses);

  if (retValue == -1)
    return -1;

  float totalDelta = CrudeSumOfSquares(deltaHypotheses, minDelta, minDistToRead,  this->DEBUG);

  if (deltaHypotheses[1]-minDelta < 0)
    refLikelihood = 0;
  else
    refLikelihood = deltaHypotheses[1]-minDelta;

  if (deltaHypotheses[0]-minDelta < 0)
    distDelta = 0;
  else
    distDelta = deltaHypotheses[0]-minDelta;

  if (totalDelta < 0)
    return -1;

  return 0;
}

// --------------------------------------------------------------------------

int HypothesisEvaluator::calculateHPLikelihood(const string& read_sequence, const int startFlow, int refHpLen, int flowPosition, float & new_metric, bool strand) {

  new_metric = 0;
  vector<string> Hypotheses;
  if (!strand) {
    char * seqRev = new char[read_sequence.length()+1];
    reverseComplement(read_sequence, seqRev);
    Hypotheses.push_back(string((const char*) seqRev));
  } else
    Hypotheses.push_back(read_sequence);

  vector<float> DistanceObserved(1);
  vector<float> DistanceHypotheses(0);
  int return_val;

  return_val = CalculateDistances(Hypotheses, startFlow, refHpLen, flowPosition, 0,
                                  DistanceObserved, DistanceHypotheses);
  if (return_val == -1)
    return return_val;

  new_metric = DistanceObserved[0];
  return 0;
}

