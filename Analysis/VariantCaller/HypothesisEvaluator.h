/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     HypothesisEvaluator.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#ifndef HYPOTHESISEVALUATOR_H
#define HYPOTHESISEVALUATOR_H

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <assert.h>
#include "stdlib.h"
#include "ctype.h"
#include "TreephaserSSE.h"
#include "DPTreephaser.h"
#include "SpliceVariantsToReads.h"



using namespace std;
using namespace ion;

class HypothesisEvaluator {
  public:

    vector<float>   measurementValues ;
    ion::FlowOrder  treePhaserFlowOrder;
    vector<float>   treePhaserParams;

    int nFlows;
    int DEBUG;


    HypothesisEvaluator(const vector<float>& measures, const ion::FlowOrder& flowOrder,const vector<float>& crIeDrParams, int _DEBUG=0) {
      treePhaserFlowOrder = flowOrder;
      assert(crIeDrParams.size() == 3);
      treePhaserParams = crIeDrParams;

      assert((int)measures.size() <= flowOrder.num_flows());
      nFlows = measures.size();
      measurementValues.reserve(flowOrder.num_flows());
      measurementValues = measures;
      measurementValues.resize(flowOrder.num_flows(),0);

      DEBUG = _DEBUG;
    };

    void LoadDataToRead(BasecallerRead &read);
    unsigned int SolveBeginningOfRead(DPTreephaser &working_treephaser, BasecallerRead &read,
                                      const vector<string>& Hypotheses, int startFlow);
    void   AddHypothesisToPrefix(BasecallerRead &current_read, const string &cur_hypothesis, unsigned int prefix_size);
    void EvaluateAllHypotheses(DPTreephaser &working_treephaser, BasecallerRead &read,
                               vector<BasecallerRead> &hypothesesReadsVector,const vector<string>& Hypotheses, int startFlow, int applyNormalization, int &max_last_flow);
    int EvaluateOneHypothesis(DPTreephaser &working_treephaser, BasecallerRead &current_hypothesis, int applyNormalization);
    int LastIncorporatingFlow(BasecallerRead &current_hypothesis);
    void SquaredDistancesAllHypotheses(vector<BasecallerRead> &hypothesesReadsVector, int max_last_flow,
                                       vector<float>& DistanceObserved,
                                       vector<float>& DistanceHypotheses);
    int MatchedFilter(DPTreephaser &working_treephaser, vector<BasecallerRead> &hypothesesReadsVector,int max_last_flow,
                       int refHpLen, int flowPosition, int startFlow,
                       vector<float>& DistanceObserved,
                       vector<float>& DistanceHypotheses);
    int CalculateDistances(const vector<string>& Hypotheses,
                           const int startFlow,
                           const int refHPlength,
                           const int flowPosition,
                           const int applyNormalization,
                           vector<float>& DistanceObserved,
                           vector<float>& DistanceHypotheses);

    int calculateCrudeLikelihood(vector<string> &hypotheses, int start_flow,
                                      float & refLikelihood, float & minDelta, float & minDistToRead, float & distDelta);

    int  calculateHPLikelihood(const string& read_sequence, const int startFlow, int refHpLength, int flowPosition, float & new_metric, bool strand);

};

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
          int verbose);

#endif // HYPOTHESISEVALUATOR_H
