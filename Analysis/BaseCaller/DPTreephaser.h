/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * DPTreephaser.h
 * DP Tree Phase Solver
 *
 *  Created on: Jul 7, 2011, based on DPDephaser
 *      Author: ckoller
 */

#ifndef DPTREEPHASER_H
#define DPTREEPHASER_H

#include <string>
#include <vector>

using namespace std;



class BasecallerRead {
public:
  void SetDataAndKeyNormalize(const float *flowValues, int _numFlows, const int *keyVec, int numKeyFlows);

  void flowToString(const string &flowOrder, string &seq);

  void FitBaselineVector(int numSteps, int stepSize, int startStep = 0);
  void FitNormalizerVector(int numSteps, int stepSize, int startStep = 0);

  void Normalize(int fromFlow, int toFlow);

  void AdaptiveNormalizationOfPredictions(int numSteps, int stepSize);


  int             numFlows;
  float           keyNormalizer;
  vector<float>   measurements;     // Measured, key-normalized flow intensities
  vector<char>    solution;         // HP-sequence determined by the solver. All entries are integer
  vector<float>   prediction;       // Predicted flow intensities corresponding to "solved" sequence
  vector<float>   normalizedMeasurements;

  float           miscNormalizer;

};


class TreephaserPath {
public:
  TreephaserPath(int numFlows) :
    state(numFlows), prediction(numFlows), solution(numFlows) {
    flow = 0;
    windowStart = 0;
    windowEnd = 0;
    pathMetric = 0.0;
    residualLeftOfWindow = 0.0;
    perFlowMetric = 0.0;
    dotCounter = 0;
    inUse = false;
  }

  bool inUse;

  // Phasing state of this path

  int flow; // in phase flow
  vector<float> state;
  int windowStart;
  int windowEnd;

  // Path metrics and related values

  float pathMetric;
  float residualLeftOfWindow; // Residual left of the window;
  float perFlowMetric;
  int dotCounter;

  // Input-output state of this path

  vector<float> prediction;
  vector<char>  solution;
};

class DPTreephaser {

public:
  DPTreephaser(const char *_flowOrder, int _numFlows, int _numPaths);
  ~DPTreephaser() {
  }

  void SetModelParameters(double cf, double ie, double dr);

  int ComputeQVmetrics(BasecallerRead& read); // Computes "oneMerHeight" and "deltaPenalty" using BasicGreedySolver

  void NormalizeAndSolve3(BasecallerRead& read, int maxFlows); // Adaptive normalization
  void NormalizeAndSolve4(BasecallerRead& read, int maxFlows); // Old normalization

  void NormalizeAndSolve5(BasecallerRead& read, int maxFlows); // Windowed adaptive normalization


  void Solve(BasecallerRead& read, int maxFlows, int restartFlows = 0);
  void Simulate2(BasecallerRead& read, int maxFlows);
  void Simulate3(BasecallerRead& read, int maxFlows);


  vector<double> HpScaleWeight; // For Normalization
  vector<float> oneMerHeight; // For QV metrics
  vector<float> penaltyMismatch; // For QV metrics: min penalty to earlier base (deletion before this base)
  vector<float> penaltyResidual; // For QV metrics: min penalty to later base (this base is insertion)

protected:

  void advanceState(TreephaserPath *child, const TreephaserPath *parent, int nuc, int toflow);

  int numFlows; // Number of flows
  int numPaths; // Maximum number of paths considered
  int usePaths;

  float Multiplier; // Multiplier used for normalization
  float minDistance; // Squared distance of solution to measurements
  float expThreshold; // Threshold for expanding paths
  float negMultiplier; // Extra weight on the negative residuals
  float dotValue; // percentage of expected Signal that constitutes a "dot"
  int maxHP; // maximum possible homopolymer length
  float minFraction; // Minimum fraction to be taken into account

  int maxPathDelay; // Paths that are delayed more are killed
  bool useNonLinHPs; // Nonlinear Adjustment of homopolymer height

  vector<TreephaserPath> path;

  vector<float> transitionBase[4], transitionFlow[4];
  vector<int> flowOrder;

  vector<vector<float> > perNucHPadjustment;

};

#endif // DPTREEPHASER_H
