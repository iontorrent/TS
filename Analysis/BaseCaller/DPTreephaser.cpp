/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * DPTreephaser.cpp
 *
 *  Created on: July 11, 2011 adapted from DPDephaser
 *      Author: ckoller
 */

#include <cassert>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "DPTreephaser.h"


DPTreephaser::DPTreephaser(const char *_flowOrder, int _numFlows, int _numPaths)
{
  numFlows = _numFlows;
  numPaths = _numPaths;

  // Set default values to options
  expThreshold = 0.2;
  minFraction = 1e-6;
  maxPathDelay = 40;
  negMultiplier = 2.0;
  dotValue = 0.3;
  maxHP = 11;
  usePaths = numPaths;

  float HpScale[5] = { 0, 1, 0.8, 0.64, 0.37 };
  HpScaleWeight.assign(HpScale, HpScale + 5);
  Multiplier = 1;

  // Computing coefficients for nonlinear HP adjustment "TACG"
  useNonLinHPs = false;
  perNucHPadjustment.resize(4);
  float nucAdjustment[4] = { -0.01, -0.01, -0.02, 0 };
  for (int nuc = 0; nuc < 4; nuc++) {
    perNucHPadjustment[nuc].resize(maxHP);
    for (int iHP = 0; iHP < maxHP; iHP++)
      perNucHPadjustment[nuc][iHP] = exp(nucAdjustment[nuc] * iHP);
  }

  // Initialize state storage
  for (int i = 0; i < 4; i++) {
    transitionBase[i].resize(numFlows);
    transitionFlow[i].resize(numFlows);
  }

  flowOrder.resize(numFlows);
  int iFlowShort = 0;
  for (int iFlow = 0; iFlow < numFlows; iFlow++) {
    switch (_flowOrder[iFlowShort]) {
    case 'T':
      flowOrder[iFlow] = 0;
      break;
    case 'A':
      flowOrder[iFlow] = 1;
      break;
    case 'C':
      flowOrder[iFlow] = 2;
      break;
    case 'G':
      flowOrder[iFlow] = 3;
      break;
    }
    iFlowShort++;
    if (_flowOrder[iFlowShort] == 0)
      iFlowShort = 0;
  }

  path.resize(numPaths, numFlows);
}

//-------------------------------------------------------------------------

void DPTreephaser::SetModelParameters(double cf, double ie, double dr)
{
  double distToBase[4] = { 0, 0, 0, 0 };
  for (int iFlow = 0; iFlow < numFlows; iFlow++) {
    distToBase[flowOrder[iFlow]] = 1;
    for (int nuc = 0; nuc < 4; nuc++) {
      transitionBase[nuc][iFlow] = distToBase[nuc] * (1 - dr) * (1 - ie);
      transitionFlow[nuc][iFlow] = (1 - distToBase[nuc]) + distToBase[nuc] * (1 - dr) * ie;
      distToBase[nuc] *= cf;
    }
  }
}

//-------------------------------------------------------------------------



void BasecallerRead::SetDataAndKeyNormalize(float *flowValues, int _numFlows, int *keyVec, int numKeyFlows)
{
  numFlows = _numFlows;
  measurements.resize(numFlows);
  normalizedMeasurements.resize(numFlows);
  solution.assign(numFlows, 0);
  prediction.assign(numFlows, 0);

  float onemerSum = 0.0;
  float onemerCount = 0.0;

  for (int iFlow = 0; iFlow < numKeyFlows; iFlow++) {
    if (keyVec[iFlow] == 1) {
      onemerSum += flowValues[iFlow];
      onemerCount += 1.0;
    }
  }

  keyNormalizer = 1;
  miscNormalizer = 1;
  for (int iFlow = 0; iFlow < numFlows; iFlow++)
    measurements[iFlow] = flowValues[iFlow];

  if ((onemerSum > 0) && (onemerCount > 0)) {
    keyNormalizer = onemerCount / onemerSum;

    for (int iFlow = 0; iFlow < numFlows; iFlow++)
      measurements[iFlow] *= keyNormalizer;
  }

  for (int iFlow = 0; iFlow < numFlows; iFlow++)
    normalizedMeasurements[iFlow] = measurements[iFlow];
}

void BasecallerRead::flowToString(const string &flowOrder, string &seq)
{
  seq = "";
  for(unsigned int iFlow=0; iFlow < solution.size(); iFlow++) {
    if(solution[iFlow] > 0) {
      char base = flowOrder[iFlow % flowOrder.length()];
      for(int iNuc = solution[iFlow]; iNuc > 0; iNuc--)
        seq += base;
    }
  }
}



// ----------------------------------------------------------------------
// New normalization strategy

void BasecallerRead::FitBaselineVector(int numSteps, int stepSize, int startStep)
{
  assert(numSteps > 0);

  float latestNormalizer = 0;
  int fromFlow = 0;
  int toFlow = stepSize;

  float stepNormalizers[numSteps];
  float medianSet[stepSize];

  for (int iStep = startStep; iStep < numSteps; iStep++) {

    if (fromFlow >= numFlows)
      break;
    toFlow = std::min(toFlow, numFlows);

    if (toFlow < numFlows) {
      int medianCount = 0;
      for (int iFlow = fromFlow; iFlow < toFlow; iFlow++)
        if (prediction[iFlow] < 0.3)
          medianSet[medianCount++] = measurements[iFlow] - prediction[iFlow];

      if (medianCount > 5) {
        std::nth_element(medianSet, medianSet + medianCount/2, medianSet + medianCount);
        latestNormalizer = medianSet[medianCount / 2];
      }
    }

    stepNormalizers[iStep] = latestNormalizer;

    fromFlow = toFlow;
    toFlow += stepSize;
  }

  // Three modes:
  //  1 - before the middle of the first region: flat
  //  2 - in between regions: linear interpolation
  //  3 - beyond the middle of the last region: flat


  int halfStep = stepSize / 2;

  int iFlow = 0;
  if (startStep > 0)
    iFlow = halfStep + startStep * stepSize;

  for (; (iFlow < halfStep) && (iFlow < numFlows); iFlow++)
    normalizedMeasurements[iFlow] = measurements[iFlow] - stepNormalizers[0];

  for (int iStep = startStep + 1; iStep < numSteps; iStep++) {
    for (int pos = 0; (pos < stepSize) && (iFlow < numFlows); pos++, iFlow++)
      normalizedMeasurements[iFlow] = measurements[iFlow]
          - (stepNormalizers[iStep - 1] * (stepSize - pos) + stepNormalizers[iStep] * pos) / stepSize;
  }

  for (; iFlow < numFlows; iFlow++)
    normalizedMeasurements[iFlow] = measurements[iFlow] - stepNormalizers[numSteps - 1];

}

void BasecallerRead::FitNormalizerVector(int numSteps, int stepSize, int startStep)
{
  assert(numSteps > 0);

  float latestNormalizer = 1;
  int fromFlow = 0;
  int toFlow = stepSize;

  float stepNormalizers[numSteps];
  float medianSet[stepSize];

  for (int iStep = startStep; iStep < numSteps; iStep++) {

    if (fromFlow >= numFlows)
      break;
    toFlow = std::min(toFlow, numFlows);

    if (toFlow < numFlows) {
      int medianCount = 0;
      for (int iFlow = fromFlow; iFlow < toFlow; iFlow++)
        if ((prediction[iFlow] > 0.5) && (normalizedMeasurements[iFlow] > 0))
          medianSet[medianCount++] = normalizedMeasurements[iFlow] / prediction[iFlow];

      if (medianCount > 5) {
        std::nth_element(medianSet, medianSet + medianCount/2, medianSet + medianCount);
        if (medianSet[medianCount / 2] > 0)
          latestNormalizer = medianSet[medianCount / 2];
      }
    }

    stepNormalizers[iStep] = latestNormalizer;

    fromFlow = toFlow;
    toFlow += stepSize;
  }

  // Three modes:
  //  1 - before the middle of the first region: flat
  //  2 - in between regions: linear interpolation
  //  3 - beyond the middle of the last region: flat

  float normalizer = 1;
  int halfStep = stepSize / 2;

  int iFlow = 0;
  if (startStep > 0)
    iFlow = halfStep + startStep * stepSize;

  for (; (iFlow < halfStep) && (iFlow < numFlows); iFlow++) {
    normalizer = stepNormalizers[0];
    normalizedMeasurements[iFlow] /= normalizer;
  }

  for (int iStep = startStep + 1; iStep < numSteps; iStep++) {
    for (int pos = 0; (pos < stepSize) && (iFlow < numFlows); pos++, iFlow++) {
      normalizer = stepNormalizers[iStep - 1] * (stepSize - pos) + stepNormalizers[iStep] * pos;
      normalizer /= stepSize;
      normalizedMeasurements[iFlow] /= normalizer;
    }
  }

  for (; iFlow < numFlows; iFlow++) {
    normalizer = stepNormalizers[numSteps - 1];
    normalizedMeasurements[iFlow] /= normalizer;
  }

}




// New improved normalization strategy
void DPTreephaser::NormalizeAndSolve3(BasecallerRead& well, int maxFlows)
{
  int stepSize = 50;

  for (int iStep = 1; iStep < 100; iStep++) {

      int solveLength = (iStep+1) * stepSize;
      solveLength = std::min(solveLength,maxFlows);

      Solve(well,solveLength);
      well.FitBaselineVector(iStep,stepSize);
      well.FitNormalizerVector(iStep,stepSize);

      if (solveLength == maxFlows)
          break;
  }

  Solve(well,maxFlows);
}



// Old normalization, but uses BasecallerRead object
void DPTreephaser::NormalizeAndSolve4(BasecallerRead& well, int maxFlows)
{
  for (int it = 0; it < 7; it++) {
    int solveFlow = 100 + 20 * it;
    if (solveFlow < maxFlows) {
      Solve(well,solveFlow);
      well.Normalize(11, 80 + 20 * it);
    }
  }
  Solve(well,maxFlows);
}




// Sliding window adaptive normalization
void DPTreephaser::NormalizeAndSolve5(BasecallerRead& well, int maxFlows)
{
  int stepSize = 50;

  for (int iStep = 1; iStep < 100; iStep++) {

      int solveLength = (iStep+1) * stepSize;
      solveLength = std::min(solveLength, maxFlows);

      int restartLength = std::max(solveLength - 100, 0);
      int restartStep = 0; //std::max(iStep - 3, 0);

      Solve(well,solveLength, restartLength);
      well.FitBaselineVector(iStep, stepSize, restartStep);
      well.FitNormalizerVector(iStep, stepSize, restartStep);

      if (solveLength == maxFlows)
          break;
  }

  Solve(well,maxFlows);
}




void BasecallerRead::Normalize(int fromFlow, int toFlow)
{
  const float HpScaleWeight[5] = { 0, 1, 0.8, 0.64, 0.37 };

  double xy = 0;
  double yy = 0;

  for (int iFlow = fromFlow; ((iFlow < toFlow) && (iFlow < numFlows)); iFlow++) {
    if ((solution[iFlow] > 0) && (((int)solution[iFlow]) < 5)) {

//      xy += HpScaleWeight[(int)solution[iFlow]] * measurements[iFlow] * prediction[iFlow];
      xy += HpScaleWeight[(int)solution[iFlow]] * normalizedMeasurements[iFlow] * prediction[iFlow];
      yy += HpScaleWeight[(int)solution[iFlow]] * prediction[iFlow] * prediction[iFlow];
    }
  }

  double Divisor = 1;
  if ((yy > 0) && (xy > 0))
    Divisor = xy / yy;

  for (int iFlow = 0; iFlow < numFlows; iFlow++)
//    normalizedMeasurements[iFlow] = measurements[iFlow] / Divisor;
    normalizedMeasurements[iFlow] /= Divisor;

  miscNormalizer *= Divisor;
}


//-------------------------------------------------------------------------

void DPTreephaser::advanceState(TreephaserPath *child, const TreephaserPath *parent, int nuc, int toflow)
{
  if (child != parent) {  // The advance is not done in place

    // Advance flow
    child->flow = parent->flow;
    while ((child->flow < toflow) && (flowOrder[child->flow] != nuc))
      child->flow++;

    // Initialize window
    child->windowStart = parent->windowStart;
    child->windowEnd = parent->windowEnd;

    if (parent->flow == child->flow) {

      // This nuc simply prolongs current homopolymer, inherits state from parent
      for (int iFlow = child->windowStart; iFlow < child->windowEnd; iFlow++)
        child->state[iFlow] = parent->state[iFlow];
      return;
    }

    // This nuc begins a new homopolymer
    float alive = 0;
    for (int iFlow = parent->windowStart; iFlow < child->windowEnd; iFlow++) {

      // State progression according to phasing model
      if (iFlow < parent->windowEnd)
        alive += parent->state[iFlow];
      child->state[iFlow] = alive * transitionBase[nuc][iFlow];
      alive *= transitionFlow[nuc][iFlow];

      // Window maintenance
      if ((iFlow == child->windowStart) && (iFlow < (child->windowEnd - 1)) && (child->state[iFlow] < minFraction))
        child->windowStart++;

      if ((iFlow == (child->windowEnd - 1)) && (child->windowEnd < toflow) && (alive > minFraction))
        child->windowEnd++;
    }

  } else {    // The advance is done in place

    // Advance flow
    int oldFlow = child->flow;
    while ((child->flow < toflow) && (flowOrder[child->flow] != nuc))
      child->flow++;

    if (oldFlow == child->flow) // Same homopolymer, no changes in the state
      return;

    // This nuc begins a new homopolymer
    float alive = 0;
    int oldWindowStart = child->windowStart;
    int oldWindowEnd = child->windowEnd;
    for (int iFlow = oldWindowStart; iFlow < child->windowEnd; iFlow++) {

      // State progression according to phasing model
      if (iFlow < oldWindowEnd)
        alive += child->state[iFlow];
      child->state[iFlow] = alive * transitionBase[nuc][iFlow];
      alive *= transitionFlow[nuc][iFlow];

      // Window maintenance
      if ((iFlow == child->windowStart) && (iFlow < (child->windowEnd - 1)) && (child->state[iFlow] < minFraction))
        child->windowStart++;

      if ((iFlow == (child->windowEnd - 1)) && (child->windowEnd < toflow) && (alive > minFraction))
        child->windowEnd++;
    }
  }
}



void DPTreephaser::Simulate2(BasecallerRead& read, int maxFlows)
{
  read.prediction.assign(numFlows, 0);

  TreephaserPath *state = &(path[0]);

  state->flow = 0;
  state->state[0] = 1;
  state->windowStart = 0;
  state->windowEnd = 1;

  for (int mainFlow = 0; (mainFlow < numFlows) && (mainFlow < maxFlows); mainFlow++) {
    for (int iHP = 0; iHP < read.solution[mainFlow]; iHP++) {

      advanceState(state, state, flowOrder[mainFlow], maxFlows);

      float myAdjustment = 1.0;
      if (useNonLinHPs)
        myAdjustment = perNucHPadjustment[flowOrder[mainFlow]][iHP];

      for (int iFlow = state->windowStart; iFlow < state->windowEnd; iFlow++)
        read.prediction[iFlow] += state->state[iFlow] * myAdjustment;
    }
  }
}


void DPTreephaser::Simulate3(BasecallerRead &data, int maxFlows)
{
  for (int iFlow = 0; iFlow < numFlows; iFlow++)
    data.prediction[iFlow] = 0;

  path[0].state.resize(numFlows);
  path[0].state[0] = 1;
  int startFlowWindow = 0;
  int endFlowWindow = 1;

  for (int currentFlow = 0; (currentFlow < numFlows) && (currentFlow < maxFlows); currentFlow++) {
    for (int iHP = 0; iHP < data.solution[currentFlow]; iHP++) {

      if (iHP > 0) {
        for (int iFlow = startFlowWindow; (iFlow < endFlowWindow) && (iFlow < numFlows); iFlow++)
          data.prediction[iFlow] += path[0].state[iFlow];
        continue;
      }

      float alive = 0;
      int nextEndFlowWindow = endFlowWindow;
      for (int iFlow = startFlowWindow; iFlow < numFlows; iFlow++) {

        if (iFlow < endFlowWindow)
          alive += path[0].state[iFlow];
        path[0].state[iFlow] = alive * transitionBase[flowOrder[currentFlow]][iFlow];
        alive *= transitionFlow[flowOrder[currentFlow]][iFlow];

        data.prediction[iFlow] += path[0].state[iFlow];

        if (iFlow == startFlowWindow)
          if (path[0].state[iFlow] < 1e-4)
            startFlowWindow++;

        if (iFlow == (nextEndFlowWindow - 1)) {
          if (alive > 1e-4)
            nextEndFlowWindow++;
          else
            break;
        }
      }
      endFlowWindow = nextEndFlowWindow;
    }
  }
}




// Solve3 - main tree search procedure that determines the base sequence.
// Another temporary version, uses external class for storing read data

void DPTreephaser::Solve(BasecallerRead& read, int maxFlows, int restartFlows)
{
  assert(maxFlows <= numFlows);

  // Initialize stack: just one root path
  for (int iPath = 1; iPath < numPaths; iPath++)
    path[iPath].inUse = false;

  path[0].flow = 0;
  path[0].state[0] = 1;
  path[0].windowStart = 0;
  path[0].windowEnd = 1;
  path[0].pathMetric = 0;
  path[0].perFlowMetric = 0;
  path[0].residualLeftOfWindow = 0;
  path[0].dotCounter = 0;
  path[0].prediction.assign(numFlows, 0);
  path[0].solution.assign(numFlows, 0);
  path[0].inUse = true;

  int spaceOnStack = usePaths - 1;
  minDistance = maxFlows;

  if (restartFlows > 0) {
    // The solver will not attempt to solve initial restartFlows
    // - Simulate restartFlows instead of solving
    // - If it turns out that solving was finished before restartFlows, simply exit without any changes to the read.

    assert(read.solution.size() >= (unsigned int)restartFlows);
    int lastIncorporatingFlow = 0;

    for (int mainFlow = 0; (mainFlow < numFlows) && (mainFlow < restartFlows); mainFlow++) {

      path[0].solution[mainFlow] = read.solution[mainFlow];

      for (int iHP = 0; iHP < path[0].solution[mainFlow]; iHP++) {

        advanceState(&path[0], &path[0], flowOrder[mainFlow], maxFlows);

        float myAdjustment = 1.0;
        if (useNonLinHPs)
          myAdjustment = perNucHPadjustment[flowOrder[mainFlow]][iHP];

        for (int iFlow = path[0].windowStart; iFlow < path[0].windowEnd; iFlow++)
          path[0].prediction[iFlow] += path[0].state[iFlow] * myAdjustment;

        lastIncorporatingFlow = mainFlow;
      }
    }

    if (lastIncorporatingFlow < (restartFlows-10)) // This read ended before restartFlows. No point resolving it.
      return;

    for (int iFlow = 0; iFlow < path[0].windowStart; iFlow++) {
      float residual = read.normalizedMeasurements[iFlow] - path[0].prediction[iFlow];
      path[0].residualLeftOfWindow += residual * residual;
    }
  }

  // Initializing variables
  read.solution.assign(numFlows, 0);
  read.prediction.assign(numFlows, 0);

  // Main loop to select / expand / delete paths
  while (1) {

    // ------------------------------------------
    // Step 1: Prune the content of the stack and make sure there are at least 4 empty slots

    // Remove paths that are more than 'maxPathDelay' behind the longest one
    if (spaceOnStack < (numPaths - 3)) {
      int maxFlow = 0;
      for (int iPath = 0; iPath < usePaths; iPath++)
        if (path[iPath].inUse)
          maxFlow = max(maxFlow, path[iPath].flow);

      if (maxFlow > maxPathDelay) {
        for (int iPath = 0; iPath < usePaths; iPath++) {
          if (path[iPath].inUse && (path[iPath].flow < (maxFlow - maxPathDelay))) {
            path[iPath].inUse = false;
            spaceOnStack++;
          }
        }
      }
    }

    // If necessary, remove paths with worst perFlowMetric
    while (spaceOnStack < 4) {
      // find maximum per flow metric
      float ExtremeVal = -0.1;
      int newIdx = usePaths;
      for (int iPath = 0; iPath < usePaths; iPath++) {
        if (path[iPath].inUse && (path[iPath].perFlowMetric > ExtremeVal)) {
          ExtremeVal = path[iPath].perFlowMetric;
          newIdx = iPath;
        }
      }

      // killing path with largest per flow metric
      if (!(newIdx < usePaths)) {
        printf("Failed assertion in Treephaser\n");
        for (int iPath = 0; iPath < usePaths; iPath++) {
          if (path[iPath].inUse)
            printf("Path %d, inUse = true, perFlowMetric = %f\n", iPath, path[iPath].perFlowMetric);
          else
            printf("Path %d, inUse = false, perFlowMetric = %f\n", iPath, path[iPath].perFlowMetric);
        }
        fflush(NULL);
      }

      assert (newIdx < usePaths);
      path[newIdx].inUse = false;
      spaceOnStack++;
    }

    // ------------------------------------------
    // Step 2: Select a path to expand or break if there is none

    TreephaserPath *parent = NULL;
    float ExtremeVal = 1000;
    for (int iPath = 0; iPath < usePaths; iPath++) {
      if (path[iPath].inUse && (path[iPath].pathMetric < ExtremeVal)) {
        ExtremeVal = path[iPath].pathMetric;
        parent = &(path[iPath]);
      }
    }
    if (!parent)
      break;

    // ------------------------------------------
    // Step 3: Construct four expanded paths and calculate feasibility metrics
    assert (spaceOnStack >= 4);

    TreephaserPath *children[4];

    for (int nuc = 0, iPath = 0; nuc < 4; iPath++)
      if (!path[iPath].inUse)
        children[nuc++] = &(path[iPath]);

    float penalty[4] = { 0, 0, 0, 0 };

    for (int nuc = 0; nuc < 4; nuc++) {

      TreephaserPath *child = children[nuc];

      advanceState(child, parent, nuc, maxFlows);

      // Apply easy termination rules

      if (child->flow >= maxFlows) {
        penalty[nuc] = 25; // Mark for deletion
        continue;
      }

      int currentHP = parent->solution[child->flow];
      if (currentHP == maxHP) {
        penalty[nuc] = 25; // Mark for deletion
        continue;
      }

      child->pathMetric = parent->residualLeftOfWindow;
      child->residualLeftOfWindow = parent->residualLeftOfWindow;

      float myAdjustment = 1.0;
//      if (useNonLinHPs)
//        myAdjustment = perNucHPadjustment[nuc][currentHP];

      float penaltyN = 0;
      float penalty1 = 0;

      for (int iFlow = parent->windowStart; (iFlow < child->windowEnd) ; iFlow++) {

        child->prediction[iFlow] = parent->prediction[iFlow] + child->state[iFlow] * myAdjustment;

        float residual = read.normalizedMeasurements[iFlow] - child->prediction[iFlow];
        float sqResidual = residual * residual;

        // Metric calculation
        if (iFlow < child->windowStart) {
          child->residualLeftOfWindow += sqResidual;
          child->pathMetric += sqResidual;
        } else if (residual <= 0)
          child->pathMetric += sqResidual;

        if (residual <= 0)
          penaltyN += sqResidual;
        else if (iFlow < child->flow)
          penalty1 += sqResidual;
      }


      penalty[nuc] = penalty1 + negMultiplier * penaltyN;
      penalty1 += penaltyN;

      child->perFlowMetric = (child->pathMetric + 0.5 * penalty1) / child->flow;

    } //looping over nucs


    // Find out which nuc has the least penalty (the greedy choice nuc)
    int bestNuc = 0;
    if (penalty[bestNuc] > penalty[1])
      bestNuc = 1;
    if (penalty[bestNuc] > penalty[2])
      bestNuc = 2;
    if (penalty[bestNuc] > penalty[3])
      bestNuc = 3;

    // ------------------------------------------
    // Step 4: Use calculated metrics to decide which paths are worth keeping

    for (int nuc = 0; nuc < 4; nuc++) {

      TreephaserPath *child = children[nuc];

      // Path termination rules

      if (penalty[nuc] >= 20)
        continue;

      if (child->pathMetric > minDistance)
        continue;

      // This is the only rule that depends on finding the "best nuc"
      if ((penalty[nuc] - penalty[bestNuc]) >= expThreshold)
        continue;

      float newSignal = (read.normalizedMeasurements[child->flow] - parent->prediction[child->flow]) / child->state[child->flow];
      child->dotCounter = (newSignal < dotValue) ? (parent->dotCounter + 1) : 0;
      if (child->dotCounter > 1)
        continue;

      //            if ((nuc != bestNuc) && (child->dotCounter > 0))
      //                continue;

      // Path survived termination rules and will be kept on stack
      child->inUse = true;
      spaceOnStack--;

      // Fill out the remaining portion of the prediction
      for (int iFlow = 0; iFlow < parent->windowStart; iFlow++)
        child->prediction[iFlow] = parent->prediction[iFlow];

      for (int iFlow = child->windowEnd; iFlow < maxFlows; iFlow++)
        child->prediction[iFlow] = 0;

      // Fill out the solution
      child->solution = parent->solution;
      child->solution[child->flow]++;
    }

    // ------------------------------------------
    // Step 5. Check if the selected path is in fact the best path so far

    // Computing sequence squared distance
    float SequenceDist = parent->residualLeftOfWindow;
    for (int iFlow = parent->windowStart; iFlow < maxFlows; iFlow++) {
      float residual = read.normalizedMeasurements[iFlow] - parent->prediction[iFlow];
      SequenceDist += residual * residual;
    }

    // Updating best path
    if (SequenceDist < minDistance) {
      read.prediction.swap(parent->prediction);
      read.solution.swap(parent->solution);
      minDistance = SequenceDist;
    }

    parent->inUse = false;
    spaceOnStack++;

  } // main decision loop
}



// ------------------------------------------------------------------------
// Compute quality metrics
int DPTreephaser::ComputeQVmetrics(BasecallerRead& read)
{

  oneMerHeight.assign(numFlows, 1);

  int numBases = 0;
  for (int iFlow = 0; iFlow < numFlows; iFlow++)
    numBases += read.solution[iFlow];

  if (numBases == 0)
    return 0;

  penaltyMismatch.assign(numBases, 0);
  penaltyResidual.assign(numBases, 0);

  int toflow = numFlows;

  TreephaserPath *parent = &(path[0]);
  TreephaserPath *children[4] = { &(path[1]), &(path[2]), &(path[3]), &(path[4]) };

  parent->flow = 0;
  parent->state[0] = 1;
  parent->windowStart = 0;
  parent->windowEnd = 1;
  parent->prediction.assign(numFlows, 0);
  parent->solution.assign(numFlows, 0);

  int iBase = 0;
  float recentInPhaseSignal = 1;

  // main loop for base calling

  for (int mainFlow = 0; mainFlow < toflow; mainFlow++) {

    for (int iHP = 0; iHP < read.solution[mainFlow]; iHP++) {

      float penalty[4] = { 0, 0, 0, 0 };

      for (int nuc = 0; nuc < 4; nuc++) {

        TreephaserPath *child = children[nuc];

        advanceState(child, parent, nuc, toflow);

        // Apply easy termination rules

        if (child->flow >= toflow) {
          penalty[nuc] = 25; // Mark for deletion
          continue;
        }

        int currentHP = parent->solution[child->flow];
        if (currentHP == maxHP) {
          penalty[nuc] = 25; // Mark for deletion
          continue;
        }

        float myAdjustment = 1.0;
        if (useNonLinHPs)
          myAdjustment = perNucHPadjustment[nuc][currentHP];

        for (int iFlow = parent->windowStart; iFlow < child->windowEnd; iFlow++) {

          child->prediction[iFlow] = parent->prediction[iFlow] + child->state[iFlow] * myAdjustment;

          float residual = read.normalizedMeasurements[iFlow] - child->prediction[iFlow];

          if ((residual <= 0) || (iFlow < child->flow))
            penalty[nuc] += residual*residual;
        }

      } //looping over nucs


      // find current incorporating base
      int calledNuc = flowOrder[mainFlow];
      assert(children[calledNuc]->flow == mainFlow);

      recentInPhaseSignal = children[calledNuc]->state[mainFlow];

      // Get delta penalty to next best solution
      penaltyMismatch[iBase] = -1; // min delta penalty to earlier base hypothesis
      penaltyResidual[iBase] = 0;

      if( mainFlow - parent->windowStart > 0 )
        penaltyResidual[iBase] = penalty[calledNuc] / ( mainFlow - parent->windowStart );


      for (int nuc = 0; nuc < 4; nuc++) {
        if (nuc == calledNuc)
            continue;
        float iDeltaPenalty = penalty[calledNuc] - penalty[nuc];
        penaltyMismatch[iBase] = std::max( penaltyMismatch[iBase], iDeltaPenalty);
      }

      // Fill out the remaining portion of the prediction
      for (int iFlow = 0; iFlow < parent->windowStart; iFlow++)
        children[calledNuc]->prediction[iFlow] = parent->prediction[iFlow];

      for (int iFlow = children[calledNuc]->windowEnd; iFlow < toflow; iFlow++)
        children[calledNuc]->prediction[iFlow] = 0;

      // Called state is the starting point for next base
      TreephaserPath *swap = parent;
      parent = children[calledNuc];
      children[calledNuc] = swap;

      iBase++;
    }

    oneMerHeight[mainFlow] = recentInPhaseSignal;
    if (oneMerHeight[mainFlow] < 0.01)
      oneMerHeight[mainFlow] = 0.01;
  }

  return numBases;
}










// ----------------------------------------------------------------------
// New normalization strategy

void BasecallerRead::AdaptiveNormalizationOfPredictions(int numSteps, int stepSize)
{
  assert(numSteps > 0);
  int halfStep = stepSize / 2;

  // Additive correction

  float latestNormalizer = 0;
  int fromFlow = 0;
  int toFlow = stepSize;

  float stepAdditive[numSteps];
  float stepMultiplicative[numSteps];
  float medianSet[stepSize];

  for (int iStep = 0; iStep < numSteps; iStep++) {

    if (fromFlow >= numFlows)
      break;
    toFlow = std::min(toFlow, numFlows);

    if (toFlow < numFlows) {
      int medianCount = 0;
      for (int iFlow = fromFlow; iFlow < toFlow; iFlow++)
        if (prediction[iFlow] < 0.3)
          medianSet[medianCount++] = measurements[iFlow] - prediction[iFlow];

      if (medianCount > 5) {
        std::nth_element(medianSet, medianSet + medianCount/2, medianSet + medianCount);
        latestNormalizer = medianSet[medianCount / 2];
      }
    }

    stepAdditive[iStep] = latestNormalizer;

    fromFlow = toFlow;
    toFlow += stepSize;
  }


  int iFlow = 0;
  for (; (iFlow < halfStep) && (iFlow < numFlows); iFlow++)
    normalizedMeasurements[iFlow] = measurements[iFlow] - stepAdditive[0];

  for (int iStep = 1; iStep < numSteps; iStep++) {
    for (int pos = 0; (pos < stepSize) && (iFlow < numFlows); pos++, iFlow++)
      normalizedMeasurements[iFlow] = measurements[iFlow]
          - (stepAdditive[iStep - 1] * (stepSize - pos) + stepAdditive[iStep] * pos) / stepSize;
  }

  for (; iFlow < numFlows; iFlow++)
    normalizedMeasurements[iFlow] = measurements[iFlow] - stepAdditive[numSteps - 1];


  // Multiplicative correction

  latestNormalizer = 1;
  fromFlow = 0;
  toFlow = stepSize;

  for (int iStep = 0; iStep < numSteps; iStep++) {

    if (fromFlow >= numFlows)
      break;
    toFlow = std::min(toFlow, numFlows);

    if (toFlow < numFlows) {
      int medianCount = 0;
      for (int iFlow = fromFlow; iFlow < toFlow; iFlow++)
        if ((prediction[iFlow] > 0.5) && (normalizedMeasurements[iFlow] > 0))
          medianSet[medianCount++] = normalizedMeasurements[iFlow] / prediction[iFlow];

      if (medianCount > 5) {
        std::nth_element(medianSet, medianSet + medianCount/2, medianSet + medianCount);
        if (medianSet[medianCount / 2] > 0)
          latestNormalizer = medianSet[medianCount / 2];
      }
    }

    stepMultiplicative[iStep] = latestNormalizer;

    fromFlow = toFlow;
    toFlow += stepSize;
  }

  iFlow = 0;
  for (; (iFlow < halfStep) && (iFlow < numFlows); iFlow++) {
    prediction[iFlow] *= stepMultiplicative[0];
    prediction[iFlow] += stepAdditive[0];
  }

  for (int iStep = 1; iStep < numSteps; iStep++) {
    for (int pos = 0; (pos < stepSize) && (iFlow < numFlows); pos++, iFlow++) {
      prediction[iFlow] *= (stepMultiplicative[iStep - 1] * (stepSize - pos) + stepMultiplicative[iStep] * pos) / stepSize;
      prediction[iFlow] += (stepAdditive[iStep - 1] * (stepSize - pos) + stepAdditive[iStep] * pos) / stepSize;
    }
  }

  for (; iFlow < numFlows; iFlow++) {
    prediction[iFlow] *= stepMultiplicative[numSteps - 1];
    prediction[iFlow] += stepAdditive[numSteps - 1];
  }

}



