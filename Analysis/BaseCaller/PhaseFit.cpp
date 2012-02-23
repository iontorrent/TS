/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <iomanip>
#include "PhaseFit.h"
#include "Stats.h"

PhaseFit::PhaseFit() {
    read.resize(0);
    signal.resize(0);
    residual_raw_vec.resize(0);
    residual_weighted_vec.resize(0);
    err.resize(0);
    flowString = "";
    flowCycle.resize(0);
    concentration.resize(0);
    cf.resize(0);
    ie.resize(0);
    dr.resize(0);
    nFlow = 0;
    droopType = ONLY_WHEN_INCORPORATING;
    maxAdv = 0;
    droopAdvancerFirst.resize(0);
    extendAdvancerFirst.resize(0);
    droopAdvancer.resize(0);
    extendAdvancer.resize(0);
    flowWeight.resize(0);
    ignoreHPs = false;
    extraTaps = 0;
    residualType = SQUARED;
    residualSummary = MEAN;
}

void PhaseFit::InitializeFlowSpecs(
      string                  _flowString,
      vector<weight_vec_t>    _concentration,
      weight_vec_t            _cf,
      weight_vec_t            _ie,
      weight_vec_t            _dr,
      unsigned int            _nFlow,
      DroopType               _droopType,
      unsigned int            _maxAdv
) {
  read.resize(0);
  signal.resize(0);

  flowString = _flowString;
  flowCycle.resize(flowString.length());
  for(unsigned int iFlow=0; iFlow<flowString.length(); iFlow++)
    flowCycle[iFlow] = charToNuc(flowString[iFlow]);

  concentration = _concentration;
  cf = _cf;
  ie = _ie;
  dr = _dr;
  nFlow = _nFlow;
  droopType = _droopType;
  maxAdv = _maxAdv;
}

void PhaseFit::AddRead(weight_vec_t &seqFlow, weight_vec_t &sig, weight_t maxErr) {

  // Convert vector of continuous values to ints
  hpLen_vec_t hpFlow;
  vector<unsigned int> untrustedFlow;
  weight_vec_t::iterator iSeqFlow=seqFlow.begin();
  for(unsigned int iFlow=0; iSeqFlow != seqFlow.end(); iFlow++, iSeqFlow++) {
    // Round the signal
    hpLen_t rounded = (hpLen_t) floor(*iSeqFlow + 0.5);
    hpFlow.push_back(rounded);
   
    // Check if it was larger than we'd like
    weight_t err = abs(*iSeqFlow - (weight_t) rounded);
    if(err > maxErr)
      untrustedFlow.push_back(iFlow);
  }

  // Set weights
  weight_vec_t newHpWeight(seqFlow.size(),1);
  if(untrustedFlow.size() > 0) {
    unsigned int first_to_ignore = std::max(((int)untrustedFlow[0])-4,1);
    for(weight_vec_t::iterator iHpWeight=newHpWeight.begin()+first_to_ignore; iHpWeight != newHpWeight.end(); iHpWeight++)
      *iHpWeight = 0;
  }

//cout << "Weights:  ";
//for(unsigned int i=0; i<newHpWeight.size(); i++)
//  cout << ", " << setiosflags(ios::fixed) << setprecision(2) << newHpWeight[i];
//cout << "\n";

  // Set up the read
  PhaseSim newRead;
  newRead.setDroopType(droopType);
  newRead.setFlowCycle(flowString);
  newRead.setSeq(hpFlow);
  newRead.setAdvancerContexts(maxAdv);
  newRead.setExtraTaps(extraTaps);

  // Store the results
  read.push_back(newRead);
  hpWeight.push_back(newHpWeight);
  signal.push_back(sig);
}

void PhaseFit::AddRead(string seq, weight_vec_t &sig) {
  PhaseSim newRead;

  newRead.setDroopType(droopType);
  newRead.setFlowCycle(flowString);
  newRead.setSeq(seq);
  newRead.setAdvancerContexts(maxAdv);
  newRead.setExtraTaps(extraTaps);

  read.push_back(newRead);
  signal.push_back(sig);
}

int PhaseFit::LevMarFit(int max_iter, int nParam, float *params) {
  unsigned int nRead = signal.size();
  int nData = nRead * nFlow;

  // Call LevMarFitter::Initialize() - the 3rd arg is null as we don't need access to
  // the LevMarFitter's internal x array in the calls to Evaluate
  Initialize(nParam,nData,NULL);

  // Update LevMarFitter::residualWeight array if we are using flow-specific weighting or
  // if we are ignoring homopolymers of size > 1
  if( (!ignoreHPs) || (flowWeight.size() > 0) || (hpWeight.size() > 0) )
    updateResidualWeight(ignoreHPs, flowWeight, hpWeight);

  // Gather the observed data into an array of floats
  float *y   = new float[nData];
  unsigned int iY=0;
  for(unsigned int iRead=0; iRead<nRead; iRead++)
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++)
      y[iY++] = (float) signal[iRead][iFlow];

  err.resize(nData);
  int nIter = LevMarFitter::Fit(max_iter, y, params);

  // Store the (possibly weighted) residuals
  float *fval = new float[len];
  Evaluate(fval,params);
  residual_raw_vec.resize(nRead);
  residual_weighted_vec.resize(nRead);
  for(unsigned int iRead=0,iY=0; iRead<nRead; iRead++) {
    residual_raw_vec[iRead].resize(nFlow);
    residual_weighted_vec[iRead].resize(nFlow);
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iY++) {
      weight_t res = y[iY] - fval[iY];
      residual_raw_vec[iRead][iFlow] = res;
      residual_weighted_vec[iRead][iFlow] = residualWeight[iY] * res;
    }
  }
  delete [] fval;

  delete [] y;

  return(nIter);
}

void PhaseFit::updateResidualWeight(bool ignoreHPs, weight_vec_t &flowWeight, vector<weight_vec_t> &hpWeight) {
  // residualWeight is initialized in the call to LevMarFitter::Initialize(nparams,len,x)
  // so this function should be called just after that call and 
  
  if(len > 0) {
    // Initialize all weights to 1
    for(int iRes=0; iRes<len; iRes++)
      residualWeight[iRes] = 1;

    // Apply per-flow multipliers
    unsigned int nRead = read.size();
    if(flowWeight.size() > 0) {
      unsigned int iRes = 0;
      for(unsigned int iRead=0; iRead < nRead; iRead++) {
        for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iRes++) {
            if(iFlow >= flowWeight.size())
              throw("flowWeight vector is shorter than the number of flows in one or more of the reads.");
            residualWeight[iRes] *= flowWeight[iFlow];
        }
      }
    }

    // Apply per-flow multipliers
    if(hpWeight.size() > 0) {
      if(hpWeight.size()!=nRead)
        throw("hpWeight vector should be either empty or of the same length as the number of reads");
      unsigned int iRes = 0;
      for(unsigned int iRead=0; iRead < nRead; iRead++) {
        for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iRes++) {
            residualWeight[iRes] *= hpWeight[iRead][iFlow];
        }
      }
    }

    // Optionally ignore residuals for homopolymer runs of length larger than 1
    if(ignoreHPs) {
      unsigned int iRes = 0;
      for(unsigned int iRead=0; iRead < nRead; iRead++) {
        const hpLen_vec_t seqFlow = read[iRead].getSeqFlow();
        for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iRes++) {
          if(seqFlow[iFlow] > 1)
            residualWeight[iRes] = 0;
        }
      }
    }
  }

//cout << "weight for first read:\n";
//for(unsigned int iFlow=0, iRes=0; iFlow<nFlow; iFlow++, iRes++)
//  cout << ", " << setiosflags(ios::fixed) << setprecision(2) << residualWeight[iRes];
//cout << "\n";

}

void PhaseFit::PrintResidual(float *params) {
  unsigned int nRead = signal.size();
  unsigned int nData = nRead * nFlow;

  float *tmp = new float[nData];
  Evaluate(tmp,params);

  float *y   = new float[nData];
  for(unsigned int iRead=0, iY=0; iRead<nRead; iRead++)
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++)
      y[iY++] = (float) signal[iRead][iFlow];

  float res = CalcResidual(y, tmp, len);

  cout << "Residual = " << setiosflags(ios::fixed) << setprecision(3) << res << "\n";
  for(unsigned int iRead=0, iY=0; iRead<std::min(nRead,(unsigned int)2); iRead++) {
    cout << "raw res [" << iRead << "]: ";
    unsigned int iYsave = iY;
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++,iY++)
      cout << " " << setiosflags(ios::fixed) << setprecision(3) << y[iY] - tmp[iY];
    cout << "\n";
    iY = iYsave;
    cout << "wt res  [" << iRead << "]: ";
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++,iY++)
      cout << " " << setiosflags(ios::fixed) << setprecision(3) << residualWeight[iY] * (y[iY] - tmp[iY]);
    cout << "\n";
  }

  delete [] y;
  delete [] tmp;
}


float PhaseFit::CalcResidual(float *refVals, float *testVals, int numVals, LaVectorDouble *err_vec) {

  // Compute raw residuals and if necessary, return them
  for (int i=0;i < numVals;i++)
    err[i] = residualWeight[i] * (refVals[i]-testVals[i]);
  if (err_vec)
    for (int i=0;i < numVals;i++)
      (*err_vec)(i) = err[i];

  // Transform residuals
  switch(residualType) {
    case(SQUARED):
      for (int i=0;i < numVals;i++)
        err[i] = pow(err[i], 2.0);
    break;

    case(ABSOLUTE):
      for (int i=0;i < numVals;i++)
        err[i] = abs(err[i]);
    break;

    case(GEMAN_MCCLURE):
      for (int i=0;i < numVals;i++)
        err[i] = ionStats::geman_mcclure(err[i]);
    break;

    default:
      throw("Invalid residual type");
  }

  // Summarize the residuals
  float r=0;
  float numerator=0;
  float denominator=0;
  unsigned int nRead = signal.size();
  weight_vec_t readErr;

  switch(residualSummary) {
    case(MEAN):
      for (int i=0;i < numVals;i++)
        if(residualWeight[i] > 0) {
          numerator += err[i];
          denominator += residualWeight[i];
        }
      r = numerator/denominator;
    break;

    case(MEDIAN):
      readErr.reserve(numVals);
      for (int i=0;i < numVals;i++)
        if(residualWeight[i] > 0)
          readErr.push_back(err[i]);
      r = ionStats::median(readErr);
    break;

    case(MEAN_OF_MEDIAN):
      readErr.reserve(nFlow);
      for(unsigned int iRead=0, iErr=0; iRead<nRead; iRead++) {
        readErr.resize(0);
        for(unsigned int iFlow=0; iFlow<nFlow; iFlow++,iErr++) {
          if(residualWeight[iErr] > 0)
            readErr.push_back(err[iErr]);
        }
        if(readErr.size() > 0) {
          numerator += readErr.size() * ionStats::median(readErr);
          denominator += readErr.size();
        }
      }
      r = numerator/denominator;
    break;

    case(MEDIAN_OF_MEAN):
      readErr.reserve(nRead);
      readErr.resize(0);
      for(unsigned int iRead=0, iErr=0; iRead<nRead; iRead++) {
        numerator = 0;
        denominator = 0;
        for(unsigned int iFlow=0; iFlow<nFlow; iFlow++,iErr++) {
          if(residualWeight[iErr] > 0) {
            numerator += err[iErr];
            denominator += residualWeight[iErr];
          }
        }
        if(denominator > 0)
          readErr.push_back(numerator/denominator);
      }
      r = ionStats::median(readErr);
    break;

    default:
      throw("Invalid residual summary method");
  }

  return r;
}
