/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef SYNCHDATCONVERT_H
#define SYNCHDATCONVERT_H

#include "Mask.h"
#include "GridMesh.h"
#include "T0Calc.h"
#include "T0Model.h"
#include "TraceChunk.h"
#include "Utils.h"
#include "HandleExpLog.h"
#include "SynchDat.h"

class TraceConfig {

public:
  TraceConfig() {
    start_detailed_time = -5;
    stop_detailed_time = 16;
    left_avg = 5;
    rate_sigma_intercept = 0;
    rate_sigma_slope = 0.6;
    // rate_sigma_intercept = 0.9160620;
    // rate_sigma_slope = 0.7400019;
    // rate_sigma_intercept = -0.2577416f;
    // rate_sigma_slope = 1.0195878f;
    t0_tmid_intercept = 1.5;
    //    t0_tmid_intercept = .5;
    t0_tmid_slope = 1.0;
    time_start_slop = 0;
    t0_prior_pre_millis = 333; // about 5 frames at 15fps
    t0_prior_post_millis = 333; // about 5 frames at 15fps
    t0_prior_weight = 0.0f;
    row_step = 100;
    col_step = 100;
    doTopCoder = false;
    doDebug = false;
    numEvec = 6;
    precision = 10.0;
    numFlows = -1;
    numCores = 4;
    errCon = 0;
    rankGood = 0;
    pivot = 0;
    t0_hard = 0;
    tmid_hard = 0;
    sigma_hard = 0;
    use_hard_est = false;
    grind_acq_0 = 0;
    t0_hard_end = 3000;
    isThumbnail = false;
  }

  int start_detailed_time; ///< From TimeCompression::SetUpTime()
  int stop_detailed_time; ///< From TimeCompression::SetUpTime()
  int left_avg; ///< From TimeCompression::SetUpTime()
  int row_step;
  int col_step;
  float rate_sigma_intercept; ///< intercept for estimating sigma from 1/slope
  float rate_sigma_slope; ///< coefficient for estimating sigma from 1/slope
  float t0_tmid_intercept; ///< intercept for estimating t_mid_nuc from t0
  float t0_tmid_slope; ///< coefficient for estimating t_mid_nuc from t0
  float time_start_slop; ///< minimum frames back from t0
  int t0_prior_pre_millis;
  int t0_prior_post_millis;
  float t0_prior_weight;
  bool doTopCoder;
  bool  doDebug;
  int numEvec;
  double precision;
  int numFlows;
  string compressionType;
  int numCores;
  double errCon;
  int rankGood;
  double pivot;
  double t0_hard;
  double tmid_hard;
  double sigma_hard;
  int t0_hard_end;
  bool use_hard_est;
  std::string bg_param;
  int grind_acq_0;
  bool isThumbnail;
};

template <typename shortvec> void GenerateBfT0Prior(TraceConfig &config,
                       shortvec &img, 
                       int baseFrameRate,
                       size_t numRow,
                       size_t numCol,
                       size_t numFrame,
                       int *timeStamps,
                       size_t rowStep,
		       size_t colStep,
                       Mask *mask,
                       T0Calc &t0,
                       GridMesh<T0Prior> &t0Prior) {
  t0.SetWindowSize(4);
  t0.SetMinFirstHingeSlope(-5.0/(float) baseFrameRate);
  t0.SetMaxFirstHingeSlope(300.0/(float) baseFrameRate);
  t0.SetMinSecondHingeSlope(-20000.0/(float) baseFrameRate);
  t0.SetMaxSecondHingeSlope(-10/(float) baseFrameRate);
  t0.Init(numRow, numCol, numFrame, rowStep, colStep, 1);
  t0.SetTimeStamps(timeStamps, numFrame);
  if (mask != NULL) {
    t0.SetMask(mask);
  }
  t0.SetStepSize(10);
  t0.FillInT0Prior(t0Prior, config.t0_prior_pre_millis, config.t0_prior_post_millis, config.t0_prior_weight, config.t0_hard_end);
  t0.CalcAllSumTrace(img);
  t0.CalcT0FromSum();
  if (config.doDebug) { 
    string refFile = "t0_bf_reference.txt";
    ofstream out(refFile.c_str());
    t0.WriteResults(out);
    out.close();
  }
}

template <typename shortvec> void GenerateAcqT0Prior(TraceConfig &config,
                        shortvec &img, 
                        int baseFrameRate,
                        size_t numRow,
                        size_t numCol,
                        size_t numFrame,
                        int *timeStamps,
                        size_t rowStep,
                        size_t colStep,
                        Mask *mask,
                        T0Calc &t0,
                        GridMesh<T0Prior> &t0Prior) {
  /* How wide of a window on either side to use for linear model. */
  t0.SetWindowSize(5);
  /* Set constraints on allowable slope. */
  t0.SetMinFirstHingeSlope(-10/(float) baseFrameRate);
  t0.SetMaxFirstHingeSlope(1/(float) baseFrameRate);
  t0.SetMinSecondHingeSlope(2/(float) baseFrameRate);
  t0.SetMaxSecondHingeSlope(100/(float) baseFrameRate);
  /* Configure the size of the image and the grid size we want to look at. */
  t0.Init(numRow, numCol, numFrame, rowStep, colStep, 1);
  /* What time is each frame. */
  t0.SetTimeStamps(timeStamps, numFrame);
  t0.SetStepSize(10);
  if (mask != NULL) {
    t0.SetMask(mask);
  }
  t0.SetT0Prior(t0Prior);
  /* Calculate the sum for all regions to get avg trace. */
  t0.CalcAllSumTrace(img);
  /* Look for point with best reduction in ssq error using hinge over line. */
  t0.CalcT0FromSum();
  /* Output some statistics for debugging. */
  if (config.doDebug) {
    string refFile = "t0_dat_reference.txt";
    ofstream out(refFile.c_str());
    t0.CalculateSlopePostT0(2);
    t0.WriteResults(out);
    out.close();
  }
}

void EstimateSigmaValue(T0Calc &t0, SigmaTMidNucEstimation &sigmaEst, GridMesh<SigmaEst> &sigmaTMid, int numNeighbors=2) {
  assert(sigmaTMid.GetNumBin() == t0.GetNumRegions());
  t0.CalculateSlopePostT0(numNeighbors);
  for (size_t i = 0; i < t0.GetNumRegions(); i++) {
    float slopeEst = t0.GetSlope(i);
    float t0Est = t0.GetT0(i);
    SigmaEst &est = sigmaTMid.GetItem(i);
    est.mSigma = 0;
    est.mTMidNuc = 0;
    if (t0Est > 0 && slopeEst > 0) {
      est.mT0 = t0Est;
      est.mRate = slopeEst;
      sigmaEst.Predict(t0Est, slopeEst, est.mSigma, est.mTMidNuc);
    }
    else {
      int rowStart,rowEnd,colStart, colEnd;
      t0.GetRegionCoords(i, rowStart, rowEnd, colStart, colEnd);
    }
  }
}

#endif // SYNCHDATCONVERT_H
