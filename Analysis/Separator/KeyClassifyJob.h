/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef KEYCLASSIFYJOB_H
#define KEYCLASSIFYJOB_H
#include <vector>
#include <string>

#include "PJob.h"
#include "Traces.h"
#include "KClass.h"
#include "KeyClassifier.h"
#include "Mask.h"
#include "Utils.h"

class KeyClassifyJob : public PJob {

 public:
  
  KeyClassifyJob() { 
		mMask = NULL; 
		mWells = NULL; 
		mKeys = NULL;
    mTime = NULL; 
		mReport = NULL; 
		mTraces = NULL;
		mMinSnr = 0;
		mColStart = 0;
		mColEnd = 0;
		mRowEnd = 0;
		mRowStart = 0;
                mFlows = 0;
	}
  
  void Init(int rowStart, int rowEnd, int colStart, int colEnd,
	    double snr, Mask *mask, std::vector<KeyFit> *wells,
	    std::vector<KeySeq> *keys,  Col<double> *time,
	    std::vector<KeyReporter<double> *> *report,
            TraceStore<double> *traces, int flows) {
	    //std::vector<Traces> *traces) {
    mRowStart = rowStart;
    mRowEnd = rowEnd;
    mColStart = colStart;
    mColEnd = colEnd;
    mMinSnr = snr;
    mKeys = keys;
    mReport = report;
    mMask = mask;
    mWells = wells;
    mTraces = traces;
    mTime = time;
    mFlows = flows;
  };

  /** Process work. */
  virtual void Run() {
    for (int rowIx = mRowStart; rowIx < mRowEnd; rowIx++) {
      for (int colIx = mColStart; colIx < mColEnd; colIx++) {
	size_t idx = mTraces->WellIndex(rowIx, colIx);
	(*mWells)[idx].wellIdx = idx;
	mKc.InitialClassifyWell((*mMask), mBg, (*mKeys), (*mTraces), mFlows, (*mTime), (*mReport), mMinSnr, mDist, mValues, (*mWells)[idx]);
      }
    }
  }
  /** Cleanup any resources. */
  virtual void TearDown() {}
  /** Exit this pthread (killing thread) */
  void Exit() {
    pthread_exit(NULL);
  }

 private:
  int mRowStart, mRowEnd, mColStart, mColEnd;
  Mask *mMask;
  //std::vector<Traces> *mTraces;
  TraceStore<double> *mTraces;
  double mMinSnr;
  std::vector<double> mDist;
  std::vector<std::vector<float> *> mValues;
  std::vector<KeyFit> *mWells;
  KClass mKc;
  int mFlows;
  ZeromerDiff<double> mBg;
  std::vector<KeySeq> *mKeys;
  Col<double> *mTime;
  std::vector<KeyReporter<double> *> *mReport;
};

#endif // KEYCLASSIFYJOB_H
