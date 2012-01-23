/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef KEYCLASSIFYTAUEJOB_H
#define KEYCLASSIFYTAUEJOB_H
#include <vector>
#include <string>

#include "PJob.h"
#include "Traces.h"
#include "KClass.h"
#include "KeyClassifier.h"
#include "Mask.h"
#include "Utils.h"

class KeyClassifyTauEJob : public PJob {

 public:
  
  KeyClassifyTauEJob() { mMask = NULL;
    mWells = NULL;
    mKeys = NULL;
    mTime = NULL;
    mReport = NULL;
    mTraces = NULL;
    mTauEMesh = NULL;
    mMinSnr = 0;
    mColStart = 0;
    mColEnd = 0;
    mRowEnd = 0;
    mRowStart = 0;
    mTauEEst = 0;
    mFlows = 0;
    mBg = NULL;
  }

  void Init(int rowStart, int rowEnd, int colStart, int colEnd,
	    double snr, Mask *mask, std::vector<KeyFit> *wells,
	    std::vector<KeySeq> *keys,  Col<double> *time,
            double basicTauE, Col<double> *incorp,
            ZeromerModelBulk<double> *bg,
	    std::vector<KeyReporter<double> *> *report,
            //	    std::vector<Traces> *traces,
            TraceStore<double> *traces,
            int flows,
            GridMesh<SampleQuantiles<double> > *tauEMesh, 
            SampleQuantiles<float> &tauEQuant) {
    mRowStart = rowStart;
    mRowEnd = rowEnd;
    mColStart = colStart;
    mColEnd = colEnd;
    mMinSnr = snr;
    mKeys = keys;
    mIncorp = incorp;
    mReport = report;
    mMask = mask;
    mFlows = flows;
    mWells = wells;
    mTraces = traces;
    mTime = time;
    mTauEMesh = tauEMesh;
    mTauEEst = 0;
    mBg = bg;
    int count = 0;
    int limit = min(20,(int)(mTauEMesh->GetRowStep() * mTauEMesh->GetColStep() * .1));
    for (size_t i = 0; i < mTauEMesh->GetNumBin(); i++) {
      if (mTauEMesh->GetItem(i).GetNumSeen() > limit) {
        count++;
        mTauEEst += mTauEMesh->GetItem(i).GetQuantile(.5);
      }
    }
    if (count != 0 && (count * limit > 50)) {
      mTauEEst = mTauEEst / count;
    }
    else if (tauEQuant.GetNumSeen() > 50) {
      mTauEEst = tauEQuant.GetMedian();
    }
    else {
      mTauEEst = basicTauE; // if all else fails..
    }
  };

  /** Process work. */
  virtual void Run() {
    for (int rowIx = mRowStart; rowIx < mRowEnd; rowIx++) {
      for (int colIx = mColStart; colIx < mColEnd; colIx++) {
	size_t idx = mTraces->WellIndex(rowIx, colIx); 
	(*mWells)[idx].wellIdx = idx;
	mKc.ClassifyWellKnownTau((*mMask), (*mBg), (*mKeys), (*mTraces), mFlows, (*mTime), (*mIncorp),
                                 (*mReport), (*mTauEMesh), mTauEEst, mMinSnr, mDist, mValues, (*mWells)[idx]);
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
  //  std::vector<Traces> *mTraces;
  TraceStore<double> *mTraces;
  double mMinSnr;
  int mFlows;
  std::vector<double> mDist;
  std::vector<std::vector<float> *> mValues;
  std::vector<KeyFit> *mWells;
  KClass mKc;
  double mTauEEst;
  //  ZeromerDiff<double> mBg;
  ZeromerModelBulk<double> *mBg;
  std::vector<KeySeq> *mKeys;
  Col<double> *mTime;
  Col<double> *mIncorp;
  std::vector<KeyReporter<double> *> *mReport;
  GridMesh<SampleQuantiles<double> > *mTauEMesh;
};

#endif // KEYCLASSIFYTAUEJOB_H
