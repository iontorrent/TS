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
#define SEP_FRAME_END 18
#define SEP_USEFUL_SIGNAL 20
template <class T>
class NucDarkMatterReporter : public KeyReporter<T> {

 public:

  NucDarkMatterReporter() {
    m0merNucs.resize(4);
    m1merNucs.resize(4);
    mMinPeak = 30;
    mMinSnr = 10;
    mMinSeen = 50;
  }

  void SetTraceStore(TraceStore<T> *store) {
    mStore = store;
  }

  void SetKeys(std::vector<KeySeq> *keys) {
    mKeys = keys;
  }

  void Report(const KeyFit &fit, 
              const Mat<T> &wellFlows,
              const Mat<T> &refFlows,
              const Mat<T> &predicted) {
    if (m0merNucs[0].size() == 0) {
      for (size_t nucIx = 0; nucIx < m0merNucs.size(); nucIx++) {
        m0merNucs[nucIx].resize(wellFlows.n_rows);
        m1merNucs[nucIx].resize(wellFlows.n_rows);
        for (size_t frameIx = 0; frameIx < m0merNucs[nucIx].size(); frameIx++) {
          m0merNucs[nucIx][frameIx].Clear();
          m0merNucs[nucIx][frameIx].Init(1000);
          m1merNucs[nucIx][frameIx].Clear();
          m1merNucs[nucIx][frameIx].Init(1000);
        }
      }
    }

    if (fit.keyIndex >= 0 && fit.peakSig >= mMinPeak && fit.snr >= mMinSnr) {
      for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
        if (mKeys->at(fit.keyIndex).flows[flowIx] == 0) {
          int nucIx = mStore->GetNucForFlow(flowIx);
          for (size_t frameIx = 0; frameIx < wellFlows.n_rows; frameIx++) {
            m0merNucs[nucIx][frameIx].AddValue(wellFlows.at(frameIx,flowIx) - predicted.at(frameIx,flowIx));
          }
        }
        else if (mKeys->at(fit.keyIndex).flows[flowIx] == 1) {
          int nucIx = mStore->GetNucForFlow(flowIx);
          for (size_t frameIx = 0; frameIx < wellFlows.n_rows; frameIx++) {
            m1merNucs[nucIx][frameIx].AddValue(wellFlows.at(frameIx,flowIx) - predicted.at(frameIx,flowIx));
          }
        }
      }
    }
  }

  void GetDarkMatter(Mat<T> &matter, Mat<T> &onemer) {
    matter.set_size(mStore->GetNumFrames(), mStore->GetNumNucs());
    matter.fill(0);

    onemer.set_size(mStore->GetNumFrames(), mStore->GetNumNucs());
    onemer.fill(0);

    for (size_t nucIx = 0; nucIx < m0merNucs.size(); nucIx++) {
      if (m0merNucs[nucIx].size() > 0 && m0merNucs[nucIx][0].GetNumSeen() > mMinSeen) {
        for (size_t frameIx = 0; frameIx < m0merNucs[nucIx].size(); frameIx++) {
          matter(frameIx, nucIx) = m0merNucs[nucIx][frameIx].GetMedian();
        }
      }
    }

    for (size_t nucIx = 0; nucIx < m0merNucs.size(); nucIx++) {
      if (m1merNucs[nucIx].size() > 0 && m1merNucs[nucIx][0].GetNumSeen() > mMinSeen) {
        for (size_t frameIx = 0; frameIx < m1merNucs[nucIx].size(); frameIx++) {
          onemer(frameIx,nucIx) = m1merNucs[nucIx][frameIx].GetMedian();
        }
      }
    }
  }

  int mMinSeen;  
  double mMinPeak;
  double mMinSnr;
  TraceStore<T> *mStore;
  std::vector<KeySeq> *mKeys;
  std::vector<std::vector<SampleQuantiles<T> > > m0merNucs;
  std::vector<std::vector<SampleQuantiles<T> > > m1merNucs;
};

class KeyClassifyTauEJob : public PJob {

 public:
  
  KeyClassifyTauEJob() { mMask = NULL;
    mWells = NULL;
    mKeys = NULL;
    mTraces = NULL;
    mReport = NULL;
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
    mTime.set_size(time->n_rows);
    mTime[0] = time->at(0);
    for (size_t i = 1; i < mTime.n_rows; i++) {
      //mTime[i] = (time->at(i) - time->at(i-1)) / 2.0f;
      mTime[i] = (time->at(i) - time->at(i-1));
    }
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
    std::vector<KeyReporter<double> *> firstReporter;
    NucDarkMatterReporter<double> fitReporter;
    fitReporter.SetTraceStore(mTraces);
    fitReporter.SetKeys(mKeys);
    firstReporter.push_back(&fitReporter);
    for (int rowIx = mRowStart; rowIx < mRowEnd; rowIx++) {
      for (int colIx = mColStart; colIx < mColEnd; colIx++) {
	size_t idx = mTraces->WellIndex(rowIx, colIx); 
	(*mWells)[idx].wellIdx = idx;
	mKc.ClassifyWellKnownTau((*mMask), (*mBg), (*mKeys), (*mTraces), mFlows, mTime, NULL, NULL, SEP_FRAME_END,
                                 firstReporter, (*mTauEMesh), mTauEEst, mMinSnr, mDist, mValues, (*mWells)[idx]);
      }
    }
    Mat<double> darkMatter, onemers;
    fitReporter.GetDarkMatter(darkMatter, onemers);
    Mat<double> D = onemers - darkMatter;
    Col<double> signal = mean(D, 1);
    size_t frameEnd = min((size_t)SEP_FRAME_END, (size_t)signal.n_rows);
    while (signal[frameEnd] > SEP_USEFUL_SIGNAL && frameEnd < signal.n_rows -1) {
      frameEnd++;
    }
    for (size_t nucIx = 0; nucIx < onemers.n_cols; nucIx++) {
      double val = norm(onemers.col(nucIx),2);
      if (val > 0) {
        onemers.col(nucIx) = onemers.col(nucIx) / val;
      }
    }
    for (int rowIx = mRowStart; rowIx < mRowEnd; rowIx++) {
      for (int colIx = mColStart; colIx < mColEnd; colIx++) {
	size_t idx = mTraces->WellIndex(rowIx, colIx); 
	(*mWells)[idx].wellIdx = idx;
	mKc.ClassifyWellKnownTau((*mMask), (*mBg), (*mKeys), (*mTraces), mFlows, mTime, &darkMatter, &onemers,frameEnd,
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
  Col<double> mTime;
  Col<double> *mIncorp;
  std::vector<KeyReporter<double> *> *mReport;
  GridMesh<SampleQuantiles<double> > *mTauEMesh;
};

#endif // KEYCLASSIFYTAUEJOB_H
