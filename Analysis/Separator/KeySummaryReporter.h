/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef KEYSUMMARYREPORTER_H
#define KEYSUMMARYREPORTER_H

#include <string>
#include <vector>
#include <pthread.h>
#include "KeyReporter.h"
#include "SampleStats.h"
#include "SampleQuantiles.h"
#include "KeyClassifier.h"
#include "GridMesh.h"
#include "AvgKeyReporter.h"
#include "IonErr.h"
#define KEY_SAMPLE_SIZE 1000

template <class T>
class KeyRegionSummary {

public:
  KeyRegionSummary(int sampSize=5000) {
    mSampleSize = sampSize;
  }
  
  void Init(int nRows, int nCols, int rowStep, int colStep, const KeySeq &key) {
    mKey = key;
    mGlobalTraces.resize(key.usableKeyFlows);
  }

  void Report(const KeyFit &fit, 
              const Mat<T> &wellFlows,
              const Mat<T> &refFlows,
              const Mat<T> &predicted) {
    for (size_t flowIx = 0; flowIx < mGlobalTraces.size() && flowIx < wellFlows.n_cols; flowIx++) {
      if (mGlobalTraces[flowIx].size() == 0 && wellFlows.n_rows > 0) {
        if (mGlobalTraces[flowIx].size() > 0) {
          ION_ABORT("Shouldn't have different sizes.");
        }
        mGlobalTraces[flowIx].resize(wellFlows.n_rows);
        for (size_t i = 0; i < mGlobalTraces[flowIx].size(); i++) {
          mGlobalTraces[flowIx][i].Init(mSampleSize);
        }
      }
      size_t size = min(mGlobalTraces[flowIx].size(), (size_t)wellFlows.n_rows);
      for (size_t frameIx = 0; frameIx < size; frameIx++) {
        float d = wellFlows.at(frameIx,flowIx) - predicted.at(frameIx,flowIx);
        mGlobalTraces[flowIx][frameIx].AddValue(d);
      }
    }
  }

  void Finish() {
    mGlobalAvg.resize(mGlobalTraces.size());
    for (size_t flowIx = 0; flowIx < mGlobalTraces.size(); flowIx++) {
      if (mGlobalAvg[flowIx].empty()) {
        size_t maxSize = 0;
        for (size_t i = 0; i < mGlobalTraces.size(); i++) {
          maxSize = max(mGlobalTraces[i].size(),maxSize);
        }
        mGlobalAvg[flowIx].resize(maxSize);
        for (size_t frameIx = 0; frameIx < mGlobalAvg[flowIx].size(); frameIx++) {
          if (mGlobalTraces[flowIx][frameIx].GetNumSeen() > 10) {
            mGlobalAvg[flowIx][frameIx].AddValue(mGlobalTraces[flowIx][frameIx].GetMedian());
          }
        }
      }
    }
  }

  std::vector<SampleStats<T> > &GetFlowTrace(int flowIx) {
    return mGlobalAvg[flowIx];
  }

private:
  int mSampleSize;
  KeySeq mKey;
  std::vector<std::vector<SampleQuantiles<float> > > mGlobalTraces;
  std::vector<std::vector<SampleStats<T> > > mGlobalAvg;
  const static int mMinCount = 100;
};

template <class T>
class KeySummaryReporter : public KeyReporter<T> {
  
public:
  
  void Init(const std::string &flowOrder, const std::string &prefix, 
            int nRows, int nCols,
            int rowStep, int colStep, const std::vector<KeySeq> &keys) {
    mFlowOrder = flowOrder;
    mPrefix = prefix;
    pthread_mutex_init(&mLock, NULL);
    mKeyRegions.resize(keys.size());
    for (size_t i = 0; i < mKeyRegions.size(); i++) {
      mKeyRegions[i].Init(nRows, nCols, rowStep, colStep, keys[i]);
      
    }
    mKeys = keys;
    mMinPeakSig.resize(keys.size());
    std::fill(mMinPeakSig.begin(), mMinPeakSig.end(), std::numeric_limits<double>::max() * -1);
  }

  ~KeySummaryReporter() {
    pthread_mutex_destroy(&mLock);
  }

  void SetMinKeyThreshold(int key, double val) {
    mMinPeakSig[key] = val;
  }

  
  void Report(const KeyFit &fit, 
	      const Mat<T> &wellFlows,
	      const Mat<T> &refFlows,
	      const Mat<T> &predicted) {
    if (fit.keyIndex < 0 || fit.mad > 50 || refFlows.n_cols == 0) {
      return;
    }
    if (fit.peakSig < mMinPeakSig[fit.keyIndex]) {
      return;
    }
    pthread_mutex_lock(&mLock);
    mKeyRegions[(int)fit.keyIndex].Report(fit, wellFlows, refFlows, predicted);
    pthread_mutex_unlock(&mLock);
  }

  void Finish() {
    for (size_t keyIx = 0; keyIx < mKeyRegions.size(); keyIx++) {
      mKeyRegions[keyIx].Finish();
    }
    for (size_t keyIx = 0; keyIx < mKeyRegions.size(); keyIx++) {
      string name = AvgKeyReporter<double>::NameForKey(mKeys[keyIx], mFlowOrder);
      string file = mPrefix + "/avgNukeTrace_" + name + ".txt"; 
      ofstream avgTrace;
      avgTrace.open(file.c_str());
      for (size_t flowIx = 0; flowIx < mKeys[keyIx].usableKeyFlows; flowIx++) {
        avgTrace << flowIx;
        std::vector<SampleStats<double> > &flowTrace = mKeyRegions[keyIx].GetFlowTrace(flowIx);
        for (size_t frameIx = 0; frameIx < flowTrace.size(); frameIx++) {
          avgTrace << " " << flowTrace[frameIx].GetMean();
        }
        avgTrace << endl;
      }
      vector<int> seen(mKeys[keyIx].flows.size(), 0);
      for (size_t flowIx = 0; flowIx < mKeys[keyIx].flows.size(); flowIx++) {
        if (mKeys[keyIx].flows[flowIx] == 0) {
          seen[flowIx] = 1;
        }
      }
      for (size_t nucIx = 0; nucIx < mFlowOrder.length() && !AvgKeyReporter<double>::AllSeen(seen); nucIx++) {
        for (size_t flowIx = 0; flowIx < mKeys[keyIx].usableKeyFlows; flowIx++) {
          if (mKeys[keyIx].flows[flowIx] == 1 && mFlowOrder[flowIx % mFlowOrder.length()] == mFlowOrder[nucIx]) {
            seen[flowIx] = 1;
            avgTrace << mFlowOrder[flowIx % mFlowOrder.length()];
            std::vector<SampleStats<double> > &flowTrace = mKeyRegions[keyIx].GetFlowTrace(flowIx);
            for (size_t frameIx = 0; frameIx < flowTrace.size(); frameIx++) {
              avgTrace << " " << flowTrace[frameIx].GetMean();
            }	
            avgTrace << endl;
          }
        }
      }
      avgTrace.close();
    }
  }

private:
  std::string mFlowOrder;
  std::vector<KeyRegionSummary<T> > mKeyRegions;
  std::vector<double> mMinPeakSig;
  std::string mPrefix;
  std::vector<KeySeq> mKeys;
  pthread_mutex_t mLock;  
};

#endif // KEYSUMMARYREPORTER_H
