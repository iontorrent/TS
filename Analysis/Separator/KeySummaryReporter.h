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

template <class T>
class KeyRegionSummary {

public:
  KeyRegionSummary(int sampSize=5000) {
    mSampleSize = sampSize;
  }
  
  void Init(int nRows, int nCols, int nFrames, int rowStep, int colStep, const KeySeq &key) {
    mKey = key;
    mGlobalTraces.resize(key.usableKeyFlows);
    mGlobalWeight = 0;
    for (size_t i = 0; i < mGlobalTraces.size(); i++) {
      mGlobalTraces[i].resize(nFrames);
      for (size_t fIx = 0; fIx < mGlobalTraces[i].size(); fIx++) {
        std::fill(mGlobalTraces[i].begin(), mGlobalTraces[i].end(), 0.0f);
      }
    }
  }

  void Report(EvaluateKey &evaluator, int keyIx) {
    int flowFrameStride = evaluator.m_num_flows * evaluator.m_num_frames;
    int keyOffset = keyIx * flowFrameStride;
    mGlobalWeight += evaluator.m_key_counts[keyIx];
    for (size_t frameIx = 0; frameIx < evaluator.m_num_frames; frameIx++) {
      for (size_t flowIx = 0; flowIx < evaluator.m_num_flows; flowIx++) {        
        mGlobalTraces[flowIx][frameIx] += (evaluator.m_flow_key_avg[keyOffset + flowIx * evaluator.m_num_frames + frameIx] * evaluator.m_key_counts[keyIx]);
      }
    }
  }

  void Report(const KeyFit &fit, 
              const Mat<T> &wellFlows,
              const Mat<T> &refFlows,
              const Mat<T> &predicted) {
  }

  void Finish() {
    mGlobalAvg.resize(mGlobalTraces.size());
    for (size_t flowIx = 0; flowIx < mGlobalTraces.size(); flowIx++) {        
      mGlobalAvg[flowIx].resize(mGlobalTraces[flowIx].size());
      for (size_t frameIx = 0; frameIx < mGlobalTraces[flowIx].size(); frameIx++) {
        mGlobalTraces[flowIx][frameIx] /= mGlobalWeight;
        mGlobalAvg[flowIx][frameIx].AddValue(mGlobalTraces[flowIx][frameIx]);
      }
    }
  }

  std::vector<SampleStats<T> > &GetFlowTrace(int flowIx) {
    return mGlobalAvg[flowIx];
  }

private:
  int mSampleSize;
  KeySeq mKey;
  std::vector<std::vector<double > > mGlobalTraces;
  double mGlobalWeight;
  std::vector<std::vector<SampleStats<T> > > mGlobalAvg;
};

template <class T>
class KeySummaryReporter : public KeyReporter<T> {
  
public:

  KeySummaryReporter() { 
    mFrames = 0;
    pthread_mutex_init(&mLock, NULL);
  }

  ~KeySummaryReporter() {
    pthread_mutex_destroy(&mLock);
  }
  
  void Init(const std::string &flowOrder, const std::string &prefix, 
            int nRows, int nCols, int nFrames,
            int rowStep, int colStep, const std::vector<KeySeq> &keys) {
    mFlowOrder = flowOrder;
    mPrefix = prefix;
    mFrames = nFrames;

    mKeyRegions.resize(keys.size());
    for (size_t i = 0; i < mKeyRegions.size(); i++) {
      mKeyRegions[i].Init(nRows, nCols, nFrames, rowStep, colStep, keys[i]);
    }
    mKeys = keys;
    mMinPeakSig.resize(keys.size());
    std::fill(mMinPeakSig.begin(), mMinPeakSig.end(), std::numeric_limits<double>::max() * -1);
  }

  void SetMinKeyThreshold(int key, double val) {
    mMinPeakSig[key] = val;
  }

  void Report(EvaluateKey &evaluator) {
    pthread_mutex_lock(&mLock);
    for (size_t keyIx = 0; keyIx < mKeys.size(); keyIx++) {
      mKeyRegions[keyIx].Report(evaluator, keyIx);
    }
    pthread_mutex_unlock(&mLock);
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
  ION_DISABLE_COPY_ASSIGN(KeySummaryReporter)

  std::string mFlowOrder;
  std::vector<KeyRegionSummary<T> > mKeyRegions;
  std::vector<double> mMinPeakSig;
  std::string mPrefix;
  int mFrames;
  std::vector<KeySeq> mKeys;
  pthread_mutex_t mLock;  
};

#endif // KEYSUMMARYREPORTER_H
