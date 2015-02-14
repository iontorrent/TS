/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef AVGKEYREPORTER_H
#define AVGKEYREPORTER_H

#include <string>
#include <pthread.h>
#include "SampleStats.h"
#include "SampleQuantiles.h"

#define KEY_SAMPLE_SIZE 1000

template <class T>
class AvgKeyReporter { 
  
 public:
  AvgKeyReporter(const std::vector<KeySeq> &keys, const std::string &prefix, 
                 const std::string &flowOrder, const std::string &nukeDir, int nFlows, int nFrames) {
    string s = prefix + ".avg-traces.txt";
    mTraces.open(s.c_str());
    mKeys = keys;
    mNukeDir = nukeDir;
    mFlowOrder = flowOrder;
    mMinPeakSig.resize(keys.size());
    std::fill(mMinPeakSig.begin(), mMinPeakSig.end(), std::numeric_limits<double>::max() * -1);
    mAvgTraces.resize(keys.size());
    for (size_t key_ix = 0; key_ix < keys.size(); key_ix++) {
      mAvgTraces[key_ix].resize(nFlows);
      for (int flow_ix = 0; flow_ix < nFlows; flow_ix++) {
        mAvgTraces[key_ix][flow_ix].resize(nFrames);
      }
      //      std::fill(mAvgTraces[key_ix][flow_ix].begin(),mAvgTraces[key_ix][flow_ix].end(),0.0f);
    }
    pthread_mutex_init(&lock, NULL);
  }


  ~AvgKeyReporter() {
    pthread_mutex_destroy(&lock);
    mTraces.close();
  }

  void SetMinKeyThreshold(int key, double val) {
    mMinPeakSig[key] = val;
  }

  void Report(EvaluateKey & evaluator) {
    pthread_mutex_lock(&lock);
    for (size_t keyIx = 0; keyIx < mKeys.size(); keyIx++) {
      if (evaluator.m_key_counts[keyIx] < 500) {
        continue;
      }
      int flowFrameStride = evaluator.m_num_flows * evaluator.m_num_frames;
      int keyOffset = keyIx * flowFrameStride;
      for (size_t flowIx = 0; flowIx < mAvgTraces[keyIx].size(); flowIx++) {
        for (size_t frameIx = 0; frameIx < mAvgTraces[keyIx][flowIx].size(); frameIx++) {
          double d = evaluator.m_flow_key_avg[keyOffset + flowIx * evaluator.m_num_frames + frameIx];
          mAvgTraces[keyIx][flowIx][frameIx].AddValue(d);
        }
      }      
    }
    pthread_mutex_unlock(&lock);

  }

  static std::string NameForKey(const KeySeq &key, const std::string &flowOrder) {
    string name; 
    for (size_t i = 0; i < key.flows.size(); i++) {
      if(key.flows[i] > 0) {
        name = name + flowOrder[i % flowOrder.length()];
      }
    }
    return name;
  }

  static bool AllSeen(const std::vector<int> &seen) {
    for (size_t i = 0; i < seen.size(); i++) {
      if (seen[i] == 0) 
        return false;
    }
    return true;
  }

  void Finish() {
    for (size_t keyIx = 0; keyIx < mAvgTraces.size(); keyIx++) {
      for (size_t flowIx = 0; flowIx < mAvgTraces[keyIx].size(); flowIx++) {
        mTraces << keyIx << "\t" << flowIx;
        for (size_t frameIx = 0; frameIx < mAvgTraces[keyIx][flowIx].size(); frameIx++) {
          //	  mTraces << "\t" << mAvgTraces[keyIx][flowIx][frameIx].GetTrimmedMean(.1, .9);
          mTraces << "\t" << mAvgTraces[keyIx][flowIx][frameIx].GetMean();
        }
        mTraces << endl;
      }
    }
    if (!mNukeDir.empty()) {
      for (size_t keyIx = 0; keyIx < mAvgTraces.size(); keyIx++) {
        string name = NameForKey(mKeys[keyIx], mFlowOrder);
        string file = mNukeDir + "/avgNukeTraceOld_" + name + ".txt"; 
        ofstream avgTrace;
        avgTrace.open(file.c_str());
        for (size_t flowIx = 0; flowIx < mKeys[keyIx].usableKeyFlows; flowIx++) {
          avgTrace << flowIx;
          for (size_t frameIx = 0; frameIx < mAvgTraces[keyIx][flowIx].size(); frameIx++) {
            avgTrace << " " << mAvgTraces[keyIx][flowIx][frameIx].GetMean();
          }
          avgTrace << endl;
        }
        vector<int> seen(mKeys[keyIx].flows.size(), 0);
        for (size_t flowIx = 0; flowIx < mKeys[keyIx].flows.size(); flowIx++) {
          if (mKeys[keyIx].flows[flowIx] == 0) {
            seen[flowIx] = 1;
          }
        }
        for (size_t nucIx = 0; nucIx < mFlowOrder.length() && !AllSeen(seen); nucIx++) {
          for (size_t flowIx = 0; flowIx < mKeys[keyIx].usableKeyFlows; flowIx++) {
            if (mKeys[keyIx].flows[flowIx] == 1 && mFlowOrder[flowIx % mFlowOrder.length()] == mFlowOrder[nucIx]) {
              seen[flowIx] = 1;
              avgTrace << mFlowOrder[flowIx % mFlowOrder.length()];
              for (size_t frameIx = 0; frameIx < mAvgTraces[keyIx][flowIx].size(); frameIx++) {
                avgTrace << " " << mAvgTraces[keyIx][flowIx][frameIx].GetMean();
              }	
              avgTrace << endl;
            }
          }
        }
        avgTrace.close();
      }
    }
  }
  
 private:
  //  std::vector<std::vector<std::vector<SampleQuantiles<double> > > > mAvgTraces;
  std::vector<std::vector<std::vector<SampleStats<double> > > > mAvgTraces;
  std::vector<KeySeq> mKeys; // This memory owned elsewhere...
  std::string mFlowOrder;
  std::string mNukeDir;
  std::vector<double> mMinPeakSig;
  ofstream mTraces;
  pthread_mutex_t lock;  
};

#endif // AVGKEYREPORTER_H
