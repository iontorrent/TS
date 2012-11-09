/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SAMPLEKEYREPORTER_H
#define SAMPLEKEYREPORTER_H

#include <string>
#include "KeyClassifier.h"
#include "KeyReporter.h"

template <class T>
class SampleKeyReporter : public KeyReporter<T> {

 public:
  SampleKeyReporter(const std::string &prefix, int numWells) {
    pthread_mutex_init(&lock, NULL);
    mPrefix = prefix;
    mReport.resize(numWells);
    fill(mReport.begin(), mReport.end(), false);
  }

  ~SampleKeyReporter() {
    pthread_mutex_destroy(&lock);
    if (mTraces.is_open()) {
      mTraces.close();
      mSignal.close();
      mReferences.close();
      mPredicted.close();
    }
  }

  void SetReportSet(const std::vector<int> &indexes) {
    if (indexes.size() == 0) {
      return;
    }
    string s = mPrefix + ".key-traces.txt";
    mTraces.open(s.c_str());

    s = mPrefix + ".key-references.txt";
    mReferences.open(s.c_str());

    s = mPrefix + ".key-predicted.txt";
    mPredicted.open(s.c_str());

    s = mPrefix + ".key-signal.txt";
    mSignal.open(s.c_str());

    for (size_t i = 0; i < indexes.size(); i++) {
      mReport[indexes[i]] = true;
    }
  }

  void WriteMat(const KeyFit &fit, const Mat<T> &mat, std::ofstream &out) {
    // this transposes the matrix
    for (size_t flowIx = 0; flowIx < mat.n_cols; flowIx++)  {
      out << fit.wellIdx << '\t' << flowIx << '\t' << (int)fit.keyIndex;
      for (size_t frameIx = 0; frameIx < mat.n_rows; frameIx++) {
	out << '\t' << mat.at(frameIx, flowIx);
      }
      out << endl;
    }
  }

  void Report(const KeyFit &fit, 
	      const Mat<T> &wellFlows,
	      const Mat<T> &refFlows,
	      const Mat<T> &predicted) {
    if (mReport[fit.wellIdx]) {
      pthread_mutex_lock(&lock);
      WriteMat(fit, wellFlows, mTraces);
      WriteMat(fit, refFlows, mReferences);
      WriteMat(fit, predicted, mPredicted);
      mSignal << fit.wellIdx << '\t' << (int)fit.keyIndex;
      for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
	double sig = 0;
        //	for (size_t frameIx = 0; frameIx < FRAME_SIGNAL; frameIx++) {
	for (size_t frameIx = 0; frameIx < wellFlows.n_rows; frameIx++) {
	  sig += wellFlows.at(frameIx,flowIx) - predicted.at(frameIx,flowIx);
	}
	mSignal << '\t' << sig;
      }
      mSignal << endl;
      pthread_mutex_unlock(&lock);
    }
    
  }

private:
  std::vector<int> mIndexes;
  std::vector<bool> mReport;
  std::string mPrefix;
  ofstream mTraces;
  ofstream mReferences;
  ofstream mPredicted;
  ofstream mSignal;
  pthread_mutex_t lock;  
};

#endif // SAMPLEKEYREPORTER_H
