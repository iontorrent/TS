/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef INCORPREPORTER_H
#define INCORPREPORTER_H

#include <string>
#include <pthread.h>
#include <math.h>
#include "KeyReporter.h"
#include "SampleStats.h"
#include "SampleQuantiles.h"
#include "KeyClassifier.h"

#define KEY_SAMPLE_SIZE 1000

template <class T>
class IncorpReporter : public KeyReporter<T> {
  
 public:
  IncorpReporter(std::vector<KeySeq> *keys, double minSnr) {
    pthread_mutex_init(&lock, NULL);
    mKeys = keys;
    mMinSnr = minSnr;
  }
  
  ~IncorpReporter() {
    pthread_mutex_destroy(&lock);
  }
  
  void Report(const KeyFit &fit, 
	      const Mat<T> &wellFlows,
	      const Mat<T> &refFlows,
	      const Mat<T> &predicted) {
    if (fit.keyIndex < 0 || fit.mad > 50 || refFlows.n_cols == 0 || fit.snr < mMinSnr || !isfinite(fit.snr)) {
      return;
    }
    SampleStats<float> peak;
    for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
      if (mKeys->at(fit.keyIndex).flows[flowIx] == 1) {
        double m = 0.0;
        for (size_t frameIx = 0; frameIx < wellFlows.n_rows; frameIx++) {
          double d = wellFlows.at(frameIx,flowIx) - predicted.at(frameIx,flowIx);
          m = max(d,m);
        }
        peak.AddValue(m);
      }
    }
    if (peak.GetMean() < 40) {
      return;
    }
    pthread_mutex_lock(&lock);
    if (wellFlows.n_rows > mTraceMean.size()) {
      mTraceMean.resize(wellFlows.n_rows);
    }
    
    for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
      for (size_t frameIx = 0; frameIx < wellFlows.n_rows && frameIx < mTraceMean.size(); frameIx++) {
	if (mKeys->at(fit.keyIndex).flows[flowIx] == 1) {
	  double d = wellFlows.at(frameIx,flowIx) - predicted.at(frameIx,flowIx);
          if (isfinite(d)) {
            mTraceMean[frameIx].AddValue(d);
          }
	}
      }
    }
    pthread_mutex_unlock(&lock);
  }
  
  Col<double> GetMeanTrace() {
    Col<double> trace(mTraceMean.size());
    for (size_t i = 0; i < mTraceMean.size(); i++) {
      trace.at(i) = mTraceMean[i].GetMean();
    }
    return trace;
  }


  const std::vector<SampleQuantiles<double> > &GetQuantileTraces() { return mTraceMean; }

 private:
	//  std::vector<SampleQuantiles<double> >  mTraceMean;
	std::vector<SampleStats<double> > mTraceMean;
  std::vector<KeySeq> *mKeys; // This memory owned elsewhere...
  pthread_mutex_t lock;  
  double mMinSnr;
};

#endif // INCORPREPORTER_H
