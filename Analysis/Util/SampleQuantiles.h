/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SAMPLEQUANTILES_H
#define SAMPLEQUANTILES_H
#include <vector>
#include <math.h>
#include <assert.h>
#include "Stats.h"
#include "ReservoirSample.h"
#include "SampleStats.h"

template <class T, class COUNT=long> 
class SampleQuantiles {

public:

  /** Constructor, how big of a sample do we want to use for quantiles */
  SampleQuantiles() {
    mSorted = false;
  }

  SampleQuantiles(int sampleSize) {
    Init(sampleSize);
  }
  
  void Init(int sampleSize) {
    mSorted = false;
    mSample.Init(sampleSize);
  }

  void Clear(int seed = 1) {
    mSorted = false;
    mSample.Clear(seed);
  }

  /** Add an item to our current mean/variance statistics */
  void AddValue(T x) {
    mSample.Add(x);
  }
  
  /** Add an entire vector of items to our mean variance statistics. */
  void AddValues(const std::vector<T> &values) {
    typename std::vector<T>::const_iterator i;
    for(i = values.begin(); i != values.end(); i++) {
      AddValue(*i);
    }
  }
  
  /** Add an entire vector of items to our mean variance statistics. */
  void AddValues(const T *values, size_t size) {
    for(size_t i = 0; i < size; ++i) {
      AddValue(values[i]);
    }
  }

  int GetNumSeen() {
    return mSample.GetNumSeen();
  }

  int GetCount() {
    return GetNumSeen();
  }

  double GetMedian() {
    return GetQuantile(.5);
  }
    
  double GetTrimmedMean(float startQ, float endQ) {
    SortData();
    SampleStats<T> mean;
    int startIdx = (int)(startQ * mSample.GetCount() + .5);
    int endIdx = (int)(endQ * mSample.GetCount() + .5);
    for (int i = startIdx; i < endIdx; i++) {
      mean.AddValue(mSample.GetVal(i));
    }
    return mean.GetMean();
  }

  double GetQuantile(float quantile) {
    SortData();
    std::vector<T> & data = mSample.GetData();
    if (data.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return ionStats::quantile_sorted(data, quantile);
  }

  double GetIQR() {
    return (GetQuantile(.75) - GetQuantile(.25));
  }

  double GetIqrSd() {
    SortData();
    std::vector<T> & data = mSample.GetData();
    assert(data.size() > 0);
    double diff =  ionStats::quantile_sorted(data, .75) - ionStats::quantile_sorted(data, .25);
    return diff / 1.35;
  }

private:
  void SortData() {
    if (!mSorted) {
      mSample.Finished();
      std::vector<T> & data = mSample.GetData();
      std::sort(data.begin(), data.end());
      mSorted = true;
    }
  }

  bool mSorted;
  ReservoirSample<T> mSample;

};

#endif // SAMPLEQUANTILES_H
