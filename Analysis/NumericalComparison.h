/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef NUMERICALCOMPARISON_H
#define NUMERICALCOMPARISON_H
#include "SampleStats.h"

/**
 * Class to keep track of differences between two sets of paired
 * numbers. Designed initially for integration testing to keep see if
 * two distributions are the same.
 */ 
template <class T, class COUNT=long> 
class NumericalComparison {

public: 

  /** Constructor */
  NumericalComparison() {
    mEpsilon = 0;
    mNumDiff = 0;
    mNumSeen = 0;
  }

  /** Constructor, epsilon is the maximum allowed difference to be
      considered the same. */
  NumericalComparison(const T &epsilon) {
    mEpsilon = epsilon;
    mNumDiff = 0;
    mNumSeen = 0;
  }

  /** 
   * Add a pair of values and see if they are the same or different,
   * also logging the correlation of the two. Returns true if this
   * particiular x & y are closer than epsilon, false otherwise.
  */
  bool AddPair(const T &x, const T &y) {
    bool closeEnough = true;
    mNumSeen++;

    T xy = x * y;
    mX.AddValue(x);
    mY.AddValue(y);
    mXY.AddValue(xy);

    T diff = x - y;
    if (y > x) {
      diff = y - x;
    }

    if (diff > mEpsilon) {
      mNumDiff++;
      closeEnough = false;
    }
    return closeEnough;
  }

  T GetEpsilon() const { return mEpsilon; }

  COUNT GetNumDiff() const { return mNumDiff; }

  COUNT GetNumSame() const { return mNumSeen - mNumDiff; }

  COUNT GetCount() const { return mNumSeen; }

  double GetCorrelation() const {
    double correlation = mXY.GetMean() - (mX.GetMean() * mY.GetMean());
    correlation = correlation / (mX.GetSD() * mY.GetSD());
    return correlation;
  }

  const SampleStats<T> &GetXStats() const { return mX; };

  const SampleStats<T> &GetYStats() const { return mY; };


 private:
  SampleStats<T> mX;
  SampleStats<T> mY;
  SampleStats<T> mXY;

  COUNT mNumDiff;
  COUNT mNumSeen;
  T mEpsilon;
};

#endif // NUMERICALCOMPARISON_H 
