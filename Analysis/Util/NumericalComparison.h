/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef NUMERICALCOMPARISON_H
#define NUMERICALCOMPARISON_H
#include <ostream>
#include <stdio.h>
#include <stdint.h>
#include <cmath>
#include "SampleStats.h"

#define NC_CORRELATION_EPSILON .00001

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
    Init(0);
  }

  /** Constructor, epsilon is the maximum allowed difference to be
      considered the same. */
  NumericalComparison(const T &epsilon) {
    Init(epsilon);
  }

  /** Initialize */
  void Init(const T &epsilon) {
    mEpsilon = epsilon;
    mNumDiff = 0;
    mNumSeen = 0;
  }

  /** Should we tolerate non-finite values like NaNs and Infs */
  void SetTolerateNaN(bool value) { mTolerateNans = value; }

  /** Set the name of the column */
  void SetName(const std::string &name) { mName = name; }

  /** 
   * Add a pair of values and see if they are the same or different,
   * also logging the correlation of the two. Returns true if this
   * particiular x & y are closer than epsilon, false otherwise.
  */
  bool AddPair(const T &x, const T &y) {
    if (std::isnan(x) && std::isnan(y)) {
      return true;
    }
    if (std::isinf(x) && std::isinf(y)) {
      return true;
    }
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

  bool CorrelationOk(double min_value, double epsilon=NC_CORRELATION_EPSILON) {
    if (GetCorrelation() + epsilon >= min_value) {
      return true;
    }
    return false;
  }

  const SampleStats<T> &GetXStats() const { return mX; };

  const SampleStats<T> &GetYStats() const { return mY; };

  void Out(std::ostream &out, float tag_percent=3.0f) {
    char buff[256];
    char tag = ' ';
    if (GetNumDiff() * 100.0f/GetCount() >= tag_percent) {
      tag = '*';
    }
    snprintf(buff, sizeof(buff), "%-12s\t%8u (%6.2f%%)\t%8u (%6.2f%%)\t%4.2f%c", 
	     mName.c_str(),
	     (uint32_t)GetNumSame(),(GetNumSame() * 100.0f)/GetCount(), 
	     (uint32_t)GetNumDiff(),(GetNumDiff() * 100.0f)/GetCount(), 
	     GetCorrelation(), tag);
    out << buff << std::endl;
  }

 private:
  SampleStats<T> mX;
  SampleStats<T> mY;
  SampleStats<T> mXY;
  std::string mName;
  COUNT mNaN;
  COUNT mNumDiff;
  COUNT mNumSeen;
  bool mTolerateNans; // quietly ignore nans, and infs
  T mEpsilon;
};

#endif // NUMERICALCOMPARISON_H 
