/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SAMPLESTATS_H
#define SAMPLESTATS_H

#include <stdlib.h>
#include <vector>
#include <math.h>

/**
 Knuth's stable mean/variance calc. Defined as a template but really
 only desiged for numeric (int,short,float,double types) as it uses
 doubles for the accumulators.

Donald E. Knuth (1998). The Art of Computer Programming, volume 2:
 Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.

 <pre>
 n = 0
 mean = 0
 M2 = 0
 
 def calculate_online_variance(x):
    n = n + 1
    delta = x - mean
    mean = mean + delta/n
    M2 = M2 + delta*(x - mean)  # This expression uses the new value of mean
 
    variance_n = M2/n
    variance = M2/(n - 1)
    return variance
</pre>
*/
template <class T, class COUNT=long> 
class SampleStats {

public:

  /** Constructor */
  SampleStats() {
    mN = 0;
    mMean = 0;
    mM2 = 0;
  }

  void Clear() {
    mN = 0;
    mMean = 0;
    mM2 = 0;
  }

  /** Add an item to our current mean/variance statistics */
  void AddValue(T x) {
    mN++;
    double delta = x - mMean;
    mMean = mMean + delta / mN;
    mM2 = mM2 + delta * (x - mMean);
  }

  /** Add an entire vector of items to our mean variance statistics. */
  void AddValues(const std::vector<T> &values) {
    typename std::vector<T>::const_iterator i;
    for(i = values.begin(); i != values.end(); i++) {
      AddValue(*i);
    }
  }

  /** Add an entire vector of items to our mean variance statistics. */
  void AddValues(const T* values, size_t size) {
    for(size_t i = 0; i < size; ++i) {
      AddValue(values[i]);
    }
  }
  
  /** Get the mean of items seen. */
  double GetMean() const {
    return mMean;
  }
  
  /** Get variance of items seen. */
  double GetSampleVar() const {
    return mM2 / (mN - 1);
  }

  /** Get variance of items seen. */
  double GetVar() const {
    return mM2 / (mN);
  }

  /** Get the standard deviation (sqrt(variance))  */
  double GetSD() const {
    return sqrt(GetVar());
  }
  
  /** How many values have been seen. */
  COUNT GetCount() { return mN; }

 private:
  COUNT mN;      ///< Number of itemss seen. 
  double mMean;  ///< Current mean estimate
  double mM2;    ///< Sum of squared diffferences from mean for variance

};

#endif // SAMPLESTATS_H
