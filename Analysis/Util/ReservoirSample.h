/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef RESERVOIRSAMPLE_H
#define RESERVOIRSAMPLE_H

#include <vector>
#include <stdlib.h>
#include <assert.h>
#include "RandSchrange.h"
/**
   Generic class for doing sampling without replacement on a stream of data.
   Convenient as it doesn't use any more memory than the number of items desired
   int the samplie size.
   Exampl to get a sample of size 100
   ReservoirSample<float> samp(100)
   float f = 0f;
   while(GetNextFloat(f)) {
   samp.Add(f);
   }
   samp.Finished();
   std::vector<float> &finishedSample = samp.GetData();

*/
template <class T>
class ReservoirSample {
public:

  /** Constructor. */
  ReservoirSample() {
    mK = 0;
    mSeen = 0;
    mFinished = false;
  }

  /** Constructor for sample of size k, note seed is implicitly 1 */
  ReservoirSample(size_t k) {
    mK = 0;
    mSeen = 0;
    mFinished = false;
    Init(k);
  }

  /** Constructor for sample of size k, note seed is implicitly 1 */
  void Init(size_t k) {
    Init(k, 1);
  }

  /** Constructor for sample of size k and ability to specify integrer seed. */
  void Init(size_t k, int seed) {
    mK = k;
    mRand.SetSeed(seed);
    mRes.clear();
    mRes.reserve(k);
  }

  void Clear(int seed = 1) {
    mRand.SetSeed(seed);
    mSeen = 0;
    mRes.clear();
    mFinished = false;
  }

  /** Insert a value to be considered for sampling. */
  void Add(const T &t) {
    assert(!mFinished);
    if (mRes.size() < mK) {
      mRes.push_back(t);
    }
    else {
      size_t r = mRand.Rand() % mSeen;
      if (r < mK) {
	mRes[r] = t;
      }
    }
    mSeen++;
  }

  /** Get the number of items observed (not number sampled) */
  size_t GetNumSeen() {
    return mSeen;
  }

  /** Finish sampling. Sampling while still adding items is discouraged as
      items not observed yet have zero probability of being included. */
  void Finished() {
    mFinished = true;
  }

  /** Get the number of items sampled. Must call Finished() first 
      This can be smaller than the sample size if there were less items
      in the population than the sample size. */
  size_t GetCount() {
    assert(mFinished);
    return mRes.size();
  }

  /** Get a particular sampled item. Must call Finished() first */
  const T& GetVal(size_t index) {
    assert(mFinished);
    return mRes[index];
  }

  /** Get all of the sampled data. Must call Finished() first */
  std::vector<T> & GetData() {
    assert(mFinished);
    return mRes;
  }

private:
  bool mFinished;      ///< Are we finished sampling yet.
  std::vector<T> mRes; ///< Internal reservoir of current items sampled
  size_t mK;           ///< Size of original sample size
  size_t mSeen;        ///< Number of items observed for sampling
  RandSchrange mRand;  ///< Internal random number generator to avoid having calling order of threads change results.
};

#endif // RESERVOIRSAMPLE_H
