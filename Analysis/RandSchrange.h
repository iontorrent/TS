/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef RANDSCHRANGE_H
#define RANDSCHRANGE_H

/**
 * Random number generator using Park & Millers numbers and Schrange's 
 * handling of integer overflow
 */
class RandSchrange {

 public:

  const static int RandMax =  2147483646; // RANDMAX = M - 1

  RandSchrange(int seed = 1) {
    SetSeed(seed);
  }
  
  void SetSeed(int seed) {
    assert(seed != 0);
    mNext = seed;
  }

  int Rand()  {
    mNext = A * (mNext % q) - r * (mNext / q);
    if (mNext < 0) {
      mNext += M;
    }
    return mNext;
  }

 private: 
  const static int A = 16807;
  const static int M = 2147483647;   // 2^31 - 1
  const static int q = 127773;       // M / A
  const static int r = 2836;         // M % A

  int mNext;
};

#endif // RANDSCHRANGE_H
