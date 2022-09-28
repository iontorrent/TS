/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef AFFINESWOPTIMIZATION_H
#define AFFINESWOPTIMIZATION_H

#ifndef __cplusplus
#error a C++ compiler is required
#endif

#include <cstring>
#include <sstream>
#include "Solution.h"
#include "AffineSWOptimizationHash.h"

//#define AFFINESWOPTIMIZATION_USE_HASH

using namespace std;

class AffineSWOptimization {
public:
  AffineSWOptimization(int type);

  int process(const uint8_t *target, int32_t tlen,
              const uint8_t *query, int32_t qlen,
              int qsc, int qec,
              int mm, int mi, int o, int e, int dir,
              int *opt, int *te, int *qe, int *n_best, int* fitflag);

  ~AffineSWOptimization();

  int getMaxQlen() { return s->getMaxQlen(); }
  int getMaxTlen() { return s->getMaxTlen(); }
private:
  int myType;
  Solution *s;
  AffineSWOptimizationHash *hash;
  string a, b;
};
#endif // AFFINESWOPTIMIZATION_H
