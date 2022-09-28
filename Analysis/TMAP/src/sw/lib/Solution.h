/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SOLUTION_H
#define SOLUTION_H

#ifndef __cplusplus
#error a C++ compiler is required
#endif

#include <cstring>
#include <sstream>

using namespace std;

class Solution {
public:
  Solution();

  virtual int process(const string& b, const string& a, int qsc, int qec,
                 int mm, int mi, int o, int e, int dir,
                 int *opt, int *te, int *qe, int *n_best, int* fitflag) = 0;

  virtual ~Solution();

  int getMaxQlen() { return max_qlen; }
  int getMaxTlen() { return max_tlen; }
  
protected:
  int max_qlen;
  int max_tlen;
};

#endif // SOLUTION_H
