/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef AFFINESWOPTIMIZATIONHASH_H
#define AFFINESWOPTIMIZATIONHASH_H

#ifndef __cplusplus
#error a C++ compiler is required
#endif

#include <cstring>
#include <sstream>

using namespace std;

typedef struct {
    unsigned long long int hash;
    int tlen;
    int dir;
    string b;
    int opt;
    int te;
    int qe;
    int n_best;
    int fitflag;
} hash_t;

class AffineSWOptimizationHash {
public:
  AffineSWOptimizationHash();

  bool process(const string &b, const string &a, int qsc, int qec,
                  int mm, int mi, int o, int e, int dir,
                  int *opt, int *te, int *qe, int *n_best, int* fitflag);

  void add(const string &b, const string &a, int _qsc, int _qec,
           int mm, int mi, int o, int e, int dir,
           int *opt, int *te, int *qe, int *n_best, int* fitflag);

  ~AffineSWOptimizationHash();
private:
  int qsc;
  int qec;
  int size;
  string query;
  hash_t *hash;
};
#endif // AFFINESWOPTIMIZATIONHASH_H
