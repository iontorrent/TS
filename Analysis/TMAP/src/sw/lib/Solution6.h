/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SOLUTION6_H
#define SOLUTION6_H

#ifndef __cplusplus
#error a C++ compiler is required
#endif

#include <cstring>
#include <sstream>
#include "Solution.h"

using namespace std;

class Solution6 : public Solution {
    typedef union { short s[8]; __m128i m; } m128si16;
public:
    Solution6();

    virtual int process(const string& bs, const string& as, int qsc, int qec,
                        int mm, int mi, int oe, int e, int dir,
                        int *opt, int *te, int *qe, int *n_best, int* fitflag);
};

#endif // SOLUTION6_H
