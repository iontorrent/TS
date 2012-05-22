/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef POISSONCDF_H
#define POISSONCDF_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

class PoissonCDFApproxMemo{
  public:
    float **poiss_cdf;
    float **dpoiss_cdf; // derivative of function
    float **ipoiss_cdf; // integral of function at totat intensity
    float *t;
    int max_events; // 0-n cdfs
    int max_dim; // 0-whatever intensity
    float scale; // scale for max_dim

    PoissonCDFApproxMemo();
    ~PoissonCDFApproxMemo();
    void Delete();
    void Allocate(int,int,float);
    void GenerateValues();
    void DumpValues();
};


#endif // POISSONCDF_H