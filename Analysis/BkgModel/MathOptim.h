/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * MathOptim.h
 *
 *  Created on: Jun 7, 2010
 *      Author: Mark Beauchemin
 */

#ifndef MATHOPTIM_H
#define MATHOPTIM_H

#include "BkgMagicDefines.h"
//#define MAX_HPLEN 11


union Vec4 {
    float v __attribute__ ((vector_size (16)));
    float e[4];
};

class PoissonCDFApproxMemo{
  public:
    float **poiss_cdf;
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

// wrap the poisson lookup table
// we do a mixture of two adjacent homopolymer levels
class MixtureMemo{
  public:
  float A;
  float *my_mixL; // custom interpolation
  float *my_mixR; // custom interpolation
  int max_dim;
  float inv_scale;
  float total_live;
  float occ_l, occ_r; // historical
  MixtureMemo();
  ~MixtureMemo();
  float Generate(float A, PoissonCDFApproxMemo  *my_math);
  void ScaleMixture(float SP);
  float GetStep(float x);
  void Delete();
};

float ErfApprox(float x);
float Expm2Approx(float x);
float ExpApprox(float x);

#endif // MATHOPTIM_H
