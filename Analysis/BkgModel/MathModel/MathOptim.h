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
#include <stdio.h>
#include <stdlib.h>

#include "PoissonCdf.h"

//#define MAX_HPLEN 11


union Vec4 {
    float v __attribute__ ((vector_size (16)));
    float e[4];
};



// wrap the poisson lookup table
// we do a mixture of two adjacent homopolymer levels
class MixtureMemo{
  public:
  float A;
  float dA; // initial derivative for pact
  float *my_mixL; // custom interpolation
  float *my_mixR; // custom interpolation
  float *my_deltaL;
  float *my_deltaR;
  float *my_totalL; // total hydrogens generated
  float *my_totalR;
  int left;
  int right;
  float lx;
  float ly;
  float rx;
  float ry;
  int max_dim;
  int max_entry;
  float inv_scale;
  float scale;
  float total_live;
  float occ_l, occ_r; // historical
  
  MixtureMemo();
  ~MixtureMemo();
  float Generate(float A, PoissonCDFApproxMemo  *my_math);
  void ScaleMixture(float SP);
  float GetStep(float x);
  float GetDStep(float x);
  float GetIStep(float x);
  void UpdateState(float total_intensity, float &active_polymerase, float &generated_hplus);
  void UpdateActivePolymeraseState(float total_intensity, float &active_polymerase);
  void UpdateGeneratedHplus(float &generated_hplus); // uses current active intensity & interpolation
  void Delete();
};

float ErfApprox(float x);
float Expm2Approx(float x);
float ExpApprox(float x);

#endif // MATHOPTIM_H
