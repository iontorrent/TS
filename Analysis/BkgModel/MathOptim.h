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
//#define MAX_HPLEN 11


union Vec4 {
    float v __attribute__ ((vector_size (16)));
    float e[4];
};

class Dual{
  public:
    float a; // real
    float da; // derivative 1
    float dk; // derivative 2
    Dual(float a0, float b0=0.0, float c0=0.0): a(a0), da(b0), dk(c0) {};
    Dual(){a=da=dk=0.0;};
    bool operator =(const float &y){
      a = y;
      da=0.0;
      dk=0.0;
      return true; // really?  what should this return usually?
    };
    void operator +=(const Dual &y){
      a+=y.a; da+=y.da; dk += y.dk;
    };
    void operator -=(const Dual &y){
      a-=y.a; da-=y.da; dk -= y.dk;
    };
    void operator *=(const Dual &y){
      da = a*y.da+y.a*da; dk = a*y.dk+dk*y.a;
      a *=y.a; // used in other expressions, so must be last
    };
    void operator /=(const Dual &y){
       da= (y.a*da-a*y.da)/(y.a*y.a);
       dk = (y.a*dk-a*y.dk)/(y.a*y.a);
       a/= y.a; // used in other expressions, must be last
    };
    void Reciprocal(const Dual &y){
      a = 1.0/y.a; // overwriting, so this is fine
      da = (y.a-y.da)*a*a;
      dk = (y.a-y.dk)*a*a;
    };
    void Dump(char *my_note){
      printf("%s\t%f\t%f\t%f\n",my_note,a,da,dk);
    }
};

Dual operator+(const Dual &x, const Dual &y);
Dual operator-(const Dual &x, const Dual &y);
Dual operator*(const Dual &x, const Dual &y);
Dual operator/(const Dual &x, const Dual &y);



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
  int max_dim;
  int max_entry;
  float inv_scale;
  float scale;
  float total_live;
  float occ_l, occ_r; // historical
  MixtureMemo();
  ~MixtureMemo();
  float Generate(float A, PoissonCDFApproxMemo  *my_math);
  Dual Generate(Dual &A, PoissonCDFApproxMemo *my_math);
  void ScaleMixture(float SP);
  float GetStep(float x);
  Dual GetStep(Dual x);
  float GetDStep(float x);
  float GetIStep(float x);
  void UpdateState(float total_intensity, float &active_polymerase, float &generated_hplus);
  void Delete();
};

float ErfApprox(float x);
float Expm2Approx(float x);
float ExpApprox(float x);

#endif // MATHOPTIM_H
