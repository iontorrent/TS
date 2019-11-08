/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef POISSONCDF_H
#define POISSONCDF_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined( __SSE__ ) && !defined( __CUDACC__ )
    #include <x86intrin.h>
#else
    typedef float __m128 __attribute__ ((__vector_size__ (16), __may_alias__));
#endif

class PoissonCDFApproxMemo{
  public:
    float **poiss_cdf;
    float **dpoiss_cdf; // derivative of function
    float **ipoiss_cdf; // integral of function at totat intensity
    float *t;

    __m128 **poissLUT; //lookup table optimized for 2D interpolation

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

#ifdef ION_COMPILE_CUDA
    #include <cuda_runtime.h>		// for __host__ and __device__
#endif

// Functions for computing the poisson cdf (aka the normalized incomplete gamma function).
// This is based on the code from Numerical Recipies in C,
// or the code from Alan Kaminsky's clear recopying of said code for CUDA.
//
// The trick to calculating this stuff is that you want to use different functions
// depending on where in the function domain you want to work.
//
// These constants control the overall accuracy of the calculation.
// As it turns out, for our uses, we can turn the accuracy *way* down with only
// a slight effect on the results. The proposed values are here; the original values
// are left in comments.
// #define GAMMA_ITMAX 100
// #define GAMMA_EPS 3.0e-7f
#define GAMMA_ITMAX 10          // 100          Adjusted for speed over accuracy.
#define GAMMA_EPS 1e-1f         // 3.0e-7       Adjusted for speed over accuracy.
#define GAMMA_FPMIN 1e-30f

// Here's the switch that turns it on or off.
// As it turns out, the lookup table approach seems to be faster...
// although that's probably worth revisiting occasionally.
#define USE_TABLE_GAMMA 1

/**
 * Returns the incomplete gamma function P(a,x), evaluated by its series representation.
 * Assumes a > 0 and x >= 0.
 */

#ifdef ION_COMPILE_CUDA
__host__ __device__ 
#endif
inline float gser( float a, float x )
{
  float ap, del, sum;
  int i;

  ap = a;
  del = 1.0f / a;
  sum = del;
  for( i = 1 ; i <= GAMMA_ITMAX ; ++i )
  {
    ap += 1.0f;
    del *= x / ap;
    sum += del;
    if ( fabs( del ) < fabs( sum ) * GAMMA_EPS )
    {
      return sum * exp( -x + a * log( x ) - ::lgamma( a ) );
    }
  }

  // Too many iterations...
  return 1.f;
}

/**
 * Returns the complementary incomplete gamma function Q(a,x),
 * evaluated by its continued fraction representation.
 * Assumes a > 0 and x >= 0.
 */
#ifdef ION_COMPILE_CUDA
__host__ __device__ 
#endif
inline float gcf( float a, float x )
{
  float b, c, d, h, an, del;
  int i;

  b = x + 1.f - a;
  c = 1.f / GAMMA_FPMIN;
  d = 1.f / b;
  h = d;
  for( i = 1 ; i <= GAMMA_ITMAX ; ++i )
  {
    an = -i * ( i - a );
    b += 2.f;
    d = an * d + b;
    if ( fabs( d ) < GAMMA_FPMIN ) d = GAMMA_FPMIN;
    c = b + an / c;
    if ( fabs( c ) < GAMMA_FPMIN ) c = GAMMA_FPMIN;
    d = 1.f / d;
    del = d * c;
    h *= del;
    if ( fabs( del - 1.f ) < GAMMA_EPS )
    {
      return exp( -x + a * log( x ) - ::lgamma( a ) ) * h;
    }
  }

  // Too many iterations...
  return 0.f; 
}

/**
 * Returns the complementary incomplete gamma function Q(a,x) = 1 - P(a,x).
 * If a is an integer, this would be the cumulative distribution function we're looking for.
 */
#ifdef ION_COMPILE_CUDA
__host__ __device__ 
#endif
inline float gammq( float a, float x )
{
  /* Clamp to legitimate values. Throwing an error here wouldn't really help. */
  if ( a <= 0.f ) a = GAMMA_FPMIN;
  if ( x < 0.f ) x = 0.f;

  return x < a + 1.f ? 1.f - gser( a, x ) : gcf( a , x );
}

/**
 * If X is a Poi(lambda) random variable, and s is an integer, then Pr( X < s ) = PoissonCDF.
 */
#ifdef ION_COMPILE_CUDA
__host__ __device__ 
#endif
inline float PoissonCDF( float s, float lambda )
{
  return gammq( s, lambda );
}

#endif // POISSONCDF_H
