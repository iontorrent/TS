/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "MathUtil.h"
#include "math.h"
#include <algorithm>

// Expressive little functions to simplify doing "vector" operations because we use float pointers everywhere
// the compiler at max optimization will turn these all into inline stuff, but I want to >read< code in vector style

// utility functions because C++ is too low level a language to be useful
void MultiplyVectorByScalar(float *my_vec, float my_scalar, int len)
{
    for (int i=0; i<len; i++)
        my_vec[i] *= my_scalar;
}

void AddScaledVector(float *start_vec, float *my_vec, float my_scalar, int len)
{
    for (int i=0; i<len; i++)
        start_vec[i] += my_scalar * my_vec[i];
}

void CopyVector(float *destination_vec, float *my_vec, int len)
{
    memcpy(destination_vec, my_vec, sizeof(float)*len);
}

void AccumulateVector(float *destination_vec, float *my_vec, int len)
{
    for (int i=0; i<len; i++)
        destination_vec[i] += my_vec[i];
}

void DiminishVector(float *destination_vec, float *my_vec, int len)
{
    for (int i=0; i<len; i++)
        destination_vec[i] -= my_vec[i];
}

// this is an 'in-place' version (input1 = input1 .* input2)
void MultiplyVectorByVector(float *my_vec, float *my_other_vec, int len)
{
    for (int i=0; i<len; i++)
        my_vec[i] *= my_other_vec[i];
}

// ...and a not 'in-place' version (output = input1 .* input2)
void MultiplyVectorByVector(float *my_output_vec, float *my_vec, float *my_other_vec, int len)
{
    for (int i=0; i<len; i++)
        my_output_vec[i] = my_vec[i] * my_other_vec[i];
}

float CheckVectorDiffSSQ(float *my_vec, float *my_other_vec, int len)
{
    float retval = 0;
    for (int i=0; i<len; i++)
        retval += (my_vec[i]-my_other_vec[i])*(my_vec[i]-my_other_vec[i]);
    return(retval);
}

// generic local derivatives
void CALC_PartialDeriv(float *p1,float *p2,int len,float dp)
{
    float *pend = p2+len;
    for (;p2 < pend;p2++) *p2 = (*p2 - *p1++)/dp;
}

void CALC_PartialDeriv_W_EMPHASIS(float *p1,float *p2,float *pe,int len,float dp,float pelen)
{
    float *pend = p2+len;
    int ipe;
    for (ipe=0;p2 < pend;p2++)
    {
        *p2 = (*p2 - *p1++)*pe[ipe++]/dp;
        if (ipe >= pelen) ipe = 0;
    }
}


// computes the partial derivative for a parameter with emphasis.  len is the total length
// of data points to compute over, and all vectors should have len values.  dp is the amount
// the parameter was stepped by when computing the function values in p2.
void CALC_PartialDeriv_W_EMPHASIS_LONG(float *p1,float *p2,float *pe,int len,float dp)
{
    for (int i=0;i < len;i++)
        p2[i] = (p2[i] - p1[i])*pe[i]/dp;
}


// do interpolation fun

void Bary_Coord( float *bary_c, float test_x, float test_y, float *x, float *y)
{
  // x, y = length 3 entries for triangle coordinates
  // bary_coordinates = length 3 return values
  bary_c[0] = (y[1]-y[2])*(test_x-x[2])+(x[2]-x[1])*(test_y-y[2]);
  bary_c[1] = (y[2]-y[0])*(test_x-x[2])+(x[0]-x[2])*(test_y-y[2]);
  float detT = (y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2]);
  if (detT!=0.0) // triangle not co-linear
  {
    bary_c[0] /= detT;
    bary_c[1] /= detT;
    bary_c[2] = 1-bary_c[0]-bary_c[1];
  }
  else
  {
    bary_c[0] = 0.0;
    bary_c[1] = 1.0; // arbitrarily choose one point
    bary_c[2] = 0.0;
  }
}

void Bary_Interpolate(float *output, float **Triple, float *bary_c, int npts)
{
  // given curves at points of triangle
  // interpolate between them using the given weights
  for (int ii=0; ii<npts; ii++)
  {
    output[ii] = 0.0;
    for (int t=0; t<3; t++)
    {
      output[ii] += bary_c[t] * Triple[t][ii];
    }
  }
}
