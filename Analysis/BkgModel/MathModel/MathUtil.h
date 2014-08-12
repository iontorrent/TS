/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MATHUTIL_H
#define MATHUTIL_H

#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>

// Mark may go ahead and vectorize this as he likes
void MultiplyVectorByScalar(float *my_vec, float my_scalar, int len);
void MultiplyVectorByVector(float *my_vec, float *my_other_vec, int len);
void MultiplyVectorByVector(float *my_output_vec, float *my_vec, float *my_other_vec, int len);
void AddScaledVector(float *start_vec, const float *my_vec, float my_scalar, int len);
float CheckVectorDiffSSQ(float *my_vec, float *my_other_vec, int len);
void CopyVector(float *destination_vec, float *my_vec, int len);
void AccumulateVector(float *destination_vec, float *my_vec, int len);
void DiminishVector(float *destination_vec, float *my_vec, int len);

void CALC_PartialDeriv(float *p1,float *p2,int len,float dp);
void CALC_PartialDeriv_W_EMPHASIS(float *p1,float *p2,float *pe,int len,float dp,float pelen);
void CALC_PartialDeriv_W_EMPHASIS_LONG(float *p1,float *p2,float *pe,int len,float dp);

void Bary_Coord( float *bary_c, float test_x, float test_y, float *x, float *y);
void Bary_Interpolate(float *output, float **Triple, float *bary_c, int npts);   

#endif // MATHUTIL_H
