/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "MiscVec.h"
#include <string.h>
#include <algorithm>
#include <math.h>

#ifndef __INTEL_COMPILER

#include "VectorMacros.h"  // control madness of included headers



void MultiplyVectorByScalar_Vec (float *my_vec, float my_scalar, int len)
{
  v4sf dest;
  v4sf mul = {my_scalar,my_scalar,my_scalar,my_scalar};
  for (int i=0; i<len; i+=VEC_INC)
  {
    LOAD_4FLOATS_FRAMES (dest, my_vec, i, len);
    dest *= mul;
    UNLOAD_4FLOATS_FRAMES (my_vec, dest, i, len);
  }
}

void Dfderr_Step_Vec (int numfb, float** dst, float** et, float** em, int len)
{
  v4sf dst_v, et_v, em_v;
  int i, fb;
  for (fb=0;fb<numfb;fb+=VEC_INC)
  {
    for (i=0;i < len;i++)
    {
      LOAD_4FLOATS_FLOWS (et_v,et,fb,i,numfb);
      LOAD_4FLOATS_FLOWS (em_v,em,fb,i,numfb);

      dst_v = et_v * em_v;

      UNLOAD_4FLOATS_FLOWS (dst,dst_v,fb,i,numfb);
    }
  }
}

void Dfdgain_Step_Vec (int numfb, float** dst, float** src, float** em, int len, float gain)
{
  v4sf dst_v, src_v, em_v;
  v4sf gain_v = (v4sf) {gain, gain, gain, gain};
  int i, fb;
  for (fb=0;fb<numfb;fb+=VEC_INC)
  {
    for (i=0;i < len;i++)
    {
      LOAD_4FLOATS_FLOWS (src_v,src,fb,i,numfb);
      LOAD_4FLOATS_FLOWS (em_v,em,fb,i,numfb);

      dst_v = src_v * em_v / gain_v;

      UNLOAD_4FLOATS_FLOWS (dst,dst_v,fb,i,numfb);
    }
  }
}

#endif // INTEL_COMPILER