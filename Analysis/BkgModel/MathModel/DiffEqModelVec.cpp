/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DiffEqModelVec.h"
#include <string.h>
#include <algorithm>
#include <math.h>

#ifndef __INTEL_COMPILER

#include "VectorMacros.h"  // control madness of included headers

void MathModel::PurpleSolveTotalTrace_Vec (float **vb_out, float **blue_hydrogen, float **red_hydrogen, int len, const float *deltaFrame, float *tauB, float *etbR, float gain, int flow_block_size)
{
  v4sf xt;
  v4sf dt;
  v4sf etbR_vec;
  v4sf tauBV;
  v4sf one_over_two_tauBV;
  v4sf one_over_one_plus_xt;

  v4sf out_new,out_old;
  v4sf rh_new,bh_new,rh_old,bh_old;
  v4sf zero={0.0f,0.0f,0.0f,0.0f};
  v4sf one={1.0f,1.0f,1.0f,1.0f};
  v4sf two={2.0f,2.0f,2.0f,2.0f};
  int  i,fb;

  float aligned_etbr[flow_block_size] __attribute__ ( (aligned (16)));
  float aligned_tau[flow_block_size] __attribute__ ( (aligned (16)));
  memcpy (aligned_etbr, etbR, sizeof (float) *flow_block_size);
  memcpy (aligned_tau, tauB, sizeof (float) *flow_block_size);

  // We may work on some unneeded points at either end, but the data will be tossed.
  for (fb=0;fb<flow_block_size;fb+=VEC_INC)
  {

    etbR_vec = * (v4sf *) &aligned_etbr[fb];
    tauBV = * (v4sf*) &aligned_tau[fb];
    one_over_two_tauBV = one/(two*tauBV);

    // intermediate values
    bh_new = zero;
    rh_new = zero;
    out_new = zero;
    for (i=0;i < len;i++)
    {
      out_old = out_new;
      rh_old = rh_new;
      bh_old = bh_new;
      
      dt = (v4sf) {deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};
      xt = dt*one_over_two_tauBV;
      one_over_one_plus_xt = one/ (one+xt);

      LOAD_4FLOATS_FLOWS (rh_new,red_hydrogen,fb,i,flow_block_size);
      LOAD_4FLOATS_FLOWS (bh_new,blue_hydrogen,fb,i,flow_block_size);

      out_new = ((rh_new-rh_old) + (etbR_vec+xt)*bh_new-(etbR_vec-xt)*bh_old + (one-xt)*out_old)*one_over_one_plus_xt;

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,out_new,fb,i,flow_block_size);
    }
  }
}

void MathModel::BlueSolveBackgroundTrace_Vec (float **vb_out, float **blue_hydrogen,  int len, 
    const float *deltaFrame, const float *tauB, const float *etbR, int flow_block_size)
{
  v4sf xt;
  v4sf dt;
  v4sf etbR_vec;
  v4sf tauBV;
  v4sf one_over_two_tauBV;
  v4sf one_over_one_plus_xt;

  v4sf out_new,out_old;
  v4sf bh_new,bh_old;
  v4sf zero={0.0f,0.0f,0.0f,0.0f};
  v4sf one={1.0f,1.0f,1.0f,1.0f};
  v4sf two={2.0f,2.0f,2.0f,2.0f};
  int  i,fb;

  float aligned_etbr[flow_block_size] __attribute__ ( (aligned (16)));
  float aligned_tau[flow_block_size] __attribute__ ( (aligned (16)));
  memcpy (aligned_etbr, etbR, sizeof (float) *flow_block_size);
  memcpy (aligned_tau, tauB, sizeof (float) *flow_block_size);

  for (fb=0;fb<flow_block_size;fb+=VEC_INC)
  {

    etbR_vec = * (v4sf *) &aligned_etbr[fb];
    tauBV = * (v4sf*) &aligned_tau[fb];
    one_over_two_tauBV = one/(two*tauBV);

    // intermediate values
    bh_new = zero;

    out_new = zero;
    for (i=0;i < len;i++)
    {
      out_old = out_new;

      bh_old = bh_new;

      dt = (v4sf) {deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};
      xt = dt*one_over_two_tauBV;
      one_over_one_plus_xt = one/ (one+xt);


      LOAD_4FLOATS_FLOWS (bh_new,blue_hydrogen,fb,i,flow_block_size);

      out_new = ((etbR_vec+xt)*bh_new-(etbR_vec-xt)*bh_old + (one-xt)*out_old)*one_over_one_plus_xt;

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,out_new,fb,i,flow_block_size);
    }
  }
}


void MathModel::RedSolveHydrogenFlowInWell_Vec  (float * const *vb_out, const float * const *red_hydrogen, int len, const float *deltaFrame, const float *tauB,
    int flow_block_size
  )
{
  v4sf xt;
  v4sf dt;
  v4sf tauBV;
  v4sf one_over_two_tauBV;
  v4sf one_over_one_plus_xt;

  v4sf out_new,out_old;
  v4sf rh_new,rh_old;
  v4sf zero={0.0f,0.0f,0.0f,0.0f};
  v4sf one={1.0f,1.0f,1.0f,1.0f};
  v4sf two={2.0f,2.0f,2.0f,2.0f};
  int  i,fb;

  float aligned_tau[flow_block_size] __attribute__ ( (aligned (16)));

  memcpy (aligned_tau, tauB, sizeof (float) *flow_block_size);

  for (fb=0;fb<flow_block_size;fb+=VEC_INC)
  {


    tauBV = * (v4sf*) &aligned_tau[fb];
    one_over_two_tauBV = one/(two*tauBV);

    // intermediate values

    rh_new = zero;
    out_new = zero;
    for (i=0;i < len;i++)
    {
      out_old = out_new;
      rh_old = rh_new;


      dt = (v4sf) {deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};
      xt = dt*one_over_two_tauBV;
      one_over_one_plus_xt = one/ (one+xt);

      LOAD_4FLOATS_FLOWS (rh_new,red_hydrogen,fb,i,flow_block_size);

      out_new = ((rh_new-rh_old) + (one-xt)*out_old)*one_over_one_plus_xt;

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,out_new,fb,i,flow_block_size);
    }
  }
}

#endif
