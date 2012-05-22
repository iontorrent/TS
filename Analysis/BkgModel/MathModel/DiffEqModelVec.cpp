/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DiffEqModelVec.h"
#include <string.h>
#include <algorithm>
#include <math.h>

#ifndef __INTEL_COMPILER

#include "VectorMacros.h"  // control madness of included headers

void OldPurpleSolveTotalTrace_Vec (int numfb, float **vb_out, float **blue_hydrogen, float **red_hydrogen, int len, float *deltaFrame, float *tauB, float *etbR, float gain)
{
  v4sf dv,dvn,dv_rs;
  v4sf aval;
  v4sf dt;
  v4sf shift_ratio;
  v4sf tauBV;
  v4sf one_over_tauBV;

  v4sf total;
  v4sf rh,bh;
  v4sf zero={0.0f,0.0f,0.0f,0.0f};
  v4sf one={1.0f,1.0f,1.0f,1.0f};
  v4sf two={2.0f,2.0f,2.0f,2.0f};
  int  i,fb;

  float aligned_etbr[numfb] __attribute__ ( (aligned (16)));
  float aligned_tau[numfb] __attribute__ ( (aligned (16)));
  memcpy (aligned_etbr, etbR, sizeof (float) *numfb);
  memcpy (aligned_tau, tauB, sizeof (float) *numfb);

  for (fb=0;fb<numfb;fb+=VEC_INC)
  {
    shift_ratio = * (v4sf*) &aligned_etbr[fb] - one;
    aval = zero;
    dv = zero;
    dv_rs = zero;
    tauBV = * (v4sf*) &aligned_tau[fb];
    one_over_tauBV = one/tauBV;

    for (i=0;i < len;i++)
    {
      dt = (v4sf) {deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};
      dt /=two;

      LOAD_4FLOATS_FLOWS (rh,red_hydrogen,fb,i,numfb);
      LOAD_4FLOATS_FLOWS (bh,blue_hydrogen,fb,i,numfb);

      aval = dt*one_over_tauBV;
      
      dvn = (rh + shift_ratio * bh - dv_rs*one_over_tauBV - dv*aval) / (one+aval);
      dv_rs = dv_rs + (dv+dvn) * dt;
      dv = dvn;
      total = (dv+bh);

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,total,fb,i,numfb);
    }
  }

}

void NewPurpleSolveTotalTrace_Vec (int numfb, float **vb_out, float **blue_hydrogen, float **red_hydrogen, int len, float *deltaFrame, float *tauB, float *etbR, float gain)
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

  float aligned_etbr[numfb] __attribute__ ( (aligned (16)));
  float aligned_tau[numfb] __attribute__ ( (aligned (16)));
  memcpy (aligned_etbr, etbR, sizeof (float) *numfb);
  memcpy (aligned_tau, tauB, sizeof (float) *numfb);

  for (fb=0;fb<numfb;fb+=VEC_INC)
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

      LOAD_4FLOATS_FLOWS (rh_new,red_hydrogen,fb,i,numfb);
      LOAD_4FLOATS_FLOWS (bh_new,blue_hydrogen,fb,i,numfb);

      out_new = ((rh_new-rh_old) + (etbR_vec+xt)*bh_new-(etbR_vec-xt)*bh_old + (one-xt)*out_old)*one_over_one_plus_xt;

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,out_new,fb,i,numfb);
    }
  }
}

void PurpleSolveTotalTrace_Vec (int numfb, float **vb_out, float **blue_hydrogen, float **red_hydrogen, int len, float *deltaFrame, float *tauB, float *etbR, float gain)
{
  //OldPurpleSolveTotalTrace_Vec(numfb,vb_out,blue_hydrogen,red_hydrogen,len,deltaFrame,tauB,etbR,gain);
  NewPurpleSolveTotalTrace_Vec(numfb,vb_out,blue_hydrogen,red_hydrogen,len,deltaFrame,tauB,etbR,gain);
}

void OldBlueSolveBackgroundTrace_Vec (int numfb, float **vb_out, float **blue_hydrogen, int len,
                                   float *deltaFrame, float *tauB, float *etbR)
{
  v4sf dv,dvn,dv_rs;
  v4sf aval;
  v4sf dt;
  v4sf shift_ratio;
  v4sf tauBV;
  v4sf one_over_tauBV;
  v4sf total;
  v4sf bh;
  v4sf zero={0.0f,0.0f,0.0f,0.0f};
  v4sf one={1.0f,1.0f,1.0f,1.0f};
  v4sf two={2.0f,2.0f,2.0f,2.0f};
  int  i,fb;

  float aligned_etbr[numfb] __attribute__ ( (aligned (16)));
  float aligned_tau[numfb] __attribute__ ( (aligned (16)));
  memcpy (aligned_etbr, etbR, sizeof (float) *numfb);
  memcpy (aligned_tau, tauB, sizeof (float) *numfb);

  for (fb=0;fb<numfb;fb+=VEC_INC)
  {
    shift_ratio = * (v4sf *) &aligned_etbr[fb] - one;
    aval = zero;
    dv = zero;
    dv_rs = zero;
    tauBV = * (v4sf *) &aligned_tau[fb];
    one_over_tauBV = one/tauBV;

    for (i=0;i < len;i++)
    {
      dt = (v4sf) {deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};
      dt /= two;
      LOAD_4FLOATS_FLOWS (bh,blue_hydrogen,fb,i,numfb);

      aval = dt*one_over_tauBV;

      dvn = (shift_ratio * bh - dv_rs*one_over_tauBV - dv*aval) / (one+aval);
      dv_rs = dv_rs + (dv+dvn) * dt;
      dv = dvn;
      total = (dv+bh);

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,total,fb,i,numfb);
    }
  }

}
void NewBlueSolveBackgroundTrace_Vec (int numfb, float **vb_out, float **blue_hydrogen,  int len, float *deltaFrame, float *tauB, float *etbR)
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

  float aligned_etbr[numfb] __attribute__ ( (aligned (16)));
  float aligned_tau[numfb] __attribute__ ( (aligned (16)));
  memcpy (aligned_etbr, etbR, sizeof (float) *numfb);
  memcpy (aligned_tau, tauB, sizeof (float) *numfb);

  for (fb=0;fb<numfb;fb+=VEC_INC)
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


      LOAD_4FLOATS_FLOWS (bh_new,blue_hydrogen,fb,i,numfb);

      out_new = ((etbR_vec+xt)*bh_new-(etbR_vec-xt)*bh_old + (one-xt)*out_old)*one_over_one_plus_xt;

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,out_new,fb,i,numfb);
    }
  }
}

void BlueSolveBackgroundTrace_Vec (int numfb, float **vb_out, float **blue_hydrogen,  int len, float *deltaFrame, float *tauB, float *etbR)
{
  //OldBlueSolveBackgroundTrace_Vec(numfb,vb_out,blue_hydrogen,len,deltaFrame,tauB,etbR);
  NewBlueSolveBackgroundTrace_Vec(numfb,vb_out,blue_hydrogen,len,deltaFrame,tauB,etbR);
}

void OldRedSolveHydrogenFlowInWell_Vec (int numfb, float **vb_out, float **red_hydrogen, int len, float *deltaFrame, float *tauB)
{
  v4sf dv,dvn,dv_rs;
  v4sf aval;
  v4sf dt;
  v4sf tauBV;
  v4sf one_over_tauBV;
  v4sf rh;
  v4sf zero = {0.0f,0.0f,0.0f,0.0f};
  v4sf one={1.0f,1.0f,1.0f,1.0f};
  v4sf two={2.0f,2.0f,2.0f,2.0f};
  int  i,fb;

  float aligned_tauB[numfb] __attribute__( (aligned (16)));
  memcpy(aligned_tauB, tauB, sizeof (float) *numfb);

  for (fb=0;fb<numfb;fb+=VEC_INC)
  {
    tauBV = * (v4sf *) &aligned_tauB[fb];
    aval = zero;
    dv = zero;
    dv_rs = zero;
    one_over_tauBV = one/tauBV;
    for (i=0;i < len;i++)
    {
      dt = (v4sf) {deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};
      dt /= two;
      LOAD_4FLOATS_FLOWS (rh,red_hydrogen,fb,i,numfb);

      aval = dt*one_over_tauBV;

      dvn = (rh - dv_rs*one_over_tauBV - dv*aval) / (one+aval);
      dv_rs = dv_rs + (dv+dvn) * dt;
      dv = dvn;

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,dv,fb,i,numfb);
    }
  }

}

void NewRedSolveHydrogenFlowInWell_Vec (int numfb, float **vb_out, float **red_hydrogen, int len, float *deltaFrame, float *tauB)
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

  float aligned_tau[numfb] __attribute__ ( (aligned (16)));

  memcpy (aligned_tau, tauB, sizeof (float) *numfb);

  for (fb=0;fb<numfb;fb+=VEC_INC)
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

      LOAD_4FLOATS_FLOWS (rh_new,red_hydrogen,fb,i,numfb);

      out_new = ((rh_new-rh_old) + (one-xt)*out_old)*one_over_one_plus_xt;

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,out_new,fb,i,numfb);
    }
  }
}

void RedSolveHydrogenFlowInWell_Vec  (int numfb, float **vb_out, float **red_hydrogen, int len, float *deltaFrame, float *tauB)
{
  //OldRedSolveHydrogenFlowInWell_Vec(numfb,vb_out,red_hydrogen,len,deltaFrame,tauB);
  NewRedSolveHydrogenFlowInWell_Vec(numfb,vb_out,red_hydrogen,len,deltaFrame,tauB);
}


#endif
