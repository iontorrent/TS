/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "Vectorization.h"
#include <string.h>
#include <algorithm>
#include <math.h>

#ifndef __INTEL_COMPILER
void PurpleSolveTotalTrace_Vec (int numfb, float **vb_out, float **blue_hydrogen, float **red_hydrogen, int len, float *deltaFrame, float *tauB, float *etbR, float gain)
{
  v4sf dv,dvn,dv_rs;
  v4sf aval;
  v4sf dt;
  v4sf shift_ratio;
  v4sf tauBV;
  v4sf ttauBV;
  v4sf total;
  v4sf rh,bh;
  v4sf one={1.0,1.0,1.0,1.0};
  v4sf two={2.0,2.0,2.0,2.0};
  int  i,fb;

  float aligned_etbr[numfb] __attribute__ ( (aligned (16)));
  float aligned_tau[numfb] __attribute__ ( (aligned (16)));
  memcpy (aligned_etbr, etbR, sizeof (float) *numfb);
  memcpy (aligned_tau, tauB, sizeof (float) *numfb);

  for (fb=0;fb<numfb;fb+=VEC_INC)
  {
    shift_ratio = * (v4sf*) &aligned_etbr[fb] - one;
    aval = (v4sf) {0,0,0,0};
    dv = (v4sf) {0,0,0,0};
    dv_rs = (v4sf) {0,0,0,0};
    tauBV = * (v4sf*) &aligned_tau[fb];
    ttauBV = tauBV*two;
    for (i=0;i < len;i++)
    {
      dt = (v4sf) {deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};

      LOAD_4FLOATS_FLOWS (rh,red_hydrogen,fb,i,numfb);
      LOAD_4FLOATS_FLOWS (bh,blue_hydrogen,fb,i,numfb);

      aval = dt/ ttauBV;

      dvn = (rh + shift_ratio * bh - dv_rs/tauBV - dv*aval) / (one+aval);
      dv_rs = dv_rs + (dv+dvn) * (dt/two);
      dv = dvn;
      total = (dv+bh);

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,total,fb,i,numfb);
    }
  }

}

void BlueSolveBackgroundTrace_Vec (int numfb, float **vb_out, float **blue_hydrogen, int len,
                                   float *deltaFrame, float *tauB, float *etbR)
{
  v4sf dv,dvn,dv_rs;
  v4sf aval;
  v4sf dt;
  v4sf shift_ratio;
  v4sf tauBV;
  v4sf ttauBV;
  v4sf total;
  v4sf bh;
  v4sf one={1.0,1.0,1.0,1.0};
  v4sf two={2.0,2.0,2.0,2.0};
  int  i,fb;

  float aligned_etbr[numfb] __attribute__ ( (aligned (16)));
  float aligned_tau[numfb] __attribute__ ( (aligned (16)));
  memcpy (aligned_etbr, etbR, sizeof (float) *numfb);
  memcpy (aligned_tau, tauB, sizeof (float) *numfb);

  for (fb=0;fb<numfb;fb+=VEC_INC)
  {
    shift_ratio = * (v4sf *) &aligned_etbr[fb] - one;
    aval = (v4sf) {0,0,0,0};
    dv = (v4sf) {0,0,0,0};
    dv_rs = (v4sf) {0,0,0,0};
    tauBV = * (v4sf *) &aligned_tau[fb];
    ttauBV = tauBV*two;
    for (i=0;i < len;i++)
    {
      dt = (v4sf) {deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};

      LOAD_4FLOATS_FLOWS (bh,blue_hydrogen,fb,i,numfb);

      aval = dt/ ttauBV;

      dvn = (shift_ratio * bh - dv_rs/tauBV - dv*aval) / (one+aval);
      dv_rs = dv_rs + (dv+dvn) * (dt/two);
      dv = dvn;
      total = (dv+bh);

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,total,fb,i,numfb);
    }
  }

}

void RedSolveHydrogenFlowInWell_Vec (int numfb, float **vb_out, float **red_hydrogen, int len, float *deltaFrame, float *tauB)
{
  v4sf dv,dvn,dv_rs;
  v4sf aval;
  v4sf dt;
  v4sf tauBV;
  v4sf ttauBV;
  v4sf rh;
  v4sf one={1.0,1.0,1.0,1.0};
  v4sf two={2.0,2.0,2.0,2.0};
  int  i,fb;

  float aligned_tauB[numfb] __attribute__( (aligned (16)));
  memcpy(aligned_tauB, tauB, sizeof (float) *numfb);

  for (fb=0;fb<numfb;fb+=VEC_INC)
  {
    tauBV = * (v4sf *) &aligned_tauB[fb];
    ttauBV = tauBV*two;
    aval = (v4sf) {0,0,0,0};
    dv = (v4sf) {0,0,0,0};
    dv_rs = (v4sf) {0,0,0,0};
    for (i=0;i < len;i++)
    {
      dt = (v4sf) {deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};

      LOAD_4FLOATS_FLOWS (rh,red_hydrogen,fb,i,numfb);

      aval = dt/ ttauBV;

      dvn = (rh - dv_rs/tauBV - dv*aval) / (one+aval);
      dv_rs = dv_rs + (dv+dvn) * (dt/two);
      dv = dvn;

      // record the result
      UNLOAD_4FLOATS_FLOWS (vb_out,dv,fb,i,numfb);
    }
  }

}

#ifndef FLOW_STEP
#define FLOW_STEP 4
#endif

#ifndef n_to_uM_conv
#define n_to_uM_conv    (0.000062f)
#endif

//assumptions: get 4 flows passed here
// deltaFrameSeconds always the same
// SUB_STEPS always the same
void ParallelLinearComputeCumulativeIncorporationHydrogens_Vec (float **ival_offset, int npts, float *deltaFrameSeconds,
    float **nuc_rise_ptr, int SUB_STEPS, int *my_start,
    float *A, float *SP,
    float *kr, float *kmax, float *d)
{
  int i;
  int common_start;

  v4sf totocc, totgen;
  v4sf pact,pact_new;
  v4sf  c_dntp_top, c_dntp_bot;
  v4sf  hplus_events_sum, hplus_events_current; // mean events per molecule, cumulative and current
  v4sf allone={1.0,1.0,1.0,1.0};
  v4sf allzero={0.0,0.0,0.0,0.0};
  v4sf allhalf={0.5,0.5,0.5,0.5};
  //v4sf twopi = {6.28,6.28,6.28,6.28};
  v4sf mslope;
  v4sf moffset;
  v4sf mscale;

  v4sf tA,sA;
  LOAD_4FLOATS_FRAMES (sA,A,0,4);
  float xA[4];
  for (int q=0; q<FLOW_STEP; q++)
    xA[q] = std::max (A[q],1.0f);
  LOAD_4FLOATS_FRAMES (tA,xA,0,4);
  //tA = std::max(sA,allone);

  // slope for current approximation
  float xslope[4];
  for (int q=0;  q<FLOW_STEP; q++)
    xslope[q] = std::min (-1.0f/sqrt (6.28f*std::max (A[q],1.0f)),-0.5/std::max (A[q],1.0f));
  LOAD_4FLOATS_FRAMES (mslope,xslope,0,4);
  //mslope = - allone/sqrt(twopi*tA);
  //mslope = std::min(mslope, -allhalf/tA);

  moffset = allhalf-mslope*tA;

  for (int q=0; q<FLOW_STEP; q++)
    xA[q] = std::min (A[q],1.0f);
  LOAD_4FLOATS_FRAMES (mscale,xA,0,4);
  //mscale = std::min(sA,allone);
  mscale = __builtin_ia32_maxps (mscale,allone);

  v4sf tSP;
  LOAD_4FLOATS_FRAMES (tSP,SP,0,4);
  mscale *= tSP;
  mslope *= mscale;
  moffset *= mscale;
  pact = mscale;
  totocc = tSP*sA;
  totgen = totocc;
  c_dntp_bot = allzero;
  c_dntp_top = allzero;
  hplus_events_sum = allzero;
  hplus_events_current = allzero;

  for (int q=0; q<FLOW_STEP; q++)
    memset (ival_offset[q],0,sizeof (float[npts]));  // zero the points we don't compute

  v4sf tkr;
  LOAD_4FLOATS_FRAMES (tkr,kr,0,4);
  v4sf td;
  LOAD_4FLOATS_FRAMES (td,d,0,4);
  v4sf tconv = {n_to_uM_conv,n_to_uM_conv,n_to_uM_conv,n_to_uM_conv};
  v4sf scaled_kr;
  scaled_kr = tkr*tconv/td;
  v4sf half_kr;
  half_kr = tkr*allhalf;

  v4sf tkmax;
  LOAD_4FLOATS_FRAMES (tkmax,kmax,0,4);
  v4sf c_dntp_bot_plus_kmax;
  c_dntp_bot_plus_kmax = allone/tkmax;
  v4sf c_dntp_old_effect;
  c_dntp_old_effect = allzero;
  v4sf c_dntp_new_effect;
  c_dntp_new_effect = allzero;

  //@TODO this should be parallel magic
  float sum_totgen = 0.0;
  for (int q=0; q<FLOW_STEP; q++)
    sum_totgen += ( (float *) &totgen) [q];

  // this is just an index
  common_start = my_start[0];
  for (int q=0; q<FLOW_STEP; q++)
    if (common_start<my_start[q])
      common_start = my_start[q];
  // first non-zero index of the computed [dNTP] array for this nucleotide
  int c_dntp_top_ndx = common_start*SUB_STEPS;

  for (i=common_start;i < npts;i++)
  {
    if (sum_totgen > 0.0)
    {
      v4sf ldt;
      float tldt = deltaFrameSeconds[i]/SUB_STEPS;
      ldt = (v4sf) {tldt,tldt,tldt,tldt};
      for (int st=1; (st <= SUB_STEPS) && (sum_totgen > 0.0);st++)  // someone needs computation
      {
        // update top of well equilibrium
        LOAD_4FLOATS_FLOWS (c_dntp_top,nuc_rise_ptr,0,c_dntp_top_ndx,4);
        c_dntp_top_ndx += 1;

        // assume instantaneous equilibrium within the well
        c_dntp_bot = c_dntp_top/ (allone+scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = allone/ (c_dntp_bot+tkmax);

        // Now compute effect of concentration on enzyme rate
        c_dntp_old_effect = c_dntp_new_effect;
        c_dntp_new_effect = c_dntp_bot*c_dntp_bot_plus_kmax;

        // update events per molecule
        hplus_events_current = ldt*half_kr* (c_dntp_new_effect+c_dntp_old_effect);
        hplus_events_sum += hplus_events_current;

        // how many active molecules left at end of time period given poisson process with total intensity of events
        pact_new = hplus_events_sum*mslope+moffset;
        //pact_new = std::max(pact_new,allzero);
        //pact_new = std::min(pact_new,mscale);
        pact_new = __builtin_ia32_maxps (pact_new,allzero);
        pact_new = __builtin_ia32_minps (pact_new,mscale);

        // how many hplus were generated
        totgen -= (pact+pact_new) *allhalf*hplus_events_current;
        pact = pact_new;
        //totgen = std::max(totgen,allzero);
        totgen = __builtin_ia32_maxps (totgen,allzero);

        // or is there a "max" across command?
        sum_totgen = 0.0;
        for (int q=0; q<FLOW_STEP; q++)
          sum_totgen += ( (float *) &totgen) [q];
      }

    }
    v4sf tIval;
    tIval = totocc-totgen;
    UNLOAD_4FLOATS_FLOWS (ival_offset,tIval,0,i,4);

  }
}



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
#endif
