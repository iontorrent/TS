/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "DiffEqModel.h"
#include "math.h"
#include <algorithm>

// ripped out to separate the math from the optimization procedure


// incorporation traces not yet quite right

// Compute the incorporation "red" hydrogen trace only without any "blue" hydrogen from the bulk or cross-talk
// does not adjust for gain?
void OldRedSolveHydrogenFlowInWell (float *vb_out, float *red_hydrogen, int len, int i_start,float *deltaFrame, float tauB)
{
  float dv = 0.0f;
  float dvn = 0.0f;
  float dv_rs = 0.0f;
  float dt;
  float aval;
  float one_over_tauB = 1.0f/tauB;
  float one_over_one_plus_aval;
  memset (vb_out,0,sizeof (float[i_start]));

  for (int i=i_start; i<len; i++)
  {
    // calculate the 'background' part (the accumulation/decay of the protons in the well
    // normally accounted for by the background calc)
    dt = deltaFrame[i]/2.0f;
    aval = dt*one_over_tauB;
    one_over_one_plus_aval = 1.0f/ (1.0f+aval);

    // calculate new dv
    dvn = (red_hydrogen[i] - dv_rs*one_over_tauB - dv*aval) * one_over_one_plus_aval;
    dv_rs += (dv+dvn) *dt;

    vb_out[i] = dv = dvn;
  }
}

void NewRedSolveHydrogenFlowInWell (float *vb_out, float *red_hydrogen, int len, int i_start, float *deltaFrame, float tauB)
{

  memset (vb_out,0,sizeof (float[i_start]));
  float one_over_two_tauB = 1.0f/ (2.0f*tauB);
  float one_over_one_plus_aval = 1.0f;
  float aval = 0.0f;
  //handle i_start==0 cleanly - not that this ever really happens if we do our job right
  int i=i_start;
  aval = deltaFrame[i]*one_over_two_tauB;
  one_over_one_plus_aval = 1.0f/ (1.0f+aval);
  vb_out[i] = (red_hydrogen[i]) *one_over_one_plus_aval;
  i++;
  // all the rest of the frames continue by recursion
  for (; i<len; i++)
  {
    aval = deltaFrame[i]*one_over_two_tauB;
    one_over_one_plus_aval = 1.0f/ (1.0f+aval);
    // this is very pretty the innovation + decay of existing hydrogen ions
    // note that the decay of existing hydrogen ions is the Pade (1,1) approximation to exp(-t/tauB)
    vb_out[i] = (red_hydrogen[i]-red_hydrogen[i-1]+ (1.0f-aval) *vb_out[i-1]) *one_over_one_plus_aval;
  }
}

void RedSolveHydrogenFlowInWell (float *vb_out, float *red_hydrogen, int len, int i_start,float *deltaFrame, float tauB)
{
  NewRedSolveHydrogenFlowInWell (vb_out,red_hydrogen,len,i_start,deltaFrame,tauB);
  //OldRedSolveHydrogenFlowInWell (vb_out,red_hydrogen,len,i_start,deltaFrame,tauB);
}

void IntegrateRedFromRedTraceObserved (float *red_hydrogen, float *red_obs, int len, int i_start, float *deltaFrame, float tauB)
{

  memset (red_hydrogen,0,sizeof (float[i_start]));
  float one_over_two_tauB = 1.0f/ (2.0f*tauB);

  float aval = 0.0f;
  //handle i_start==0 cleanly - not that this ever really happens if we do our job right
  int i=i_start;
  aval = deltaFrame[i]*one_over_two_tauB;

  red_hydrogen[i] = red_obs[i] * (1.0f+aval);
  i++;
  // all the rest of the frames continue by recursion
  for (; i<len; i++)
  {
    aval = deltaFrame[i]*one_over_two_tauB;
// just invert the Red recursion above
    red_hydrogen[i] = (1.0f+aval)*red_obs[i] -(1.0f-aval)*red_obs[i-1]+ red_hydrogen[i-1];
  }
}

// generates the background trace for a well given the "empty" well measurement of blue_hydrogen ions
void OldBlueSolveBackgroundTrace (float *vb_out, float *blue_hydrogen, int len, float *deltaFrame, float tauB, float etbR)
{
  float dv,dvn,dv_rs;
  float aval;
  float dt;
  float shift_ratio = etbR-1.0f;
  float one_over_tauB = 1.0f/tauB;
  float one_over_one_plus_aval;

  dv = 0.0f;
  dv_rs = 0.0f;
  dt = -1.0f;
  dvn = 0.0f;

  for (int i=0;i < len;i++)
  {
    dt = deltaFrame[i]/2.0f;
    aval = dt*one_over_tauB;
    one_over_one_plus_aval = 1.0f/ (1.0f + aval);

    // calculate new dv
    dvn = (shift_ratio*blue_hydrogen[i] - dv_rs*one_over_tauB - dv*aval) * one_over_one_plus_aval;
    dv_rs += (dv+dvn) *dt;
    dv = dvn;

    vb_out[i] = (dv+blue_hydrogen[i]);
  }
}

void NewBlueSolveBackgroundTrace (float *vb_out, float *blue_hydrogen, int len, float *deltaFrame, float tauB, float etbR)
{
  int   i;
  float xt;

  float one_over_two_taub = 1.0f / (2.0f*tauB);
  float one_over_one_plus_aval = 0.0f;

// initialization is special because nothing before it to recur
  i=0;
  xt = deltaFrame[i]*one_over_two_taub;
  one_over_one_plus_aval = 1.0f/ (1.0f+xt);
  vb_out[i] = ( (etbR+xt) *blue_hydrogen[i]) *one_over_one_plus_aval;
  i++;
  for (;i < len;i++)
  {
    xt=deltaFrame[i]*one_over_two_taub;
    one_over_one_plus_aval = 1.0f/ (1.0f+xt);
    // again a very pretty formulation:  innovation from environment with decay of previous value
    // note the Pade (1,1) approximant to exp(-t/tauB) appearing as the "decay" term
    vb_out[i] = ( (etbR+xt) *blue_hydrogen[i] - (etbR-xt) *blue_hydrogen[i-1]+ (1.0f-xt) *vb_out[i-1]) *one_over_one_plus_aval;
  }
}

void NewBlueSolveBackgroundTrace (double *vb_out, const double *blue_hydrogen, int len, const double *deltaFrame, float tauB, float etbR)
{
  int   i;
  float xt;

  float one_over_two_taub = 1.0f / (2.0f*tauB);
  float one_over_one_plus_aval = 0.0f;

// initialization is special because nothing before it to recur
  i=0;
  xt = deltaFrame[i]*one_over_two_taub;
  one_over_one_plus_aval = 1.0f/ (1.0f+xt);
  vb_out[i] = ( (etbR+xt) *blue_hydrogen[i]) *one_over_one_plus_aval;
  i++;
  for (;i < len;i++)
  {
    xt=deltaFrame[i]*one_over_two_taub;
    one_over_one_plus_aval = 1.0f/ (1.0f+xt);
    // again a very pretty formulation:  innovation from environment with decay of previous value
    // note the Pade (1,1) approximant to exp(-t/tauB) appearing as the "decay" term
    vb_out[i] = ( (etbR+xt) *blue_hydrogen[i] - (etbR-xt) *blue_hydrogen[i-1]+ (1.0f-xt) *vb_out[i-1]) *one_over_one_plus_aval;
  }
}


void BlueSolveBackgroundTrace (float *vb_out, float *blue_hydrogen, int len, float *deltaFrame, float tauB, float etbR)
{
  NewBlueSolveBackgroundTrace (vb_out,blue_hydrogen,len,deltaFrame,tauB,etbR);
  //OldBlueSolveBackgroundTrace(vb_out,blue_hydrogen,len,deltaFrame,tauB,etbR);
}

void OldPurpleSolveTotalTrace (float *vb_out, float *blue_hydrogen, float *red_hydrogen, int len, float *deltaFrame, float tauB, float etbR)
{
  float dv,dvn,dv_rs;
  float aval;
  int   i;
  float dt;
  float shift_ratio = etbR-1.0f;
  float one_over_taub = 1.0f / tauB;
  float one_over_one_plus_aval;

  dv = 0.0f;
  dv_rs = 0.0f;
  dt = -1.0f;
  dvn = 0.0f;
  for (i=0;i < len;i++)
  {
    dt = deltaFrame[i]/2.0f;
    aval = dt * one_over_taub;
    one_over_one_plus_aval = 1.0f/ (1.0f + aval);

    dvn = (red_hydrogen[i] + shift_ratio*blue_hydrogen[i] - dv_rs*one_over_taub - dv*aval) * one_over_one_plus_aval;
    dv_rs += (dv+dvn) *dt;
    dv = dvn;

    vb_out[i] = (dv+blue_hydrogen[i]);
  }
}

//@TODO: vectorized versions of the new fucctions
//@TODO: note that now it is most natural to use "delta-red" hydrogen flux rather than cumulative red hydrogen
void NewPurpleSolveTotalTrace (float *vb_out, float *blue_hydrogen, float *red_hydrogen, int len, float *deltaFrame, float tauB, float etbR)
{
  int   i;
  float xt;

  float one_over_two_taub = 1.0f / (2.0f*tauB);
  float one_over_one_plus_aval = 0.0f;

// initialization is special because nothing before it to recur
  i=0;
  xt = deltaFrame[i]*one_over_two_taub;
  one_over_one_plus_aval = 1.0f/ (1.0f+xt);
  vb_out[i] = ( (red_hydrogen[i]) + (etbR+xt) *blue_hydrogen[i]) *one_over_one_plus_aval;
  i++;
  for (;i < len;i++)
  {
    xt=deltaFrame[i]*one_over_two_taub;
    one_over_one_plus_aval = 1.0f/ (1.0f+xt);
    // again a very pretty formulation:  innovation from environment with decay of previous value
    // note the Pade (1,1) approximant to exp(-t/tauB) appearing as the "decay" term
    vb_out[i] = ( (red_hydrogen[i]-red_hydrogen[i-1]) + (etbR+xt) *blue_hydrogen[i]- (etbR-xt) *blue_hydrogen[i-1]+ (1.0f-xt) *vb_out[i-1]) *one_over_one_plus_aval;
  }
}

void PurpleSolveTotalTrace (float *vb_out, float *blue_hydrogen, float *red_hydrogen, int len, float *deltaFrame, float tauB, float etbR)
{
  NewPurpleSolveTotalTrace (vb_out,blue_hydrogen,red_hydrogen,len, deltaFrame, tauB, etbR);
  //OldPurpleSolveTotalTrace (vb_out,blue_hydrogen,red_hydrogen,len, deltaFrame, tauB, etbR);

}

// return >red_hydrogen< estimate
// from purple_obs and blue_hydrogen
void IntegrateRedFromObservedTotalTrace ( float *red_hydrogen, float *purple_obs, float *blue_hydrogen,  int len, float *deltaFrame, float tauB, float etbR)
{
  int   i;
  float xt;

  float one_over_two_taub = 1.0f / (2.0f*tauB);

// initialization is special because nothing before it to recur
  i=0;
  xt = deltaFrame[i]*one_over_two_taub;

  red_hydrogen[i] = (1.0f+xt)*purple_obs[i] -(etbR+xt)*blue_hydrogen[i];
  i++;
  for (;i < len;i++)
  {
    xt=deltaFrame[i]*one_over_two_taub;
    // invert the recursion above
    red_hydrogen[i] = (1.0f+xt)*purple_obs[i] -(1.0f-xt)*purple_obs[i-1] - ((etbR+xt)*blue_hydrogen[i]-(etbR-xt)*blue_hydrogen[i-1]) +red_hydrogen[i-1];
  }
}

// compute the trace for a single flow
void RedTrace (float *red_out, float *ivalPtr, int npts, float *deltaFrameSeconds, float *deltaFrame, float *nuc_rise_ptr, int SUB_STEPS, int my_start,
               float C, float A, float SP, float kr, float kmax, float d, float molecules_to_micromolar_conversion, float sens, float gain, float tauB,
               PoissonCDFApproxMemo *math_poiss)
{
  ComputeCumulativeIncorporationHydrogens (ivalPtr, npts, deltaFrameSeconds, nuc_rise_ptr, ISIG_SUB_STEPS_SINGLE_FLOW, my_start,  C, A, SP, kr, kmax, d, molecules_to_micromolar_conversion, math_poiss);
  MultiplyVectorByScalar (ivalPtr, sens,npts); // transform hydrogens to signal       // variables used for solving background signal shape
  RedSolveHydrogenFlowInWell (red_out,ivalPtr,npts,my_start,deltaFrame,tauB);
  MultiplyVectorByScalar (red_out,gain,npts);
}

