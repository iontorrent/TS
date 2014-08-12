/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "Hydrogen.h"
#include "math.h"
#include <algorithm>
#include "MathUtil.h"


// shielding layer to insulate from choice
void MathModel::ComputeCumulativeIncorporationHydrogens (
    float *ival_offset, int npts, const float *deltaFrameSeconds,
    const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
    float A, float SP,
    float kr, float kmax, float d, float molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss,
    int incorporationModelType ) // default value for external calls
{
  bool purely_local = false;
  if (math_poiss==NULL)
  {
    math_poiss = new PoissonCDFApproxMemo;
    math_poiss->Allocate (MAX_POISSON_TABLE_COL,MAX_POISSON_TABLE_ROW,POISSON_TABLE_STEP);
    math_poiss->GenerateValues();
    purely_local = true;
  }

  // handle sign here by "function composition"
  float tA = A;
  if (A<0.0f)
    tA = -A;

  switch( incorporationModelType ){
    case 1:
      MathModel::ReducedComputeCumulativeIncorporationHydrogens (ival_offset,npts,deltaFrameSeconds,nuc_rise_ptr,SUB_STEPS,my_start,C,tA,SP,kr,kmax,d,molecules_to_micromolar_conversion, math_poiss);
      break;
    case 2:
      MathModel::Reduced2ComputeCumulativeIncorporationHydrogens (ival_offset,npts,deltaFrameSeconds,nuc_rise_ptr,SUB_STEPS,my_start,C,tA,SP,kr,kmax,d,molecules_to_micromolar_conversion, math_poiss);
      break;
    case 3:
        MathModel::Reduced3ComputeCumulativeIncorporationHydrogens (ival_offset,npts,deltaFrameSeconds,nuc_rise_ptr,SUB_STEPS,my_start,C,tA,SP,kr,kmax,d,molecules_to_micromolar_conversion, math_poiss);
        break;

    case 0:
    default:
      MathModel::SimplifyComputeCumulativeIncorporationHydrogens (ival_offset,npts,deltaFrameSeconds,nuc_rise_ptr,SUB_STEPS,my_start,C,tA,SP,kr,kmax,d, molecules_to_micromolar_conversion, math_poiss);
      break;
  }

  if (purely_local)
  {
    delete math_poiss;
    math_poiss=NULL;
  }
  // flip sign: we never have negative incorporations, but we can have cross-talk over-subtracted which we pretend has the same shape
  if (A<0.0f)
    MultiplyVectorByScalar (ival_offset,-1.0f,npts);
}



#define FLOW_STEP 4

void MathModel::ParallelSimpleComputeCumulativeIncorporationHydrogens (
    float **ival_offset, int npts, const float *deltaFrameSeconds,
    const float *const *nuc_rise_ptr, int SUB_STEPS, int *my_start,
    float *A, float *SP,
    float *kr, float *kmax, float *d, float *molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss, int incorporationModelType)
{

  // handle sign here by function composition
  float tA[FLOW_STEP];
  for (int q=0; q<FLOW_STEP; q++)
    if (A[q]<0.0f)
      tA[q] = -A[q];
    else
      tA[q] = A[q];

  switch( incorporationModelType ){
    case 1:
          for (int q=0; q<FLOW_STEP; q++){
              ReducedComputeCumulativeIncorporationHydrogens (ival_offset[q],npts,deltaFrameSeconds,nuc_rise_ptr[q],SUB_STEPS,
                                                              my_start[q],0,tA[q],SP[q],kr[q],kmax[q],d[q],molecules_to_micromolar_conversion[q], math_poiss);
          }
          break;
    case 2:
        for (int q=0; q<FLOW_STEP; q++){
            Reduced2ComputeCumulativeIncorporationHydrogens (ival_offset[q],npts,deltaFrameSeconds,nuc_rise_ptr[q],SUB_STEPS,
                                                            my_start[q],0,tA[q],SP[q],kr[q],kmax[q],d[q],molecules_to_micromolar_conversion[q], math_poiss);
        }
        break;
    case 3:
      for (int q=0; q<FLOW_STEP; q++){
          Reduced3ComputeCumulativeIncorporationHydrogens (ival_offset[q],npts,deltaFrameSeconds,nuc_rise_ptr[q],SUB_STEPS,
                                                          my_start[q],0,tA[q],SP[q],kr[q],kmax[q],d[q],molecules_to_micromolar_conversion[q], math_poiss);
      }
      break;
    case 0:
    default:
          UnsignedParallelSimpleComputeCumulativeIncorporationHydrogens (ival_offset,npts,deltaFrameSeconds,nuc_rise_ptr,SUB_STEPS,my_start,tA,SP,kr,kmax,d,molecules_to_micromolar_conversion,math_poiss);
          break;
  }


  // flip sign - we never really have negative incorporation, but we can "over-subtract" cross-talk
  for (int q=0; q<FLOW_STEP; q++)
    if (A[q]<0.0f)
      MultiplyVectorByScalar (ival_offset[q],-1.0f,npts);
}

//assumptions: get 4 flows passed here
// deltaFrameSeconds always the same
// SUB_STEPS always the same
// pretend we are vectorizing this function
// by the crude example of doing arrays pretending to be vectors for all.
void MathModel::UnsignedParallelSimpleComputeCumulativeIncorporationHydrogens (
    float **ival_offset, int npts, const float *deltaFrameSeconds,
    const float * const *nuc_rise_ptr, int SUB_STEPS, int *my_start,
    float *A, float *SP,
    float *kr, float *kmax, float *d, float *molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss)
{
  int i;
  int common_start;

  float totocc[FLOW_STEP], totgen[FLOW_STEP];
  float pact[FLOW_STEP],pact_new[FLOW_STEP];
  float  c_dntp_top[FLOW_STEP], c_dntp_bot[FLOW_STEP];
  float  hplus_events_sum[FLOW_STEP], hplus_events_current[FLOW_STEP]; // mean events per molecule, cumulative and current
  float enzyme_dt[FLOW_STEP];
  float Aint[FLOW_STEP];

// A a pointer, so isolate by copying
  for (int q=0;q<FLOW_STEP;q++)
    Aint[q] = A[q];

  MixtureMemo mix_memo[4];
  float tA[FLOW_STEP];
  for (int q=0; q<FLOW_STEP; q++)
    tA[q] = mix_memo[q].Generate (Aint[q],math_poiss); // don't damage A now that it is a pointer

  for (int q=0; q<FLOW_STEP; q++)
    mix_memo[q].ScaleMixture (SP[q]);
  for (int q=0; q<FLOW_STEP; q++)
    pact[q] = mix_memo[q].total_live;  // active polymerases
  for (int q=0; q<FLOW_STEP; q++)
    totocc[q] = SP[q]*tA[q];  // how many hydrogens we'll eventually generate

  for (int q=0; q<FLOW_STEP; q++)
    totgen[q] = totocc[q];  // number remaining to generate

  for (int q=0; q<FLOW_STEP; q++)
    c_dntp_bot[q] = 0.0f; // concentration of dNTP in the well
  for (int q=0; q<FLOW_STEP; q++)
    c_dntp_top[q] = 0.0; // concentration at top
  for (int q=0; q<FLOW_STEP; q++)
    hplus_events_sum[q] = 0.0f;
  for (int q=0; q<FLOW_STEP; q++)
    hplus_events_current[q] = 0.0f; // Events per molecule

  for (int q=0; q<FLOW_STEP; q++)
    memset (ival_offset[q],0,sizeof (float[npts]));  // zero the points we don't compute

  float scaled_kr[FLOW_STEP];
  for (int q=0; q<FLOW_STEP; q++)
    scaled_kr[q] = kr[q]*molecules_to_micromolar_conversion[q]/d[q]; // convert molecules of polymerase to active concentraction
  float half_kr[FLOW_STEP];
  for (int q=0; q<FLOW_STEP; q++)
    half_kr[q] = kr[q] *0.5f/SUB_STEPS; // for averaging


  float c_dntp_bot_plus_kmax[FLOW_STEP];
  for (int q=0; q<FLOW_STEP; q++)
    c_dntp_bot_plus_kmax[q] = 1.0f/kmax[q];

  float c_dntp_old_effect[FLOW_STEP];
  for (int q=0; q<FLOW_STEP; q++)
    c_dntp_old_effect[q] = 0.0f;
  float c_dntp_new_effect[FLOW_STEP];
  for (int q=0; q<FLOW_STEP; q++)
    c_dntp_new_effect[q] = 0.0f;


  float sum_totgen = 0.0f;
  for (int q=0; q<FLOW_STEP; q++)
    sum_totgen += totgen[q];

  // find the earliest time frame we need to start across all flows
  common_start = my_start[0];
  for (int q=0; q<FLOW_STEP; q++)
  {
    // if my_start is earlier than the current one
    if (common_start>my_start[q])
      common_start = my_start[q];
  }
  if (common_start<0)
  {
    common_start = 0;
    printf ("Error: i_start outside of range\n");
  }

  // first non-zero index of the computed [dNTP] array for this nucleotide
  int c_dntp_top_ndx = common_start*SUB_STEPS;

  for (i=common_start;i < npts;i++)
  {
    if (sum_totgen > 0.0f)
    {
      // cannot accelerate here because half_kr is distinct per
      for (int q=0; q<FLOW_STEP; q++)
        enzyme_dt[q] = deltaFrameSeconds[i]*half_kr[q];

      for (int st=1; (st <= SUB_STEPS) && (sum_totgen > 0.0f);st++)  // someone needs computation
      {
        // update top of well concentration in the bulk for FLOW_STEP flows
        for (int q=0; q<FLOW_STEP; q++)
          c_dntp_top[q] = nuc_rise_ptr[q][c_dntp_top_ndx];
        c_dntp_top_ndx += 1;

        // assume instantaneous equilibrium within the well
        for (int q=0; q<FLOW_STEP; q++)
          c_dntp_bot[q] = c_dntp_top[q]/ (1.0f+ scaled_kr[q]*pact[q]*c_dntp_bot_plus_kmax[q]); // the level at which new nucs are used up as fast as they diffuse in
        for (int q=0; q<FLOW_STEP; q++)
          c_dntp_bot_plus_kmax[q] = 1.0f/ (c_dntp_bot[q] + kmax[q]); // scale for michaelis-menten kinetics, assuming nucs are limiting factor

        // Now compute effect of concentration on enzyme rate
        for (int q=0; q<FLOW_STEP; q++)
          c_dntp_old_effect[q] = c_dntp_new_effect[q];
        for (int q=0; q<FLOW_STEP; q++)
          c_dntp_new_effect[q] = c_dntp_bot[q]*c_dntp_bot_plus_kmax[q]; // current effect of concentration on enzyme rate

        // update events per molecule
        for (int q=0; q<FLOW_STEP; q++)
          hplus_events_current[q] = enzyme_dt[q]* (c_dntp_new_effect[q]+c_dntp_old_effect[q]); // events per molecule is average rate * time of rate
        for (int q=0; q<FLOW_STEP; q++)
          hplus_events_sum[q] += hplus_events_current[q];

        // how many active molecules left at end of time period given poisson process with total intensity of events
        for (int q=0; q<FLOW_STEP; q++)
          pact_new[q] = mix_memo[q].GetStep (hplus_events_sum[q]);


        // how many hplus were generated
        for (int q=0; q<FLOW_STEP; q++)
          totgen[q] -= ( (pact[q]+pact_new[q]) * 0.5f) * hplus_events_current[q]; // active molecules * events per molecule
        for (int q=0; q<FLOW_STEP; q++)
          pact[q] = pact_new[q];
        for (int q=0; q<FLOW_STEP; q++)
          totgen[q] = std::max (totgen[q],0.0f);
        // or is there a "max" within command?
        sum_totgen = 0.0f;
        for (int q=0; q<FLOW_STEP; q++)
          sum_totgen += totgen[q];
      }

    }
    for (int q=0; q<FLOW_STEP; q++)
      ival_offset[q][i] = (totocc[q]-totgen[q]);

  }
}


// try to simplify
void MathModel::SimplifyComputeCumulativeIncorporationHydrogens (
    float *ival_offset, int npts, const float *deltaFrameSeconds,
    const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
    float A, float SP,
    float kr, float kmax, float d, float molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss)
{
  int i;
  float totocc, totgen;
//    mixed_poisson_struct mix_ctrl;
  MixtureMemo mix_memo;

  float pact,pact_new;
  float   c_dntp_bot;
  float  hplus_events_sum, hplus_events_current; // mean events per molecule, cumulative and current

  float enzyme_dt;


  A = mix_memo.Generate (A,math_poiss);

  mix_memo.ScaleMixture (SP);

  pact = mix_memo.total_live;  // active polymerases
  totocc = SP*A;  // how many hydrogens we'll eventually generate

  totgen = totocc;  // number remaining to generate

  c_dntp_bot = 0.0f; // concentration of dNTP in the well
 
  hplus_events_sum = hplus_events_current = 0.0f; // Events per molecule

  memset (ival_offset,0,sizeof (float[my_start]));  // zero the points we don't compute

  float scaled_kr = kr*molecules_to_micromolar_conversion/d; // convert molecules of polymerase to active concentraction
  float half_kr = kr *0.5f/SUB_STEPS; // for averaging

  // first non-zero index of the computed [dNTP] array for this nucleotide
  int c_dntp_top_ndx = my_start*SUB_STEPS;
  float c_dntp_bot_plus_kmax = 1.0f/kmax;
  float c_dntp_old_effect = 0.0f;
  float c_dntp_new_effect = 0.0f;
  int st;

  for (i=my_start;i < npts;i++)
  {
    if (totgen > 0.0f)
    {
      enzyme_dt = half_kr*deltaFrameSeconds[i];

      for (st=1; (st <= SUB_STEPS) && (totgen > 0.0f);st++)
      {

        // assume instantaneous equilibrium within the well
        c_dntp_bot = nuc_rise_ptr[c_dntp_top_ndx];
        c_dntp_top_ndx++;

        c_dntp_bot /= (1.0f+ scaled_kr*pact*c_dntp_bot_plus_kmax); // the level at which new nucs are used up as fast as they diffuse in
        c_dntp_bot_plus_kmax = 1 / (c_dntp_bot + kmax); // scale for michaelis-menten kinetics, assuming nucs are limiting factor

        // Now compute effect of concentration on enzyme rate
        c_dntp_old_effect = c_dntp_new_effect;
        c_dntp_new_effect = c_dntp_bot*c_dntp_bot_plus_kmax; // current effect of concentration on enzyme rate

        // update events per molecule
        hplus_events_current = enzyme_dt* (c_dntp_new_effect+c_dntp_old_effect); // events per molecule is average rate * time of rate
        hplus_events_sum += hplus_events_current;

        // how many active molecules left at end of time period given poisson process with total intensity of events
        // exp(-t) * (1+t+t^2/+t^3/6+...) where we interpolate between polynomial lengths by A
        // exp(-t) ( 1+... + frac*(t^k/k!)) where k = ceil(A-1) and frac = A-floor(A), for A>=1
        pact_new = mix_memo.GetStep (hplus_events_sum);
        pact += pact_new;
        pact *= 0.5f;
        // how many hplus were generated
        totgen -= pact * hplus_events_current;  // active molecules * events per molecule
        pact = pact_new;
      }

      if (totgen < 0.0f) totgen = 0.0f;

      ival_offset[i] = (totocc-totgen);
    }
    else
    {
      ival_offset[i] = totocc;
    }

  }
}

//Reduced model that ignores diffusion rate
void MathModel::ReducedComputeCumulativeIncorporationHydrogens (float *ival_offset, int npts, const float *deltaFrameSeconds,
    const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
    float A, float SP,
    float kr, float kmax, float d, float molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss)
{
  int i;
  float totocc, totgen;
//    mixed_poisson_struct mix_ctrl;
  MixtureMemo mix_memo;

  float pact,pact_new;
  float   c_dntp_bot;
  float  hplus_events_sum, hplus_events_current; // mean events per molecule, cumulative and current

  float enzyme_dt;

  (void)d; (void)molecules_to_micromolar_conversion;

  A = mix_memo.Generate (A,math_poiss);

  mix_memo.ScaleMixture (SP);

  pact = mix_memo.total_live;  // active polymerases
  totocc = SP*A;  // how many hydrogens we'll eventually generate

  totgen = totocc;  // number remaining to generate

  c_dntp_bot = 0.0f; // concentration of dNTP in the well

  hplus_events_sum = hplus_events_current = 0.0f; // Events per molecule

  memset (ival_offset,0,sizeof (float[my_start]));  // zero the points we don't compute

  float half_kr = kr *0.5f/SUB_STEPS; // for averaging

  // first non-zero index of the computed [dNTP] array for this nucleotide
  int c_dntp_top_ndx = my_start*SUB_STEPS;
  float c_dntp_bot_plus_kmax = 1.0f/kmax;
  float c_dntp_old_effect = 0.0f;
  float c_dntp_new_effect = 0.0f;
  int st;

  for (i=my_start;i < npts;i++)
  {
    if (totgen > 0.0f)
    {
      enzyme_dt = half_kr*deltaFrameSeconds[i];

      for (st=1; (st <= SUB_STEPS) && (totgen > 0.0f);st++)
      {

        // assume instantaneous equilibrium within the well
        c_dntp_bot = nuc_rise_ptr[c_dntp_top_ndx];
        c_dntp_top_ndx++;

        c_dntp_bot_plus_kmax = 1 / (c_dntp_bot + kmax); // scale for michaelis-menten kinetics, assuming nucs are limiting factor

        // Now compute effect of concentration on enzyme rate
        c_dntp_old_effect = c_dntp_new_effect;
        c_dntp_new_effect = c_dntp_bot*c_dntp_bot_plus_kmax; // current effect of concentration on enzyme rate

        // update events per molecule
        hplus_events_current = enzyme_dt* (c_dntp_new_effect+c_dntp_old_effect); // events per molecule is average rate * time of rate
        hplus_events_sum += hplus_events_current;

        // how many active molecules left at end of time period given poisson process with total intensity of events
        // exp(-t) * (1+t+t^2/+t^3/6+...) where we interpolate between polynomial lengths by A
        // exp(-t) ( 1+... + frac*(t^k/k!)) where k = ceil(A-1) and frac = A-floor(A), for A>=1
        pact_new = mix_memo.GetStep (hplus_events_sum);
        pact += pact_new;
        pact *= 0.5f;
        // how many hplus were generated
        totgen -= pact * hplus_events_current;  // active molecules * events per molecule
        pact = pact_new;
      }

      if (totgen < 0.0f) totgen = 0.0f;

      ival_offset[i] = (totocc-totgen);
    }
    else
    {
      ival_offset[i] = totocc;
    }

  }
}

//Reduced model that ignores diffusion rate
void MathModel::Reduced2ComputeCumulativeIncorporationHydrogens (float *ival_offset, int npts, const float *deltaFrameSeconds,
    const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
    float A, float SP,
    float kr, float kmax, float d, float molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss)
{
  int i;
  float totocc, totgen;
//    mixed_poisson_struct mix_ctrl;
  MixtureMemo mix_memo;

  float pact,pact_new;
  float   c_dntp_bot;
  float  hplus_events_sum, hplus_events_current; // mean events per molecule, cumulative and current

  float enzyme_dt;

  (void)d; (void)molecules_to_micromolar_conversion;

  A = mix_memo.Generate (A,math_poiss);

  mix_memo.ScaleMixture (SP);

  pact = mix_memo.total_live;  // active polymerases
  totocc = SP*A;  // how many hydrogens we'll eventually generate

  totgen = totocc;  // number remaining to generate

  c_dntp_bot = 0.0f; // concentration of dNTP in the well

  hplus_events_sum = hplus_events_current = 0.0f; // Events per molecule

  memset (ival_offset,0,sizeof (float[my_start]));  // zero the points we don't compute

  float half_kr = kr *0.5f/SUB_STEPS; // for averaging

  // first non-zero index of the computed [dNTP] array for this nucleotide
  int c_dntp_top_ndx = my_start*SUB_STEPS;
  float c_dntp_old_effect = 0.0f;
  float c_dntp_new_effect = 0.0f;
  int st;

  for (i=my_start;i < npts;i++)
  {
    if (totgen > 0.0f)
    {
      enzyme_dt = half_kr*deltaFrameSeconds[i];

      for (st=1; (st <= SUB_STEPS) && (totgen > 0.0f);st++)
      {

        // assume instantaneous equilibrium within the well
        c_dntp_bot = nuc_rise_ptr[c_dntp_top_ndx] / 50.;
        c_dntp_top_ndx++;

        // Now compute effect of concentration on enzyme rate
        c_dntp_old_effect = c_dntp_new_effect;
        c_dntp_new_effect = c_dntp_bot; // current effect of concentration on enzyme rate

        // update events per molecule
        hplus_events_current = enzyme_dt* (c_dntp_new_effect+c_dntp_old_effect); // events per molecule is average rate * time of rate
        hplus_events_sum += hplus_events_current;

        // how many active molecules left at end of time period given poisson process with total intensity of events
        // exp(-t) * (1+t+t^2/+t^3/6+...) where we interpolate between polynomial lengths by A
        // exp(-t) ( 1+... + frac*(t^k/k!)) where k = ceil(A-1) and frac = A-floor(A), for A>=1
        pact_new = mix_memo.GetStep (hplus_events_sum);
        pact += pact_new;
        pact *= 0.5f;
        // how many hplus were generated
        totgen -= pact * hplus_events_current;  // active molecules * events per molecule
        pact = pact_new;
      }

      if (totgen < 0.0f) totgen = 0.0f;

      ival_offset[i] = (totocc-totgen);
    }
    else
    {
      ival_offset[i] = totocc;
    }

  }
}

//Reduced model that ignores diffusion rate
void MathModel::Reduced3ComputeCumulativeIncorporationHydrogens (float *ival_offset, int npts, const float *deltaFrameSeconds,
    const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
    float A, float SP,
    float kr, float kmax, float d, float molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss)
{
  int i;
  float totocc, totgen;
//    mixed_poisson_struct mix_ctrl;
  MixtureMemo mix_memo;

  float pact,pact_new;
  float   c_dntp_bot;
  float  hplus_events_sum, hplus_events_current; // mean events per molecule, cumulative and current

  float enzyme_dt;

  (void)d; (void)molecules_to_micromolar_conversion;

  A = mix_memo.Generate (A,math_poiss);

  mix_memo.ScaleMixture (SP);

  pact = mix_memo.total_live;  // active polymerases
  totocc = SP*A;  // how many hydrogens we'll eventually generate

  totgen = totocc;  // number remaining to generate

  c_dntp_bot = 0.0f; // concentration of dNTP in the well

  hplus_events_sum = hplus_events_current = 0.0f; // Events per molecule

  memset (ival_offset,0,sizeof (float[my_start]));  // zero the points we don't compute

  float half_kr = kr/SUB_STEPS; // for averaging

  // first non-zero index of the computed [dNTP] array for this nucleotide
  int c_dntp_top_ndx = my_start*SUB_STEPS;
  float c_dntp_new_effect = 0.0f;
  int st;

  for (i=my_start;i < npts;i++)
  {
    if (totgen > 0.0f)
    {
      enzyme_dt = half_kr*deltaFrameSeconds[i];

      for (st=1; (st <= SUB_STEPS) && (totgen > 0.0f);st++)
      {

        // assume instantaneous equilibrium within the well
        c_dntp_bot = nuc_rise_ptr[c_dntp_top_ndx] / 50.;
        c_dntp_top_ndx++;

        // Now compute effect of concentration on enzyme rate
        c_dntp_new_effect = c_dntp_bot; // current effect of concentration on enzyme rate

        // update events per molecule
        hplus_events_current = enzyme_dt*c_dntp_new_effect; // events per molecule is average rate * time of rate
        hplus_events_sum += hplus_events_current;

        // how many active molecules left at end of time period given poisson process with total intensity of events
        // exp(-t) * (1+t+t^2/+t^3/6+...) where we interpolate between polynomial lengths by A
        // exp(-t) ( 1+... + frac*(t^k/k!)) where k = ceil(A-1) and frac = A-floor(A), for A>=1
        pact_new = mix_memo.GetStep (hplus_events_sum);
        pact += pact_new;
        pact *= 0.5f;
        // how many hplus were generated
        totgen -= pact * hplus_events_current;  // active molecules * events per molecule
        pact = pact_new;
      }

      if (totgen < 0.0f) totgen = 0.0f;

      ival_offset[i] = (totocc-totgen);
    }
    else
    {
      ival_offset[i] = totocc;
    }

  }
}


// try to simplify
// use the "update state" idea for the poisson process
// may be slower because of compiler annoyances
void MathModel::SuperSimplifyComputeCumulativeIncorporationHydrogens (float *ival_offset, int npts, const float *deltaFrameSeconds,
    float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
    float A, float SP,
    float kr, float kmax, float d, float molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss)
{
  int i;

  MixtureMemo mix_memo;

  float pact;
  float   c_dntp_bot;
  float  hplus_events_sum; // mean events per molecule, cumulative and current

  float enzyme_dt;
  float totocc;


  A = mix_memo.Generate (A,math_poiss);

  mix_memo.ScaleMixture (SP);
  totocc = SP*A;

  pact = mix_memo.total_live;  // active polymerases
  float pact_threshold = 0.05f*pact; // explicit short circuit for the computation

  c_dntp_bot = 0.0f; // concentration of dNTP in the well
  
  hplus_events_sum = 0.0f; // Events per molecule

  memset (ival_offset,0,sizeof (float[my_start]));  // zero the points we don't compute

  float scaled_kr = kr*molecules_to_micromolar_conversion/d; // convert molecules of polymerase to active concentraction
  float half_kr = kr *0.5f/SUB_STEPS; // for averaging

  // first non-zero index of the computed [dNTP] array for this nucleotide
  int c_dntp_top_ndx = my_start*SUB_STEPS;
  float c_dntp_bot_plus_kmax = 1.0f/kmax;
  float c_dntp_old_effect = 0.0f;
  float c_dntp_new_effect = 0.0f;
  int st;
  float totgen=0.0f;
  // set up my interpolation cache at least once before we execute
  mix_memo.UpdateActivePolymeraseState (hplus_events_sum, pact);
  for (i=my_start;i < npts;i++)
  {
    if (pact>pact_threshold) // if we don't have any significant number of molecules remaining to track explicitly
    {
      enzyme_dt = half_kr*deltaFrameSeconds[i]; // integrated exposure of enzyme in each substep
      // update polymerase state & total events
      for (st=1; (st <= SUB_STEPS) && (pact>pact_threshold);st++)
      {
        // assume instantaneous equilibrium within the well
        c_dntp_bot = nuc_rise_ptr[c_dntp_top_ndx];
        c_dntp_top_ndx++;

        c_dntp_bot /= (1.0f+ scaled_kr*pact*c_dntp_bot_plus_kmax); // the level at which new nucs are used up as fast as they diffuse in
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + kmax); // scale for michaelis-menten kinetics, assuming nucs are limiting factor

        // Now compute effect of concentration on enzyme rate
        c_dntp_old_effect = c_dntp_new_effect;
        c_dntp_new_effect = c_dntp_bot*c_dntp_bot_plus_kmax; // current effect of concentration on enzyme rate

        // update events per molecule // events per molecule is average rate * time of rate
        hplus_events_sum += enzyme_dt* (c_dntp_new_effect+c_dntp_old_effect);; // total intensity up to this time

        // update state of molecules
        mix_memo.UpdateActivePolymeraseState (hplus_events_sum, pact); // update state of poisson process based on intensity
      }
      mix_memo.UpdateGeneratedHplus (totgen); // uses the state of the poisson process rather than calculating inside the loop
    }
    else
      totgen=totocc; // set to maximum generated if short-circuiting the iteration
    ival_offset[i]= totgen;
    //  printf("%d %f %f %f %f\n", i, A, hplus_events_sum, totgen, ival_offset[i]);
  }
}


// this is complex

// computes the incorporation signal for a single flow
// One of the two single most important functions
//
// basic parameters for a well and a flow
// iValOffset is the scratch space for this flow to put the cumulative signal
// nuc_rise_ptr & my_start describe the concentration this well will see at the top
// C = max concentration, not actually altered
// A is the "Amplitude" = mixture of homopolymer lengths, approximated by adjacent lengths
// SP is the current number of active copies of the template
// kr = kurrent rate for enzyme activity [depends on nuc type, possibly on flow]
// kmax = kurrent max rate for activity "Michaelis Menten",
// d = diffusion rate into the well of nucleotide [depends on nuc type, possibly on well]
// this equation works the way we want if pact is the # of active pol in the well
// c_dntp_int is in uM-seconds, and dt is in frames @ 15fps
// it calculates the number of incorporations based on the average active polymerase
// and the average dntp concentration in the well during this time step.
// note c_dntp_int is the integral of the dntp concentration in the well during the
// time step, which is equal to the average [dntp] in the well times the timestep duration
void MathModel::ComplexComputeCumulativeIncorporationHydrogens (
    float *ival_offset, int npts, const float *deltaFrameSeconds,
    const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
    float A, float SP,
    float kr, float kmax, float d, float molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss)
{
  int i;
  float totocc, totgen;

  MixtureMemo mix_memo;

  float pact,pact_new;
  float c_dntp_sum, c_dntp_bot;
  float c_dntp_top, c_dntp_int;

  float ldt;
  int st;

  // step 4
  float alpha;
  float expval;



  A = mix_memo.Generate (A,math_poiss);
  //A = InitializeMixture(&mix_ctrl,A,MAX_HPLEN); // initialize Poisson with correct amplitude which maxes out at MAX_HPLEN
  mix_memo.ScaleMixture (SP);
  //ScaleMixture(&mix_ctrl,SP); // scale mixture fractions to proper number of molecules
  pact = mix_memo.total_live;  // active polymerases
  totocc = SP*A;  // how many hydrogens we'll eventually generate

  totgen = totocc;  // number remaining to generate

  c_dntp_bot = 0.0; // concentration of dNTP in the well
  c_dntp_top = 0.0; // concentration at top
  c_dntp_sum = 0.0; // running sum of kr*[dNTP]

  // some pre-computed things
  float c_dntp_bot_plus_kmax = 1.0/kmax;
  float last_tmp1 = 0.0;
  float last_tmp2 = 0.0;
  float c_dntp_fast_inc = kr* (C/ (C+kmax));

  // [dNTP] in the well after which we switch to simpler model
  float fast_start_threshold = 0.99*C;
  float scaled_kr = kr*molecules_to_micromolar_conversion;

  // first non-zero index of the computed [dNTP] array for this nucleotide
  int c_dntp_top_ndx = my_start*SUB_STEPS;

  memset (ival_offset,0,sizeof (float[my_start]));  // zero the points we don't compute

  for (i=my_start;i < npts;i++)
  {
    if (totgen > 0.0)
    {
      ldt = deltaFrameSeconds[i];

      // once the [dNTP] pretty much reaches full strength in the well
      // the math becomes much simpler
      if (c_dntp_bot > fast_start_threshold)
      {
        c_dntp_int = c_dntp_fast_inc*ldt;
        c_dntp_sum += c_dntp_int;

        pact_new = mix_memo.GetStep (c_dntp_sum);

        totgen -= ( (pact+pact_new) /2.0) * c_dntp_int;
        pact = pact_new;
      }
      // can also use fast step math when c_dntp_bot ~ c_dntp_top
      else if (c_dntp_top > 0 && c_dntp_bot/c_dntp_top > 0.95)
      {
        c_dntp_bot = nuc_rise_ptr[c_dntp_top_ndx++];
        c_dntp_fast_inc = kr* (c_dntp_bot/ (c_dntp_bot+kmax));
        c_dntp_int = c_dntp_fast_inc*ldt;
        c_dntp_sum += c_dntp_int;

        pact_new = mix_memo.GetStep (c_dntp_sum);

        totgen -= ( (pact+pact_new) /2.0) * c_dntp_int;
        pact = pact_new;
      }
      else
      {
        ldt /= SUB_STEPS;
        for (st=1; (st <= SUB_STEPS) && (totgen > 0.0);st++)
        {
          c_dntp_top = nuc_rise_ptr[c_dntp_top_ndx++];

          // we're doing the "exponential euler method" approximation for taking a time-step here
          // so we don't need to do ultra-fine steps when the two effects are large and nearly canceling each other
          alpha = d+scaled_kr*pact*c_dntp_bot_plus_kmax;
          expval = ExpApprox (-alpha*ldt);

          c_dntp_bot = c_dntp_bot*expval + d*c_dntp_top* (1-expval) /alpha;

          c_dntp_bot_plus_kmax = 1.0/ (c_dntp_bot + kmax);
          last_tmp1 = c_dntp_bot * c_dntp_bot_plus_kmax;
          c_dntp_int = kr* (last_tmp2 + last_tmp1) *ldt/2.0;
          last_tmp2 = last_tmp1;

          c_dntp_sum += c_dntp_int;

          // calculate new number of active polymerase
          pact_new = mix_memo.GetStep (c_dntp_sum);

          totgen -= ( (pact+pact_new) /2.0) * c_dntp_int;
          pact = pact_new;
        }
      }

      if (totgen < 0.0) totgen = 0.0;
    }

    ival_offset[i] = (totocc-totgen);
  }
}
