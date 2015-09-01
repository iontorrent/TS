/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "StreamingKernels.h"
#include "CudaUtils.h" // for cuda < 5.0 this has to be included ONLY here!
#include "MathModel/PoissonCdf.h"

// define to creat new layout on device instead of copying from host
//#define CREATE_POISSON_LUT_ON_DEVICE

#define DEBUG_TABLE_GAMMA 0

__device__ void
ComputeHydrogenForMultiFlowFit_dev(
  int sId,
  int flow_ndx,
  int nucid,
  float * nucRise,
  float A, 
  float Krate,
  float gain,
  float SP,
  float d,
  int c_dntp_top_ndx,
  int num_frames,
  int num_beads,
  float* ival,
  int nonZeroEmpFrames);

__device__ void
ComputeSignalForMultiFlowFit_dev(
  bool useNonZeroEmphasis,
  int nonZeroEmpFrames, 
  float restrict_clonal,
  int sId,
  int flow_ndx,
  float A, 
  float tauB,
  float etbR,
  float gain,
  int num_frames,
  int num_beads,
  float* non_integer_penalty,
  float* dark_matter,
  float* pPCA_vals,
  float* sbg,
  float* ival,
  float* output,
  bool useEmphasis = false,
  float diff = 0.0f,
  float* emphasis = NULL,
  float* fval = NULL);

__device__
void ComputeMidNucTime_dev(float& tmid, const ConstParams*pCP, int nucId, int flow_ndx);

__device__
void ComputeTauB_dev(float& tauB, const ConstParams* pCP, float etbR, int sId); 

__device__ 
void ComputeEtbR_dev(float& etbR, float R, int sId, int nucid, int absFnum); 

__device__
void ComputeSP_dev(float& SP, float Copies, int flow_ndx, int sId); 

__device__
void GenerateSmoothingKernelForExponentialTailFit_dev(
  int size,
  float taub,
  int exp_start,
  float* kern, 
  const ConstParams* pCP
);

__device__ float 
CalculateMeanResidualErrorPerFlow(
  int startFrame,
  const float* fg_buffers, 
  const float* fval, 
  const float* weight,
  const int num_beads,
  const int num_frames); 

__device__    
void ModelFunctionEvaluationForExponentialTailFit_dev(
  int tail_start, 
  int num_frames, 
  int num_beads, 
  float A, 
  float taub, 
  float dc_offset, 
  float* fval,
  const ConstParams* pCP,
  float* tmp_fval = NULL);

__device__    
void CalculateResidualForExponentialTailFit_dev(
  float* obs, 
  float* pred, 
  float* start,
  float* end,
  float* err,
  float& residual);

///// Implemented functions
/*

__device__ float  CalcNucAvgDarkMatterPerFrame(
  int frame, 
  float*darkMatter)
{
 return  darkMatter[frame]+ CP[sId].darkness[0]; //CP_MULTIFLOWFIT
}

__device__ float  CalcPCADarkMatterPerFrame(
  int frame,
  float *pca_vals, 
  float *darkMatter)
{

 return  darkMatter[frame]+ CP[sId].darkness[0]; //CP_MULTIFLOWFIT
}
*/

namespace { 
    enum ModelFuncEvaluationOutputMode { NoOutput, OneParam, TwoParams };
}


__device__ void
Keplar_ModelFuncEvaluationForSingleFlowFit(
//  int * pMonitor,
  const bool twoParamFit,
  const int sId,
  const int flow_ndx,
  const int nucid,
  const float * nucRise,
  float A, 
  const float Krate,
  const float tau,
  const float gain,
  const float SP,
  const float d,
  float sens,
  int c_dntp_top_ndx,
  const int num_frames,
  const int num_beads,
  float* fval,
  float* ConstP_deltaFrame,
  int endFrames,
  const ModelFuncEvaluationOutputMode flag,
  float * jac_out = NULL,
  const float * emLeft = NULL,
  const float * emRight = NULL,
  const float frac = 0,
  const float * fval_in = NULL,
  const float * err = NULL,
  float *aa = NULL,
  float *rhs0 = NULL, 
  float *krkr = NULL,
  float *rhs1 = NULL,
  float *akr = NULL
)
{
  // At this point, every thread is an independent bead.
  // We're looping over flows, evaluating how one flow over one bead
  // fits into a particular nucleotide.

  if ( A!=A )
    A=0.0001f; // safety check

  if (A < 0.0f) {
    A = -A;
    sens = -sens;
  }

  else if (A > LAST_POISSON_TABLE_COL)
    A = LAST_POISSON_TABLE_COL;

  if ( A<0.0001f )
    A = 0.0001f; // safety


  int ileft, iright;
  float ifrac, idelta;

  // step 2
  float occ_l,occ_r;
  float totocc;
  float totgen;
  float pact;
  int i, st;

  // step 3
  float ldt;

  // step 4
  float c_dntp_int;
  float pact_new;


  ileft = ( int ) A;
  idelta = A-ileft;
  iright = ileft+1;
  ifrac = 1-idelta;
  ileft--;
  iright--;

  occ_l = ifrac; // lower mixture
  occ_r = idelta; // upper mixture


  if (ileft < 0)
  {
    occ_l = 0.0;
    ileft = 0;
  }

  if (iright == LAST_POISSON_TABLE_COL)
  {
    iright = ileft;
    occ_r = occ_l;
    occ_l = 0;
  }

  occ_l *= SP;
  occ_r *= SP;
  pact = occ_l + occ_r;
  totocc = SP*A;
  totgen = totocc;

#ifndef POISS_FLOAT4
  const float* rptr = precompute_pois_params_streaming (iright);
  const float* lptr = precompute_pois_params_streaming (ileft);
#else
  const float4 * LUTptr = precompute_pois_LUT_params_streaming (ileft, iright);
//  atomicAdd(&pMonitor[ileft], 1);
#endif

  // We reuse this constant every loop...
  float cp_sid_kmax_nucid = CP[sId].kmax[nucid];

  float c_dntp_bot = 0.0; // concentration of dNTP in the well
  float c_dntp_sum = 0.0;
  float c_dntp_old_rate = 0;
  float c_dntp_new_rate = 0;

  float scaled_kr = Krate*CP[sId].molecules_to_micromolar_conversion/d; //CP_SINGLEFLOWFIT
  float half_kr = Krate*0.5f;

  // variables used for solving background signal shape
  float aval = 0.0f;

  //new Solve HydrogenFlowInWell

  float one_over_two_tauB = 1.0f;
  float one_over_one_plus_aval = 1.0f/ (1.0f+aval);
  float red_hydro_prev; 
  float fval_local  = 0.0f;

  float red_hydro;

  c_dntp_top_ndx += flow_ndx*num_frames*ISIG_SUB_STEPS_SINGLE_FLOW;

  float c_dntp_bot_plus_kmax = 1.0f/cp_sid_kmax_nucid; //CP_SINGLEFLOWFIT

  //for (i=CP[sId].start[flow_ndx];i < num_frames;i++) //CP_SINGLEFLOWFIT
  for (i=CP[sId].start[flow_ndx];i < endFrames; i++) //CP_SINGLEFLOWFIT
  {
    if (totgen > 0.0f)
    {
      ldt = (ConstP_deltaFrame[i]/( ISIG_SUB_STEPS_SINGLE_FLOW * FRAMESPERSEC)) * half_kr; //CP_SINGLEFLOWFIT
      for (st=1; (st <= ISIG_SUB_STEPS_SINGLE_FLOW) && (totgen > 0.0f);st++)
      {
        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;

        // All the threads should be grabbing from the same nucRise location.
        c_dntp_bot = nucRise[c_dntp_top_ndx++]/ (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + cp_sid_kmax_nucid); //CP_SINGLEFLOWFIT

        c_dntp_new_rate = c_dntp_bot*c_dntp_bot_plus_kmax;
        c_dntp_int = ldt* (c_dntp_new_rate+c_dntp_old_rate);
        c_dntp_sum += c_dntp_int;

        // calculate new number of active polymerase
#ifndef POISS_FLOAT4
        pact_new = poiss_cdf_approx_streaming (c_dntp_sum,rptr) * occ_r;
//       if (occ_l > 0.0f)
        pact_new += poiss_cdf_approx_streaming (c_dntp_sum,lptr) * occ_l;
#else       
        pact_new = poiss_cdf_approx_float4(c_dntp_sum, LUTptr, occ_l, occ_r);
#endif
        totgen -= ( (pact+pact_new) * 0.5f) * c_dntp_int;
        pact = pact_new;
      }

      if (totgen < 0.0f) totgen = 0.0f;
      red_hydro = (totocc-totgen);
    }else{
      red_hydro = totocc;
    }
    
    // calculate the 'background' part (the accumulation/decay of the protons in the well
    // normally accounted for by the background calc)
    
    red_hydro *= sens;  
 
    one_over_two_tauB = 1.0f/ (2.0f*tau);
    aval = ConstP_deltaFrame[i]*one_over_two_tauB; //CP_SINGLEFLOWFIT
    one_over_one_plus_aval = 1.0f/ (1.0f+aval);
    
    if(i==CP[sId].start[flow_ndx]) //CP_SINGLEFLOWFIT
      fval_local  = red_hydro; // *one_over_one_plus_aval;
    else
      fval_local = red_hydro - red_hydro_prev + (1.0f-aval)*fval_local; // *one_over_one_plus_aval;

    red_hydro_prev = red_hydro;
 
    fval_local *=  one_over_one_plus_aval;

    switch( flag ) {
    case NoOutput:
#ifdef FVAL_L1
      fval[i] = fval_local * gain;  
#else
      fval[num_beads*i] = fval_local * gain;  
#endif
      break;

    case OneParam:
    case TwoParams:
      float weight = emRight != NULL ? frac*emLeft[i*(MAX_POISSON_TABLE_COL)] + (1.0f - frac)*emRight[i*(MAX_POISSON_TABLE_COL)] : emLeft[i*(MAX_POISSON_TABLE_COL)];

      int bxi = num_beads * i;
      float err_bxi = err[bxi]; // Grab this early so that we only get it once.
#ifdef FVAL_L1
      float jac_tmp =  weight * (fval_local*gain - fval_in[i]) * 1000.0f;
#else
      float jac_tmp =  weight * (fval_local*gain - fval_in[bxi]) * 1000.0f;
#endif
      if(flag==OneParam){
#ifdef JAC_L1
        jac_out[i] = jac_tmp;
#else
        jac_out[bxi] = jac_tmp;
#endif
        *aa += jac_tmp * jac_tmp;       
        if (!twoParamFit) 
         *rhs0 += (jac_tmp * err_bxi);
      }
      else {            // Two params.
#ifdef JAC_L1
        float my_jac_out = jac_out[i];           // Only grab it from memory once.
#else
        float my_jac_out = jac_out[bxi];         // Only grab it from memory once.
#endif
        *akr +=  my_jac_out * jac_tmp;
        *rhs0 += my_jac_out * err_bxi;
        *rhs1 += jac_tmp * err_bxi;
        *krkr += jac_tmp * jac_tmp;
      }
    }
  }
}


__device__ void
Fermi_ModelFuncEvaluationForSingleFlowFitNoOutput(
//  int * pMonitor,
  const ConstParams* pCP,
  const int flow_ndx,
  const int nucid,
  const float * nucRise,
  float A,
  const float Krate,
  const float tau,
  const float gain,
  const float SP,
  const float d,
  float sens,
  int c_dntp_top_ndx,
  const int num_frames,
  const int num_beads,
  float * fval,
  float* ConstP_deltaFrame,
  int endFrames
)
{
  // At this point, every thread is an independent bead.
  // We're looping over flows, evaluating how one flow over one bead
  // fits into a particular nucleotide.

  if ( A!=A )
    A=0.0001f; // safety check

  if (A < 0.0f) {
    A = -A;
    sens = -sens;
  }

  else if (A > LAST_POISSON_TABLE_COL)
    A = LAST_POISSON_TABLE_COL;

  if ( A<0.0001f )
    A = 0.0001f; // safety

#if USE_TABLE_GAMMA
  int ileft = ( int ) A;
  float idelta = A-ileft;
  int iright = ileft+1;
  float ifrac = 1-idelta;
  ileft--;
  iright--;

  float occ_l = ifrac; // lower mixture
  float occ_r = idelta; // upper mixture


  if (ileft < 0)
  {
    occ_l = 0.0;
    ileft = 0;
  }

  if (iright == LAST_POISSON_TABLE_COL)
  {
    iright = ileft;
    occ_r = occ_l;
    occ_l = 0;
  }

  occ_l *= SP;
  occ_r *= SP;
  float pact = occ_l + occ_r;

#ifndef POISS_FLOAT4
  const float* rptr = precompute_pois_params_streaming (iright);
  const float* lptr = precompute_pois_params_streaming (ileft);
#else
  const float4 * LUTptr = precompute_pois_LUT_params_streaming (ileft, iright);
#endif
#else
  float pact = SP;
#endif    // USE_TABLE_GAMMA
  float totocc = SP*A;
  float totgen = totocc;

  // We reuse this constant every loop...
  float cp_sid_kmax_nucid = pCP->kmax[nucid];

  float c_dntp_sum = 0.0;
  float c_dntp_old_rate = 0;
  float c_dntp_new_rate = 0;

  float scaled_kr = Krate*pCP->molecules_to_micromolar_conversion/d; //CP_SINGLEFLOWFIT
  float half_kr = Krate*0.5f;

  // variables used for solving background signal shape
  float aval = 0.0f;

  //new Solve HydrogenFlowInWell

  float one_over_two_tauB = 1.0f;
  float one_over_one_plus_aval = 1.0f/ (1.0f+aval);
  float red_hydro_prev; 
  float fval_local  = 0.0f;

  float red_hydro;

  c_dntp_top_ndx += flow_ndx*num_frames*ISIG_SUB_STEPS_SINGLE_FLOW;

  float c_dntp_bot_plus_kmax = 1.0f/cp_sid_kmax_nucid; //CP_SINGLEFLOWFIT

  bool start_frame = true;
  //for (int i=pCP->start[flow_ndx];i < num_frames;i++) //CP_SINGLEFLOWFIT
  for (int i=pCP->start[flow_ndx];i < endFrames;i++) //CP_SINGLEFLOWFIT
  {
    if (totgen > 0.0f)
    {
      float ldt = (ConstP_deltaFrame[i]/( ISIG_SUB_STEPS_SINGLE_FLOW * FRAMESPERSEC)) * half_kr; //CP_SINGLEFLOWFIT
      for (int st=1; (st <= ISIG_SUB_STEPS_SINGLE_FLOW) && (totgen > 0.0f);st++)
      {
        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;

        // All the threads should be grabbing from the same nucRise location.
        // c_dntp_bot is the concentration of dNTP in the well
        float c_dntp_bot = nucRise[c_dntp_top_ndx++]/ (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + cp_sid_kmax_nucid); //CP_SINGLEFLOWFIT

        c_dntp_new_rate = c_dntp_bot*c_dntp_bot_plus_kmax;
        float c_dntp_int = ldt* (c_dntp_new_rate+c_dntp_old_rate);
        c_dntp_sum += c_dntp_int;

        // calculate new number of active polymerase
#if USE_TABLE_GAMMA
#ifndef POISS_FLOAT4
        float pact_new = poiss_cdf_approx_streaming (c_dntp_sum,rptr) * occ_r;
//       if (occ_l > 0.0f)
        pact_new += poiss_cdf_approx_streaming (c_dntp_sum,lptr) * occ_l;
#else       
        float pact_new = poiss_cdf_approx_float4(c_dntp_sum, LUTptr, occ_l, occ_r);

#if DEBUG_TABLE_GAMMA
        printf("A=%g, c_dntp_sum=%g, table=%g, calc=%g\n"
               "    calc_interp %g %g %g %g\n"
               "    table_interp %g %g %g %g\n", 
               A, c_dntp_sum, pact_new, PoissonCDF( A, c_dntp_sum ) * SP,
               PoissonCDF( floor(A), floor(c_dntp_sum*20.f)/20.f ),
               PoissonCDF( floor(A), ceil(c_dntp_sum*20.f)/20.f ),
               PoissonCDF( ceil(A), floor(c_dntp_sum*20.f)/20.f ),
               PoissonCDF( ceil(A), ceil(c_dntp_sum*20.f)/20.f ),
               LUTptr[(int)(c_dntp_sum*20.f)].x,
               LUTptr[(int)(c_dntp_sum*20.f)].y,
               LUTptr[(int)(c_dntp_sum*20.f)].z,
               LUTptr[(int)(c_dntp_sum*20.f)].w
               );
#endif    // DEBUG_TABLE_GAMMA
#endif
#else
        float pact_new = PoissonCDF( A, c_dntp_sum ) * SP;
#endif    // USE_TABLE_GAMMA
        totgen -= ( (pact+pact_new) * 0.5f) * c_dntp_int;
        pact = pact_new;
      }

      if (totgen < 0.0f) totgen = 0.0f;
      red_hydro = (totocc-totgen);
    }else{
      red_hydro = totocc;
    }

    // calculate the 'background' part (the accumulation/decay of the protons in the well
    // normally accounted for by the background calc)
    
    red_hydro *= sens;  
 
    one_over_two_tauB = 1.0f/ (2.0f*tau);
    aval = ConstP_deltaFrame[i]*one_over_two_tauB; //CP_SINGLEFLOWFIT
    one_over_one_plus_aval = 1.0f/ (1.0f+aval);
    
    if(start_frame) { //CP_SINGLEFLOWFIT
      fval_local = red_hydro; // *one_over_one_plus_aval;
      start_frame = false;
    } else {
      fval_local = red_hydro - red_hydro_prev + (1.0f-aval)*fval_local; // *one_over_one_plus_aval;
    }

    red_hydro_prev = red_hydro;
 
    fval_local *=  one_over_one_plus_aval;

#ifdef FVAL_L1
    fval[i] = fval_local * gain;  
#else
    fval[num_beads*i] = fval_local * gain;  
#endif
  }
}


__device__ void
Fermi_ModelFuncEvaluationForSingleFlowFit(
  const int sId,
  const int flow_ndx,
  const int nucid,
  const float * const nucRise,
  float A1, 
  float A2, 
  const float Krate1,
  const float Krate2,
  const float tau,
  const float gain,
  const float SP,
  const float d,
  const float sens_in,
  int c_dntp_top_ndx,
  const int num_frames,
  const int num_beads,
  const ModelFuncEvaluationOutputMode flag,
  const float * const emLeft,
  const float * const emRight,
  const float frac,
  const float * const fval_in,
  const float * const err,
  float *const aa,
  float *const rhs0,
  float *const krkr,
  float *const rhs1,
  float *const akr,
  float *ConstP_deltaFrame,
  int endFrames
)
{
  float sens1 = sens_in;
  // At this point, every thread is an independent bead.
  // We're looping over flows, evaluating how one flow over one bead
  // fits into a particular nucleotide.

  if ( A1!=A1 )
    A1=0.0001f; // safety check

  if (A1 < 0.0f) {
    A1 = -A1;
    sens1 = -sens1;
  }

  else if (A1 > LAST_POISSON_TABLE_COL)
    A1 = LAST_POISSON_TABLE_COL;

  if ( A1<0.0001f )
    A1 = 0.0001f; // safety

  float sens2 = sens_in;

  if ( A2!=A2 )
    A2=0.0001f; // safety check

  if (A2 < 0.0f) {
    A2 = -A2;
    sens2 = -sens2;
  }

  else if (A2 > LAST_POISSON_TABLE_COL)
    A2 = LAST_POISSON_TABLE_COL;

  if ( A2<0.0001f )
    A2 = 0.0001f; // safety

#if USE_TABLE_GAMMA
  int ileft1 = ( int ) A1;
  float occ_r1 = A1-ileft1;     // upper mixture
  int iright1 = ileft1+1;
  float occ_l1 = 1-occ_r1;      // lower mixture
  ileft1--;
  iright1--;

  if (ileft1 < 0)
  {
    occ_l1 = 0.0;
    ileft1 = 0;
  }

  if (iright1 == LAST_POISSON_TABLE_COL)
  {
    iright1 = ileft1;
    occ_r1 = occ_l1;
    occ_l1 = 0;
  }

  occ_l1 *= SP;
  occ_r1 *= SP;
  float pact1 = occ_l1 + occ_r1;

  int ileft2 = ( int ) A2;
  float occ_r2 = A2-ileft2;     // upper mixture
  int iright2 = ileft2+1;
  float occ_l2 = 1-occ_r2;      // lower mixture
  ileft2--;
  iright2--;

  if (ileft2 < 0)
  {
    occ_l2 = 0.0;
    ileft2 = 0;
  }

  if (iright2 == LAST_POISSON_TABLE_COL)
  {
    iright2 = ileft2;
    occ_r2 = occ_l2;
    occ_l2 = 0;
  }

  occ_l2 *= SP;
  occ_r2 *= SP;
  float pact2 = occ_l2 + occ_r2;

#ifndef POISS_FLOAT4
  const float* rptr1 = precompute_pois_params_streaming (iright1);
  const float* lptr1 = precompute_pois_params_streaming (ileft1);
#else
  const float4 * LUTptr1 = precompute_pois_LUT_params_streaming (ileft1, iright1);
#endif

#ifndef POISS_FLOAT4
  const float* rptr2 = precompute_pois_params_streaming (iright2);
  const float* lptr2 = precompute_pois_params_streaming (ileft2);
#else
  const float4 * LUTptr2 = precompute_pois_LUT_params_streaming (ileft2, iright2);
#endif

#else   // !USE_TABLE_GAMMA
  float pact1 = SP;
  float pact2 = SP;
#endif    // USE_TABLE_GAMMA
  const float totocc1 = SP*A1;
  float totgen1 = totocc1;
  const float totocc2 = SP*A2;
  float totgen2 = totocc2;

  // We reuse this constant every loop...
  const float cp_sid_kmax_nucid = CP[sId].kmax[nucid];

  float c_dntp_sum1 = 0.0;
  float c_dntp_new_rate1 = 0;

  const float scaled_kr1 = Krate1*CP[sId].molecules_to_micromolar_conversion/d; //CP_SINGLEFLOWFIT

  float red_hydro_prev1;

  c_dntp_top_ndx += flow_ndx*num_frames*ISIG_SUB_STEPS_SINGLE_FLOW;

  float c_dntp_bot_plus_kmax1 = 1.0f/cp_sid_kmax_nucid; //CP_SINGLEFLOWFIT

  float c_dntp_sum2 = 0.0;
  float c_dntp_new_rate2 = 0;
  
  float fval_local1 = 0.f;
  float fval_local2 = 0.f;

  const float scaled_kr2 = Krate2*CP[sId].molecules_to_micromolar_conversion/d; //CP_SINGLEFLOWFIT

  float red_hydro_prev2;

  float c_dntp_bot_plus_kmax2 = 1.0f/cp_sid_kmax_nucid; //CP_SINGLEFLOWFIT

  int starting_frame = CP[sId].start[flow_ndx];
  //for (int i=starting_frame;i < num_frames;i++) //CP_SINGLEFLOWFIT
  for (int i=starting_frame;i < endFrames; i++) //CP_SINGLEFLOWFIT
  {
    float delta_frame = ConstP_deltaFrame[i];

    float red_hydro1 = totocc1;
    float red_hydro2 = totocc2;

    // Move memory fetches well ahead of where they're used.    
#ifdef FVAL_L1
    const float fval_in_i = fval_in[i];
#else
    const float fval_in_i = fval_in[num_beads * i];
#endif

    if (totgen1 > 0.0f || (totgen2 > 0.f && flag == TwoParams ) )
    {
      //CP_SINGLEFLOWFIT
      float ldt1 = (delta_frame/( ISIG_SUB_STEPS_SINGLE_FLOW * FRAMESPERSEC)) * (Krate1*0.5f);
      float ldt2 = (delta_frame/( ISIG_SUB_STEPS_SINGLE_FLOW * FRAMESPERSEC)) * (Krate2*0.5f);

      for (int st=1; st <= ISIG_SUB_STEPS_SINGLE_FLOW ;st++)
      {
        // All the threads should be grabbing from the same nucRise location.
        float nuc_rise = nucRise[ c_dntp_top_ndx++ ];

        if ( totgen1 > 0.f ) {
          // assume instantaneous equilibrium
          const float c_dntp_old_rate1 = c_dntp_new_rate1;

          // c_dntp_bot is concentration of dNTP in the well
          const float c_dntp_bot = nuc_rise / (1.0f + scaled_kr1*pact1*c_dntp_bot_plus_kmax1);
          c_dntp_bot_plus_kmax1 = 1.0f/ (c_dntp_bot + cp_sid_kmax_nucid); //CP_SINGLEFLOWFIT
  
          c_dntp_new_rate1 = c_dntp_bot*c_dntp_bot_plus_kmax1;
          float c_dntp_int1 = ldt1* (c_dntp_new_rate1+c_dntp_old_rate1);
          c_dntp_sum1 += c_dntp_int1;

          // calculate new number of active polymerase
#if USE_TABLE_GAMMA
#ifndef POISS_FLOAT4
          float pact_new1 = poiss_cdf_approx_streaming (c_dntp_sum1,rptr1) * occ_r1;
//       if (occ_l1 > 0.0f)
          pact_new1 += poiss_cdf_approx_streaming (c_dntp_sum1,lptr1) * occ_l1;
#else       
          float pact_new1 = poiss_cdf_approx_float4(c_dntp_sum1, LUTptr1, occ_l1, occ_r1);
#endif
#else
          float pact_new1 = PoissonCDF( A1, c_dntp_sum1 ) * SP;
#endif    // USE_TABLE_GAMMA
          totgen1 -= ( (pact1+pact_new1) * 0.5f) * c_dntp_int1;
          pact1 = pact_new1;
        }

        if ( totgen2 > 0.f && flag == TwoParams )
        {
          // assume instantaneous equilibrium
          const float c_dntp_old_rate2 = c_dntp_new_rate2;

          // c_dntp_bot is concentration of dNTP in the well
          const float c_dntp_bot = nuc_rise / (1.0f + scaled_kr2*pact2*c_dntp_bot_plus_kmax2);
          c_dntp_bot_plus_kmax2 = 1.0f/ (c_dntp_bot + cp_sid_kmax_nucid); //CP_SINGLEFLOWFIT

          c_dntp_new_rate2 = c_dntp_bot*c_dntp_bot_plus_kmax2;
          float c_dntp_int2 = ldt2* (c_dntp_new_rate2+c_dntp_old_rate2);
          c_dntp_sum2 += c_dntp_int2;

          // calculate new number of active polymerase
#if USE_TABLE_GAMMA
#ifndef POISS_FLOAT4
          float pact_new2 = poiss_cdf_approx_streaming (c_dntp_sum2,rptr2) * occ_r2;
//       if (occ_l2 > 0.0f)
          pact_new2 += poiss_cdf_approx_streaming (c_dntp_sum2,lptr2) * occ_l2;
#else       
          float pact_new2 = poiss_cdf_approx_float4(c_dntp_sum2, LUTptr2, occ_l2, occ_r2);
#endif
#else
          float pact_new2 = PoissonCDF( A2, c_dntp_sum2 ) * SP;
#endif    // USE_TABLE_GAMMA
          totgen2 -= ( (pact2+pact_new2) * 0.5f) * c_dntp_int2;
          pact2 = pact_new2;
        }
      }

      if (totgen1 < 0.0f) totgen1 = 0.0f;
      red_hydro1 -= totgen1;

      if ( flag == TwoParams ) {
        if (totgen2 < 0.0f) totgen2 = 0.0f;
        red_hydro2 -= totgen2;
      }
    }

    float err_bxi = err[num_beads * i]; // Grab this early so that we only get it once.

    // calculate the 'background' part (the accumulation/decay of the protons in the well
    // normally accounted for by the background calc)
    
    red_hydro1 *= sens1;  
 
    // variables used for solving background signal shape
    const float one_over_two_tauB = 1.0f/ (2.0f*tau);
    const float aval = delta_frame*one_over_two_tauB; //CP_SINGLEFLOWFIT
    const float one_over_one_plus_aval = 1.0f/ (1.0f+aval);
    
    if( i == starting_frame ) //CP_SINGLEFLOWFIT
      fval_local1 = red_hydro1; // *one_over_one_plus_aval;
    else
      fval_local1 = red_hydro1 - red_hydro_prev1 + (1.0f-aval)*fval_local1; // *one_over_one_plus_aval;

    red_hydro_prev1 = red_hydro1;
 
    fval_local1 *=  one_over_one_plus_aval;

    float weight = emRight != NULL ? frac*emLeft[i*(MAX_POISSON_TABLE_COL)] + (1.0f - frac)*emRight[i*(MAX_POISSON_TABLE_COL)] : emLeft[i*(MAX_POISSON_TABLE_COL)];

    float jac_1 =  weight * (fval_local1*gain - fval_in_i) * 1000.0f;
    *aa += jac_1 * jac_1;       
    *rhs0 += (jac_1 * err_bxi);

    if ( flag == TwoParams )
    {
      // calculate the 'background' part (the accumulation/decay of the protons in the well
      // normally accounted for by the background calc)
      red_hydro2 *= sens2;  
 
      if( i == starting_frame ) //CP_SINGLEFLOWFIT
        fval_local2 = red_hydro2; // *one_over_one_plus_aval;
      else
        fval_local2 = red_hydro2 - red_hydro_prev2 + (1.0f-aval)*fval_local2; // *one_over_one_plus_aval;

      red_hydro_prev2 = red_hydro2;
 
      fval_local2 *=  one_over_one_plus_aval;

      float jac_2 =  weight * (fval_local2*gain - fval_in_i) * 1000.0f;
      *akr +=  jac_1 * jac_2;
      *rhs1 += jac_2 * err_bxi;
      *krkr += jac_2 * jac_2;
    } // end flag == TwoParams
  } // loop over i
}

#if 0
__device__ void
ModelFuncEvaluationAndProjectiontForSingleFlowFit(
  int sId,
  int flow_ndx,
  int nucid,
  float * nucRise,
  float A, 
  float Krate,
  float tau,
  float gain,
  float SP,
  float d,
  float sens,
  int c_dntp_top_ndx,
  int num_frames,
  int num_beads,
  float epsilon,
  float* fval,
  float* emLeft,
  float* emRight,
  float frac,
  float* err,
  float &delta
)
{
  if (A < 0.0f) {
    A = -A;
    sens = -sens;
  }
  else if (A > LAST_POISSON_TABLE_COL)
    A = LAST_POISSON_TABLE_COL;


  int ileft, iright;
  float ifrac;

  // step 2
  float occ_l,occ_r;
  float totocc;
  float totgen;
  float pact;
  int i, st;

  // step 3
  float ldt;

  // step 4
  float c_dntp_int;
  float pact_new;

  // initialize diffusion/reaction simulation for this flow
  ileft = (int) A;
  iright = ileft + 1;
  ifrac = iright - A;
  occ_l = ifrac;
  occ_r = A - ileft;

  ileft--;
  iright--;

  if (ileft < 0)
  {
    occ_l = 0.0;
  }

  if (iright == LAST_POISSON_TABLE_COL)
  {
    iright = ileft;
    occ_r = occ_l;
    occ_l = 0;
  }

  occ_l *= SP;
  occ_r *= SP;
  pact = occ_l + occ_r;
  totocc = SP*A;
  totgen = totocc;

#ifndef POISS_FLOAT4
  const float* rptr = precompute_pois_params_streaming (iright);
  const float* lptr = precompute_pois_params_streaming (ileft);
#else
  const float4* LUTptr = precompute_pois_LUT_params_streaming (ileft, iright);
#endif

  float c_dntp_bot = 0.0; // concentration of dNTP in the well
  float c_dntp_sum = 0.0;
  float c_dntp_old_rate = 0;
  float c_dntp_new_rate = 0;

  float c_dntp_bot_plus_kmax = 1.0f/CP[sId].kmax[nucid]; //CP_SINGLEFLOWFIT

  float scaled_kr = Krate*CP[sId].molecules_to_micromolar_conversion/d; //CP_SINGLEFLOWFIT
  float half_kr = Krate*0.5f;

  // variables used for solving background signal shape
  float aval = 0.0f;

  //new Solve HydrogenFlowInWell

  float one_over_two_tauB = 1.0f;
  float one_over_one_plus_aval = 1.0f/ (1.0f+aval);
  float red_hydro_prev; 
  float fval_local  = 0.0f;

  float red_hydro;

  c_dntp_top_ndx += flow_ndx*num_frames*ISIG_SUB_STEPS_SINGLE_FLOW;

  float num = 0;
  float den = 0.0001f;
  for (i=CP[sId].start[flow_ndx];i < num_frames;i++) //CP_SINGLEFLOWFIT
  {
    if (totgen > 0.0f)
    {
      ldt = (CP[sId].deltaFrames[i]/( ISIG_SUB_STEPS_SINGLE_FLOW * FRAMESPERSEC)) * half_kr; //CP_SINGLEFLOWFIT
      for (st=1; (st <= ISIG_SUB_STEPS_SINGLE_FLOW) && (totgen > 0.0f);st++)
      {
        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;
        c_dntp_bot = nucRise[c_dntp_top_ndx++]/ (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + CP[sId].kmax[nucid]); //CP_SINGLEFLOWFIT

        c_dntp_new_rate = c_dntp_bot*c_dntp_bot_plus_kmax;
        c_dntp_int = ldt* (c_dntp_new_rate+c_dntp_old_rate);
        c_dntp_sum += c_dntp_int;

        // calculate new number of active polymerase
#ifndef POISS_FLOAT4
        pact_new = poiss_cdf_approx_streaming (c_dntp_sum,rptr) * occ_r;
 //       if (occ_l > 0.0f)
         pact_new += poiss_cdf_approx_streaming (c_dntp_sum,lptr) * occ_l;
#else       
          pact_new = poiss_cdf_approx_float4(c_dntp_sum, LUTptr, occ_l, occ_r);
#endif
        
        totgen -= ( (pact+pact_new) * 0.5f) * c_dntp_int;
        pact = pact_new;
      }

      if (totgen < 0.0f) totgen = 0.0f;
      red_hydro = (totocc-totgen);
    }else{
      red_hydro = totocc;
    }
    
    // calculate the 'background' part (the accumulation/decay of the protons in the well
    // normally accounted for by the background calc)
    
    red_hydro *= sens;  
 
   
    one_over_two_tauB = 1.0f/ (2.0f*tau);
    aval = CP[sId].deltaFrames[i]*one_over_two_tauB; //CP_SINGLEFLOWFIT
    one_over_one_plus_aval = 1.0f/ (1.0f+aval);
    
    if(i==CP[sId].start[flow_ndx]) //CP_SINGLEFLOWFIT
      fval_local  = red_hydro; // *one_over_one_plus_aval;
    else
      fval_local = red_hydro - red_hydro_prev + (1.0f-aval)*fval_local; // *one_over_one_plus_aval;

    red_hydro_prev = red_hydro;
 
    fval_local *=  one_over_one_plus_aval;

      float weight = (emRight != NULL )?( frac*emLeft[i*(MAX_POISSON_TABLE_COL)] + (1.0f - frac)*emRight[i*(MAX_POISSON_TABLE_COL)]) :( emLeft[i*(MAX_POISSON_TABLE_COL)]);

     delta = (fval_local*gain) - fval[i*num_beads];
     num += epsilon*delta*err[i*num_beads]*weight*weight; 
     den += delta*delta*weight*weight;  
  }
  delta = num/den;
}
#endif


__device__ void
ComputeHydrogenForMultiFlowFit_dev(
  int sId,
  int flow_ndx,
  int nucid, 
  float * nucRise,
  float A, 
  float Krate,
  float gain,
  float SP,
  float d,
  int c_dntp_top_ndx,
  int num_frames,
  int num_beads,
  float* ival,
  int nonZeroEmpFrames)
{
  float sens = CP[sId].sens*SENSMULTIPLIER; //CP_MULTIFLOWFIT
  if (A < 0.0f) {
    A = -A;
    sens = -sens;
  }
  else if (A > LAST_POISSON_TABLE_COL)
    A = LAST_POISSON_TABLE_COL;

  if ( A<0.0001f )
    A = 0.0001f; // safety

  int ileft, iright;
  float ifrac;

  // step 2
  float occ_l,occ_r;
  float totocc;
  float totgen;
  float pact;
  int i, st;

  // step 3
  float ldt;

  // step 4
  float c_dntp_int;

  // initialize diffusion/reaction simulation for this flow
  ileft = (int) A;
  iright = ileft + 1;
  ifrac = iright - A;
  occ_l = ifrac;
  occ_r = A - ileft;

  ileft--;
  iright--;

  if (ileft < 0)
  {
    occ_l = 0.0;
    ileft = 0;
  }

  if (iright >= LAST_POISSON_TABLE_COL)
  {
    iright = ileft = LAST_POISSON_TABLE_COL-1;
    occ_r = occ_l;
    occ_l = 0;
  }

  occ_l *= SP;
  occ_r *= SP;
  pact = occ_l + occ_r;
  totocc = SP*A;
  totgen = totocc;

#ifndef POISS_FLOAT4
  const float* rptr = precompute_pois_params_streaming (iright);
  const float* lptr = precompute_pois_params_streaming (ileft);
#else
 const float4* LUTptr = precompute_pois_LUT_params_streaming (ileft, iright);
#endif

  float c_dntp_bot = 0.0; // concentration of dNTP in the well
  float c_dntp_sum = 0.0;
  float c_dntp_old_rate = 0;
  float c_dntp_new_rate = 0;

  float c_dntp_bot_plus_kmax = 1.0f/CP[sId].kmax[nucid]; //CP_MULTIFLOWFIT

  float scaled_kr = Krate*CP[sId].molecules_to_micromolar_conversion/d; //CP_MULTIFLOWFIT
  float half_kr = Krate*0.5f;

  c_dntp_top_ndx += flow_ndx*num_frames*ISIG_SUB_STEPS_MULTI_FLOW;

  for(i=0;i<CP[sId].start[flow_ndx]; i++) { //CP_MULTIFLOWFIT
    *ival = 0;
    ival += num_beads;
  }

  for (i=CP[sId].start[flow_ndx];i < nonZeroEmpFrames;i++) //CP_MULTIFLOWFIT
  {
    if (totgen > 0.0f)
    {
      ldt = (CP[sId].deltaFrames[i]/( ISIG_SUB_STEPS_MULTI_FLOW * FRAMESPERSEC)) * half_kr; //CP_MULTIFLOWFIT
      for (st=1; (st <= ISIG_SUB_STEPS_MULTI_FLOW) && (totgen > 0.0f);st++)
      {
        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;
        c_dntp_bot = nucRise[c_dntp_top_ndx++]/ (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + CP[sId].kmax[nucid]); //CP_MULTIFLOWFIT

        c_dntp_new_rate = c_dntp_bot*c_dntp_bot_plus_kmax;
        c_dntp_int = ldt* (c_dntp_new_rate+c_dntp_old_rate);
        c_dntp_sum += c_dntp_int;

        // calculate new number of active polymerase
#if USE_TABLE_GAMMA
#ifndef POISS_FLOAT4
        float pact_new = poiss_cdf_approx_streaming (c_dntp_sum,rptr) * occ_r;
 //       if (occ_l > 0.0f)
         pact_new += poiss_cdf_approx_streaming (c_dntp_sum,lptr) * occ_l;
#else       
        float pact_new = poiss_cdf_approx_float4(c_dntp_sum, LUTptr, occ_l, occ_r);
#endif
#else
        float pact_new = PoissonCDF( A, c_dntp_sum ) * SP;
#endif    // USE_TABLE_GAMMA

        totgen -= ( (pact+pact_new) * 0.5f) * c_dntp_int;
        pact = pact_new;
      }

      if (totgen < 0.0f) totgen = 0.0f;
      *ival = (totocc-totgen) * sens;
    }else{
      *ival = totocc * sens;
    }
    ival += num_beads;
  }
}

__device__ void
ComputeSignalForMultiFlowFit_dev(
  bool useNonZeroEmphasis,
  int nonZeroEmpFrames,
  float restrict_clonal,
  int sId,
  int flow_ndx,
  float A, 
  float tauB,
  float etbR,
  float gain,
  int num_frames,
  int num_beads,
  float* non_integer_penalty,
  float* dark_matter,
  float* pPCA_vals,
  float* sbg,
  float* ival,
  float* output,
  bool useEmphasis,
  float diff,
  float* emphasis,
  float* fval)
{
  float xt;
  float fval_local, purple_hydr;
  float clonal_error_term = 0.0f;
  int i=0;

  if ((A < restrict_clonal) && (flow_ndx > KEY_LEN)) {
    int intcall = A + 0.5f;
    clamp_streaming(intcall, 0, MAGIC_MAX_CLONAL_HP_LEVEL);
    clonal_error_term = fabs(A - intcall) * non_integer_penalty[intcall];
  }

  float one_over_two_taub = 1.0f / (2.0f*tauB);
  xt = CP[sId].deltaFrames[i]*one_over_two_taub; //CP_MULTIFLOWFIT

  float one_over_one_plus_aval = 1.0f/ (1.0f+xt);

  sbg += flow_ndx*num_frames;
  purple_hydr = ( *ival + (etbR+xt)*sbg[i])*one_over_one_plus_aval;
  
  //fval_local =   dark_matter[i]*CP[sId].darkness[0] +  //CP_MULTIFLOWFIT
  fval_local =   ApplyDarkMatterToFrame(dark_matter, pPCA_vals, i, num_frames, num_beads, sId);  
  fval_local += purple_hydr*gain + clonal_error_term * ((float) (i&1) - 0.5f);
  *output = useEmphasis ? (fval_local - *fval)*emphasis[i] / diff : fval_local;
  output += num_beads;
  i++;

  int frames = useNonZeroEmphasis ? nonZeroEmpFrames : num_frames;
  for (; i<frames; ++i)
  {
    xt = CP[sId].deltaFrames[i]*one_over_two_taub; //CP_MULTIFLOWFIT
    one_over_one_plus_aval = 1.0f/(1.0f+xt);
    purple_hydr = ((ival[i*num_beads] - ival[(i-1)*num_beads]) 
         + (etbR+xt)*sbg[i] - (etbR-xt) * sbg[i-1]+ (1.0f-xt) * purple_hydr) * one_over_one_plus_aval;
    fval_local = purple_hydr*gain + ApplyDarkMatterToFrame(dark_matter, pPCA_vals , i, num_frames, num_beads, sId);  
    // dark_matter[i]*CP[sId].darkness[0]; //CP_MULTIFLOWFIT

    if (i < MAXCLONALMODIFYPOINTSERROR)
      fval_local += clonal_error_term * ((float) (i&1) - 0.5f);

    *output = useEmphasis ? 
        (fval_local - fval[i*num_beads])*emphasis[i] / diff : fval_local;
    output += num_beads;
  }

}

// smoothing kernel to provide weights for smoothing exponential tail 
__device__
void GenerateSmoothingKernelForExponentialTailFit_dev(
  int size,
  float taubInv,
  int exp_start,
  float* kern, 
  const ConstParams* pCP
)
{
  float dt;
  for (int i=0; i<size; ++i) {
    dt = (pCP->frameNumber[i+exp_start] - pCP->frameNumber[exp_start + 3])*taubInv;
    kern[i] = __expf(dt);   
  }
}

__device__ float 
ResidualCalculationPerFlow(
  int startFrame,
  const float* fg_buffers, 
  const float* fval, 
  const float* emLeft,
  const float* emRight,
  const float frac, 
        float* err, 
  const int num_beads,
  const int nonZeroEmpFrames) {

  float e;  
  float weight;
  float wtScale = 0;
  float residual = 0;
  int i;


  for (i=0; i<startFrame; ++i) {
    weight = (emRight != NULL) ?( frac* (*emLeft) + (1.0f - frac)*emRight[i*(MAX_POISSON_TABLE_COL)]) :( (*emLeft));
    emLeft += (MAX_POISSON_TABLE_COL);

#if __CUDA_ARCH__ >= 350
    *err = e = weight * __ldg(fg_buffers);
#else
    *err = e = weight * (*fg_buffers);
#endif
    residual += e*e;
    wtScale += weight*weight;
    err += num_beads;
    fg_buffers += num_beads;
#ifdef FVAL_L1
    fval ++;
#else
    fval += num_beads;
#endif
 
  }

  for (i=startFrame; i<nonZeroEmpFrames; ++i) {
    weight = (emRight != NULL) ?( frac* (*emLeft) + (1.0f - frac)*emRight[i*(MAX_POISSON_TABLE_COL)]) :( (*emLeft)); //[i*(MAX_POISSON_TABLE_COL)];
    emLeft += (MAX_POISSON_TABLE_COL);

#if __CUDA_ARCH__ >= 350
    *err = e = weight * (__ldg(fg_buffers) - *fval);
#else
    *err = e = weight * (*fg_buffers - *fval);
#endif
    residual += e*e;
    wtScale += weight*weight;
    err += num_beads;
    fg_buffers += num_beads;
#ifdef FVAL_L1
    fval ++;
#else
    fval += num_beads;
#endif
    
  }
  residual /= wtScale;

  return residual;
}

__device__ void 
ResidualForAlternatingFit(
  float* fg_buffers, 
  float* fval, 
  float* emLeft,
  float* emRight,
  float frac, 
  float* err, 
  float& residual,
  int num_beads,
  int num_frames) {
  int i;
  float e;
  
  residual = 0;

  float weight;
  float wtScale = 0;
  for (i=0; i<num_frames; ++i) {
    weight = emRight != NULL ? frac*emLeft[i*(MAX_POISSON_TABLE_COL)] + (1.0f - frac)*emRight[i*(MAX_POISSON_TABLE_COL)] : emLeft[i*(MAX_POISSON_TABLE_COL)];
    e = fg_buffers[num_beads*i] - fval[num_beads*i];
    err[num_beads*i] = e;
    residual += e*e*weight*weight;
    wtScale += weight*weight;
  }
  residual = residual/wtScale;
}


__device__ float 
CalculateMeanResidualErrorPerFlow(
  int startFrame,
  const float* fg_buffers, 
  const float* fval, 
  const float* weight, // highest hp weighting emphasis vector
  const int num_beads,
  const int num_frames) 
{
  float wtScale = 0.0f;
  float residual = 0;
  float e;

  for (int i=0; i<num_frames; ++i) {

    wtScale += *weight * *weight;

    if (i < startFrame)
      e = *weight * *fg_buffers;
    else
      e = *weight * (*fg_buffers - *fval);

    residual += e*e;

    weight += (LAST_POISSON_TABLE_COL + 1);
    fg_buffers+=num_beads;
#ifdef FVAL_L1
    fval++;
#else
    fval += num_beads;
#endif
  }
  residual = sqrtf(residual/wtScale);

  return residual;
}

__device__ void 
CalculateMeanResidualErrorPerFlowForAlternatingFit(
  float* err, 
  float* weight, // highest hp weighting emphasis vector
  float& residual,
  int num_beads,
  int num_frames) 
{
  int i;
  float e;
  float wtScale = 0.0f;

  residual = 0;
  
  for (i=0; i<num_frames; ++i) {
    wtScale += weight[i]*weight[i];
    e = weight[i] * err[i*num_beads];
    residual += e*e;
  }
  residual = sqrtf(residual/wtScale);
}



__device__ float dotProduct(float *ptr1, float * ptr2, int length, int stride)
{
  float result = 0;
  for(int i = 0; i < length; i++)
        result += ptr1[i*stride] *ptr2[i*stride];

  return result;
}

__device__ void dotProduct(float *result, float *ptr1, float * ptr2, int length, int stride)
{

  for(int i = 0; i < length; i++)
        *result += ptr1[i*stride] *ptr2[i*stride];

}

__device__ void dotProduct(float2 *result2, float *ptr1, float * ptr2, int length, int stride)
{
  float2 tempA;
  float2 tempB;
  for(int i = 0; i < length; i++){
    tempA = *((float2*)(&ptr1[i*stride]));
    tempB = *((float2*)(&ptr2[i*stride]));

    result2->x += tempA.x*tempB.x;
    result2->y += tempA.y*tempB.y;
  }
}
__device__ void dotProduct(float4 *result4, float *ptr1, float * ptr2, int length, int stride)
{
  float4 tempA;
  float4 tempB;
  for(int i = 0; i < length; i++){
    tempA = *((float4*)(&ptr1[i*stride]));
    tempB = *((float4*)(&ptr2[i*stride]));

    result4->x += tempA.x*tempB.x;
    result4->y += tempA.y*tempB.y;
    result4->z += tempA.z*tempB.z;
    result4->w += tempA.w*tempB.w;
  }
}

__device__ float CalculateJTJEntry(  unsigned int mask, 
                                     float* input,  
                                     int idb,
                                     int num_beads,
                                     int num_frames,
                                     int flow_block_size
                                     )
{
 
  unsigned int stepIdx;
  float * basePtr1;
  float * basePtr2;  
  float result = 0;

  if ((mask & 0xFFFFF) == 0) return 0;

  stepIdx  = mask >> PARAM1_STEPIDX_SHIFT;
 // printf("%u/", stepIdx ); 
  basePtr1 = input + stepIdx * num_beads*num_frames *flow_block_size + idb;
  stepIdx = (mask >> PARAM2_STEPIDX_SHIFT) & 63; // 63 == 0011 1111 
 // printf("%u: ", stepIdx ); 

  basePtr2 =  input + stepIdx * num_beads*num_frames *flow_block_size + idb; 

  for(int flow_ndx = 0; flow_ndx<flow_block_size; flow_ndx++){
    bool doDotProductForFlow = (mask >> flow_ndx) & 1;
//    printf("%d", doDotProductForFlow ); 

    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + flow_ndx*num_frames*num_beads;
      float * ptr2 = basePtr2 + flow_ndx*num_frames*num_beads;
      result += dotProduct(ptr1,ptr2,num_frames,num_beads);
      //dotProduct(&result, ptr1,ptr2,num_frames,num_beads);

    }
  }
  //printf(" " ); 

  return result;
}

__device__ float2 CalculateJTJEntryVec2(  unsigned int mask, 
                                     float* input,  
                                     int idb,
                                     int num_beads,
                                     int num_frames,
                                     int flow_block_size 
                                     )
{
 
  unsigned int stepIdx;
  float * basePtr1;
  float * basePtr2;  
  float2 result2;
  result2.x = 0;
  result2.y = 0;
  
  if ((mask & 0xFFFFF) == 0) 
    return result2;

  stepIdx  = mask >> PARAM1_STEPIDX_SHIFT;
  basePtr1 = input + stepIdx * num_beads*num_frames *flow_block_size + idb;
  stepIdx = (mask >> PARAM2_STEPIDX_SHIFT) & 63; // 63 == 0011 1111 
  basePtr2 =  input + stepIdx * num_beads*num_frames *flow_block_size + idb; 
  for(int flow_ndx = 0; flow_ndx<flow_block_size; flow_ndx++){
    bool doDotProductForFlow = (mask >> flow_ndx) & 1;
    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + flow_ndx*num_frames*num_beads;
      float * ptr2 = basePtr2 + flow_ndx*num_frames*num_beads;
      dotProduct(&result2, ptr1,ptr2,num_frames,num_beads);
    }
  }
  return result2;
}

__device__ float4 CalculateJTJEntryVec4(  unsigned int mask, 
                                     float* input,  
                                     int idb,
                                     int num_beads,
                                     int num_frames,
                                     int flow_block_size
                                     )
{
  unsigned int stepIdx;
  float * basePtr1;
  float * basePtr2;  
  float4 result4;
  result4.x = 0;
  result4.y = 0;
  result4.z = 0;
  result4.w = 0;

  stepIdx  = mask >> PARAM1_STEPIDX_SHIFT;
  basePtr1 = input + stepIdx * num_beads*num_frames *flow_block_size + idb;
  stepIdx = (mask >> PARAM2_STEPIDX_SHIFT) & 63; // 63 == 0011 1111 
  basePtr2 =  input + stepIdx * num_beads*num_frames *flow_block_size + idb; 
  for(int flow_ndx = 0; flow_ndx<flow_block_size; flow_ndx++){
    bool doDotProductForFlow = (mask >> flow_ndx) & 1;
    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + flow_ndx*num_frames*num_beads;
      float * ptr2 = basePtr2 + flow_ndx*num_frames*num_beads;
      dotProduct(&result4, ptr1,ptr2,num_frames,num_beads);
    }
  }
  return result4;
} 





__device__ float CalculateRHSEntry(  unsigned int mask, 
                                     float* input,  
                                     int idb,
                                     int num_steps,   
                                     int num_beads,
                                     int num_frames,
                                     int flow_block_size
                                     )
{
 
  int stepIdx;
  float * basePtr1;
  float * basePtr2;  
  float result = 0;

  stepIdx  = mask >> PARAM1_STEPIDX_SHIFT;
 // printf("%d %d \n", stepIdx, num_steps);
  basePtr1 = input + stepIdx * num_beads*num_frames *flow_block_size + idb;
  
  basePtr2 =  input + (num_steps-1) * num_beads*num_frames *flow_block_size + idb;

  for(int flow_ndx = 0; flow_ndx<flow_block_size; flow_ndx++){
    bool doDotProductForFlow = (mask >> flow_ndx) & 1;
    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + flow_ndx*num_frames*num_beads;
      float * ptr2 = basePtr2 + flow_ndx*num_frames*num_beads;
      result += dotProduct(ptr1,ptr2,num_frames,num_beads);
      //dotProduct(&result, ptr1,ptr2,num_frames,num_beads);

    }
  }
  return result;

}

__device__ float2 CalculateRHSEntryVec2(  unsigned int mask, 
                                     float* input,  
                                     int idb,
                                     int num_steps,   
                                     int num_beads,
                                     int num_frames,
                                     int flow_block_size
                                     )
{
 
  int stepIdx;
  float * basePtr1;
  float * basePtr2;  
  float2 result2;
  result2.x = 0;
  result2.y = 0;

  stepIdx  = mask >> PARAM1_STEPIDX_SHIFT;
 // printf("%d %d \n", stepIdx, num_steps);
  basePtr1 = input + stepIdx * num_beads*num_frames *flow_block_size + idb;
  
  basePtr2 =  input + (num_steps-1) * num_beads*num_frames *flow_block_size + idb;

  for(int flow_ndx = 0; flow_ndx<flow_block_size; flow_ndx++){
    bool doDotProductForFlow = (mask >> flow_ndx) & 1;
    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + flow_ndx*num_frames*num_beads;
      float * ptr2 = basePtr2 + flow_ndx*num_frames*num_beads;
      dotProduct(&result2, ptr1,ptr2,num_frames,num_beads);
      //dotProduct(&result, ptr1,ptr2,num_frames,num_beads);

    }
  }
  return result2;

}
__device__ float4 CalculateRHSEntryVec4(  unsigned int mask, 
                                     float* input,  
                                     int idb,
                                     int num_steps,   
                                     int num_beads,
                                     int num_frames,
                                     int flow_block_size
                                     )
{
 
  int stepIdx;
  float * basePtr1;
  float * basePtr2;  
  float4 result4;
  result4.x = 0;
  result4.y = 0;
  result4.z = 0;
  result4.w = 0;
  stepIdx  = mask >> PARAM1_STEPIDX_SHIFT;
 // printf("%d %d \n", stepIdx, num_steps);
  basePtr1 = input + stepIdx * num_beads*num_frames *flow_block_size + idb;
  
  basePtr2 =  input + (num_steps-1) * num_beads*num_frames *flow_block_size + idb;

  for(int flow_ndx = 0; flow_ndx<flow_block_size; flow_ndx++){
    bool doDotProductForFlow = (mask >> flow_ndx) & 1;
    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + flow_ndx*num_frames*num_beads;
      float * ptr2 = basePtr2 + flow_ndx*num_frames*num_beads;
      dotProduct(&result4, ptr1,ptr2,num_frames,num_beads);
      //dotProduct(&result, ptr1,ptr2,num_frames,num_beads);

    }
  }
  return result4;

}

__device__ float CalculateNonDiagLowerTriangularElements_dev(
    int bead_ndx,
    int row, 
    float** curJtj, 
    float* ltr, 
    float** curLtr, 
    int stride)
{
  //if (bead_ndx == 33) printf("Non Diag Ele Calculation\n");
  float dotP = 0;
  float runningSumNonDiagonalEntries = 0;
  float curRowElement = 0;
  for (int i=0; i<row; ++i) {
    curRowElement = ((*curJtj)[bead_ndx] - runningSumNonDiagonalEntries) / ltr[bead_ndx];
    //if (bead_ndx == 96) printf("r: %d, c: %d, curRowElement: %f\n", row, i, curRowElement);
    dotP += (curRowElement*curRowElement);
    (*curLtr)[i*stride + bead_ndx] = curRowElement;
    runningSumNonDiagonalEntries = 0;
    ltr += stride;
    for (int j=0; j<=i; ++j) {
      //if (bead_ndx == 33) printf("j: %d, ltr: %f, curltr: %f\n", j, ltr[bead_ndx], (*curLtr)[j*stride + bead_ndx]);
      runningSumNonDiagonalEntries += (ltr[bead_ndx]*((*curLtr)[j*stride + bead_ndx]));
      ltr += stride;
    }
    (*curJtj) += stride;
    
  }

  (*curLtr) += row*stride;

  return dotP;  
}

// Solving for Ly = b
__device__ void SolveLowerTriangularMatrix_dev(
    float* y, // y solution vector
    float* ltr, // lower triangular matrix 
    float* rhs, // b vector
    int bead_ndx,
    int num_params,
    int stride)
{
  //printf("Solve Lower Triangular Matrix\n");
  float sum;
  int i,j;
  for (i=0; i<num_params; ++i) 
  {
    sum = 0;
    for (j=0; j<i; ++j) 
    {
      sum += y[j*stride + bead_ndx] * ltr[bead_ndx];
      ltr += stride;    
    }
    y[i*stride + bead_ndx] = (rhs[bead_ndx] - sum) / ltr[bead_ndx];
    //printf("sum: %f, param: %d rhs: %f, y: %f\n", sum, i, rhs[bead_ndx], y[i*stride + bead_ndx]);
    //if (bead_ndx == 96) printf("sum: %f, rhs: %f, y: %f\n", sum, rhs[bead_ndx], y[i*stride + bead_ndx]);
    ltr += stride;
    rhs += stride;
  }
}

// Solving for LTx = y hwere LT is upper triangular 
__device__ void SolveUpperTriangularMatrix_dev(
    float* x, // x solution vector
    float* ltr, // lower triangular matrix 
    float* y, // y vector
    int bead_ndx,
    int num_params,
    int stride)
{
  //printf("Solve Upper Triangular Matrix\n");
  float sum;
  int i, j;
  int lastRowIdx = ((num_params * (num_params + 1)) / 2) - 1;
  int idx = lastRowIdx;
  for (i=(num_params - 1); i>=0; --i) 
  {
    sum = 0;
    for (j=num_params; j>(i+1); --j) 
    {
      sum += (ltr[idx*stride + bead_ndx] * x[(j-1)*stride + bead_ndx]);
      //printf("ltr: %f, x: %f, idx: %d\n", ltr[idx*stride + bead_ndx], x[(j-1)*stride + bead_ndx], idx);
      idx = idx - j + 1;
    }
    //if (bead_ndx == 96) printf("y: %f\n", y[i*stride + bead_ndx]);
    x[i*stride + bead_ndx] = (y[i*stride + bead_ndx] - sum)/ltr[idx*stride + bead_ndx];
    //if (bead_ndx == 96) printf("sum: %f, param: %d, y: %f, x: %f, idx: %d\n", sum, i, y[i*stride + bead_ndx], x[i*stride + bead_ndx], idx);
    lastRowIdx--;
    idx = lastRowIdx;
  }
}

// Zero out the JTJ matrix before building the matrix
// It might be a device function called from the kernel performing lev mar fitting
// Solve Ax = b
// Write A as A= L(LT) where lT implies transpose of L. Here L is lower triangular matrix
// L(LT)x = b
// Assume (LT)x = y
// Ly = b 
// Solve for y and back substitue in (LT)x = y to solve for x
// Here A is JTJ matrix, x is delta step for the params to fit and b is the squared residual times (JT)
__device__ void CholeskySolve_dev(
  float lambda,
  float* jtj, // matrix from build matrix kernel 
  float* scratch_mat,
  float* rhs,
  float* delta,
  int bead_ndx,
  int num_params,
  int num_beads
  // bit mask for beads we want to compute. Need to filter beads 
  // whose JTJ matrix is not positive definite
)
{
  //printf("Cholesky Solve\n");
  int row;
  float dotProduct; // lrr is diagonal entry in lower triangular matrix where c  and r are column and row
  float* curJtjPtr = jtj;
  float* ltr = scratch_mat;
  float* curLtr = scratch_mat;
  //printf("lambda: %f\n", lambda);
  for (row=0; row<num_params; ++row) 
  {
    // product of square of non diagonal entries in a row in lower triangular matrix
    dotProduct = CalculateNonDiagLowerTriangularElements_dev(bead_ndx, row, &curJtjPtr, ltr, &curLtr, num_beads);
    // diagonal entry calculation
    curLtr[bead_ndx] = sqrtf(curJtjPtr[bead_ndx]*(1.0f + lambda) - dotProduct);
    //if (bead_ndx == 96) printf("row: %d, arr: %f, dotP: %f, lrr: %f\n", row, curJtjPtr[bead_ndx], dotProduct, curLtr[bead_ndx]);
    curLtr += num_beads;
    curJtjPtr += num_beads;
  }

  SolveLowerTriangularMatrix_dev(delta, ltr, rhs, bead_ndx, num_params, num_beads);
  SolveUpperTriangularMatrix_dev(delta, ltr, delta, bead_ndx, num_params, num_beads);
}

__device__ void CalculateNewBeadParams_dev(
  float* orig_params,
  float* new_params,
  float* delta,
  unsigned int* paramIdxMap,
  int bead_ndx,
  int num_params,
  int num_beads,
  int sId,
  int flow_block_size
)
{
  unsigned int paramIdx;
  //printf("New Params\n");
 /* for (int i=0; i<num_params; ++i)
  {
    paramIdx = paramIdxMap[i];
    printf("old: %f new: %f pIdx: %d\n", params[paramIdx*num_beads + bead_ndx], params[paramIdx*num_beads + bead_ndx] + delta[i*num_beads + bead_ndx], paramIdx);
    params[paramIdx*num_beads + bead_ndx] += delta[i*num_beads + bead_ndx];
  }*/  

  unsigned int AmplIdx = BEAD_OFFSET(Ampl[0]);
  unsigned int RIdx = BEAD_OFFSET(R);
  unsigned int CopiesIdx = BEAD_OFFSET(Copies);
  unsigned int DmultIdx = BEAD_OFFSET(dmult);
  float paramVal;
  for (int i=0; i<num_params; ++i) 
  {
    paramIdx = paramIdxMap[i];
    if (paramIdx == RIdx) {
      paramVal = orig_params[paramIdx*num_beads + bead_ndx] + delta[i*num_beads + bead_ndx];
      clamp_streaming(paramVal, CP[sId].beadParamsMinConstraints.R, CP[sId].beadParamsMaxConstraints.R); //CP_MULTIFLOWFIT //CP_MULTIFLOWFIT
    }
    if (paramIdx == CopiesIdx) {
      paramVal = orig_params[paramIdx*num_beads + bead_ndx] + delta[i*num_beads + bead_ndx];
      clamp_streaming(paramVal, CP[sId].beadParamsMinConstraints.Copies, CP[sId].beadParamsMaxConstraints.Copies); //CP_MULTIFLOWFIT //CP_MULTIFLOWFIT
    }
    if (paramIdx == DmultIdx) {
      paramVal = orig_params[paramIdx*num_beads + bead_ndx] + delta[i*num_beads + bead_ndx];
      clamp_streaming(paramVal, CP[sId].beadParamsMinConstraints.dmult, CP[sId].beadParamsMaxConstraints.dmult); //CP_MULTIFLOWFIT //CP_MULTIFLOWFIT
    }
    if (paramIdx >= AmplIdx && paramIdx <= (AmplIdx + flow_block_size - 1)) {
      paramVal = orig_params[paramIdx*num_beads + bead_ndx] + delta[i*num_beads + bead_ndx];
      clamp_streaming(paramVal, CP[sId].beadParamsMinConstraints.Ampl, CP[sId].beadParamsMaxConstraints.Ampl); //CP_MULTIFLOWFIT //CP_MULTIFLOWFIT
    }
    //printf("old: %f new: %f pIdx: %d\n", params[paramIdx*num_beads + bead_ndx], paramVal, paramIdx);
    new_params[paramIdx*num_beads + bead_ndx] = paramVal;
  }
}

__device__ void UpdateBeadParams_dev(
  float* orig_params,
  float* new_params,
  unsigned int* paramIdxMap,
  int bead_ndx,
  int num_params,
  int num_beads 
)
{
  unsigned int paramIdx;
  //printf("Updated Params in Lev Mar Iter\n");
  for (int i=0; i<num_params; ++i)
  {
    paramIdx = paramIdxMap[i];
    //printf("new: %f pIdx: %d\n", new_params[paramIdx*num_beads + bead_ndx], paramIdx);
    orig_params[paramIdx*num_beads + bead_ndx] = new_params[paramIdx*num_beads + bead_ndx];
  }  
}

__device__ void CalculateMultiFlowFitResidual_dev(
  float& residual,
  float* pObservedTrace,
  float* pModelTrace,
  float* pEmphasisVec,
  int flow_ndx,
  int num_beads,
  int num_frames,
  int nonZeroEmpFrames
)
{
  float eval;
  pObservedTrace += flow_ndx*num_beads*num_frames;
  for (int j=0; j<nonZeroEmpFrames; ++j)
  {
    eval = (*pObservedTrace - *pModelTrace)*pEmphasisVec[j];
    residual += eval*eval;
    pObservedTrace += num_beads;
    pModelTrace += num_beads;
  }
}

__device__ float DecideOnEmphasisVectorsForInterpolation(
  const int* nonZeroEmpFramesVec,
  const float** emLeft,
  const float** emRight,
  const float Ampl,
  const float* emphasis,
  const int num_frames,
  int &nonZeroEmpFrames
)
{
  float frac;
  int left;
  if (Ampl < LAST_POISSON_TABLE_COL) {
    left = (int) Ampl;
    frac = (left + 1.0f - Ampl);
    if (left < 0) {
      left = 0;
      frac = 1.0f;
    }
    *emLeft = &emphasis[left];
    *emRight = &emphasis[left + 1];
  }else{
    left = LAST_POISSON_TABLE_COL;
    *emLeft = &emphasis[left]; 
    *emRight = NULL;
    frac = 1.0f;
  }

  nonZeroEmpFrames = (left == LAST_POISSON_TABLE_COL) ? nonZeroEmpFramesVec[left] :
            max(nonZeroEmpFramesVec[left], nonZeroEmpFramesVec[left + 1]);
  return frac;
}

__device__ void DynamicConstraintKrate(
  float copies,
  float Ampl,
  float& kmult,
  bool& twoParamFit)
{
  float magic = 2.0f/copies;
  float thresh = Ampl > 0.0f ? Ampl : 0.0f;

  float lower_bound = 2.0f*magic/ (magic+thresh);
  float upper_bound = 1.0f/lower_bound;
  if (lower_bound > 1.0f)
  {
    kmult = 1.0f;
    twoParamFit = false;
  }
  else
  {
    if (kmult > upper_bound) 
      kmult = upper_bound;
    if (kmult < lower_bound)
      kmult = lower_bound;
    twoParamFit = true;
  }
}

__device__    
void ModelFunctionEvaluationForExponentialTailFit_dev(
  int start, 
  int num_frames, 
  int num_beads, 
  float A, 
  float taubInv, 
  float dc_offset, 
  float* fval,
  const ConstParams* pCP,
  float* tmp_fval)
{
  fval += start*num_beads;
  if (tmp_fval)
    tmp_fval += start*num_beads;

  float val;
  for (int i=start; i<num_frames; ++i) {
    
     val = A * __expf(-(pCP->frameNumber[i] - pCP->frameNumber[start])*taubInv)
                   + dc_offset;
     if (tmp_fval) {
       *tmp_fval = (val - *fval) / 0.001f;
       tmp_fval += num_beads;
     }
     else {
       *fval = val;
     }
     fval += num_beads;
  }
}

__device__    
void CalculateResidualForExponentialTailFit_dev(
  float* obs, 
  float* pred, 
  int start,
  int end,
  float* err,
  int num_beads,
  float& residual)
{
  residual = 0; 
  float e;

  obs += start*num_beads;
  pred += start*num_beads;
  err += start*num_beads;
  for (int i=start; i<end; ++i) {
    e = *obs - *pred;
    *err = e;
    residual += e*e;
    obs += num_beads;
    pred += num_beads;
    err += num_beads;
  }
}
 

/*****************************************************************************

              SINGLE FLOW FIT KERNELS 

*****************************************************************************/


// Let number of beads be N and frames be F. The size for each input argument in
// comments is in bytes.
__global__ void 
PerFlowGaussNewtonFit_k(
  // inputs
  float* fg_buffers, // NxF
  float* emphasisVec, 
  float* nucRise, 
  float * pBeadParamsBase, //N
  bead_state* pState,

  // scratch space in global memory
  float* err, // NxF
#ifndef FVAL_L1
  float* fval, // NxF
  float* tmp_fval, // NxF
#endif
  float* meanErr,
  // other inputs 
  float minAmpl,
  float maxKmult,
  float minKmult,
  float adjKmult,
  bool fitKmult,
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId,
  int flow_block_size
) 
{
  //useDynamicEmphasis = false;
#ifdef FVAL_L1
  float fval[MAX_COMPRESSED_FRAMES_GPU];
  float tmp_fval[MAX_COMPRESSED_FRAMES_GPU];
#endif

  extern __shared__ float emphasis[];
  int numWarps = blockDim.x/32;
  int threadWarpIdx = threadIdx.x%32;
  int warpIdx = threadIdx.x/32; 
  for(int i=warpIdx; i<num_frames; i += numWarps)
  {
     if (threadWarpIdx < MAX_POISSON_TABLE_COL)
      emphasis[(MAX_POISSON_TABLE_COL)*i + threadWarpIdx ] = emphasisVec[num_frames*threadWarpIdx + i ];
  }
  __syncthreads();

  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+31)/32) * 32;
  pBeadParamsBase += bead_ndx;
  pState += bead_ndx; 
  

  float *pCopies = &pBeadParamsBase[BEAD_OFFSET(Copies)*num_beads];
  float *pAmpl = &pBeadParamsBase[BEAD_OFFSET(Ampl[0])*num_beads];
  float *pKmult = &pBeadParamsBase[BEAD_OFFSET(kmult[0])*num_beads];
   

#ifdef FVAL_L1
//  fval = fval_l1;
//  tmp_fval = tmp_fval_l1;
#else
  fval += bead_ndx;
  tmp_fval += bead_ndx;
#endif

  err += bead_ndx;
  meanErr += bead_ndx;
 
  fg_buffers += bead_ndx;


  if (pState->corrupt || !pState->clonal_read || pState->pinned) return;

  float avg_err;
    
  float* deltaFrames = CP[sId].useRecompressTailRawTrace ? 
        CP[sId].deltaFrames_std : CP[sId].deltaFrames;
  int* nonZeroEmpFramesVec = CP[sId].useRecompressTailRawTrace ? 
        CP[sId].std_non_zero_emphasis_frames : CP[sId].non_zero_emphasis_frames;
  for(int flow_ndx=0; flow_ndx<flow_block_size; flow_ndx++){

    int nucid = CP[sId].flowIdxMap[flow_ndx]; //CP_SINGLEFLOWFIT
    float sens = CP[sId].sens*SENSMULTIPLIER;  //CP_SINGLEFLOWFIT

    float copies = *pCopies;
    float R = *(pCopies + num_beads);
    float d = *(pCopies + 2*num_beads);
    float gain = *(pCopies + 3 * num_beads) ;
    
    d *= CP[sId].d[nucid]; //CP_SINGLEFLOWFIT


    //offset for next value gets added to address at end of flow_ndx loop
    
    float krate = *pKmult;

    float Ampl = *pAmpl;
    float etbR;
    float tauB; 
    float SP;  
 
 
    ComputeEtbR_dev(etbR, &CP[sId], R, copies, pBeadParamsBase[BEAD_OFFSET(phi)*num_beads],
                          sId, nucid, realFnum+flow_ndx); //CP_SINGLEFLOWFIT
    ComputeTauB_dev(tauB, &CP[sId], etbR, sId); //CP_SINGLEFLOWFIT
    ComputeSP_dev(SP, &CP[sId], copies, realFnum+flow_ndx, sId); //CP_SINGLEFLOWFIT
 
    bool twoParamFit = fitKmult || ( copies * Ampl > adjKmult );
 
    float residual, newresidual; // lambdaThreshold;
    int i;
    // These values before start are always zero since there is no nucrise yet. Don't need to
    // zero it out. Have to change the residual calculation accordingly for the frames before the
    // start.
    for (i =0; i < CP[sId].start[flow_ndx]; i++) { //CP_SINGLEFLOWFIT

#ifdef FVAL_L1
      //fval[i] = 0;
      //tmp_fval[i] = 0;
#else
      //fval[num_beads*i] = 0;
      //tmp_fval[num_beads*i] = 0;
#endif

    }

    // first step
    // Evaluate model function using input Ampl and Krate and get starting residual
    Fermi_ModelFuncEvaluationForSingleFlowFitNoOutput(&CP[sId], flow_ndx, nucid, nucRise, 
        Ampl, krate*CP[sId].krate[nucid], tauB, gain, SP, d,  //CP_SINGLEFLOWFIT
        sens, ISIG_SUB_STEPS_SINGLE_FLOW* CP[sId].start[flow_ndx], //CP_SINGLEFLOWFIT
        num_frames, num_beads, fval, deltaFrames, num_frames);

    const float *emLeft, *emRight;
    float frac;

    // calculating weighted sum of square residuals for the convergence test
    int nonZeroEmpFrames = 0;
    frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,Ampl,emphasis, num_frames, nonZeroEmpFrames);
    residual = ResidualCalculationPerFlow(CP[sId].start[flow_ndx], fg_buffers, fval, emLeft, emRight, frac, err,
      num_beads, nonZeroEmpFrames);
  
    // new Ampl and Krate generated from the Lev mar Fit
    float newAmpl, newKrate;

    // convergence test variables 
    //int flowDone = 0;

    float delta0 = 0, delta1 = 0;

    // Lev Mar Fit Outer Loop
    int iter;
    for (iter = 0; iter < ITER; ++iter) {

      // new Ampl and krate by adding delta to existing values
      newAmpl = Ampl + 0.001f;
      newKrate = (twoParamFit)?(krate + 0.001f):(krate);
 
      // Evaluate model function for new Ampl keeping Krate constant
      float aa = 0, akr= 0, krkr = 0, rhs0 = 0, rhs1 = 0;

      Fermi_ModelFuncEvaluationForSingleFlowFit(sId, flow_ndx, nucid, nucRise,
          newAmpl, Ampl, krate*CP[sId].krate[nucid], newKrate*CP[sId].krate[nucid],
          tauB, gain, SP, d,  //CP_SINGLEFLOWFIT
          sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW,  //CP_SINGLEFLOWFIT
          num_frames, num_beads, twoParamFit ? TwoParams : OneParam,
          emLeft, emRight, frac, fval, 
          err, &aa, &rhs0, &krkr, &rhs1, &akr, deltaFrames, nonZeroEmpFrames);

     // Now start the solving.        
      if(twoParamFit){ 
        float det = 1.0f / (aa*krkr - akr*akr);
        delta1 = (-akr*rhs0 + aa*rhs1)*det;
        delta0 = (krkr*rhs0 - akr*rhs1)*det;
      }else
        delta0 = rhs0 / aa;

      if( !isnan(delta0) && !isnan(delta1)){
        // add delta to params to obtain new params
        newAmpl = Ampl + delta0;
        if(twoParamFit)newKrate = krate + delta1;

        clamp_streaming(newAmpl, minAmpl, (float)LAST_POISSON_TABLE_COL);
        if(twoParamFit)clamp_streaming(newKrate, minKmult, maxKmult);

        // Evaluate using new params
        Fermi_ModelFuncEvaluationForSingleFlowFitNoOutput(&CP[sId], flow_ndx, nucid, nucRise, 
            newAmpl, newKrate*CP[sId].krate[nucid], tauB, gain, SP,  //CP_SINGLEFLOWFIT
            d, sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW,  //CP_SINGLEFLOWFIT
            num_frames, num_beads, tmp_fval, deltaFrames, num_frames);

        // residual calculation using new parameters
        if (useDynamicEmphasis) {
          int newNonZeroEmpFrames;
          frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,newAmpl,emphasis, num_frames,
                      newNonZeroEmpFrames);
          nonZeroEmpFrames = max(nonZeroEmpFrames, newNonZeroEmpFrames);
        }
        newresidual = ResidualCalculationPerFlow(CP[sId].start[flow_ndx], fg_buffers, tmp_fval, emLeft, emRight, frac, err, num_beads, nonZeroEmpFrames);
        
        if (newresidual < residual) {
          Ampl = newAmpl;
          if(twoParamFit)krate = newKrate;
          // copy new function val to fval
          for (i=CP[sId].start[flow_ndx]; i<num_frames; ++i){ //CP_SINGLEFLOWFIT
#ifdef FVAL_L1
            fval[i] = tmp_fval[i];
#else
            fval[num_beads*i] = tmp_fval[num_beads*i];
#endif
          }
          residual = newresidual;
        }
        else {
          if (useDynamicEmphasis) {
            frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,Ampl,emphasis, 
              num_frames, nonZeroEmpFrames);
          }
        }
      }

      if ((delta0*delta0) < 0.0000025f){
        iter++;
        break;
      }
    } // end ITER loop
    //atomicAdd(&pMonitor[iter-1], 1);

    if(flow_ndx==0) avg_err = pState->avg_err * realFnum;  

    if(twoParamFit) *pKmult = krate;
    *pAmpl= Ampl;

 
    residual = CalculateMeanResidualErrorPerFlow(CP[sId].start[flow_ndx], fg_buffers, fval, emphasis+LAST_POISSON_TABLE_COL,
      num_beads, num_frames); 
  
    avg_err += residual;
    meanErr[num_beads * flow_ndx] = residual;

    pAmpl += num_beads;
    pKmult += num_beads;
    fg_buffers += num_frames*num_beads;
  } // end flow_ndx loop

  avg_err /= (realFnum + flow_block_size);
  pState->avg_err = avg_err;
  int high_err_cnt = 0;
  avg_err *= WASHOUT_THRESHOLD;
  for (int flow_ndx = flow_block_size - 1; flow_ndx >= 0 
                               && (meanErr[num_beads* flow_ndx] > avg_err); flow_ndx--)
    high_err_cnt++;

  if (high_err_cnt > WASHOUT_FLOW_DETECTION)
    pState->corrupt = true;

}


__global__ void 
PerFlowHybridFit_k(
  // inputs
  float* fg_buffers, // NxF
  float* emphasisVec, 
  float* nucRise, 
  float * pBeadParamsBase, //N
  bead_state* pState,
  // scratch space in global memory
  float* err, // NxF
#ifndef FVAL_L1
  float* fval, // NxF
  float* tmp_fval, // NxF
#endif
  float* meanErr,
  // other inputs 
  float minAmpl,
  float maxKmult,
  float minKmult,
  float adjKmult,
  bool fitKmult,
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId,
  int switchToLevMar,
  int flow_block_size
) 
{

#ifdef FVAL_L1
  float fval[MAX_COMPRESSED_FRAMES_GPU];
  float tmp_fval[MAX_COMPRESSED_FRAMES_GPU];
#endif

  extern __shared__ float emphasis[];
  int numWarps = blockDim.x/32;
  int threadWarpIdx = threadIdx.x%32;
  int warpIdx = threadIdx.x/32; 
  for(int i=warpIdx; i<num_frames; i += numWarps)
  {
     if (threadWarpIdx < MAX_POISSON_TABLE_COL)
      emphasis[(MAX_POISSON_TABLE_COL)*i + threadWarpIdx ] = emphasisVec[num_frames*threadWarpIdx + i ];
  }
  __syncthreads();

  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+31)/32) * 32;
  pBeadParamsBase += bead_ndx;
  pState += bead_ndx; 
  

  float *pCopies = &pBeadParamsBase[BEAD_OFFSET(Copies)*num_beads];
  float *pAmpl = &pBeadParamsBase[BEAD_OFFSET(Ampl[0])*num_beads];
  float *pKmult = &pBeadParamsBase[BEAD_OFFSET(kmult[0])*num_beads];

   

#ifdef FVAL_L1
//  fval = fval_l1;
//  tmp_fval = tmp_fval_l1;
#else
  fval += bead_ndx;
  tmp_fval += bead_ndx;
#endif

  err += bead_ndx;
  meanErr += bead_ndx;
 
  fg_buffers += bead_ndx;


  if (pState->corrupt || !pState->clonal_read || pState->pinned) return;

  float avg_err;
    
  float* deltaFrames = CP[sId].useRecompressTailRawTrace ? 
        CP[sId].deltaFrames_std : CP[sId].deltaFrames;
  int* nonZeroEmpFramesVec = CP[sId].useRecompressTailRawTrace ? 
        CP[sId].std_non_zero_emphasis_frames : CP[sId].non_zero_emphasis_frames;
  for(int flow_ndx=0; flow_ndx<flow_block_size; flow_ndx++){

    
    int nucid = CP[sId].flowIdxMap[flow_ndx]; //CP_SINGLEFLOWFIT
    float sens = CP[sId].sens*SENSMULTIPLIER;  //CP_SINGLEFLOWFIT

    float copies = *pCopies;
    float R = *(pCopies + num_beads);
    float d = *(pCopies + 2*num_beads);
    float gain = *(pCopies + 3 * num_beads) ;
    
    d *= CP[sId].d[nucid]; //CP_SINGLEFLOWFIT


    //offset for next value gets added to address at end of flow_ndx loop
    
    float krate = *pKmult;

    float Ampl = *pAmpl;
    float etbR;
    float tauB; // = tmp.x; // *ptauB;
    float SP; //= tmp.y; // *pSP;  
 
 
    ComputeEtbR_dev(etbR, &CP[sId], R, copies, pBeadParamsBase[BEAD_OFFSET(phi)*num_beads],
                          sId, nucid, realFnum+flow_ndx); //CP_SINGLEFLOWFIT
    ComputeTauB_dev(tauB, &CP[sId], etbR, sId); //CP_SINGLEFLOWFIT
    ComputeSP_dev(SP, &CP[sId], copies, realFnum+flow_ndx, sId); //CP_SINGLEFLOWFIT
  
    bool twoParamFit = fitKmult || ( copies * Ampl > adjKmult );
 
    float residual, newresidual; // lambdaThreshold;
    int i;
    // These values before start are always zero since there is no nucrise yet. Don't need to
    // zero it out. Have to change the residual calculation accordingly for the frames before the
    // start.
    for (i =0; i < CP[sId].start[flow_ndx]; i++) { //CP_SINGLEFLOWFIT

#ifdef FVAL_L1
      //fval[i] = 0;
      //tmp_fval[i] = 0;
#else
      //fval[num_beads*i] = 0;
      //tmp_fval[num_beads*i] = 0;
#endif

    }

    // first step
    // Evaluate model function using input Ampl and Krate and get starting residual
    Fermi_ModelFuncEvaluationForSingleFlowFitNoOutput(&CP[sId], flow_ndx, nucid, nucRise, 
        Ampl, krate*CP[sId].krate[nucid], tauB, gain, SP, d,  //CP_SINGLEFLOWFIT
        sens, ISIG_SUB_STEPS_SINGLE_FLOW* CP[sId].start[flow_ndx], //CP_SINGLEFLOWFIT
        num_frames, num_beads, fval, deltaFrames, num_frames);

    const float *emLeft, *emRight;
    float frac;

    // calculating weighted sum of square residuals for the convergence test
    int nonZeroEmpFrames = 0;
    frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,Ampl,emphasis, num_frames, nonZeroEmpFrames);
    residual = ResidualCalculationPerFlow(CP[sId].start[flow_ndx], fg_buffers, fval, emLeft, emRight, frac, err,
      num_beads, nonZeroEmpFrames);
  
    // new Ampl and Krate generated from the Lev mar Fit
    float newAmpl, newKrate;

    // convergence test variables 
    float delta0 = 0, delta1 = 0;

    float det;

    // Indicates whether a flow has converged
    //int flowDone = 0;

    float lambda = 1E-20;

    // Lev Mar Fit Outer Loop
    int iter;
    for (iter = 0; iter < ITER; ++iter) {

      // new Ampl and krate by adding delta to existing values
      newAmpl = Ampl + 0.001f;
      newKrate = (twoParamFit)?(krate + 0.001f):(krate);
 
      // Evaluate model function for new Ampl keeping Krate constant
      float aa = 0, akr= 0, krkr = 0, rhs0 = 0, rhs1 = 0;

      Fermi_ModelFuncEvaluationForSingleFlowFit(sId, flow_ndx, nucid, nucRise,
          newAmpl, Ampl, krate*CP[sId].krate[nucid], newKrate*CP[sId].krate[nucid],
          tauB, gain, SP, d,  //CP_SINGLEFLOWFIT
          sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW,  //CP_SINGLEFLOWFIT
          num_frames, num_beads, twoParamFit ? TwoParams : OneParam,
          emLeft, emRight, frac, fval, 
          err, &aa, &rhs0, &krkr, &rhs1, &akr, deltaFrames, nonZeroEmpFrames);

     // Now start the solving.
     if(iter< switchToLevMar){


      if(twoParamFit){ 
        float det = 1.0f / (aa*krkr - akr*akr);
        delta1 = (-akr*rhs0 + aa*rhs1)*det;
        delta0 = (krkr*rhs0 - akr*rhs1)*det;
      }else
        delta0 = rhs0 / aa;

      if( !isnan(delta0) && !isnan(delta1)){
        // add delta to params to obtain new params
        newAmpl = Ampl + delta0;
        if(twoParamFit)newKrate = krate + delta1;

        clamp_streaming(newAmpl, minAmpl, (float)LAST_POISSON_TABLE_COL);
        if(twoParamFit)clamp_streaming(newKrate, minKmult, maxKmult);

        // Evaluate using new params
        Fermi_ModelFuncEvaluationForSingleFlowFitNoOutput(&CP[sId], flow_ndx, nucid, nucRise, 
            newAmpl, newKrate*CP[sId].krate[nucid], tauB, gain, SP,  //CP_SINGLEFLOWFIT
            d, sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW,  //CP_SINGLEFLOWFIT
            num_frames, num_beads, tmp_fval, deltaFrames, num_frames);

        // residual calculation using new parameters
        if (useDynamicEmphasis){
          int newNonZeroEmpFrames;
          frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,newAmpl,emphasis, num_frames,
                      newNonZeroEmpFrames);
          nonZeroEmpFrames = max(nonZeroEmpFrames, newNonZeroEmpFrames);
        }
        newresidual = ResidualCalculationPerFlow(CP[sId].start[flow_ndx], fg_buffers, tmp_fval, emLeft, emRight, frac, err,
           num_beads, nonZeroEmpFrames);
        
        if (newresidual < residual) {
          Ampl = newAmpl;
          if(twoParamFit)krate = newKrate;
          // copy new function val to fval
          for (i=CP[sId].start[flow_ndx]; i<num_frames; ++i){ //CP_SINGLEFLOWFIT
#ifdef FVAL_L1
            fval[i] = tmp_fval[i];
#else
            fval[num_beads*i] = tmp_fval[num_beads*i];
#endif
          }
          residual = newresidual;
        }
        else {
          if (useDynamicEmphasis) {
            frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,Ampl,emphasis, 
              num_frames, nonZeroEmpFrames);
          }
        }
      }


    }else{ //LevMar Instead

     bool cont_proc = false;        
     while (!cont_proc) {
      if(twoParamFit){ 
        det = 1.0f / (aa*krkr*(1.0f + lambda)*(1.0f + lambda) - akr*akr);
        delta0 = (krkr*(1.0f + lambda)*rhs0 - akr*rhs1)*det;
        delta1 = (-akr*rhs0 + aa*(1.0f + lambda)*rhs1)*det;

      }else
        delta0 = rhs0 / (aa*(1.0f + lambda));

       // NAN check
      bool nan_detected = false;
      if( !isnan(delta0) && !isnan(delta1)){
        // add delta to params to obtain new params
        newAmpl = Ampl + delta0;
        if(twoParamFit)newKrate = krate + delta1;

        clamp_streaming(newAmpl, minAmpl, (float)LAST_POISSON_TABLE_COL);
        if(twoParamFit)clamp_streaming(newKrate, minKmult, maxKmult);

        // Evaluate using new params
        Fermi_ModelFuncEvaluationForSingleFlowFitNoOutput(&CP[sId], flow_ndx, nucid, nucRise, 
            newAmpl, newKrate*CP[sId].krate[nucid], tauB, gain, SP,  //CP_SINGLEFLOWFIT
            d, sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW, //CP_SINGLEFLOWFIT
            num_frames, num_beads, tmp_fval, deltaFrames, num_frames);

        // residual calculation using new parameters
        if (useDynamicEmphasis) {
          int newNonZeroEmpFrames;
          frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,newAmpl,emphasis, num_frames,
                      newNonZeroEmpFrames);
          nonZeroEmpFrames = max(nonZeroEmpFrames, newNonZeroEmpFrames);
        }
        newresidual = ResidualCalculationPerFlow(CP[sId].start[flow_ndx], fg_buffers, tmp_fval, emLeft, emRight, frac, err,
           num_beads, nonZeroEmpFrames);
      }
      else 
        nan_detected = true;
   
      // this might be killing...Need to rethink for some alternative here
      // If new residual is less than the earlier recorded residual, accept the solution and
      // obtain new parameters and copy them to original parameters and copy the new model function 
      // to the earlier recorded model function till this point
      if (newresidual < residual && !nan_detected) {
        lambda /= 10.0f;
        if (lambda < FLT_MIN)
          lambda = FLT_MIN;
        Ampl = newAmpl;
        if(twoParamFit)krate = newKrate;
        // copy new function val to fval
        for (i=CP[sId].start[flow_ndx]; i<num_frames; ++i){ //CP_SINGLEFLOWFIT
#ifdef FVAL_L1
            fval[i] = tmp_fval[i];
#else
            fval[num_beads*i] = tmp_fval[num_beads*i];
#endif
          }
        residual = newresidual;
        cont_proc = true;
      }
      else {
        lambda *= 10.0f;
      }

      if (lambda > 1.0f) {
        cont_proc = true;
        if (useDynamicEmphasis) {
          frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,Ampl,emphasis, 
              num_frames, nonZeroEmpFrames);
        }
      }
     }


    }

    if ((delta0*delta0) < 0.0000025f){
      iter++;
      break;
    }


    } // end ITER loop
//    atomicAdd(&pMonitor[iter-1], 1);

    if(flow_ndx==0) avg_err = pState->avg_err * realFnum;  

    if(twoParamFit) *pKmult = krate;
    *pAmpl= Ampl;

 
    residual = CalculateMeanResidualErrorPerFlow(CP[sId].start[flow_ndx], fg_buffers, fval, emphasis+LAST_POISSON_TABLE_COL,
      num_beads, num_frames); 
  
    avg_err += residual;
    meanErr[num_beads * flow_ndx] = residual;

    pAmpl += num_beads;
    pKmult += num_beads;
    fg_buffers += num_frames*num_beads;
  } // end flow_ndx loop

  avg_err /= (realFnum + flow_block_size);
  pState->avg_err = avg_err;
  int high_err_cnt = 0;
  avg_err *= WASHOUT_THRESHOLD;
  for (int flow_ndx = flow_block_size - 1; flow_ndx >= 0 
                           && (meanErr[num_beads* flow_ndx] > avg_err); flow_ndx--)
    high_err_cnt++;

  if (high_err_cnt > WASHOUT_FLOW_DETECTION)
    pState->corrupt = true;


}



__global__ void 
PerFlowLevMarFit_k(
  // inputs
  float* fg_buffers, // NxF
  float* emphasisVec, 
  float* nucRise, 
  float * pBeadParamsBase, //N
  bead_state* pState,
  // scratch space in global memory
  float* err, // NxF
#ifndef FVAL_L1
  float* fval, // NxF
  float* tmp_fval, // NxF
#endif
  float* meanErr,
  // other inputs 
  float minAmpl,
  float maxKmult,
  float minKmult,
  float adjKmult,
  bool fitKmult,
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId,
  int flow_block_size
) 
{
#ifdef FVAL_L1
  float fval[MAX_COMPRESSED_FRAMES_GPU];
  float tmp_fval[MAX_COMPRESSED_FRAMES_GPU];
#endif


  extern __shared__ float emphasis[];
  int numWarps = blockDim.x/32;
  int threadWarpIdx = threadIdx.x%32;
  int warpIdx = threadIdx.x/32; 
  for(int i=warpIdx; i<num_frames; i += numWarps)
  {
     if (threadWarpIdx < MAX_POISSON_TABLE_COL)
      emphasis[(MAX_POISSON_TABLE_COL)*i + threadWarpIdx ] = emphasisVec[num_frames*threadWarpIdx + i ];
  }
  __syncthreads();
  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+31)/32) * 32;
  pBeadParamsBase += bead_ndx;
  pState += bead_ndx; 
  

  float *pCopies = &pBeadParamsBase[BEAD_OFFSET(Copies)*num_beads];
  float *pAmpl = &pBeadParamsBase[BEAD_OFFSET(Ampl[0])*num_beads];
  float *pKmult = &pBeadParamsBase[BEAD_OFFSET(kmult[0])*num_beads];

 #ifdef FVAL_L1
//  fval = fval_l1;
//  tmp_fval = tmp_fval_l1;
#else
  fval += bead_ndx;
  tmp_fval += bead_ndx;
#endif
  
  err += bead_ndx;
  meanErr += bead_ndx;
  
  fg_buffers += bead_ndx;


  if (pState->corrupt || !pState->clonal_read || pState->pinned) return;

  float avg_err;
    
  float* deltaFrames = CP[sId].useRecompressTailRawTrace ? 
        CP[sId].deltaFrames_std : CP[sId].deltaFrames;
  int* nonZeroEmpFramesVec = CP[sId].useRecompressTailRawTrace ? 
        CP[sId].std_non_zero_emphasis_frames : CP[sId].non_zero_emphasis_frames;
  for(int flow_ndx=0; flow_ndx<flow_block_size; flow_ndx++){
    int nucid = CP[sId].flowIdxMap[flow_ndx]; //CP_SINGLEFLOWFIT
    float sens = CP[sId].sens*SENSMULTIPLIER;  //CP_SINGLEFLOWFIT

    float copies = *pCopies;
    float R = *(pCopies + num_beads);
    float d = *(pCopies + 2*num_beads);
    float gain = *(pCopies + 3 * num_beads) ;
    
    d *= CP[sId].d[nucid]; //CP_SINGLEFLOWFIT


    //offset for next value gets added to address at end of flow_ndx loop
    
    float krate = *pKmult;
    float Ampl = *pAmpl;
    float etbR;
    float tauB; // = tmp.x; // *ptauB;
    float SP; //= tmp.y; // *pSP;  
 
 
    ComputeEtbR_dev(etbR, &CP[sId], R, copies, pBeadParamsBase[BEAD_OFFSET(phi)*num_beads],
                          sId, nucid, realFnum+flow_ndx); //CP_SINGLEFLOWFIT
    ComputeTauB_dev(tauB, &CP[sId], etbR, sId); //CP_SINGLEFLOWFIT
    ComputeSP_dev(SP, &CP[sId], copies, realFnum+flow_ndx, sId); //CP_SINGLEFLOWFIT
 
    bool twoParamFit = fitKmult || ( copies * Ampl > adjKmult );
 
    float residual, newresidual; // lambdaThreshold;
    int i, iter;

    // These values before start are always zero since there is no nucrise yet. Don't need to
    // zero it out. Have to change the residual calculation accordingly for the frames before the
    // start.
    for (i=0; i < CP[sId].start[flow_ndx]; i++) { //CP_SINGLEFLOWFIT
#ifdef FVAL_L1
      //fval[i] = 0;
      //tmp_fval[i] = 0;
#else
      //fval[num_beads*i] = 0;
      //tmp_fval[num_beads*i] = 0;
#endif
    }

    // first step
    // Evaluate model function using input Ampl and Krate and get starting residual
    Fermi_ModelFuncEvaluationForSingleFlowFitNoOutput(&CP[sId], flow_ndx, nucid, nucRise, 
        Ampl, krate*CP[sId].krate[nucid], tauB, gain, SP, d,  //CP_SINGLEFLOWFIT
        sens, ISIG_SUB_STEPS_SINGLE_FLOW* CP[sId].start[flow_ndx], //CP_SINGLEFLOWFIT
        num_frames, num_beads, fval, deltaFrames, num_frames);

    const float *emLeft, *emRight;
    float frac;

    // calculating weighted sum of square residuals for the convergence test
    int nonZeroEmpFrames = 0;
    frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,Ampl,emphasis, num_frames, nonZeroEmpFrames);
    residual = ResidualCalculationPerFlow(CP[sId].start[flow_ndx], fg_buffers, fval, emLeft, emRight, frac, err,
      num_beads, nonZeroEmpFrames);
 
    // new Ampl and Krate generated from the Lev mar Fit
    float newAmpl, newKrate;

    // convergence test variables 
    float delta0 = 0, delta1 = 0;

    // determinant for the JTJ matrix in Lev Mar Solve
    float det;

    // Indicates whether a flow has converged
    int flowDone = 0;

    float lambda = 1E-20;

    // Lev Mar Fit Outer Loop
    for (iter = 0; iter < 40; ++iter) {

      // convergence test...need to think of an alternate approach
      if ((delta0*delta0) < 0.0000025f)
        flowDone++;
      else
        flowDone = 0;

      // stop the loop for this bead here
      if (flowDone  >= 2)
      {
        break;
      }
      // new Ampl and krate by adding delta to existing values
      newAmpl = Ampl + 0.001f;
      newKrate = (twoParamFit)?(krate + 0.001f):(krate);
 
      // Evaluate model function for new Ampl keeping Krate constant
      float aa = 0, akr= 0, krkr = 0, rhs0 = 0, rhs1 = 0;

      Fermi_ModelFuncEvaluationForSingleFlowFit(sId, flow_ndx, nucid, nucRise,
          newAmpl, Ampl, krate*CP[sId].krate[nucid], newKrate*CP[sId].krate[nucid],
          tauB, gain, SP, d,  //CP_SINGLEFLOWFIT
          sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW,  //CP_SINGLEFLOWFIT
          num_frames, num_beads, twoParamFit ? TwoParams : OneParam,
          emLeft, emRight, frac, fval, 
          err, &aa, &rhs0, &krkr, &rhs1, &akr, deltaFrames, nonZeroEmpFrames);

     // Now start the solving.
     bool cont_proc = false;        
     while (!cont_proc) {
      if(twoParamFit){ 
        det = 1.0f / (aa*krkr*(1.0f + lambda)*(1.0f + lambda) - akr*akr);
        delta0 = (krkr*(1.0f + lambda)*rhs0 - akr*rhs1)*det;
        delta1 = (-akr*rhs0 + aa*(1.0f + lambda)*rhs1)*det;

      }else
        delta0 = rhs0 / (aa*(1.0f + lambda));

       // NAN check
      bool nan_detected = false;
      if( !isnan(delta0) && !isnan(delta1)){
        // add delta to params to obtain new params
        newAmpl = Ampl + delta0;
        if(twoParamFit)newKrate = krate + delta1;

        clamp_streaming(newAmpl, minAmpl, (float)LAST_POISSON_TABLE_COL);
        if(twoParamFit)clamp_streaming(newKrate, minKmult, maxKmult);

        // Evaluate using new params
        Fermi_ModelFuncEvaluationForSingleFlowFitNoOutput(&CP[sId], flow_ndx, nucid, nucRise, 
            newAmpl, newKrate*CP[sId].krate[nucid], tauB, gain, SP,  //CP_SINGLEFLOWFIT
            d, sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW, //CP_SINGLEFLOWFIT
            num_frames, num_beads, tmp_fval, deltaFrames, num_frames);

        // residual calculation using new parameters
        if (useDynamicEmphasis) {
          int newNonZeroEmpFrames;
          frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,newAmpl,emphasis, num_frames,
                      newNonZeroEmpFrames);
          nonZeroEmpFrames = max(nonZeroEmpFrames, newNonZeroEmpFrames);
        }
        newresidual = ResidualCalculationPerFlow(CP[sId].start[flow_ndx], fg_buffers, tmp_fval, emLeft, emRight, frac, err,
           num_beads, nonZeroEmpFrames);
      }
      else 
        nan_detected = true;
   
      // this might be killing...Need to rethink for some alternative here
      // If new residual is less than the earlier recorded residual, accept the solution and
      // obtain new parameters and copy them to original parameters and copy the new model function 
      // to the earlier recorded model function till this point
      if (newresidual < residual && !nan_detected) {
        lambda /= 10.0f;
        if (lambda < FLT_MIN)
          lambda = FLT_MIN;
        Ampl = newAmpl;
        if(twoParamFit)krate = newKrate;
        // copy new function val to fval
        for (i=CP[sId].start[flow_ndx]; i<num_frames; ++i){ //CP_SINGLEFLOWFIT
#ifdef FVAL_L1
            fval[i] = tmp_fval[i];
#else
            fval[num_beads*i] = tmp_fval[num_beads*i];
#endif
          }
        residual = newresidual;
        cont_proc = true;
      }
      else {
        lambda *= 10.0f;
      }

      if (lambda > 1.0f) {
        cont_proc = true;
        if (useDynamicEmphasis) {
          frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,Ampl,emphasis, 
              num_frames, nonZeroEmpFrames);
        }
      }
     }

    } // end ITER loop
//    atomicAdd(&pMonitor[iter-1], 1);

    if(flow_ndx==0) avg_err = pState->avg_err * realFnum;  

    if(twoParamFit) *pKmult = krate;
    *pAmpl= Ampl;
 
    residual = CalculateMeanResidualErrorPerFlow(CP[sId].start[flow_ndx], fg_buffers, fval, emphasis+LAST_POISSON_TABLE_COL,
      num_beads, num_frames); 
  
    avg_err += residual;
    meanErr[num_beads * flow_ndx] = residual;

    pAmpl += num_beads;
    pKmult += num_beads;
    fg_buffers += num_frames*num_beads;
  } // end flow_ndx loop

  avg_err /= (realFnum + flow_block_size);
  pState->avg_err = avg_err;
  int high_err_cnt = 0;
  avg_err *= WASHOUT_THRESHOLD;
  for (int flow_ndx = flow_block_size - 1; flow_ndx >= 0 
                           && (meanErr[num_beads* flow_ndx] > avg_err); flow_ndx--)
    high_err_cnt++;

  if (high_err_cnt > WASHOUT_FLOW_DETECTION)
    pState->corrupt = true;

}

// Let number of beads be N and frames be F. The size for each input argument in
// comments is in bytes.
__global__ void 
PerFlowRelaxedKmultGaussNewtonFit_k(
  // inputs
  const float* fg_buffers, // NxF
  const float* emphasisVec, 
  const float* nucRise, 
  float * pBeadParamsBase, //N
  bead_state* pState,

  // scratch space in global memory
  float* err, // NxF
#ifndef FVAL_L1
  float* fval, // NxF
  float* tmp_fval, // NxF
#endif
  float* jac, // NxF 
  float* meanErr,
  // other inputs 
  const float minAmpl,
  float maxKmult,
  float minKmult,
  const float adjKmult,
  const bool fitKmult,
  const int realFnum,
  int num_beads, // 4
  const int num_frames, // 4
  const bool useDynamicEmphasis,
//  int * pMonitor,
  const int sId,
  const int flow_block_size
) 
{
  //useDynamicEmphasis = false;
#ifdef FVAL_L1
  float fval[MAX_COMPRESSED_FRAMES_GPU];
  float tmp_fval[MAX_COMPRESSED_FRAMES_GPU];
#endif

  // Preload the emphasis table. This is fairly quick.
  extern __shared__ float emphasis[];
  const int numWarps = blockDim.x/32;
  const int threadWarpIdx = threadIdx.x%32;
  const int warpIdx = threadIdx.x/32; 
  for(int i=warpIdx; i<num_frames; i += numWarps)
  {
     if (threadWarpIdx < MAX_POISSON_TABLE_COL)
      emphasis[(MAX_POISSON_TABLE_COL)*i + threadWarpIdx ] = 
        emphasisVec[num_frames*threadWarpIdx + i ];
  }
  __syncthreads();

  const int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+31)/32) * 32;
  pBeadParamsBase += bead_ndx;
  pState += bead_ndx; 
  

  const float * pCopies = &pBeadParamsBase[BEAD_OFFSET(Copies)*num_beads];
  float * pAmpl =   &pBeadParamsBase[BEAD_OFFSET(Ampl[0])*num_beads];
  float *pKmult = &pBeadParamsBase[BEAD_OFFSET(kmult[0])*num_beads];

   

#ifndef FVAL_L1
  fval += bead_ndx;
  tmp_fval += bead_ndx;
#endif

  jac += bead_ndx; // For Keplar
  err += bead_ndx;
  meanErr += bead_ndx;
  fg_buffers += bead_ndx;


  if (pState->corrupt || !pState->clonal_read || pState->pinned) return;

  float avg_err;
    
  float* deltaFrames = CP[sId].useRecompressTailRawTrace ? 
        CP[sId].deltaFrames_std : CP[sId].deltaFrames;
  int* nonZeroEmpFramesVec = CP[sId].useRecompressTailRawTrace ? 
        CP[sId].std_non_zero_emphasis_frames : CP[sId].non_zero_emphasis_frames;

  for(int flow_ndx=0; flow_ndx<flow_block_size; flow_ndx++) {

    const int nucid = CP[sId].flowIdxMap[flow_ndx]; //CP_SINGLEFLOWFIT
    const float sens = CP[sId].sens*SENSMULTIPLIER;  //CP_SINGLEFLOWFIT

    const float copies = *pCopies;
    const float R = *(pCopies + num_beads);
    float d = *(pCopies + 2*num_beads);
    const float gain = *(pCopies + 3 * num_beads) ;
    
    d *= CP[sId].d[nucid]; //CP_SINGLEFLOWFIT


    //offset for next value gets added to address at end of flow_ndx loop
    
    float krate = *pKmult;
    float localMinKmult = minKmult;
    float localMaxKmult= maxKmult;

    float Ampl = *pAmpl;
    float etbR;
    float tauB; 
    float SP;  
 
 
    ComputeEtbR_dev(etbR, &CP[sId], R, copies, pBeadParamsBase[BEAD_OFFSET(phi)*num_beads],
                          sId, nucid, realFnum+flow_ndx); //CP_SINGLEFLOWFIT
    ComputeTauB_dev(tauB, &CP[sId], etbR, sId); //CP_SINGLEFLOWFIT
    ComputeSP_dev(SP, &CP[sId], copies, realFnum+flow_ndx, sId); //CP_SINGLEFLOWFIT
 
    const bool twoParamFit = fitKmult || ( copies * Ampl > adjKmult );
  
    if (twoParamFit)
      krate = minKmult;
 
    float residual, newresidual;
    // These values before start are always zero since there is no nucrise yet. Don't need to
    // zero it out. Have to change the residual calculation accordingly for the frames before the
    // start.

    int relax_kmult_pass = 0;
    while (relax_kmult_pass < 2)
    {
      // first step
      // Evaluate model function using input Ampl and Krate and get starting residual
#if __CUDA_ARCH__ >= 350
      Keplar_ModelFuncEvaluationForSingleFlowFit(twoParamFit,sId, flow_ndx, nucid, nucRise, 
        Ampl, krate*CP[sId].krate[nucid], tauB, gain, SP, d,  //CP_SINGLEFLOWFIT
        sens, ISIG_SUB_STEPS_SINGLE_FLOW* CP[sId].start[flow_ndx], //CP_SINGLEFLOWFIT
        num_frames, num_beads, fval, deltaFrames, num_frames, NoOutput );
#else
      Fermi_ModelFuncEvaluationForSingleFlowFitNoOutput(&CP[sId], flow_ndx, nucid, nucRise, 
        Ampl, krate*CP[sId].krate[nucid], tauB, gain, SP, d,  //CP_SINGLEFLOWFIT
        sens, ISIG_SUB_STEPS_SINGLE_FLOW* CP[sId].start[flow_ndx], //CP_SINGLEFLOWFIT
        num_frames, num_beads, fval, deltaFrames, num_frames);
#endif
      const float *emLeft, *emRight;

      // calculating weighted sum of square residuals for the convergence test
      //const float EmphSel = (relax_kmult_pass == 1) ? (Ampl + 2.0f) : Ampl;
      const float EmphSel = Ampl;
      int nonZeroEmpFrames;
      float frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,EmphSel,emphasis, num_frames, nonZeroEmpFrames);
      residual = ResidualCalculationPerFlow(CP[sId].start[flow_ndx], fg_buffers, fval, emLeft, emRight, frac, err,
                      num_beads, nonZeroEmpFrames);

      // new Ampl and Krate generated from the Lev mar Fit
      float newAmpl, newKrate;

      float delta0 = 0, delta1 = 0;
      int iter;
      int done = 0;
      for (iter = 0; iter < ITER; ++iter) {

        if ((delta0*delta0) < 0.0000025f)
          done++;
        else 
          done = 0;
        
        if (done > 1)
          break;

        // new Ampl and krate by adding delta to existing values
        newAmpl = Ampl + 0.001f;
        newKrate = (twoParamFit)?(krate + 0.001f):(krate);

        // Evaluate model function for new Ampl keeping Krate constant
        float aa = 0, akr= 0, krkr = 0, rhs0 = 0, rhs1 = 0;

#if __CUDA_ARCH__ >= 350
        Keplar_ModelFuncEvaluationForSingleFlowFit(twoParamFit,sId, flow_ndx, nucid, nucRise,
            newAmpl, krate*CP[sId].krate[nucid], tauB, gain, SP, d,  //CP_SINGLEFLOWFIT
	    sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW,  //CP_SINGLEFLOWFIT
	    num_frames, num_beads, tmp_fval, deltaFrames, nonZeroEmpFrames, OneParam, jac, emLeft, emRight, frac, fval, 
	    err, &aa, &rhs0, &krkr, &rhs1, &akr);


	if (twoParamFit) 
	  Keplar_ModelFuncEvaluationForSingleFlowFit(twoParamFit,sId, flow_ndx, nucid, nucRise, 
	      Ampl, newKrate*CP[sId].krate[nucid], tauB, gain, SP, d,  //CP_SINGLEFLOWFIT
	      sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW,  //CP_SINGLEFLOWFIT
	      num_frames, num_beads, tmp_fval, deltaFrames, nonZeroEmpFrames, TwoParams, jac, emLeft, emRight, frac, fval, 
	      err, &aa, &rhs0, &krkr, &rhs1, &akr);
#else
        Fermi_ModelFuncEvaluationForSingleFlowFit(sId, flow_ndx, nucid, nucRise,
            newAmpl, Ampl, krate*CP[sId].krate[nucid], newKrate*CP[sId].krate[nucid],
            tauB, gain, SP, d,  //CP_SINGLEFLOWFIT
            sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW,  //CP_SINGLEFLOWFIT
            num_frames, num_beads, twoParamFit ? TwoParams : OneParam,
            emLeft, emRight, frac, fval, 
            err, &aa, &rhs0, &krkr, &rhs1, &akr, deltaFrames, nonZeroEmpFrames);
#endif
        // Now start the solving.        
        if(twoParamFit){ 
          const float det = 1.0f / (aa*krkr - akr*akr);
          delta1 = (-akr*rhs0 + aa*rhs1)*det;
          delta0 = (krkr*rhs0 - akr*rhs1)*det;
        }else
          delta0 = rhs0 / aa;

        if( !isnan(delta0) && !isnan(delta1)){
          // add delta to params to obtain new params
          newAmpl = Ampl + delta0;
          if(twoParamFit)newKrate = krate + delta1;

          clamp_streaming(newAmpl, minAmpl, (float)LAST_POISSON_TABLE_COL);
          if(twoParamFit)clamp_streaming(newKrate, localMinKmult, localMaxKmult);

            // Evaluate using new params
            if (useDynamicEmphasis) {
              int newNonZeroEmpFrames;
              frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,newAmpl,emphasis, num_frames,
                      newNonZeroEmpFrames);
              nonZeroEmpFrames = max(nonZeroEmpFrames, newNonZeroEmpFrames);
            }

#if __CUDA_ARCH__ >= 350
	    Keplar_ModelFuncEvaluationForSingleFlowFit(twoParamFit,sId, flow_ndx, nucid, nucRise, 
	        newAmpl, newKrate*CP[sId].krate[nucid], tauB, gain, SP,  //CP_SINGLEFLOWFIT
		d, sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW,  //CP_SINGLEFLOWFIT
		num_frames, num_beads, tmp_fval, deltaFrames, num_frames, NoOutput );
#else
            Fermi_ModelFuncEvaluationForSingleFlowFitNoOutput(&CP[sId], flow_ndx, nucid, nucRise, 
                newAmpl, newKrate*CP[sId].krate[nucid], tauB, gain, SP,  //CP_SINGLEFLOWFIT
                d, sens, CP[sId].start[flow_ndx]*ISIG_SUB_STEPS_SINGLE_FLOW,  //CP_SINGLEFLOWFIT
                num_frames, num_beads, tmp_fval, deltaFrames, num_frames);
#endif
            // residual calculation using new parameters
            newresidual = ResidualCalculationPerFlow(CP[sId].start[flow_ndx], fg_buffers, tmp_fval, emLeft, emRight, frac, err,
                  num_beads, nonZeroEmpFrames);

            if (newresidual < residual) {
              Ampl = newAmpl;
              if(twoParamFit)krate = newKrate;
                // copy new function val to fval
                for (int i=CP[sId].start[flow_ndx]; i<num_frames; ++i){ //CP_SINGLEFLOWFIT
#ifdef FVAL_L1
                  fval[i] = tmp_fval[i];
#else
                  fval[num_beads*i] = tmp_fval[num_beads*i];
#endif
                }
                residual = newresidual;
              }
              else {
                if (useDynamicEmphasis) {
                  frac = DecideOnEmphasisVectorsForInterpolation(nonZeroEmpFramesVec, &emLeft,&emRight,Ampl,emphasis, 
                      num_frames, nonZeroEmpFrames);
                }
              }
            }
            else {
              delta0 = 0;
              delta1 = 0;
            }

      } // end ITER loop

      // probably slower incorporation
      if (fabs(krate - localMinKmult) < 0.01f) {
        if (sqrtf(residual) > 20.0f) {
          localMaxKmult = localMinKmult;
          //krate = 0.3f;
          localMinKmult = 0.3f;
          relax_kmult_pass++;
          continue;
        }
      }
      relax_kmult_pass = 2;
    }
    if(flow_ndx==0) avg_err = pState->avg_err * realFnum;  

    if(twoParamFit) *pKmult = krate;
    *pAmpl= Ampl;

 
    residual = CalculateMeanResidualErrorPerFlow(CP[sId].start[flow_ndx], fg_buffers, fval, emphasis+LAST_POISSON_TABLE_COL,
      num_beads, num_frames); 
  
    avg_err += residual;
    meanErr[num_beads * flow_ndx] = residual;

    pAmpl += num_beads;
    pKmult += num_beads;
    fg_buffers += num_frames*num_beads;
  } // end flow_ndx loop

  avg_err /= (realFnum + flow_block_size);
  pState->avg_err = avg_err;
  int high_err_cnt = 0;
  avg_err *= WASHOUT_THRESHOLD;
  for (int flow_ndx = flow_block_size - 1; flow_ndx >= 0 
                           && (meanErr[num_beads* flow_ndx] > avg_err); flow_ndx--)
    high_err_cnt++;

  if (high_err_cnt > WASHOUT_FLOW_DETECTION)
    pState->corrupt = true;


}




///////// Pre-processing kernel (bkg correct and well params calculation)
__global__ void PreSingleFitProcessing_k(// Here FL stands for flows
  // inputs from data reorganization
  float* pCopies, // N
  float* pR, // N
  float* pPhi, // N
  float* pgain, // N
  float* pAmpl, // FLxN
  float* sbg, // FLxF 
  float* dark_matter, // FLxF
  float* pPCA_vals, // FxNUM_DM_PCA
  float* fgbuffers, // FLxFxN

  // other inputs 
  int flowNum, // starting flow number to calculate absolute flow num
  int num_beads, // 4
  int num_frames, // 4
  bool alternatingFit,
  int sId,
  int flow_block_size
)
{
  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;

  int NucId, i;
  float Rval, tau, SP;
  float gain = pgain[bead_ndx];
  float *pca_vals = pPCA_vals + bead_ndx;
  float *fval, *sbgPtr;
  float *et = dark_matter;  // set to dark matter base pointer for PCA

  for (int flow_ndx=0; flow_ndx < flow_block_size; ++flow_ndx) {
  
    sbgPtr = sbg + flow_ndx*num_frames; // may shift to constant memory
    NucId = CP[sId].flowIdxMap[flow_ndx];  //CP_SINGLEFLOWFIT

    ComputeEtbR_dev(Rval, &CP[sId], pR[bead_ndx], pCopies[bead_ndx], pPhi[bead_ndx],  
                          sId, NucId, flowNum + flow_ndx); //CP_SINGLEFLOWFIT
    ComputeTauB_dev(tau, &CP[sId], Rval, sId); //CP_SINGLEFLOWFIT
    ComputeSP_dev(SP, &CP[sId], pCopies[bead_ndx], flowNum + flow_ndx, sId); //CP_SINGLEFLOWFIT

    Rval -= 1.0f;
    float dv = 0.0f;
    float dv_rs = 0.0f;
    float dvn = 0.0f;
    float curSbgVal;
    float aval;

    // need to go in constant memory since same word access for each thread in the warp
    // if PCA vectors keep base pointer otherwise bend to nuv average
    if(! CP[sId].useDarkMatterPCA ) 
      et = &dark_matter[NucId*num_frames]; 

    fval = &fgbuffers[flow_ndx*num_beads*num_frames];
 

    for (i=0; i<num_frames; i++)
    {
      aval = CP[sId].deltaFrames[i]/(2.0f * tau); //CP_SINGLEFLOWFIT

      // calculate new dv
      curSbgVal = sbgPtr[i];
      dvn = (Rval*curSbgVal - dv_rs/tau - dv*aval) / (1.0f + aval);
      dv_rs += (dv+dvn) * CP[sId].deltaFrames[i] * 0.5f; //CP_SINGLEFLOWFIT
      dv = dvn;
      float ftmp = fval[i*num_beads + bead_ndx]
                    -  ((dv+curSbgVal)*gain + ApplyDarkMatterToFrame(et, pca_vals, i, num_frames, num_beads, sId));  
      fval[i*num_beads + bead_ndx] = ftmp;
    }
  }
}

__global__ void RecompressRawTracesForSingleFlowFit_k(
  float* fgbuffers, // FLxFxN
  float* scratch,
  int startFrame,
  int oldFrames,
  int newFrames,
  int numFlows,
  int num_beads,
  int sId)
{
  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;

  float* newfgbuffers = fgbuffers;
  newfgbuffers += bead_ndx;
  fgbuffers += bead_ndx;
  scratch += bead_ndx;

  int uncompFrame = 0;
  for (int i=0; i<startFrame; ++i) {
    uncompFrame += CP[sId].std_frames_per_point[i];
  }

  float accumVal = 0;
  float prevVal = 0;
  float curVal = 0;
  int uncomp_start_frame = uncompFrame;
  int interpolFrameNum = 0;
  for (int fnum=0; fnum < numFlows; ++fnum) {
    uncompFrame = uncomp_start_frame;
    for (int i=0; i<startFrame; ++i) {
      scratch[num_beads*i] = fgbuffers[num_beads*i];
    }
    for (int i=startFrame; i<newFrames; ++i) {
      accumVal = 0;
      for (int addedFrame=1; addedFrame<=CP[sId].std_frames_per_point[i]; ++addedFrame) {
        interpolFrameNum = CP[sId].etf_interpolate_frame[uncompFrame];
        prevVal = fgbuffers[num_beads*(interpolFrameNum - 1)];
        curVal = fgbuffers[num_beads*interpolFrameNum];
        accumVal += ((prevVal - curVal) * CP[sId].etf_interpolateMul[uncompFrame]) + curVal;     
        uncompFrame++;
      }
      scratch[num_beads*i] = accumVal/CP[sId].std_frames_per_point[i];
    }
    for (int i=0; i<newFrames; ++i) {
      newfgbuffers[num_beads*i] = scratch[num_beads*i];
    } 
    fgbuffers += num_beads*oldFrames;
    newfgbuffers += num_beads*newFrames;
  }
}


// xtalk calculation from excess hydrogen by neighbours
__global__ void NeighbourContributionToXtalk_k(// Here FL stands for flows
  // inputs from data reorganization
  float* pR, // N
  float* pCopies, // N
  float* pPhi, // N
  float* sbg, // FLxF 
  float* fgbuffers, // FLxFxN
  bead_state *pState,

  // other inputs 
  int startingFlowNum, // starting flow number to calculate absolute flow num
  int currentFlowIteration,
  int num_beads, // 4
  int num_frames, // 4

  // temporaries
  float* scratch_buf, // 3xFxN
  float* nei_xtalk, // neixNxF

  int sId
)
{
  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  if (pState[bead_ndx].pinned || pState[bead_ndx].corrupt) return;

  num_beads = ((num_beads+32-1)/32) * 32;

  int NucId;
  float Rval, tau;
  float* incorp_rise = scratch_buf;
  float* lost_hydrogen = incorp_rise + num_beads*num_frames;
  float* bulk_signal = lost_hydrogen + num_beads*num_frames;
  incorp_rise += bead_ndx;
  lost_hydrogen += bead_ndx;
  bulk_signal += bead_ndx;
  fgbuffers += bead_ndx;
  nei_xtalk += bead_ndx;
  NucId = CP[sId].flowIdxMap[currentFlowIteration];  //CP_SINGLEFLOWFIT

  ComputeEtbR_dev(Rval, &CP[sId], pR[bead_ndx], pCopies[bead_ndx], pPhi[bead_ndx], 
  	sId, NucId, startingFlowNum + currentFlowIteration); //CP_SINGLEFLOWFIT
  ComputeTauB_dev(tau, &CP[sId], Rval, sId); //CP_SINGLEFLOWFIT                      

  // Calculate approximate incorporation signal
  int f = 0;
  float one_over_two_taub = 1.0f / (2.0f*tau);
  float xt = CP[sId].deltaFrames[f]*one_over_two_taub; //CP_SINGLEFLOWFIT
  incorp_rise[f] = (1.0f+xt)*fgbuffers[f] - (Rval+xt)*sbg[f];
  f++;
  for (;f<num_frames; ++f) {
    xt = CP[sId].deltaFrames[f]*one_over_two_taub; //CP_SINGLEFLOWFIT
    incorp_rise[f*num_beads] = (1.0+xt)*fgbuffers[f*num_beads] - (1.0f-xt)*fgbuffers[(f-1)*num_beads]
          - ((Rval+xt)*sbg[f]-(Rval-xt)*sbg[f-1]) + incorp_rise[(f-1)*num_beads];        
  }

  // calculate contribution to xtalk from this bead as a neighbour in the grid
  
  if (!CP_XTALKPARAMS[sId].simpleXtalk) {
    float old_tautop = 0, old_taufluid = 0;
    for (int i=0; i<CP_XTALKPARAMS[sId].neis; ++i) {
      bool changed = false;
      // Calculate lost hydrogen using tau_top
      if (old_tautop != CP_XTALKPARAMS[sId].tau_top[i]) {
        f = CP[sId].start[currentFlowIteration]; //CP_SINGLEFLOWFIT
	one_over_two_taub = 1.0f / (2.0f*CP_XTALKPARAMS[sId].tau_top[i]);
	xt = 1.0f/(1.0f + (CP[sId].deltaFrames[f]*one_over_two_taub)); //CP_SINGLEFLOWFIT
	lost_hydrogen[f*num_beads] = incorp_rise[f*num_beads]*xt;
	f++;
	for (;f<num_frames; ++f) {
	  xt = 1.0f/(1.0f + (CP[sId].deltaFrames[f]*one_over_two_taub)); //CP_SINGLEFLOWFIT
		lost_hydrogen[f*num_beads] = (incorp_rise[f*num_beads] - incorp_rise[(f-1)*num_beads] + 
				(1.0f-(CP[sId].deltaFrames[f]*one_over_two_taub))*lost_hydrogen[(f-1)*num_beads])*xt; //CP_SINGLEFLOWFIT
	}

	for (f = CP[sId].start[currentFlowIteration];f<num_frames; ++f) { //CP_SINGLEFLOWFIT
	  lost_hydrogen[f*num_beads] = incorp_rise[f*num_beads] - lost_hydrogen[f*num_beads];
	}
	changed = true;
      }

      // Calculate ions from bulk
      if (changed || ( !changed && (old_taufluid != CP_XTALKPARAMS[sId].tau_fluid[i]))) {
        f = CP[sId].start[currentFlowIteration]; //CP_SINGLEFLOWFIT
	one_over_two_taub = 1.0f / (2.0f*CP_XTALKPARAMS[sId].tau_fluid[i]);
	xt = 1.0f/(1.0f + (CP[sId].deltaFrames[f]*one_over_two_taub)); //CP_SINGLEFLOWFIT
	bulk_signal[f*num_beads] = lost_hydrogen[f*num_beads]*xt;
	f++;
	for (;f<num_frames; ++f) {
	  xt = 1.0f/(1.0f + (CP[sId].deltaFrames[f]*one_over_two_taub)); //CP_SINGLEFLOWFIT
		bulk_signal[f*num_beads] = (lost_hydrogen[f*num_beads] - lost_hydrogen[(f-1)*num_beads] + 
				(1.0f-(CP[sId].deltaFrames[f]*one_over_two_taub))*bulk_signal[(f-1)*num_beads])*xt; //CP_SINGLEFLOWFIT
	}
      }

      // Scale down the ion by neighbour multiplier
      for (f=0; f<CP[sId].start[currentFlowIteration]; ++f) { //CP_SINGLEFLOWFIT
        *nei_xtalk = 0; 
	nei_xtalk += num_beads;
      } 
      for (; f<num_frames; ++f) {
        *nei_xtalk = bulk_signal[f*num_beads] * CP_XTALKPARAMS[sId].multiplier[i]; 
	nei_xtalk += num_beads;
      }
      old_tautop = CP_XTALKPARAMS[sId].tau_top[i];
      old_taufluid = CP_XTALKPARAMS[sId].tau_fluid[i];
    }
  }
  else {
    // Calculate lost hydrogen
    f = CP[sId].start[currentFlowIteration]; //CP_SINGLEFLOWFIT
    xt = 1.0f/(1.0f + (CP[sId].deltaFrames[f]*one_over_two_taub)); //CP_SINGLEFLOWFIT
    lost_hydrogen[f*num_beads] = incorp_rise[f*num_beads]*xt;
    f++;
    for (;f<num_frames; ++f) {
      xt = 1.0f/(1.0f + (CP[sId].deltaFrames[f]*one_over_two_taub)); //CP_SINGLEFLOWFIT
      lost_hydrogen[f*num_beads] = (incorp_rise[f*num_beads] - incorp_rise[(f-1)*num_beads] + 
      		(1.0f-(CP[sId].deltaFrames[f]*one_over_two_taub))*lost_hydrogen[(f-1)*num_beads])*xt; //CP_SINGLEFLOWFIT
    }

    for (f = CP[sId].start[currentFlowIteration];f<num_frames; ++f) { //CP_SINGLEFLOWFIT
      lost_hydrogen[f*num_beads] = incorp_rise[f*num_beads] - lost_hydrogen[f*num_beads];
    }

    // Calculate ions from bulk
    float taue = Rval * tau;
    f = CP[sId].start[currentFlowIteration]; //CP_SINGLEFLOWFIT
    one_over_two_taub = 1.0f / (2.0f*taue);
    xt = 1.0f/(1.0f + (CP[sId].deltaFrames[f]*one_over_two_taub)); //CP_SINGLEFLOWFIT
    bulk_signal[f*num_beads] = lost_hydrogen[f*num_beads]*xt;
    f++;
    for (;f<num_frames; ++f) {
      xt = 1.0f/(1.0f + (CP[sId].deltaFrames[f]*one_over_two_taub)); //CP_SINGLEFLOWFIT
      bulk_signal[f*num_beads] = (lost_hydrogen[f*num_beads] - lost_hydrogen[(f-1)*num_beads] + 
           (1.0f-(CP[sId].deltaFrames[f]*one_over_two_taub))*bulk_signal[(f-1)*num_beads])*xt; //CP_SINGLEFLOWFIT
    }
  
    // Scale down the ion by neighbour multiplier
    for (int i=0; i<CP_XTALKPARAMS[sId].neis; ++i) {
      for (f=0; f<num_frames; ++f) {
        if (f < CP[sId].start[currentFlowIteration])
          *nei_xtalk = 0;
        else 
          *nei_xtalk = bulk_signal[f*num_beads] * CP_XTALKPARAMS[sId].multiplier[i]; 
        nei_xtalk += num_beads;
      }
    }
  }

}

__global__ void XtalkAccumulation_k(
  bead_state *pState,
  int num_beads,
  int num_frames,
  int* neiIdxMap, // MAX_XTALK_NEIGHBOURS x N 
  float* nei_xtalk, // neixNxF
  float* xtalk, // NxF
  int sId
)
{
  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;
  

  int orig_beads = num_beads;
  num_beads = ((num_beads+32-1)/32) * 32;

  int beadFrameProduct = num_beads*num_frames;
  xtalk += bead_ndx;
  neiIdxMap += bead_ndx;

  // Accumulate crosstalk from neighbours
  int i,f;
  for (f=0; f<num_frames; ++f) {
    xtalk[f*num_beads] = 0;
  }
  
  for (i=0; i<CP_XTALKPARAMS[sId].neis; ++i) {
    int neiIdx = neiIdxMap[i*orig_beads];
    if (neiIdx != -1) {

      if (pState[neiIdx].pinned || pState[neiIdx].corrupt) continue;

      for (int f=0; f<num_frames; ++f) {
        xtalk[f*num_beads] += nei_xtalk[i*beadFrameProduct + f*num_beads + neiIdx];          
      }
    }
  }
}

__global__ 
void CalculateGenericXtalkForSimpleModel_k(
  int num_beads,
  int num_frames,
  //int regW,
  //int regH,
  bead_state *pState,
  int *sampNeiIdxMap,
  float* nei_xtalk,
  float* xtalk, // FxN
  float* genericXtalk,
  int sId)
{
  __shared__ float smBuffer[MAX_UNCOMPRESSED_FRAMES_GPU];

  int sampNum = blockIdx.x * blockDim.x + threadIdx.x;

  if (sampNum >= (GENERIC_SIMPLE_XTALK_SAMPLE)) return;

  num_beads = ((num_beads+32-1)/32) * 32;


  if (CP_XTALKPARAMS[sId].simpleXtalk) {
    //Accumulate xtalk signal for the sample
    int i,f;
    for (f=0; f<num_frames; ++f) {
      xtalk[f*GENERIC_SIMPLE_XTALK_SAMPLE + sampNum] = 0;
    }

    sampNeiIdxMap += sampNum;
    int beadFrameProduct = num_beads * num_frames; 
    for (i=0; i<CP_XTALKPARAMS[sId].neis; ++i) {
      int neiIdx = sampNeiIdxMap[i*GENERIC_SIMPLE_XTALK_SAMPLE];
      if (neiIdx != -1) {

        if (pState[neiIdx].pinned || pState[neiIdx].corrupt) continue;

        for (int f=0; f<num_frames; ++f) {
          xtalk[f*GENERIC_SIMPLE_XTALK_SAMPLE + sampNum] += nei_xtalk[i*beadFrameProduct + f*num_beads + neiIdx];             }
      }
    }
    __syncthreads();
  }

  if (sampNum > 0) return;

  // calculate xtalk for GENERIC_SIMPLE_XTALK_SAMPLE beads
  for (int i=0; i<(MAX_UNCOMPRESSED_FRAMES_GPU); ++i) {
    smBuffer[i] = 0;
  }

  if (CP_XTALKPARAMS[sId].simpleXtalk) {
    for (int f=0; f<num_frames; ++f) {
      for (int i=0; i<(GENERIC_SIMPLE_XTALK_SAMPLE); ++i) {
        smBuffer[f] += xtalk[i];
      }
      xtalk += GENERIC_SIMPLE_XTALK_SAMPLE;
    }
  }

  float scaling = 1.0f / (GENERIC_SIMPLE_XTALK_SAMPLE);
  for (int f=0; f<num_frames; ++f) {
    genericXtalk[f] = smBuffer[f] * scaling;  
  }
 
} 

__global__ void ComputeXtalkAndZeromerCorrectedTrace_k(// Here FL stands for flows
  int currentFlowIteration,
  float* fgbuffers, // FLxFxN
  int num_beads, // 4
  int num_frames, // 4
  float* genericXtalk, // neixNxF
  float* xtalk, // FLxN
  float* pCopies, // N
  float* pR, // N
  float* pPhi, // N 
  float* pgain, // N
  float* sbg, // FLxF 
  float* dark_matter, // FLxF
  float* pPCA_vals, // FxNUM_DM_PCA
  int flowNum, // starting flow number to calculate absolute flow num
  int sId
)
{
  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;

  xtalk += bead_ndx;
  fgbuffers += bead_ndx;
  pPCA_vals += bead_ndx;

  int i;
  
  float Rval, tau, SP;
  float gain = pgain[bead_ndx];
  int NucId = CP[sId].flowIdxMap[currentFlowIteration];  //CP_SINGLEFLOWFIT

  ComputeEtbR_dev(Rval, &CP[sId], pR[bead_ndx], pCopies[bead_ndx], pPhi[bead_ndx],
                        sId, NucId, flowNum + currentFlowIteration); //CP_SINGLEFLOWFIT
  ComputeTauB_dev(tau, &CP[sId], Rval, sId); //CP_SINGLEFLOWFIT
  ComputeSP_dev(SP, &CP[sId], pCopies[bead_ndx], flowNum + currentFlowIteration, sId); //CP_SINGLEFLOWFIT

  Rval -= 1.0f;
  float dv = 0.0f;
  float dv_rs = 0.0f;
  float dvn = 0.0f;
  float curSbgVal;
  float aval;

  // need to go in constant memory since same word access for each thread in the warp
  float* et;
  if(CP[sId].useDarkMatterPCA)
    et = dark_matter;
  else
    et = &dark_matter[NucId*num_frames]; 

  for (i=0; i<num_frames; i++)
  {
    aval = CP[sId].deltaFrames[i]/(2.0f * tau); //CP_SINGLEFLOWFIT

    // calculate new dv
    curSbgVal = sbg[i] + *xtalk - genericXtalk[i];
    dvn = (Rval*curSbgVal - dv_rs/tau - dv*aval) / (1.0f + aval);
    dv_rs += (dv+dvn) * CP[sId].deltaFrames[i] * 0.5f; //CP_SINGLEFLOWFIT
    dv = dvn;
    *fgbuffers = *fgbuffers - ((dv+curSbgVal)*gain + ApplyDarkMatterToFrame(et, pPCA_vals, i, num_frames, num_beads, sId));
    fgbuffers += num_beads;
    xtalk += num_beads;
  }
}


__global__
void ExponentialTailFitting_k(
  float bkg_scale_limit,
  float bkg_tail_dc_lower_bound,
  bead_state* pState,
  float* tauAdjust, // obtained from TaubAdjustForExponentialTailFitting()
  float* Ampl,
  float* pR,
  float* pCopies,       
  float* pPhi,
  float* fg_buffers,
  float* bkg_trace, // sbg
  float* tmp_fval,
  int num_beads,
  int num_frames,
  int flowNum,
  int sId,
  int flow_block_size
)
{
  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;

  pState += bead_ndx;
  if (pState->pinned || !pState->clonal_read || pState->corrupt) return;

  tauAdjust += bead_ndx;
  Ampl += bead_ndx;
  fg_buffers += bead_ndx;
  tmp_fval += bead_ndx;

  float kern[7];
  for (int flow_ndx=0; flow_ndx < flow_block_size; ++flow_ndx) {
    

    float Rval, taub, tmid;
    int NucId = CP[sId].flowIdxMap[flow_ndx];  //CP_SINGLEFLOWFIT
    ComputeMidNucTime_dev(tmid, &CP[sId], NucId, flow_ndx);  //CP_SINGLEFLOWFIT
    ComputeEtbR_dev(Rval, &CP[sId], pR[bead_ndx], pCopies[bead_ndx], pPhi[bead_ndx], 
                          sId, NucId, flowNum + flow_ndx); //CP_SINGLEFLOWFIT
    ComputeTauB_dev(taub, &CP[sId], Rval, sId); //CP_SINGLEFLOWFIT
    taub *= *tauAdjust; // adjust taub with multipler estimated using levmar
    if (taub > 0.0f) { 

      // set up start and end point for exponential tail
      float tail_start = tmid + 6.0f + 1.75f * (*Ampl);
      int tail_start_idx = -1, tail_end_idx = -1;
      for (int i=0; i<num_frames; ++i) {
        if ((tail_start_idx == -1) && CP[sId].frameNumber[i] >= tail_start) //CP_SINGLEFLOWFIT
          tail_start_idx = i;
        if ((tail_end_idx == -1) && CP[sId].frameNumber[i] >= (tail_start + 60.0f)) //CP_SINGLEFLOWFIT
          tail_end_idx = i;
      }

      if (tail_start_idx == -1)
        continue;

      if (tail_end_idx == -1)
        tail_end_idx = num_frames;

      int tailFrames = tail_end_idx - tail_start_idx;
      if (tailFrames >= 5) {

        // Generate smoothing kernel vector. Distance from the point is +/- 3 so need
        // 7 weights
        int exp_kern_start = tailFrames < 7 ? (tail_end_idx - 7) : tail_start_idx;
        float taubInv = 1.0f / taub;
        GenerateSmoothingKernelForExponentialTailFit_dev(7, taubInv, exp_kern_start, 
            kern, &CP[sId]); //CP_SINGLEFLOWFIT

        // perform kernel smoothing on exponential tail
        // linear regression to calculate A and C in Aexp(-(t-t0)/taub) + C
        // First calculate lhs and rhs matrix entries which are obtained by taking
        // derivative of the squared residual (y - (Aexp(-(t-t0)/taub) + C))^2 w.r.t
        // A and C to 0 which gives two linear equations in A and C
        float avg_bkg_amp_tail = 0;
        float lhs_01=0,lhs_11=0, rhs_0=0, rhs_1=0;
        for (int i=tail_start_idx; i<tail_end_idx; ++i) {
          float sum=0,scale=0;
          float tmp_fval;
          for (int j=i-3, k=0; j <= (i+3); ++j, ++k) {
            if (j >= 0 && j < num_frames) {
              sum += (kern[k] * fg_buffers[j*num_beads]);
              scale += kern[k];
            }
          }
          tmp_fval = sum / scale;
          avg_bkg_amp_tail += bkg_trace[i];

          float expval = __expf(-(CP[sId].frameNumber[i] - CP[sId].frameNumber[tail_start_idx])*taubInv); //CP_SINGLEFLOWFIT
          lhs_01 += expval;
          lhs_11 += expval*expval;
          rhs_0 += tmp_fval;
          rhs_1 += tmp_fval*expval;  
        }

        float A, C;
        float detInv = 1.0f / (tailFrames*lhs_11 - lhs_01*lhs_01);
        C = (lhs_11*rhs_0 - lhs_01*rhs_1) * detInv;
        A = (-lhs_01*rhs_0 + tailFrames*rhs_1) * detInv;

        // if negative  then no incorporation
        if (A < -20.0f) {
          C = rhs_0 / tailFrames;
        }

        avg_bkg_amp_tail /= tailFrames;

        if (avg_bkg_amp_tail > bkg_tail_dc_lower_bound) {
          C /= avg_bkg_amp_tail;
          clamp_streaming(C, -bkg_scale_limit, bkg_scale_limit); 
        }
        else
          C = 0;

        // correct fg_buffers in place
        for (int i=0; i<num_frames; ++i) {
          fg_buffers[i*num_beads] -= C*bkg_trace[i];
        }
      }

    }
    Ampl += num_beads;
    fg_buffers += num_beads*num_frames;
    bkg_trace += num_frames;
  }
}

// only performed in first 20 flows. It wll be called after presingleflowfit
__global__ 
void TaubAdjustForExponentialTailFitting_k(
  bead_state* pState,
  float* fg_buffers,
  float* Ampl,
  float* pR,
  float* pCopies,
  float* pPhi,
  float* avg_trc,
  float* fval,
  float* tmp_fval,
  float* err,
  float* jac,
  int num_beads,
  int num_frames,
  float* tauAdjust, // output it is a per bead parameter
  int sId,
  int flow_block_size
)
{
  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;

  pState += bead_ndx;
  if (pState->pinned || !pState->clonal_read || pState->corrupt) return;

  tauAdjust += bead_ndx;
  Ampl += bead_ndx;
  fg_buffers += bead_ndx;
  avg_trc += bead_ndx;

  int count = 0;
  for (int i=0; i<num_frames; ++i)
    avg_trc[i*num_beads] = 0.0f;

  // collect incorporation traces from 1mer to 3mers in this flow block and average them
  // to get a typical incorporation trace
  for (int flow_ndx=0; flow_ndx<flow_block_size; ++flow_ndx) {
    float A = *Ampl;
    if((A > 0.5f) && (A < 3.0f)) {
      for (int i=0; i<num_frames; ++i) {
        avg_trc[i*num_beads] += *fg_buffers;
        fg_buffers += num_beads;
      } 
      count++;
    }  
    else {
      fg_buffers += num_frames*num_beads;
    }  
    Ampl += num_beads;
  }

  if (count > 6) {
    float Rval, taub, tmid;
    int NucId = CP[sId].flowIdxMap[0];  //CP_SINGLEFLOWFIT
    ComputeMidNucTime_dev(tmid, &CP[sId], NucId, 0);  //CP_SINGLEFLOWFIT
    ComputeEtbR_dev(Rval, &CP[sId], pR[bead_ndx], pCopies[bead_ndx],pPhi[bead_ndx],
                          sId, NucId, 0); //CP_SINGLEFLOWFIT
    ComputeTauB_dev(taub, &CP[sId], Rval, sId); //CP_SINGLEFLOWFIT
    float orig_taub = taub;

    float exp_tail_start = tmid + 6.0f + 2.0*1.5f;
    int tail_start = -1;
    
    // perform average as well as determine tail
    for (int j=0; j<num_frames; ++j) {
      avg_trc[j*num_beads] /= count;

      if ((tail_start == -1) && (CP[sId].frameNumber[j] >= exp_tail_start)) //CP_SINGLEFLOWFIT
        tail_start = j;     
    }

    // now perform lev mar fitting for Ampl, taub and dc_offset
    
    // set starting values for estimated parameters
    float dc_offset = 0.0f;
    float A = 20.0f;

    float newA, newtaub, newdc;
    int done = 0;
    float lambda = 1E-20;
    float min_taub = orig_taub*0.9f;
    float max_taub = orig_taub*1.1f;
    float delta0=0, delta1=0, delta2=0, residual, newresidual;
    
    fval += bead_ndx;
    tmp_fval += bead_ndx;
    err += bead_ndx;
    jac += bead_ndx;
   
    // calculate model function value with starting params before starting lev mar 
    ModelFunctionEvaluationForExponentialTailFit_dev(tail_start, num_frames, 
        num_beads, A, 1.0f/taub, dc_offset, fval, &CP[sId]); //CP_SINGLEFLOWFIT

    // calculate squared residual between average incorporation trace and model 
    // function 
    CalculateResidualForExponentialTailFit_dev(avg_trc, fval, tail_start, 
        num_frames, err, num_beads, residual);      

    for (int iter=0; iter<200; ++iter) {

      if (delta0*delta0 < 0.0000025f)
        done++;
      else 
        done = 0;

      if (done >=5)
        break;

      // calculate partial derivatives using pertubed parameters
      newA = A + 0.001f;
      newtaub = taub + 0.001f;
      newdc = dc_offset + 0.001f;

      // partial derivative w.r.t A
      ModelFunctionEvaluationForExponentialTailFit_dev(tail_start, num_frames, 
        num_beads, newA, 1.0f/taub, dc_offset, fval, &CP[sId], jac); //CP_SINGLEFLOWFIT

      // partial derivative w.r.t taub
      ModelFunctionEvaluationForExponentialTailFit_dev(tail_start, num_frames, 
        num_beads, A, 1.0f/newtaub, dc_offset, fval, &CP[sId],  //CP_SINGLEFLOWFIT
        jac+num_frames*num_beads);

      // partial derivative w.r.t dc_offset
      ModelFunctionEvaluationForExponentialTailFit_dev(tail_start, num_frames, 
        num_beads, A, 1.0f/taub, newdc, fval, &CP[sId],  //CP_SINGLEFLOWFIT
        jac+2*num_frames*num_beads);

      // jacobian matrix members
      float lhs_00=0, lhs_01=0, lhs_02=0, lhs_11=0, lhs_12=0, lhs_22=0;
      float rhs_0=0, rhs_1=0, rhs_2=0, det;
   
      // calculate jtj matrix entries
      for (int i=tail_start; i<num_frames; ++i) {
        lhs_00 += jac[i*num_beads]*jac[i*num_beads];
        lhs_01 += jac[i*num_beads]*jac[(num_frames + i)*num_beads];
        lhs_02 += jac[i*num_beads]*jac[(2*num_frames + i)*num_beads];
        lhs_22 += jac[(2*num_frames + i)*num_beads]*jac[(2*num_frames + i)*num_beads];
        lhs_12 += jac[(2*num_frames + i)*num_beads]*jac[(num_frames + i)*num_beads];
        lhs_11 += jac[(num_frames + i)*num_beads]*jac[(num_frames + i)*num_beads];
        rhs_0 += jac[i*num_beads]*err[i*num_beads];
        rhs_1 += jac[(num_frames + i)*num_beads]*err[i*num_beads];
        rhs_2 += jac[(2*num_frames + i)*num_beads]*err[i*num_beads];
      }

      // Solve
      bool cont_proc = false;
      while (!cont_proc) {
        float new_lhs00 = lhs_00 * (1.0f + lambda);
        float new_lhs11 = lhs_11 * (1.0f + lambda);
        float new_lhs22 = lhs_22 * (1.0f + lambda);

        // calculate determinant
        det = new_lhs00*(new_lhs11*new_lhs22 - lhs_12*lhs_12) - 
              lhs_01*(lhs_01*new_lhs22 - lhs_12*lhs_02) +
              lhs_02*(lhs_01*lhs_12 - new_lhs11*lhs_02);
        det = 1.0f/det;

        //if (bead_ndx == 0)
        //  printf("lhs00:%.2f lhs01: %.2f lhs02:%.2f lhs11:%.2f lhs12:%.2f lhs22:%.2f rhs0:%.2f rhs1:%.2f rhs2:%.2f, det:%.2f\n", lhs_00,lhs_01,lhs_02,lhs_11,lhs_12,lhs_22,rhs_0,rhs_1,rhs_2,det);

        delta0 = det*(rhs_0*(new_lhs11*new_lhs22 - lhs_12*lhs_12) +
                 rhs_1*(lhs_02*lhs_12 - lhs_01*new_lhs22) +
                 rhs_2*(lhs_01*lhs_12 - lhs_02*new_lhs11));
        delta1 = det*(rhs_0*(lhs_12*lhs_02 - lhs_01*new_lhs22) +
                 rhs_1*(new_lhs00*new_lhs22 - lhs_02*lhs_02) +
                 rhs_2*(lhs_01*lhs_02 - new_lhs00*lhs_12));
        delta2 = det*(rhs_0*(lhs_01*lhs_12 - lhs_02*new_lhs11) +
                 rhs_1*(lhs_01*lhs_02 - new_lhs00*lhs_12) +
                 rhs_2*(new_lhs00*new_lhs11 - lhs_01*lhs_01));

        // NAN check
        bool nan_detected = true;

        //if (bead_ndx == 0)
        //  printf("delta0: %.2f delta1: %.2f delta2: %.2f\n", delta0, delta1, delta2);

        if (!isnan(delta0) && !isnan(delta1) && !isnan(delta2)) {
          newA = A + delta0;
          newtaub = taub + delta1;
          newdc = dc_offset + delta2;

            
          clamp_streaming(newA, 0.0f, 500.0f);
          clamp_streaming(newtaub, min_taub, max_taub);
          clamp_streaming(newdc, -50.0f, 50.0f);
      
          //if (bead_ndx == 0)
          //  printf("A:%.2f tau:%.2f dc:%.2f\n", newA, newtaub, newdc);

          ModelFunctionEvaluationForExponentialTailFit_dev(tail_start, 
              num_frames, num_beads, newA, 1.0f/newtaub, newdc, tmp_fval,
              &CP[sId]); //CP_SINGLEFLOWFIT
          CalculateResidualForExponentialTailFit_dev(avg_trc, tmp_fval, 
              tail_start, num_frames, err, num_beads, newresidual);      

          nan_detected = false;
        }

        if (!nan_detected && newresidual < residual) {
          lambda /= 10.0f;
          if (lambda < FLT_MIN)
            lambda = FLT_MIN;
          
          A = newA;
          taub = newtaub;
          dc_offset = newdc;

          //if (bead_ndx == 0)
          //  printf("===> iter: %d Tau: %.2f residual: %.2f newresidual: %.2f\n", iter, taub, residual, newresidual);

          float* temp = fval;
          fval = tmp_fval;
          tmp_fval = temp;

          residual = newresidual;
          cont_proc = true;
        }
        else {
          lambda *= 10.0f;
        }

        if (lambda > 100.0f)
          cont_proc = true;
        
      }
    }      

    *tauAdjust = taub / orig_taub;
  }
}

/*****************************************************************************

              MULTI FLOW FIT KERNELS 

*****************************************************************************/

//////// Computing Partial Derivatives
__global__ void ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_k (
  // inputs
  int maxEmphasis,
  float restrict_clonal,
  float* pobservedTrace, 
  float* pival, // FLxNxF   //scratch
  float* pscratch_ival, // FLxNxF
  float* pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* psbg, // FLxF
  float* pemphasis, // MAX_POISSON_TABLE_COL xF 
  float* pnon_integer_penalty, // MAX_HPLEN
  float* pdarkMatterComp, // NUMNUC * F  
  float* pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep
  CpuStep* psteps, // we need a specific struct describing this config for this well fit for GPU
  unsigned int* pDotProdMasks,
  float* pJTJ,
  float* pRHS,
  int num_params,
  int num_steps,
  int num_beads,
  int num_frames,
  // outputs
  float* residual, // N 
  float* poutput, // total bead params x FL x N x F. Need to decide on its layout 
  int sId,
  int flow_block_size
) 
{
  extern __shared__ float emphasisVec[];

  for (int i=0; i<MAX_POISSON_TABLE_COL*num_frames; i+=num_frames)
  {
    if (threadIdx.x < num_frames)
      emphasisVec[i + threadIdx.x] = pemphasis[i + threadIdx.x];
  }
  __syncthreads();


  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;

  int i, j, flow_ndx;
  int stepOffset = num_beads*num_frames;
  float* ptemp, *pfval;

  float kmult, Ampl, tauB, etbR, SP;
  float gain = pbeadParamsTranspose[(BEAD_OFFSET(gain))*num_beads + bead_ndx];
  float dmult = pbeadParamsTranspose[(BEAD_OFFSET(dmult))*num_beads + bead_ndx];
  float R = pbeadParamsTranspose[(BEAD_OFFSET(R))*num_beads + bead_ndx];
  float Copies = pbeadParamsTranspose[(BEAD_OFFSET(Copies))*num_beads + bead_ndx];
  float Phi = pbeadParamsTranspose[(BEAD_OFFSET(phi))*num_beads + bead_ndx];
  float *pPCA_vals =&pbeadParamsTranspose[(BEAD_OFFSET(pca_vals))*num_beads + bead_ndx];

  pfval = poutput + bead_ndx;

  float tot_err = 0.0f;
  pobservedTrace += bead_ndx;
  pival += bead_ndx;
  pscratch_ival += bead_ndx;
 
  for (flow_ndx=0; flow_ndx<flow_block_size; ++flow_ndx) {
    // calculate emphasis vector index
    Ampl = pbeadParamsTranspose[(BEAD_OFFSET(Ampl[0]) + flow_ndx)*num_beads + bead_ndx];
    kmult = pbeadParamsTranspose[(BEAD_OFFSET(kmult[0]) + flow_ndx)*num_beads + bead_ndx];

    int emphasisIdx = (int)(Ampl) > maxEmphasis ? maxEmphasis : (int)Ampl;
    int nonZeroEmpFrames = CP[sId].non_zero_emphasis_frames[emphasisIdx];
    int nucid = CP[sId].flowIdxMap[flow_ndx]; //CP_MULTIFLOWFIT
    float * et;    
    // if PCA use basebointer to dark Matter otherwise bend pointer to current nuc average
    if(CP[sId].useDarkMatterPCA)
      et = pdarkMatterComp;
    else
      et = &pdarkMatterComp[nucid*num_frames];
 

    for (i=0; i<num_steps; ++i) {
      ptemp = poutput + i*stepOffset + bead_ndx;

      for (int k=nonZeroEmpFrames; k<num_frames; ++k) {
        ptemp[k*num_beads] = 0;
      }

      switch (psteps[i].PartialDerivMask) {
        case YERR:
        {
          float eval;
          for (j=0; j<nonZeroEmpFrames; ++j) {
            eval = (pobservedTrace[j*num_beads] - 
                        pfval[j*num_beads]) * 
                            emphasisVec[emphasisIdx*num_frames + j];
            *ptemp = eval;
            tot_err += eval*eval;
            ptemp += num_beads;
          }
        }
        break;
        case FVAL:
        {
          ComputeEtbR_dev(etbR, &CP[sId], R, Copies, Phi, sId, nucid, flow_ndx); //CP_MULTIFLOWFIT
          ComputeTauB_dev(tauB, &CP[sId] ,etbR, sId); //CP_MULTIFLOWFIT
          ComputeSP_dev(SP,  &CP[sId], Copies, flow_ndx, sId); //CP_MULTIFLOWFIT
          ComputeHydrogenForMultiFlowFit_dev(sId, flow_ndx, nucid, pnucRise, Ampl, 
                          kmult*CP[sId].krate[nucid], gain, SP,  //CP_MULTIFLOWFIT
                          dmult*CP[sId].d[nucid],  //CP_MULTIFLOWFIT
                          ISIG_SUB_STEPS_MULTI_FLOW*CP[sId].start[flow_ndx],  //CP_MULTIFLOWFIT
                          num_frames, num_beads, pival, num_frames);
          ComputeSignalForMultiFlowFit_dev(false, num_frames, restrict_clonal, sId, flow_ndx, Ampl, tauB,
                          etbR, gain, num_frames, num_beads, pnon_integer_penalty,
                          et,pPCA_vals, psbg, pival, pfval);
        }
        break;
        default:
        {
          // perturb the parameters 
          if (psteps[i].PartialDerivMask == DFDA) {
              Ampl += psteps[i].diff;
          } 
          else if (psteps[i].PartialDerivMask == DFDDKR) {
            kmult += psteps[i].diff;
          }
          else if (psteps[i].PartialDerivMask == DFDR) {
            R += psteps[i].diff;
          }
          else if (psteps[i].PartialDerivMask == DFDP) {
            Copies += psteps[i].diff;
          }
          else if (psteps[i].PartialDerivMask == DFDPDM) {
            dmult += psteps[i].diff;
          }

          float* pivtemp = pival;
          if (psteps[i].doBoth) {
            pivtemp = pscratch_ival;
            ComputeSP_dev(SP, &CP[sId], Copies, flow_ndx, sId); //CP_MULTIFLOWFIT
            ComputeHydrogenForMultiFlowFit_dev(sId, flow_ndx, nucid, pnucRise, Ampl, 
                kmult*CP[sId].krate[nucid], gain, SP,  //CP_MULTIFLOWFIT
                dmult*CP[sId].d[nucid],  //CP_MULTIFLOWFIT
                ISIG_SUB_STEPS_MULTI_FLOW*CP[sId].start[flow_ndx],  //CP_MULTIFLOWFIT
                num_frames, num_beads, pivtemp, nonZeroEmpFrames);
          }
          ComputeEtbR_dev(etbR, &CP[sId], R, Copies, Phi, sId, nucid, 0+flow_ndx); //CP_MULTIFLOWFIT
          ComputeTauB_dev(tauB, &CP[sId], etbR, sId); //CP_MULTIFLOWFIT
          ComputeSignalForMultiFlowFit_dev(true, nonZeroEmpFrames, restrict_clonal, sId, flow_ndx, Ampl, tauB,
                etbR, gain, num_frames, num_beads, pnon_integer_penalty,
                et,pPCA_vals, psbg, pivtemp, ptemp, true, 
                psteps[i].diff, emphasisVec + emphasisIdx*num_frames, pfval);

          // restore the params back
          if (psteps[i].PartialDerivMask == DFDA) {
            Ampl -= psteps[i].diff;
          } 
          else if (psteps[i].PartialDerivMask == DFDDKR) {
            kmult -= psteps[i].diff;
          }
          else if (psteps[i].PartialDerivMask == DFDR) {
            R -= psteps[i].diff;
          }
          else if (psteps[i].PartialDerivMask == DFDP) {
            Copies -= psteps[i].diff;
          }
          else if (psteps[i].PartialDerivMask == DFDPDM) {
            dmult -= psteps[i].diff;
          }
        }
      }
    } 
    pobservedTrace += stepOffset;

     // initialize jtj and rhs to 0
    ptemp = pJTJ + bead_ndx;
    for(int row=0;row<num_params;row++) {
      for(int col = 0; col <= row; col++) {
        unsigned int mask = pDotProdMasks[row*num_params+col];
        if ((mask >> flow_ndx) & 1) {
          unsigned int stepIdx1 = mask >> PARAM1_STEPIDX_SHIFT;
          unsigned int stepIdx2 = (mask >> PARAM2_STEPIDX_SHIFT) & 63;
          *ptemp += dotProduct(poutput + stepIdx1*stepOffset + bead_ndx,
                               poutput + stepIdx2*stepOffset + bead_ndx,
                               num_frames,
                               num_beads);
        }
        ptemp += num_beads;
      }
    }
    
    ptemp = pRHS + bead_ndx;
    for(int row=0;row<num_params;row++){
      unsigned int mask = pDotProdMasks[row*num_params+row];
      unsigned int stepIdx1 = mask >> PARAM1_STEPIDX_SHIFT;
      if ((mask >> flow_ndx) & 1) {
        *ptemp += dotProduct(poutput + stepIdx1*stepOffset + bead_ndx,
                             poutput + (num_steps - 1)*stepOffset + bead_ndx,
                             num_frames,
                             num_beads);
      }
      ptemp += num_beads;
    }
  }
  residual[bead_ndx] = sqrtf(tot_err / (flow_block_size*num_frames));
}

// Kernel for lev mar fitting on first 20 flows
__global__ void MultiFlowLevMarFit_k(
  // inputs
  int maxEmphasis,
  float restrict_clonal,
  float* pobservedTrace, 
  float* pival,
  float* pfval,
  float* pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* psbg, // FLxF
  float* pemphasis, // MAX_POISSON_TABLE_COL xF // needs precomputation
  float* pnon_integer_penalty, // MAX_HPLEN
  float* pdarkMatterComp, // NUMNUC * F  
  float* pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep
  float* pevalBeadParams,
  float* plambda,
  float* pjtj, // jtj matrix generated from build matrix kernel
  float* pltr, // scratch space to write lower triangular matrix
  float* pb, // rhs vector
  float* pdelta,
  unsigned int* paramIdxMap, 
  int num_params,
  int num_beads,
  int num_frames,
  // outputs
  float* presidual, // N 
  int sId,
  int flow_block_size
)
{
  extern __shared__ float emphasisVec[];

  for (int i=0; i<MAX_POISSON_TABLE_COL*num_frames; i+=num_frames)
  {
    if (threadIdx.x < num_frames)
      emphasisVec[i + threadIdx.x] = pemphasis[i + threadIdx.x];
  }
  __syncthreads();


  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;

  float lambda = plambda[bead_ndx];
  float oldResidual = presidual[bead_ndx];
  bool done = false;

  pival += bead_ndx;
  pfval += bead_ndx;
  pobservedTrace += bead_ndx;
  while(!done) {
    // solve for delta in params
    CholeskySolve_dev(lambda, pjtj, pltr, pb, pdelta, bead_ndx, num_params, num_beads);
    // calculate new beadparams
    CalculateNewBeadParams_dev(pbeadParamsTranspose, pevalBeadParams, pdelta, 
      paramIdxMap, bead_ndx, num_params, num_beads, sId, flow_block_size);

    // calculate residual and decide whether to perform further lamda tuning and run cholesky again
    float newResidual = 0; 
    float kmult, Ampl, tauB, etbR, SP;
    float gain = pevalBeadParams[(BEAD_OFFSET(gain))*num_beads + bead_ndx];
    float dmult = pevalBeadParams[(BEAD_OFFSET(dmult))*num_beads + bead_ndx];
    float R = pevalBeadParams[(BEAD_OFFSET(R))*num_beads + bead_ndx];
    float Copies = pevalBeadParams[(BEAD_OFFSET(Copies))*num_beads + bead_ndx];
    float Phi = pevalBeadParams[(BEAD_OFFSET(phi))*num_beads + bead_ndx];
    float *et = pdarkMatterComp;
    float *pPCA_vals = &pevalBeadParams[(BEAD_OFFSET(pca_vals))*num_beads + bead_ndx];

    for (int flow_ndx=0; flow_ndx<flow_block_size; ++flow_ndx)
    {
      // calculate emphasis vector index
      Ampl = pevalBeadParams[(BEAD_OFFSET(Ampl[0]) + flow_ndx)*num_beads + bead_ndx];
      kmult = pevalBeadParams[(BEAD_OFFSET(kmult[0]) + flow_ndx)*num_beads + bead_ndx];

      int emphasisIdx = (int)(Ampl) > maxEmphasis ? maxEmphasis : (int)Ampl;
      int nonZeroEmpFrames = CP[sId].non_zero_emphasis_frames[emphasisIdx];
      int nucid = CP[sId].flowIdxMap[flow_ndx]; //CP_MULTIFLOWFIT
 
      if(!CP[sId].useDarkMatterPCA)
        et = pdarkMatterComp+num_frames*nucid;

      ComputeEtbR_dev(etbR, &CP[sId], R, Copies, Phi, sId, nucid, 0+flow_ndx); //CP_MULTIFLOWFIT
      ComputeTauB_dev(tauB, &CP[sId], etbR, sId); //CP_MULTIFLOWFIT
      ComputeSP_dev(SP, &CP[sId], Copies, flow_ndx, sId); //CP_MULTIFLOWFIT
      ComputeHydrogenForMultiFlowFit_dev(sId, flow_ndx, nucid, pnucRise, Ampl, 
        kmult*CP[sId].krate[nucid], gain, SP,  //CP_MULTIFLOWFIT
        dmult*CP[sId].d[nucid],  //CP_MULTIFLOWFIT
        ISIG_SUB_STEPS_MULTI_FLOW*CP[sId].start[flow_ndx],  //CP_MULTIFLOWFIT
        num_frames, num_beads, pival, nonZeroEmpFrames);
      ComputeSignalForMultiFlowFit_dev(true, nonZeroEmpFrames, restrict_clonal, sId, flow_ndx, Ampl, tauB,
        etbR, gain, num_frames, num_beads, pnon_integer_penalty,
        et,pPCA_vals, psbg, pival, pfval);    
      CalculateMultiFlowFitResidual_dev(newResidual, pobservedTrace, pfval, 
        emphasisVec + num_frames*emphasisIdx, flow_ndx, num_beads, num_frames, nonZeroEmpFrames);
     } 
   
     newResidual = sqrtf(newResidual/(flow_block_size*num_frames));

     if (newResidual < oldResidual)
     {
       // TODO change wrt to ampl*copies
       UpdateBeadParams_dev(pbeadParamsTranspose, pevalBeadParams, paramIdxMap, bead_ndx, num_params, num_beads);
       lambda /= 30.0f; // it is LAMBDA_STEP in LevMarState.cpp
       if (lambda < FLT_MIN)
         lambda = FLT_MIN;
       plambda[bead_ndx] = lambda;
       presidual[bead_ndx] = newResidual;      
       done = true;
     }
     else
     {
       lambda *= 30.0f;
     }
     if (lambda >= 1E+10f)
     {
       done = true;
       plambda[bead_ndx] = lambda;
     }
  }
}


__global__ void BuildMatrix_k(  
  float* pPartialDeriv, // S*FLxNxF   //scatch
  unsigned int * pDotProdMasks, // pxp
  int num_steps,
  int num_params,
  int num_beads,
  int num_frames,
  // outputs
  float* pJTJ, // pxpxN
  float* pRHS, // pxN  
  int flow_block_size
  )
{

  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x; 
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;
  pJTJ += bead_ndx;
  for(int row=0;row<num_params;row++){
    for(int col = 0; col <= row; col++){

      *pJTJ =  CalculateJTJEntry( pDotProdMasks[row*num_params+col], 
                                  pPartialDeriv,  
                                  bead_ndx,
                                  num_beads,
                                  num_frames,
                                  flow_block_size );
      pJTJ += num_beads;
    }
  }
  pRHS += bead_ndx;
  for(int row=0;row<num_params;row++){
    *pRHS = CalculateRHSEntry( pDotProdMasks[row*num_params+row], 
                                     pPartialDeriv,  
                                     bead_ndx,
                                     num_steps,
                                     num_beads,
                                     num_frames,
                                     flow_block_size );
    pRHS += num_beads;
  }
}



__global__ void BuildMatrixVec2_k(  
  float* pPartialDeriv, // S*FLxNxF   //scatch
  unsigned int * pDotProdMasks, // pxp
  int num_steps,
  int num_params,
  int num_beads,
  int num_frames,
  // outputs
  float* pJTJ, // pxpxN
  float* pRHS, // pxN  
  int flow_block_size
  )
{
  int bead_ndx = blockIdx.x * (blockDim.x*2) + threadIdx.x*2; 
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;

  unsigned int * masks = pDotProdMasks; 
  pJTJ += bead_ndx;
  for(int row=0;row<num_params;row++){
    for(int col = 0; col <= row; col++){
      *((float2*)pJTJ) =  CalculateJTJEntryVec2(  masks[row*num_params+col], 
                                  pPartialDeriv,  
                                  bead_ndx,
                                  num_beads,
                                  num_frames,
                                  flow_block_size);
      pJTJ += num_beads;
    }
  }
  pRHS += bead_ndx;
  for(int row=0;row<num_params;row++){
    *((float2*)pRHS) = CalculateRHSEntryVec2( masks[row*num_params+row], 
                                     pPartialDeriv,  
                                     bead_ndx,
                                     num_steps,
                                     num_beads,
                                     num_frames,
                                     flow_block_size );
    pRHS += num_beads;
  }
}

__global__ void BuildMatrixVec4_k(  
  float* pPartialDeriv, // S*FLxNxF   //scatch
  unsigned int * pDotProdMasks, // pxp
  int num_steps,
  int num_params,
  int num_beads,
  int num_frames,
  // outputs
  float* pJTJ, // pxpxN
  float* pRHS, // pxN  
  int flow_block_size
  )
{
  int bead_ndx = blockIdx.x * (blockDim.x*4) + threadIdx.x*4; 

  
  extern  __shared__ unsigned int masks[];

  // load dotproduct masks to shared memory
  int i=threadIdx.x;
  while(i < num_params*num_params)
  {
    masks[i] = pDotProdMasks[i];
    i += blockDim.x;
  }
  __syncthreads(); 


  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;
  //num_beads += 32 - (num_beads%32);

//  unsigned int * masks = pDotProdMasks;


  pJTJ += bead_ndx;
  for(int row=0;row<num_params;row++){
    for(int col = 0; col <= row; col++){
      *((float4*)pJTJ) =  CalculateJTJEntryVec4( masks[row*num_params+col], 
                                  pPartialDeriv,  
                                  bead_ndx,
                                  num_beads,
                                  num_frames,
                                  flow_block_size);
      pJTJ += num_beads;
    }
  }
  pRHS += bead_ndx;
  for(int row=0;row<num_params;row++){
    *((float4*)pRHS) = CalculateRHSEntryVec4( masks[row*num_params+row], 
                                     pPartialDeriv,  
                                     bead_ndx,
                                     num_steps,
                                     num_beads,
                                     num_frames,
                                     flow_block_size );
    pRHS += num_beads;
  }
}


/****************************************************************************

            Amplitude estimation

****************************************************************************/

__global__ void ProjectionSearch_k(
  bead_state* pState,
  float* fg_buffers, // FLxFxN (already background corrected but no xtalk correction))
  float* emphasisVec, // FxLAST_POISSON_TABLE_COL
  float* nucRise, // ISIG_SUB_STEPS_MULTI_FLOW*F*FL 
  float* pBeadParamsBase,
  float* fval, // NxF
  int realFnum, // starting flow number in block of 20 flows
  int num_beads,
  int num_frames,
  int sId,
  int flow_block_size
)
{
  int bead_ndx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(bead_ndx >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;

#ifdef FVAL_L1
  float fval_L1[MAX_COMPRESSED_FRAMES_GPU];
  fval = &fval_L1[0];
#else
  fval += bead_ndx;
#endif
  
  pState += bead_ndx;
  if (pState->pinned || pState->corrupt) return;

  fg_buffers += bead_ndx;
  pBeadParamsBase += bead_ndx;

  float *pCopies = &pBeadParamsBase[BEAD_OFFSET(Copies)*num_beads];
  float *pAmpl = &pBeadParamsBase[BEAD_OFFSET(Ampl[0])*num_beads];
  float *pKmult = &pBeadParamsBase[BEAD_OFFSET(kmult[0])*num_beads];
  float R = *(pCopies + num_beads);
  float d = *(pCopies + 2*num_beads);
  float gain = *(pCopies + 3 * num_beads) ;

  float copies = *pCopies;
  float sens = CP[sId].sens*SENSMULTIPLIER;  //CP_SINGLEFLOWFIT
  for(int flow_ndx=0; flow_ndx<flow_block_size; flow_ndx++){

    int nucid = CP[sId].flowIdxMap[flow_ndx]; //CP_SINGLEFLOWFIT

    float dmult = d * CP[sId].d[nucid];
    float krate = *pKmult;
    float Ampl = 1.0f;

    float etbR;
    float tauB; 
    float SP;  
 
    ComputeEtbR_dev(etbR, &CP[sId], R, copies, 
                          pBeadParamsBase[BEAD_OFFSET(phi)*num_beads], sId, nucid, realFnum+flow_ndx); //CP_SINGLEFLOWFIT
    ComputeTauB_dev(tauB, &CP[sId], etbR, sId); //CP_SINGLEFLOWFIT
    ComputeSP_dev(SP, &CP[sId], copies, realFnum+flow_ndx, sId); //CP_SINGLEFLOWFIT
 
    for (int i=0; i<2; ++i) {
      Fermi_ModelFuncEvaluationForSingleFlowFitNoOutput(&CP[sId], flow_ndx, nucid, nucRise, 
        Ampl, krate*CP[sId].krate[nucid], tauB, gain, SP, dmult,  //CP_SINGLEFLOWFIT
        sens, ISIG_SUB_STEPS_SINGLE_FLOW * CP[sId].start[flow_ndx], //CP_SINGLEFLOWFIT
        num_frames, num_beads, fval, CP[sId].deltaFrames, CP[sId].non_zero_emphasis_frames[0]);

      float num = 0, den = 0.0001f;
      for (int j=CP[sId].start[flow_ndx]; j<CP[sId].non_zero_emphasis_frames[0]; ++j) {
#ifdef FVAL_L1
        num += fval[j]*fg_buffers[j*num_beads]*emphasisVec[j]*emphasisVec[j]; // multiply by emphasis vectors
        den += fval[j]*fval[j]*emphasisVec[j]*emphasisVec[j]; 
#else
        num += fval[j*num_beads]*fg_buffers[j*num_beads]*emphasisVec[j]*emphasisVec[j]; // multiply by emphasis vectors
        den += fval[j*num_beads]*fval[j*num_beads]*emphasisVec[j]*emphasisVec[j]; 
#endif
      }
      Ampl *= (num/den);
      if (isnan(Ampl))
        Ampl = 1.0f;
      else 
        clamp_streaming(Ampl, 0.001f, (float)LAST_POISSON_TABLE_COL);
    }
    *pAmpl = Ampl;
    pAmpl += num_beads;
    pKmult += num_beads;
    fg_buffers += num_beads*num_frames;
  }
}

/*****************************************************************************

              UTILITY KERNELS 

*****************************************************************************/


__global__ 
void build_poiss_LUT_k( void )  // build LUT poisson tables on device from CDF
{
  int offset = threadIdx.x; 
  int event = blockIdx.x;   //(maxEvent = MAX_HPLEN)
  int maxEvents = gridDim.x;

  float* ptrL = POISS_APPROX_TABLE_CUDA_BASE + MAX_POISSON_TABLE_ROW * ((event == 0)?(event):(event-1)) ;
  float* ptrR = POISS_APPROX_TABLE_CUDA_BASE + MAX_POISSON_TABLE_ROW * ((event < maxEvents-1)?(event):(event-1)) ;

  int offsetPlusOne = (offset < MAX_POISSON_TABLE_ROW-1)?(offset+1):(offset);
  
  float4 tmp; 
  tmp.x = ptrL[offset];
  tmp.y = ptrR[offset];
  tmp.z = ptrL[offsetPlusOne];
  tmp.w = ptrR[offsetPlusOne];

  float4* ptrLUT =  POISS_APPROX_LUT_CUDA_BASE + event * MAX_POISSON_TABLE_ROW + offset;

  *ptrLUT = tmp;

}



__global__ void transposeData_k(float *dest, float *source, int width, int height)
{
  __shared__ float tile[32][32+1];

  int xIndexIn = blockIdx.x * 32 + threadIdx.x;
  int yIndexIn = blockIdx.y * 32 + threadIdx.y;
  
    
  int Iindex = xIndexIn + (yIndexIn)*width;

  int xIndexOut = blockIdx.y * 32 + threadIdx.x;
  int yIndexOut = blockIdx.x * 32 + threadIdx.y;
  
  int Oindex = xIndexOut + (yIndexOut)*height;

  if(xIndexIn < width && yIndexIn < height) tile[threadIdx.y][threadIdx.x] = source[Iindex];

  
   __syncthreads();
  
  if(xIndexOut < height && yIndexOut < width) dest[Oindex] = tile[threadIdx.x][threadIdx.y];
}

///////// Transpose Kernel
__global__ void transposeDataToFloat_k(float *dest, FG_BUFFER_TYPE *source, int width, int height)
{
  __shared__ float tile[32][32+1];

  int xIndexIn = blockIdx.x * 32 + threadIdx.x;
  int yIndexIn = blockIdx.y * 32 + threadIdx.y;
  
    
  int Iindex = xIndexIn + (yIndexIn)*width;

  int xIndexOut = blockIdx.y * 32 + threadIdx.x;
  int yIndexOut = blockIdx.x * 32 + threadIdx.y;
  
  int Oindex = xIndexOut + (yIndexOut)*height;

  if(xIndexIn < width && yIndexIn < height) tile[threadIdx.y][threadIdx.x] = (float)(source[Iindex]);

   __syncthreads();
  
  if(xIndexOut < height && yIndexOut < width) dest[Oindex] = tile[threadIdx.x][threadIdx.y];
}





//////////////////////////////////////////////////////////////////
///////// EXTERN DECL. WRAPPER FUNCTIONS//////////////////////////

void StreamingKernels::copyFittingConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream)
{
  cudaMemcpyToSymbolAsync ( CP, ptr, sizeof(ConstParams), offset*sizeof(ConstParams),cudaMemcpyHostToDevice, stream);
}


void StreamingKernels::copyXtalkConstParamAsync(ConstXtalkParams* ptr, int offset, cudaStream_t stream)
{
  cudaMemcpyToSymbolAsync ( CP_XTALKPARAMS, ptr, sizeof(ConstXtalkParams), offset*sizeof(ConstXtalkParams),cudaMemcpyHostToDevice, stream);
}


void  StreamingKernels::PerFlowGaussNewtonFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  float* fg_buffers_base, // NxF
  float* emphasis, // F
  float* nucRise, 
  float * pBeadParamsBase, //N
  bead_state* pState,
  // scratch space in global memory
  float* err, // NxF
  float* fval, // NxF
  float* tmp_fval, // NxF
  float* meanErr,
  // other inputs 
  float minAmpl,
  float maxKmult,
  float minKmult,  
  float adjKmult,
  bool fitKmult,
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId,
  int flow_block_size
) 
{


  switch(l1type){
    case 1:
      cudaFuncSetCacheConfig(PerFlowGaussNewtonFit_k, cudaFuncCachePreferShared);
    break;
    case 2:
      cudaFuncSetCacheConfig(PerFlowGaussNewtonFit_k, cudaFuncCachePreferL1);
      break;
    default:
      cudaFuncSetCacheConfig(PerFlowGaussNewtonFit_k, cudaFuncCachePreferEqual);
  }

  PerFlowGaussNewtonFit_k<<< grid, block, smem, stream >>> (
    fg_buffers_base, // NxF
    emphasis,
    nucRise, 
    pBeadParamsBase, //N
    pState,
    err, // NxF
#ifndef FVAL_L1
    fval, // NxF
    tmp_fval, // NxF
#endif
    meanErr,
    minAmpl,
    maxKmult,
    minKmult,
    adjKmult,
    fitKmult,
    realFnum,
    num_beads, // 4
    num_frames, // 4
    useDynamicEmphasis,
//    pMonitor,
    sId,
    flow_block_size);
}



void  StreamingKernels::PerFlowHybridFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  float* fg_buffers_base, // NxF
  float* emphasis, // F
  float* nucRise, 
  // bead params
  float * pBeadParamsBase, //N
  bead_state* pState,

  // scratch space in global memory
  float* err, // NxF
  float* fval, // NxF
  float* tmp_fval, // NxF
  float* meanErr,
  // other inputs 
  float minAmpl,
  float maxKmult,
  float minKmult,  
  float adjKmult,
  bool fitKmult,
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId,
  int switchToLevMar,
  int flow_block_size
) 
{


  switch(l1type){
    case 1:
      cudaFuncSetCacheConfig(PerFlowHybridFit_k, cudaFuncCachePreferShared);
    break;
    case 2:
      cudaFuncSetCacheConfig(PerFlowHybridFit_k, cudaFuncCachePreferL1);
      break;
    default:
      cudaFuncSetCacheConfig(PerFlowHybridFit_k, cudaFuncCachePreferEqual);
  }

  PerFlowHybridFit_k<<< grid, block, smem, stream >>> (
    fg_buffers_base, // NxF
    emphasis,
    nucRise, 
    pBeadParamsBase, //N
    pState,
    err, // NxF
#ifndef FVAL_L1
    fval, // NxF
    tmp_fval, // NxF
#endif
    meanErr,
    minAmpl,
    maxKmult,
    minKmult,
    adjKmult,
    fitKmult,
    realFnum,
    num_beads, // 4
    num_frames, // 4
    useDynamicEmphasis,
//    pMonitor,
    sId,
    switchToLevMar,
    flow_block_size
  );
}


void  StreamingKernels::PerFlowLevMarFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  float* fg_buffers_base, // NxF
  float* emphasis, // F
  float* nucRise, 
  // bead params
  float * pBeadParamsBase, //N
  bead_state* pState,
  // scratch space in global memory
  float* err, // NxF
  float* fval, // NxF
  float* tmp_fval, // NxF
  float* meanErr,
  // other inputs 
  float minAmpl,
  float maxKmult,
  float minKmult,  
  float adjKmult,
  bool fitKmult,
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId,
  int flow_block_size
) 
{
  switch(l1type){
    case 1:
      cudaFuncSetCacheConfig(PerFlowLevMarFit_k, cudaFuncCachePreferShared);
    break;
    case 2:
      cudaFuncSetCacheConfig(PerFlowLevMarFit_k, cudaFuncCachePreferL1);
      break;
    default:
      cudaFuncSetCacheConfig(PerFlowLevMarFit_k, cudaFuncCachePreferEqual);
  }

  PerFlowLevMarFit_k<<< grid, block, smem, stream >>> (
    fg_buffers_base, // NxF
    emphasis,
    nucRise, 
    pBeadParamsBase, //N
    pState,
    err, // NxF
#ifndef FVAL_L1
    fval, // NxF
    tmp_fval, // NxF
#endif
    meanErr,
    minAmpl,
    maxKmult,
    minKmult,
    adjKmult,
    fitKmult,
    realFnum,
    num_beads, // 4
    num_frames, // 4
    useDynamicEmphasis,
//    pMonitor,
    sId,
    flow_block_size);
}


void  StreamingKernels::PerFlowRelaxKmultGaussNewtonFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  float* fg_buffers_base, // NxF
  float* emphasis, // F
  float* nucRise, 
  float * pBeadParamsBase, //N
  bead_state* pState,
  // scratch space in global memory
  float* err, // NxF
  float* fval, // NxF
  float* tmp_fval, // NxF
  float* jac, // NxF 
  float* meanErr,
  // other inputs 
  float minAmpl,
  float maxKmult,
  float minKmult,  
  float adjKmult,
  bool fitKmult,
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId,
  int flow_block_size
) 
{
  switch(l1type){
    case 1:
      cudaFuncSetCacheConfig(PerFlowRelaxedKmultGaussNewtonFit_k, cudaFuncCachePreferShared);
    break;
    case 2:
      cudaFuncSetCacheConfig(PerFlowRelaxedKmultGaussNewtonFit_k, cudaFuncCachePreferL1);
      break;
    default:
      cudaFuncSetCacheConfig(PerFlowRelaxedKmultGaussNewtonFit_k, cudaFuncCachePreferEqual);
  }

  PerFlowRelaxedKmultGaussNewtonFit_k<<< grid, block, smem, stream >>> (
    fg_buffers_base, // NxF
    emphasis,
    nucRise, 
    pBeadParamsBase, //N
    pState,
    err, // NxF
#ifndef FVAL_L1
    fval, // NxF
    tmp_fval, // NxF
#endif
    jac, // NxF 
    meanErr,
    minAmpl,
    maxKmult,
    minKmult,
    adjKmult,
    fitKmult,
    realFnum,
    num_beads, // 4
    num_frames, // 4
    useDynamicEmphasis,
    sId,
    flow_block_size);
}



///////// Pre-processing kernel (bkg correct and well params calculation);

void StreamingKernels::PreSingleFitProcessing(dim3 grid, dim3 block, int smem, cudaStream_t stream,// Here FL stands for flows
  // inputs from data reorganization
  float* pCopies, // N
  float* pR, // N
  float* pPhi, // N
  float* pgain, // N
  float* pAmpl, // FLxN
  float* sbg, // FLxF 
  float* dark_matter, // FLxF
  float* pPCA_vals,
  float* fgbuffers, // FLxFxN
  // other inputs 
  int flowNum, // starting flow number to calculate absolute flow num
  int num_beads, // 4
  int num_frames, // 4
  bool alternatingFit,
  int sId,
  int flow_block_size)
{
  PreSingleFitProcessing_k<<< grid, block, smem, stream >>>(
    pCopies, // N
    pR, // N
    pPhi, // N
    pgain, // N
    pAmpl, // FLxN
    sbg, // FLxF 
    dark_matter, // FLxF
    pPCA_vals,
    fgbuffers, // FLxFxN
    flowNum, // starting flow number to calculate absolute flow num
    num_beads, // 4
    num_frames, // 4
    alternatingFit,
    sId,
    flow_block_size
    );
}

//////// Computing Partial Derivatives

void StreamingKernels::ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow(
  int l1type,
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  // inputs
  int maxEmphasis,
  float restrict_clonal,
  float* pobservedTrace, 
  float* pival, // FLxNxF   //scatch
  float* pscratch_ival, // FLxNxF
  float* pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* psbg, // FLxF
  float* pemphasis, // MAX_POISSON_TABLE_COL xF // needs precomputation
  float* pnon_integer_penalty, // MAX_HPLEN
  float* pdarkMatterComp, // NUMNUC * F  
  float* pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep
  CpuStep* psteps, // we need a specific struct describing this config for this well fit for GPU
  unsigned int* pDotProdMasks,
  float* pJTJ,
  float* pRHS,
  int num_params,
  int num_steps,
  int num_beads,
  int num_frames,
  // outputs
  float* presidual,
  float* poutput, // total bead params x FL x N x F. Need to decide on its layout 
  int sId,
  int flow_block_size
)
{
  switch(l1type){
    case 1:
      cudaFuncSetCacheConfig(ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_k, cudaFuncCachePreferShared);
    break;
    case 2:
      cudaFuncSetCacheConfig(ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_k, cudaFuncCachePreferL1);
      break;
    default:
      cudaFuncSetCacheConfig(ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_k, cudaFuncCachePreferEqual);
  }


  ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_k<<<grid,block,smem,stream>>>(
    // inputs
    maxEmphasis,
    restrict_clonal,
    pobservedTrace, 
    pival, // FLxNxF   //scatch
    pscratch_ival, // FLxNxF
    pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
    psbg, // FLxF
    pemphasis, // MAX_POISSON_TABLE_COL xF // needs precomputation
    pnon_integer_penalty, // MAX_HPLEN
    pdarkMatterComp, // NUMNUC * F  
    pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep
    psteps, // we need a specific struct describing this config for this well fit for GPU
    pDotProdMasks,
    pJTJ,
    pRHS,
    num_params,
    num_steps,
    num_beads,
    num_frames,
    // outputs
    presidual,
    poutput, // total bead params x FL x N x F. Need to decide on its layout 
    sId,
    flow_block_size); 
} 




void StreamingKernels::BuildMatrix( dim3 grid, dim3 block, int smem, cudaStream_t stream, 
  float* pPartialDeriv, // S*FLxNxF   //scatch
  unsigned int * pDotProdMasks, // pxp
  int num_steps,
  int num_params,
  int num_beads,
  int num_frames,
  // outputs
  float* pJTJ, // pxpxN
  float* pRHS, // pxN  
  int vec,
  int flow_block_size
  )
{

  switch(vec){
    case 4:
      block.x = 256;
      grid.x = (num_beads + block.x*4-1)/(block.x*4);
      grid.y = 1;

      smem = num_params*num_params*sizeof(unsigned int);
      cudaFuncSetCacheConfig(BuildMatrixVec4_k, cudaFuncCachePreferL1);
      BuildMatrixVec4_k<<< grid,block,  smem, stream >>>(  
                  pPartialDeriv, // S*FLxNxF   //scatch
                  pDotProdMasks, // pxp
                  num_steps,
                  num_params,
                  num_beads,
                  num_frames,
                  pJTJ, // pxpxN
                  pRHS, // pxN  
                  flow_block_size
                  );
   
      break;
    case 2:
      grid.x = (num_beads + block.x*2-1)/(block.x*2);
      grid.y = 1;
      cudaFuncSetCacheConfig(BuildMatrixVec2_k, cudaFuncCachePreferL1);
      BuildMatrixVec2_k<<< grid,block,  smem, stream >>>(  
                  pPartialDeriv, // S*FLxNxF   //scatch
                  pDotProdMasks, // pxp
                  num_steps,
                  num_params,
                  num_beads,
                  num_frames,
                  pJTJ, // pxpxN
                  pRHS, // pxN  
                  flow_block_size
                  );

      break;
    default:
      cudaFuncSetCacheConfig(BuildMatrix_k, cudaFuncCachePreferL1);
      BuildMatrix_k<<< grid,block,  smem, stream >>>(  
                  pPartialDeriv, // S*FLxNxF   //scatch
                  pDotProdMasks, // pxp
                  num_steps,
                  num_params,
                  num_beads,
                  num_frames,
                  pJTJ, // pxpxN
                  pRHS, // pxN  
                  flow_block_size
                  );


    }
}


void StreamingKernels::MultiFlowLevMarFit(int l1type,  dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  int maxEmphasis,
  float restrict_clonal,
  float* pobservedTrace, 
  float* pival,
  float* pfval, // FLxNxFx2  //scratch for both ival and fval
  float* pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* psbg, // FLxF
  float* pemphasis, // MAX_POISSON_TABLE_COL xF // needs precomputation
  float* pnon_integer_penalty, // MAX_HPLEN
  float* pdarkMatterComp, // NUMNUC * F  
  float* pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep
  float* pevalBeadParams,
  float* plambda,
  float* pjtj, // jtj matrix generated from build matrix kernel
  float* pltr, // scratch space to write lower triangular matrix
  float* pb, // rhs vector
  float* pdelta,
  unsigned int* paramIdxMap, 
  int num_params,
  int num_beads,
  int num_frames,
  // outputs
  float* presidual, // N 
  int sId,
  int flow_block_size
  )
{

 switch(l1type){
    case 1:
      cudaFuncSetCacheConfig(MultiFlowLevMarFit_k, cudaFuncCachePreferShared);
      break;
    case 2:
      cudaFuncSetCacheConfig(MultiFlowLevMarFit_k, cudaFuncCachePreferL1);
      break;
    case 0:
    default:
      cudaFuncSetCacheConfig(MultiFlowLevMarFit_k, cudaFuncCachePreferEqual);
  }



  MultiFlowLevMarFit_k<<< grid ,block ,  smem, stream >>>(
    maxEmphasis,
    restrict_clonal,
    pobservedTrace,
    pival,
    pfval, // FLxNxFx2  //scratch for both ival and fval
    pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
    psbg, // FLxF
    pemphasis, // MAX_POISSON_TABLE_COL xF // needs precomputation
    pnon_integer_penalty, // MAX_HPLEN
    pdarkMatterComp, // NUMNUC * F  
    pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep
    pevalBeadParams,
    plambda,
    pjtj, // jtj matrix generated from build matrix kernel
    pltr, // scratch space to write lower triangular matrix
    pb, // rhs vector
    pdelta,
    paramIdxMap, 
    num_params,
    num_beads,
    num_frames,
    presidual, // N 
    sId,
    flow_block_size);
}

///////// Xtalk computation kernel wrapper

void StreamingKernels::NeighbourContributionToXtalk(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,// Here FL stands for flows
  // inputs from data reorganization
  float* pR, // N
  float* pCopies, // N
  float* pPhi, // N
  float* sbg, // FLxF 
  float* fgbuffers, // FLxFxN
  // other inputs 
  bead_state *pState,
  int startingFlowNum, // starting flow number to calculate absolute flow num
  int currentFlowIteration,
  int num_beads, // 4
  int num_frames, // 4
  float* scratch_buf,
  float* nei_xtalk,
  int sId 
)
{
  NeighbourContributionToXtalk_k<<< 
    grid, 
    block, 
    smem, 
    stream >>>(
    pR, // N
    pCopies,
    pPhi,
    sbg, // FLxF 
    fgbuffers, // FLxFxN
    pState,
    startingFlowNum, // starting flow number to calculate absolute flow num
    currentFlowIteration,
    num_beads, // 4
    num_frames, // 4
    scratch_buf,
    nei_xtalk,
    sId 
    );
}


void StreamingKernels::XtalkAccumulation(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  bead_state *pState,
  int num_beads, // 4
  int num_frames, // 4
  int* neiIdxMap, // MAX_XTALK_NEIGHBOURS x N
  float* nei_xtalk, // neixNxF
  float* xtalk, // NxF
  int sId
)
{
  XtalkAccumulation_k<<< 
    grid, 
    block, 
    smem, 
    stream >>>(
    pState,
    num_beads, // 4
    num_frames, // 4
    neiIdxMap,
    nei_xtalk,
    xtalk,
    sId);
}

void StreamingKernels::ComputeXtalkAndZeromerCorrectedTrace(// Here FL stands for flows
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,// Here FL stands for flows
  int currentFlowIteration,
  float* fgbuffers, // FLxFxN
  int num_beads, // 4
  int num_frames, // 4
  float* genericXtalk, // neixNxF
  float* xtalk, // FLxN
  float* pCopies, // N
  float* pR, // N
  float* pPhi,// N
  float* pgain, // N
  float* sbg, // FLxF 
  float* dark_matter, // FLxF
  float* pPCA_vals, 
  int flowNum, // starting flow number to calculate absolute flow num
  int sId
)
{
  ComputeXtalkAndZeromerCorrectedTrace_k<<< 
    grid, 
    block, 
    smem, 
    stream >>>(
    currentFlowIteration,
    fgbuffers, // FLxFxN
    num_beads, // 4
    num_frames, // 4
    genericXtalk,
    xtalk,
    pCopies, // N
    pR, // N
    pPhi, // N
    pgain, // N
    sbg, // FLxF 
    dark_matter, // FLxF
    pPCA_vals,
    flowNum, // starting flow number to calculate absolute flow num
    sId 
    );
}

void StreamingKernels::CalculateGenericXtalkForSimpleModel(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  int num_beads, // 4
  int num_frames, // 4
 // int regW,
 // int regH,
  bead_state *pState,
  int* sampNeiIdxMap,
  float* nei_xtalk,
  float* xtalk, // NxF
  float* genericXtalk, // GENERIC_SIMPLE_XTALK_SAMPLE x F
  int sId)
{
  CalculateGenericXtalkForSimpleModel_k<<<
    grid, 
    block, 
    smem, 
    stream >>>(
    num_beads,
    num_frames,
    //regW,
    //regH,
    pState,
    sampNeiIdxMap,
    nei_xtalk,
    xtalk, // FLxN
    genericXtalk, 
    sId);
}


void StreamingKernels::TaubAdjustForExponentialTailFitting(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  bead_state* pState,
  float* fg_buffers,
  float* Ampl,
  float* pR,
  float* pCopies,
  float* pPhi,
  float* avg_trc,
  float* fval,
  float* tmp_fval,
  float* err,
  float* jac,
  int num_beads,
  int num_frames,
  float* tauAdjust,
  int sId,
  int flow_block_size
)
{
  TaubAdjustForExponentialTailFitting_k <<<
        grid, 
        block, 
        smem, 
        stream >>>(
        pState,
        fg_buffers, // FLxFxN,
        Ampl, // FLxN
        pR, // N
        pCopies,
        pPhi,
        avg_trc,
        fval,
        tmp_fval,
        err,
        jac,
        num_beads,
        num_frames,
        tauAdjust, // output it is a per bead parameter
        sId,
        flow_block_size);
}


void StreamingKernels::ExponentialTailFitting(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  float bkg_scale_limit,
  float bkg_tail_dc_lower_bound,
  bead_state* pState,
  float* tauAdjust,
  float* Ampl,
  float* pR,
  float* pCopies,
  float* pPhi,
  float* fg_buffers,
  float* bkg_trace,
  float* tmp_fval,
  int num_beads,
  int num_frames,
  int flowNum,
  int sId,
  int flow_block_size
)
{
  ExponentialTailFitting_k <<<
      grid, 
      block, 
      smem, 
      stream >>> (
      bkg_scale_limit,
      bkg_tail_dc_lower_bound,
      pState,
      tauAdjust,
      Ampl,
      pR,
      pCopies,
      pPhi,
      fg_buffers,
      bkg_trace,
      tmp_fval,
      num_beads,
      num_frames,
      flowNum,
      sId,
      flow_block_size);
}


void StreamingKernels::ProjectionSearch(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  bead_state* pState,
  float* fg_buffers, // FLxFxN (already background and xtalk corrected if applicable))
  float* emphasisVec, // FxLAST_POISSON_TABLE_COL
  float* nucRise, // ISIG_SUB_STEPS_MULTI_FLOW*F*FL 
  float* pBeadParamsBase,
  float* fval, // NxF
  int realFnum, // starting flow number in block of 20 flows
  int num_beads,
  int num_frames,
  int sId,
  int flow_block_size
)
{
  ProjectionSearch_k<<<
      grid, 
      block, 
      smem, 
      stream>>>(
      pState,
      fg_buffers,
      emphasisVec,
      nucRise,
      pBeadParamsBase,
      fval,
      realFnum,
      num_beads,
      num_frames,
      sId,
      flow_block_size);
}

void StreamingKernels::RecompressRawTracesForSingleFlowFit(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  float* fgbuffers, // FLxFxN
  float* scratch,
  int startFrame,
  int oldFrames,
  int newFrames,
  int numFlows,
  int num_beads,
  int sId)
{
  RecompressRawTracesForSingleFlowFit_k<<<
      grid, 
      block, 
      smem, 
      stream>>>(
      fgbuffers,
      scratch,
      startFrame,
      oldFrames,
      newFrames,
      numFlows,
      num_beads,
      sId);
}



void StreamingKernels::transposeData(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, float *source, int width, int height)
{
  transposeData_k<<< grid, block, smem, stream >>>( dest, source, width, height);
}

///////// Transpose Kernel

void StreamingKernels::transposeDataToFloat(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, FG_BUFFER_TYPE *source, int width, int height)
{
  transposeDataToFloat_k<<< grid, block, smem, stream >>>( dest,source,width,height);
}




void StreamingKernels::initPoissonTables(int device, float ** poiss_cdf)
{

  cudaSetDevice(device);

  ///////// regular float version
  int poissTableSize = MAX_POISSON_TABLE_COL * MAX_POISSON_TABLE_ROW * sizeof(float);
  float * devPtr =NULL;
  cudaMalloc(&devPtr, poissTableSize); CUDA_ALLOC_CHECK(devPtr);
  cudaMemcpyToSymbol(POISS_APPROX_TABLE_CUDA_BASE , &devPtr  , sizeof (float*)); CUDA_ERROR_CHECK();
  for(int i = 0; i< (MAX_POISSON_TABLE_COL); i++)
  {
    cudaMemcpy(devPtr, poiss_cdf[i], sizeof(float)*MAX_POISSON_TABLE_ROW, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    devPtr += MAX_POISSON_TABLE_ROW;
  }

#ifndef USE_CUDA_ERF
    cudaMemcpyToSymbol (ERF_APPROX_TABLE_CUDA, ERF_APPROX_TABLE, sizeof (ERF_APPROX_TABLE)); CUDA_ERROR_CHECK();
#endif


}



void StreamingKernels::initPoissonTablesLUT(int device, void ** poissLUT)
{

  cudaSetDevice(device);
////////// float4/avx version
//  float4 ** pPoissLUT = (float4**) poissLUT;

  int poissTableSize =  MAX_LUT_TABLE_COL * MAX_POISSON_TABLE_ROW * sizeof(float4);
  float4 * devPtrLUT = NULL;  
  cudaMalloc(&devPtrLUT, poissTableSize); CUDA_ALLOC_CHECK(devPtrLUT);
  cudaMemset(devPtrLUT, 0, poissTableSize); CUDA_ERROR_CHECK();
  cudaMemcpyToSymbol(POISS_APPROX_LUT_CUDA_BASE, &devPtrLUT  , sizeof (float4*)); CUDA_ERROR_CHECK();

#ifdef CREATE_POISSON_LUT_ON_DEVICE
  // run kernel to create LUT table from CDF tables on device
  dim3 block(512,1);
  dim3 grid (MAX_POISSON_TABLE_COL, 1);
  build_poiss_LUT_k<<<grid, block >>>( ); 
  CUDA_ERROR_CHECK();
#else  
  // cast and copy host side __m128 SSE/AVX data to float4
  float4** pPoissLUT =(float4**)poissLUT;
  for(int i = 0; i< MAX_LUT_TABLE_COL; i++)
  {
    cudaMemcpy(devPtrLUT, &pPoissLUT[i][0], sizeof(float4)*MAX_POISSON_TABLE_ROW, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    devPtrLUT += MAX_POISSON_TABLE_ROW;
  }
#endif

}



void StreamingKernels::destroyPoissonTables(int device)
{
  cudaSetDevice(device);

  float * basepointer;

  cudaMemcpyFromSymbol (&basepointer,  POISS_APPROX_TABLE_CUDA_BASE , sizeof (float*)); CUDA_ERROR_CHECK();

  if(basepointer != NULL){
    cudaFree(basepointer); CUDA_ERROR_CHECK();
  }
}





