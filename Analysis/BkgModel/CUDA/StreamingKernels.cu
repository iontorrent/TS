/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "StreamingKernels.h"
#include "CudaConstDeclare.h" // for cuda < 5.0 this has to be included ONLY here!


// definte to use ne possion table layout
#define POISS_FLOAT4

// define to creat new layout on device instead of copying from host
//#define CREATE_POISSON_LUT_ON_DEVICE

// transpose emphasise vectors in shared memory to remove bank conflicts
#define TRANSPOSED_EMPH


//forward declerations of device functions

__device__
float erf_approx_streaming (float x);

__device__
float poiss_cdf_approx_streaming (float x, const float* ptr);

__device__
float poiss_cdf_approx_float4 (float x, const float4* ptr, float occ_l, float occ_r);


__device__
const float*  precompute_pois_params_streaming (int n);

__device__
const float4*  precompute_pois_LUT_params_streaming (int il, int ir);


__global__ 
void  build_poiss_LUT_k( void );


__device__
void clamp_streaming ( int &x, int a, int b);


__device__
void clamp_streaming ( double &x, double a, double b);


__device__
void clamp_streaming ( float &x, float a, float b);



__device__ void
ModelFuncEvaluationForSingleFlowFit(
//  int * pMonitor,
  bool twoParamFit,
  int sId,
  int fnum,
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
//  int ibd,
  int num_frames,
  int num_beads,
  float* fval,
  int flag = 0,
  float * jac_out = NULL,
  float * emLeft = NULL,
  float * emRight = NULL,
  float frac = 0,
  float * fval_in = NULL,
  float * err=NULL,
  float *aa = NULL,
  float *rhs0 = NULL, 
  float *krkr = NULL,
  float *rhs1 = NULL,
  float *akr = NULL
);

__device__ void
ComputeHydrogenForMultiFlowFit_dev(
  int sId,
  int fnum,
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
  float* ival);

__device__ void
ComputeSignalForMultiFlowFit_dev(
  float restrict_clonal,
  int sId,
  int fnum,
  float A, 
  float tauB,
  float etbR,
  float gain,
  int num_frames,
  int num_beads,
  float* non_integer_penalty,
  float* dark_matter,
  float* sbg,
  float* ival,
  float* output,
  bool useEmphasis = false,
  float diff = 0.0f,
  float* emphasis = NULL,
  float* fval = NULL);

//__device__
//void ComputeTauB_dev(float& tauB, float etbR, int sId); 
__device__
void ComputeTauB_dev(float& tauB, const ConstParams* pCP, float etbR, int sId); 

__device__ 
void ComputeEtbR_dev(float& etbR, float R, int sId, int nucid, int absFnum); 

__device__
void ComputeSP_dev(float& SP, float Copies, int fnum, int sId); 

__device__ void 
ResidualCalculationPerFlow(
  float* fg_buffers, 
  float* fval, 
  float* emLeft,
  float* emRight,
  float frac, 
  float* err, 
  float& residual,
//  int ibd,
  int num_beads,
  int num_frames); 

__device__ void 
CalculateMeanResidualErrorPerFlow(
  float* fg_buffers, 
  float* fval, 
  float* weight,
  float& residual,
//  int ibd,
  int num_beads,
  int num_frames); 


///// Implemented functions


__device__
float erf_approx_streaming (float x)
{

#ifdef USE_CUDA_ERF
  return erf (x);
#else

  int left, right;
  float sign = 1.0;
  float frac;
  float ret;

  if (x < 0.0)
  {
    x = -x;
    sign = -1.0;
  }

  left = (int) (x * 100.0f); // left-most point in the lookup table
  right = left + 1; // right-most point in the lookup table

  // both left and right points are inside the table...interpolate between them
  if ( (left >= 0) && (right < (int) (sizeof (ERF_APPROX_TABLE) / sizeof (float))))
  {
    frac = (x * 100.0f - left);
    ret = (1 - frac) * ERF_APPROX_TABLE_CUDA[left] + frac * ERF_APPROX_TABLE_CUDA[right];
  }
  else
  {
    if (left < 0)
      ret = ERF_APPROX_TABLE_CUDA[0];
    else
      ret = 1.0;
  }

  return (ret * sign);

#endif

}

__device__
float poiss_cdf_approx_float4 (float x, const float4* ptr, float occ_l, float occ_r)
{
  float ret;
  x *= 20.0f;
  int left = (int) x;

  float idelta = x-left;
  float ifrac = 1.0f-idelta;
  
  int max_dim_minus_one = MAX_POISSON_TABLE_ROW - 1;

  left = (left < max_dim_minus_one )?(left):(max_dim_minus_one);

  float4 mixLUT = ptr[left];

#ifndef CREATE_POISSON_LUT_ON_DEVICE
  //using _mm128 type casted as float4 --> reverse order of x,y,z,w ---> w,z,y,x
  ret = ( ifrac * ( occ_l * mixLUT.w + occ_r * mixLUT.z ) + idelta * (occ_l * mixLUT.y + occ_r * mixLUT.x )); 
#else
  ret = ( ifrac * ( occ_l * mixLUT.x + occ_r * mixLUT.y ) + idelta * (occ_l * mixLUT.z + occ_r * mixLUT.w )); 
/*
    int right = left+1;
    idelta = (float) (x - left);

    // both left and right points are inside the table...interpolate between them
    if ( (left >= 0) && (right < MAX_POISSON_TABLE_ROW))
    {
      ret = ((float)( (1 - idelta) * mixLUT.y + idelta * mixLUT.w )) * occ_r ;
    }
    else
    {
      if (left < 0)
        ret = ((float) mixLUT.y) * occ_r;
      else
        ret = ((float) mixLUT.w) * occ_r;
    }

    if ( (left >= 0) && (right < MAX_POISSON_TABLE_ROW))
    {
      ret += ((float)( (1 - idelta) * mixLUT.x + idelta * mixLUT.z )) * occ_l ;
    }
    else
    {
      if (left < 0)
        ret += ((float) mixLUT.x) * occ_l;
      else
        ret += ((float) mixLUT.z) * occ_l;
    }*/
#endif

  return ret;
}


__device__
float poiss_cdf_approx_streaming (float x, const float* ptr)
{
  int left, right;
  float frac;
  float ret = 0;

//  if (ptr != NULL)
//  {
    x *= 20.0f;
    left = (int) x; // left-most point in the lookup table
    right = left + 1; // right-most point in the lookup table

    // both left and right points are inside the table...interpolate between them
    if ( (left >= 0) && (right < MAX_POISSON_TABLE_ROW))
    {
      frac = (float) (x - left);
      ret = (float) ( (1 - frac) * ptr[left] + frac * ptr[right]);
    }
    else
    {
      if (left < 0)
        ret = (float) ptr[0];
      else
        ret = (float) (ptr[MAX_POISSON_TABLE_ROW - 1]);
    }
//  }
  return ret;
}


#ifndef USE_GLOBAL_POISSTABLE 
__device__
const float*  precompute_pois_params_streaming (int n)
{

  const float* ptr = NULL;
  switch (n)
  {
        case 0:
            ptr = POISS_0_APPROX_TABLE_CUDA;
            break;
        case 1:
            ptr = POISS_1_APPROX_TABLE_CUDA;
            break;
        case 2:
            ptr = POISS_2_APPROX_TABLE_CUDA;
            break;
        case 3:
            ptr = POISS_3_APPROX_TABLE_CUDA;
            break;
        case 4:
            ptr = POISS_4_APPROX_TABLE_CUDA;
            break;
        case 5:
            ptr = POISS_5_APPROX_TABLE_CUDA;
            break;
        case 6:
            ptr = POISS_6_APPROX_TABLE_CUDA;
            break;
        case 7:
            ptr = POISS_7_APPROX_TABLE_CUDA;
            break;
        case 8:
            ptr = POISS_8_APPROX_TABLE_CUDA;
            break;
        case 9:
            ptr = POISS_9_APPROX_TABLE_CUDA;
            break;
        case 10:
            ptr = POISS_10_APPROX_TABLE_CUDA;
            break;
        case 11:
            ptr = POISS_11_APPROX_TABLE_CUDA;
            break;
        default:
            ;
    }
    return ptr;

}
#else
__device__
const float*  precompute_pois_params_streaming (int n)
{

  const float* ptr = POISS_APPROX_TABLE_CUDA_BASE + n * MAX_POISSON_TABLE_ROW;
  return ptr;  
}

__device__
const float4*  precompute_pois_LUT_params_streaming (int il, int ir)
{
  int n;
  if( il == 0 && ir == 0 )
      n = 0; //special case for the packed case for 0 < A < 1
  else
      n = il+1; //layout: poiss_cdf[ei][i], poiss_cdf[ei+1][i], poiss_cdf[ei][i+1], poiss_cdf[ei+1][i+1]

  const float4* ptr =  POISS_APPROX_LUT_CUDA_BASE + n * MAX_POISSON_TABLE_ROW;

  return ptr;  
}
#endif

__device__
void clamp_streaming ( int &x, int a, int b)
{
  // Clamps x between a and b
  x = (x < a ? a : (x > b ? b : x));
}


__device__
void clamp_streaming ( double &x, double a, double b)
{
  // Clamps x between a and b
  x = (x < a ? a : (x > b ? b : x));
}


__device__
void clamp_streaming ( float &x, float a, float b)
{
  // Clamps x between a and b
  x = (x < a ? a : (x > b ? b : x));
}



__device__ void
ModelFuncEvaluationForSingleFlowFit(
//  int * pMonitor,
  bool twoParamFit,
  int sId,
  int fnum,
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
//  int ibd,
  int num_frames,
  int num_beads,
  float* fval,
  int flag,
  float * jac_out,
  float * emLeft,
  float * emRight,
  float frac,
  float * fval_in,
  float * err,
  float *aa,
  float *rhs0, 
  float *krkr,
  float *rhs1,
  float *akr
)
{

  if ( A!=A )
    A=0.0001f; // safety check

  if (A < 0.0f) {
    A = -A;
    sens = -sens;
  }

  else if (A > MAX_HPLEN)
    A = MAX_HPLEN;

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

  if (iright == MAX_HPLEN)
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

  float c_dntp_bot = 0.0; // concentration of dNTP in the well
  float c_dntp_sum = 0.0;
  float c_dntp_old_rate = 0;
  float c_dntp_new_rate = 0;

  float c_dntp_bot_plus_kmax = 1.0f/CP_SINGLEFLOWFIT[sId].kmax[nucid];

  float scaled_kr = Krate*CP_SINGLEFLOWFIT[sId].molecules_to_micromolar_conversion/d;
  float half_kr = Krate*0.5f;

  // variables used for solving background signal shape
  float aval = 0.0f;

  //new Solve HydrogenFlowInWell

  float one_over_two_tauB = 1.0f;
  float one_over_one_plus_aval = 1.0f/ (1.0f+aval);
  float red_hydro_prev; 
  float fval_local  = 0.0f;

  float red_hydro;

  c_dntp_top_ndx += fnum*num_frames*ISIG_SUB_STEPS_SINGLE_FLOW;

  for (i=CP_SINGLEFLOWFIT[sId].start[fnum];i < num_frames;i++)
  {
    if (totgen > 0.0f)
    {
      ldt = (CP_SINGLEFLOWFIT[sId].deltaFrames[i]/( ISIG_SUB_STEPS_SINGLE_FLOW * FRAMESPERSEC)) * half_kr;
      for (st=1; (st <= ISIG_SUB_STEPS_SINGLE_FLOW) && (totgen > 0.0f);st++)
      {
        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;
        c_dntp_bot = nucRise[c_dntp_top_ndx++]/ (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + CP_SINGLEFLOWFIT[sId].kmax[nucid]);

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
    aval = CP_SINGLEFLOWFIT[sId].deltaFrames[i]*one_over_two_tauB;
    one_over_one_plus_aval = 1.0f/ (1.0f+aval);
    
    if(i==CP_SINGLEFLOWFIT[sId].start[fnum])
      fval_local  = red_hydro; // *one_over_one_plus_aval;
    else
      fval_local = red_hydro - red_hydro_prev + (1.0f-aval)*fval_local; // *one_over_one_plus_aval;

    red_hydro_prev = red_hydro;
 
    fval_local *=  one_over_one_plus_aval;

    if( flag == 0){
      fval[num_beads*i] = fval_local * gain;  
 
    }else{

#ifndef TRANSPOSED_EMPH
      float weight = emRight != NULL ? frac*emLeft[i] + (1.0f - frac)*emRight[i] : emLeft[i];
#else
      float weight = emRight != NULL ? frac*emLeft[i*(MAX_HPLEN+1)] + (1.0f - frac)*emRight[i*(MAX_HPLEN+1)] : emLeft[i*(MAX_HPLEN+1)];
#endif
      int bxi = num_beads * i;
      float jac_tmp =  weight * (fval_local*gain - fval_in[bxi]) * 1000.0f;
      if(flag==1){
        jac_out[bxi] = jac_tmp;
        *aa += jac_tmp * jac_tmp;       
        if (!twoParamFit) 
         *rhs0 += (jac_tmp * err[bxi]);
      }
      
      if(flag == 2){
        *akr +=  jac_out[bxi] * jac_tmp;
        *rhs0 += jac_out[bxi] * err[bxi];  //err
        *rhs1 += jac_tmp * err[bxi];
        *krkr += jac_tmp*jac_tmp;
      }
    }
  }
}

__device__ void
ModelFuncEvaluationAndProjectiontForSingleFlowFit(
  int sId,
  int fnum,
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
  else if (A > MAX_HPLEN)
    A = MAX_HPLEN;


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

  if (iright == MAX_HPLEN)
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

  float c_dntp_bot_plus_kmax = 1.0f/CP_SINGLEFLOWFIT[sId].kmax[nucid];

  float scaled_kr = Krate*CP_SINGLEFLOWFIT[sId].molecules_to_micromolar_conversion/d;
  float half_kr = Krate*0.5f;

  // variables used for solving background signal shape
  float aval = 0.0f;

  //new Solve HydrogenFlowInWell

  float one_over_two_tauB = 1.0f;
  float one_over_one_plus_aval = 1.0f/ (1.0f+aval);
  float red_hydro_prev; 
  float fval_local  = 0.0f;

  float red_hydro;

  c_dntp_top_ndx += fnum*num_frames*ISIG_SUB_STEPS_SINGLE_FLOW;

  float num = 0;
  float den = 0.0001f;
  for (i=CP_SINGLEFLOWFIT[sId].start[fnum];i < num_frames;i++)
  {
    if (totgen > 0.0f)
    {
      ldt = (CP_SINGLEFLOWFIT[sId].deltaFrames[i]/( ISIG_SUB_STEPS_SINGLE_FLOW * FRAMESPERSEC)) * half_kr;
      for (st=1; (st <= ISIG_SUB_STEPS_SINGLE_FLOW) && (totgen > 0.0f);st++)
      {
        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;
        c_dntp_bot = nucRise[c_dntp_top_ndx++]/ (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + CP_SINGLEFLOWFIT[sId].kmax[nucid]);

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
    aval = CP_SINGLEFLOWFIT[sId].deltaFrames[i]*one_over_two_tauB;
    one_over_one_plus_aval = 1.0f/ (1.0f+aval);
    
    if(i==CP_SINGLEFLOWFIT[sId].start[fnum])
      fval_local  = red_hydro; // *one_over_one_plus_aval;
    else
      fval_local = red_hydro - red_hydro_prev + (1.0f-aval)*fval_local; // *one_over_one_plus_aval;

    red_hydro_prev = red_hydro;
 
    fval_local *=  one_over_one_plus_aval;
#ifndef TRANSPOSED_EMPH
      float weight = emRight != NULL ? frac*emLeft[i] + (1.0f - frac)*emRight[i] : emLeft[i];
#else
      float weight = emRight != NULL ? frac*emLeft[i*(MAX_HPLEN+1)] + (1.0f - frac)*emRight[i*(MAX_HPLEN+1)] : emLeft[i*(MAX_HPLEN+1)];
 #endif

     delta = (fval_local*gain) - fval[i*num_beads];
     num += epsilon*delta*err[i*num_beads]*weight*weight; 
     den += delta*delta*weight*weight;  
  }
  delta = num/den;
}



__device__ void
ComputeHydrogenForMultiFlowFit_dev(
  int sId,
  int fnum,
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
  float* ival)
{
  float sens = CP_MULTIFLOWFIT[sId].sens*SENSMULTIPLIER;
  if (A < 0.0f) {
    A = -A;
    sens = -sens;
  }
  else if (A > MAX_HPLEN)
    A = MAX_HPLEN;

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
    ileft = 0;
  }

  if (iright >= MAX_HPLEN)
  {
    iright = ileft = MAX_HPLEN-1;
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

  float c_dntp_bot_plus_kmax = 1.0f/CP_MULTIFLOWFIT[sId].kmax[nucid];

  float scaled_kr = Krate*CP_MULTIFLOWFIT[sId].molecules_to_micromolar_conversion/d;
  float half_kr = Krate*0.5f;

  c_dntp_top_ndx += fnum*num_frames*ISIG_SUB_STEPS_MULTI_FLOW;

  for(i=0;i<CP_MULTIFLOWFIT[sId].start[fnum]; i++) {
    *ival = 0;
    ival += num_beads;
  }

  for (i=CP_MULTIFLOWFIT[sId].start[fnum];i < num_frames;i++)
  {
    if (totgen > 0.0f)
    {
      ldt = (CP_MULTIFLOWFIT[sId].deltaFrames[i]/( ISIG_SUB_STEPS_MULTI_FLOW * FRAMESPERSEC)) * half_kr;
      for (st=1; (st <= ISIG_SUB_STEPS_MULTI_FLOW) && (totgen > 0.0f);st++)
      {
        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;
        c_dntp_bot = nucRise[c_dntp_top_ndx++]/ (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + CP_MULTIFLOWFIT[sId].kmax[nucid]);

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
      *ival = (totocc-totgen) * sens;
    }else{
      *ival = totocc * sens;
    }
    ival += num_beads;
  }
}

__device__ void
ComputeSignalForMultiFlowFit_dev(
  float restrict_clonal,
  int sId,
  int fnum,
  float A, 
  float tauB,
  float etbR,
  float gain,
  int num_frames,
  int num_beads,
  float* non_integer_penalty,
  float* dark_matter,
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

  if ((A < restrict_clonal) && (fnum > KEY_LEN)) {
    int intcall = A + 0.5f;
    clamp_streaming(intcall, 0, MAGIC_MAX_CLONAL_HP_LEVEL);
    clonal_error_term = fabs(A - intcall) * non_integer_penalty[intcall];
  }

  float one_over_two_taub = 1.0f / (2.0f*tauB);
  xt = CP_MULTIFLOWFIT[sId].deltaFrames[i]*one_over_two_taub;

  float one_over_one_plus_aval = 1.0f/ (1.0f+xt);

  sbg += fnum*num_frames;
  purple_hydr = ( *ival + (etbR+xt)*sbg[i])*one_over_one_plus_aval;
  fval_local = purple_hydr*gain + dark_matter[i]*CP_MULTIFLOWFIT[sId].darkness[0] + 
                              clonal_error_term * ((float) (i&1) - 0.5f);
  *output = useEmphasis ? (fval_local - *fval)*emphasis[i] / diff : fval_local;
  output += num_beads;
  i++;
  for (; i<num_frames; ++i)
  {
    xt = CP_MULTIFLOWFIT[sId].deltaFrames[i]*one_over_two_taub;
    one_over_one_plus_aval = 1.0f/(1.0f+xt);
    purple_hydr = ((ival[i*num_beads] - ival[(i-1)*num_beads]) 
         + (etbR+xt)*sbg[i] - (etbR-xt) * sbg[i-1]+ (1.0f-xt) * purple_hydr) * one_over_one_plus_aval;
    fval_local = purple_hydr*gain + dark_matter[i]*CP_MULTIFLOWFIT[sId].darkness[0];

    if (i < MAXCLONALMODIFYPOINTSERROR)
      fval_local += clonal_error_term * ((float) (i&1) - 0.5f);

    *output = useEmphasis ? 
        (fval_local - fval[i*num_beads])*emphasis[i] / diff : fval_local;
    output += num_beads;
  }

}
/*
__device__
void ComputeTauB_dev(float& tauB, float etbR, int sId) {
  if (CP_MULTIFLOWFIT[sId].fit_taue) {
    tauB = etbR  ? (CP_MULTIFLOWFIT[sId].tauE / etbR) : MINTAUB;
  }
  else {
    tauB = CP_MULTIFLOWFIT[sId].tau_R_m*etbR + CP_MULTIFLOWFIT[sId].tau_R_o;
  }
  clamp_streaming(tauB, MINTAUB, MAXTAUB);
}*/

__device__
void ComputeTauB_dev(float& tauB, const ConstParams* pCP , float etbR, int sId) {

  if (pCP->fit_taue) {
    tauB = etbR  ? (pCP->tauE / etbR) : MINTAUB;
  }
  else {
    tauB = pCP->tau_R_m*etbR + pCP->tau_R_o;
  }
  clamp_streaming(tauB, MINTAUB, MAXTAUB);
}

__device__ 
void ComputeEtbR_dev(float& etbR,const ConstParams* pCP , float R, int sId, int nucid, int absFnum ) {
  if (CP_MULTIFLOWFIT[sId].fit_taue) {
    etbR = R;
    if (etbR)
      etbR = pCP->NucModifyRatio[nucid] /(pCP->NucModifyRatio[nucid] + 
               (1.0f - (pCP->RatioDrift * (absFnum)/SCALEOFBUFFERINGCHANGE))*
               (1.0f / etbR - 1.0f));
  }
  else {
    etbR = R*pCP->NucModifyRatio[nucid] + 
        (1.0f - R*pCP->NucModifyRatio[nucid])*
        pCP->RatioDrift*(absFnum)/SCALEOFBUFFERINGCHANGE;
  }
}

__device__
void ComputeSP_dev(float& SP, const ConstParams *pCP, float Copies, int absFnum, int sId) {
  SP = (float)(COPYMULTIPLIER * Copies) * pow(pCP->CopyDrift,absFnum);
}

__device__ void 
ResidualCalculationPerFlow(
  float* fg_buffers, 
  float* fval, 
  float* emLeft,
  float* emRight,
  float frac, 
  float* err, 
  float& residual,
//  int ibd,
  int num_beads,
  int num_frames) {
  int i;
  float e;
  
  residual = 0;

  float weight;
  float wtScale = 0;
  for (i=0; i<num_frames; ++i) {
#ifndef TRANSPOSED_EMPH
    weight = emRight != NULL ? frac*emLeft[i] + (1.0f - frac)*emRight[i] : emLeft[i];
#else
    weight = emRight != NULL ? frac* (*emLeft) + (1.0f - frac)*emRight[i*(MAX_HPLEN+1)] : (*emLeft); //[i*(MAX_HPLEN+1)];
    emLeft += (MAX_HPLEN+1);
#endif
    *err = e = weight * (*fg_buffers - *fval);
    residual += e*e;
    wtScale += weight*weight;
    err += num_beads;
    fg_buffers += num_beads;
    fval += num_beads;
    
  }
  residual /= wtScale;
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
#ifndef TRANSPOSED_EMPH
    weight = emRight != NULL ? frac*emLeft[i] + (1.0f - frac)*emRight[i] : emLeft[i];
#else
    weight = emRight != NULL ? frac*emLeft[i*(MAX_HPLEN+1)] + (1.0f - frac)*emRight[i*(MAX_HPLEN+1)] : emLeft[i*(MAX_HPLEN+1)];
#endif
    e = fg_buffers[num_beads*i] - fval[num_beads*i];
    err[num_beads*i] = e;
    residual += e*e*weight*weight;
    wtScale += weight*weight;
  }
  residual = residual/wtScale;
}


__device__ void 
CalculateMeanResidualErrorPerFlow(
  float* fg_buffers, 
  float* fval, 
  float* weight, // highest hp weighting emphasis vector
  float& residual,
//  int ibd,
  int num_beads,
  int num_frames) 
{
  int i;
  float e;
  float wtScale = 0.0f;

  residual = 0;
  
  for (i=0; i<num_frames; ++i) {
    wtScale += *weight * *weight;
    e = *weight * 
          (*fg_buffers - *fval);
    residual += e*e;

    weight += (MAX_HPLEN + 1);
    fg_buffers+=num_beads;
    fval += num_beads;
    
  }
  residual = sqrtf(residual/wtScale);
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
                                     int num_frames )
{
 
  unsigned int stepIdx;
  float * basePtr1;
  float * basePtr2;  
  float result = 0;

  if ((mask & 0xFFFFF) == 0) return 0;

  stepIdx  = mask >> PARAM1_STEPIDX_SHIFT;
 // printf("%u/", stepIdx ); 
  basePtr1 = input + stepIdx * num_beads*num_frames *NUMFB + idb;
  stepIdx = (mask >> PARAM2_STEPIDX_SHIFT) & 63; // 63 == 0011 1111 
 // printf("%u: ", stepIdx ); 

  basePtr2 =  input + stepIdx * num_beads*num_frames *NUMFB + idb; 

  for(int fnum = 0; fnum<NUMFB; fnum++){
    bool doDotProductForFlow = (mask >> fnum) & 1;
//    printf("%d", doDotProductForFlow ); 

    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + fnum*num_frames*num_beads;
      float * ptr2 = basePtr2 + fnum*num_frames*num_beads;
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
                                     int num_frames )
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
  basePtr1 = input + stepIdx * num_beads*num_frames *NUMFB + idb;
  stepIdx = (mask >> PARAM2_STEPIDX_SHIFT) & 63; // 63 == 0011 1111 
  basePtr2 =  input + stepIdx * num_beads*num_frames *NUMFB + idb; 
  for(int fnum = 0; fnum<NUMFB; fnum++){
    bool doDotProductForFlow = (mask >> fnum) & 1;
    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + fnum*num_frames*num_beads;
      float * ptr2 = basePtr2 + fnum*num_frames*num_beads;
      dotProduct(&result2, ptr1,ptr2,num_frames,num_beads);
    }
  }
  return result2;
}

__device__ float4 CalculateJTJEntryVec4(  unsigned int mask, 
                                     float* input,  
                                     int idb,
                                     int num_beads,
                                     int num_frames )
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
  basePtr1 = input + stepIdx * num_beads*num_frames *NUMFB + idb;
  stepIdx = (mask >> PARAM2_STEPIDX_SHIFT) & 63; // 63 == 0011 1111 
  basePtr2 =  input + stepIdx * num_beads*num_frames *NUMFB + idb; 
  for(int fnum = 0; fnum<NUMFB; fnum++){
    bool doDotProductForFlow = (mask >> fnum) & 1;
    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + fnum*num_frames*num_beads;
      float * ptr2 = basePtr2 + fnum*num_frames*num_beads;
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
                                     int num_frames )
{
 
  int stepIdx;
  float * basePtr1;
  float * basePtr2;  
  float result = 0;

  stepIdx  = mask >> PARAM1_STEPIDX_SHIFT;
 // printf("%d %d \n", stepIdx, num_steps);
  basePtr1 = input + stepIdx * num_beads*num_frames *NUMFB + idb;
  
  basePtr2 =  input + (num_steps-1) * num_beads*num_frames *NUMFB + idb;

  for(int fnum = 0; fnum<NUMFB; fnum++){
    bool doDotProductForFlow = (mask >> fnum) & 1;
    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + fnum*num_frames*num_beads;
      float * ptr2 = basePtr2 + fnum*num_frames*num_beads;
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
                                     int num_frames )
{
 
  int stepIdx;
  float * basePtr1;
  float * basePtr2;  
  float2 result2;
  result2.x = 0;
  result2.y = 0;

  stepIdx  = mask >> PARAM1_STEPIDX_SHIFT;
 // printf("%d %d \n", stepIdx, num_steps);
  basePtr1 = input + stepIdx * num_beads*num_frames *NUMFB + idb;
  
  basePtr2 =  input + (num_steps-1) * num_beads*num_frames *NUMFB + idb;

  for(int fnum = 0; fnum<NUMFB; fnum++){
    bool doDotProductForFlow = (mask >> fnum) & 1;
    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + fnum*num_frames*num_beads;
      float * ptr2 = basePtr2 + fnum*num_frames*num_beads;
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
                                     int num_frames )
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
  basePtr1 = input + stepIdx * num_beads*num_frames *NUMFB + idb;
  
  basePtr2 =  input + (num_steps-1) * num_beads*num_frames *NUMFB + idb;

  for(int fnum = 0; fnum<NUMFB; fnum++){
    bool doDotProductForFlow = (mask >> fnum) & 1;
    if(doDotProductForFlow){
      float * ptr1 = basePtr1 + fnum*num_frames*num_beads;
      float * ptr2 = basePtr2 + fnum*num_frames*num_beads;
      dotProduct(&result4, ptr1,ptr2,num_frames,num_beads);
      //dotProduct(&result, ptr1,ptr2,num_frames,num_beads);

    }
  }
  return result4;

}

__device__ float CalculateNonDiagLowerTriangularElements_dev(
    int ibd,
    int row, 
    float** curJtj, 
    float* ltr, 
    float** curLtr, 
    int stride)
{
  //if (ibd == 33) printf("Non Diag Ele Calculation\n");
  float dotP = 0;
  float runningSumNonDiagonalEntries = 0;
  float curRowElement = 0;
  for (int i=0; i<row; ++i) {
    curRowElement = ((*curJtj)[ibd] - runningSumNonDiagonalEntries) / ltr[ibd];
    //if (ibd == 96) printf("r: %d, c: %d, curRowElement: %f\n", row, i, curRowElement);
    dotP += (curRowElement*curRowElement);
    (*curLtr)[i*stride + ibd] = curRowElement;
    runningSumNonDiagonalEntries = 0;
    ltr += stride;
    for (int j=0; j<=i; ++j) {
      //if (ibd == 33) printf("j: %d, ltr: %f, curltr: %f\n", j, ltr[ibd], (*curLtr)[j*stride + ibd]);
      runningSumNonDiagonalEntries += (ltr[ibd]*((*curLtr)[j*stride + ibd]));
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
    int ibd,
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
      sum += y[j*stride + ibd] * ltr[ibd];
      ltr += stride;    
    }
    y[i*stride + ibd] = (rhs[ibd] - sum) / ltr[ibd];
    //printf("sum: %f, param: %d rhs: %f, y: %f\n", sum, i, rhs[ibd], y[i*stride + ibd]);
    //if (ibd == 96) printf("sum: %f, rhs: %f, y: %f\n", sum, rhs[ibd], y[i*stride + ibd]);
    ltr += stride;
    rhs += stride;
  }
}

// Solving for LTx = y hwere LT is upper triangular 
__device__ void SolveUpperTriangularMatrix_dev(
    float* x, // x solution vector
    float* ltr, // lower triangular matrix 
    float* y, // y vector
    int ibd,
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
      sum += (ltr[idx*stride + ibd] * x[(j-1)*stride + ibd]);
      //printf("ltr: %f, x: %f, idx: %d\n", ltr[idx*stride + ibd], x[(j-1)*stride + ibd], idx);
      idx = idx - j + 1;
    }
    //if (ibd == 96) printf("y: %f\n", y[i*stride + ibd]);
    x[i*stride + ibd] = (y[i*stride + ibd] - sum)/ltr[idx*stride + ibd];
    //if (ibd == 96) printf("sum: %f, param: %d, y: %f, x: %f, idx: %d\n", sum, i, y[i*stride + ibd], x[i*stride + ibd], idx);
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
  int ibd,
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
    dotProduct = CalculateNonDiagLowerTriangularElements_dev(ibd, row, &curJtjPtr, ltr, &curLtr, num_beads);
    // diagonal entry calculation
    curLtr[ibd] = sqrtf(curJtjPtr[ibd]*(1.0f + lambda) - dotProduct);
    //if (ibd == 96) printf("row: %d, arr: %f, dotP: %f, lrr: %f\n", row, curJtjPtr[ibd], dotProduct, curLtr[ibd]);
    curLtr += num_beads;
    curJtjPtr += num_beads;
  }

  SolveLowerTriangularMatrix_dev(delta, ltr, rhs, ibd, num_params, num_beads);
  SolveUpperTriangularMatrix_dev(delta, ltr, delta, ibd, num_params, num_beads);
}

__device__ void CalculateNewBeadParams_dev(
  float* orig_params,
  float* new_params,
  float* delta,
  unsigned int* paramIdxMap,
  int ibd,
  int num_params,
  int num_beads,
  int sId 
)
{
  unsigned int paramIdx;
  //printf("New Params\n");
 /* for (int i=0; i<num_params; ++i)
  {
    paramIdx = paramIdxMap[i];
    printf("old: %f new: %f pIdx: %d\n", params[paramIdx*num_beads + ibd], params[paramIdx*num_beads + ibd] + delta[i*num_beads + ibd], paramIdx);
    params[paramIdx*num_beads + ibd] += delta[i*num_beads + ibd];
  }*/  

  unsigned int AmplIdx = BEAD_AMPL(Ampl[0]);
  unsigned int RIdx = BEAD_R(R);
  unsigned int CopiesIdx = BEAD_COPIES(Copies);
  unsigned int DmultIdx = BEAD_DMULT(dmult);
  float paramVal;
  for (int i=0; i<num_params; ++i) 
  {
    paramIdx = paramIdxMap[i];
    if (paramIdx == RIdx) {
      paramVal = orig_params[paramIdx*num_beads + ibd] + delta[i*num_beads + ibd];
      clamp_streaming(paramVal, CP_MULTIFLOWFIT[sId].beadParamsMinConstraints.R, CP_MULTIFLOWFIT[sId].beadParamsMaxConstraints.R);
    }
    if (paramIdx == CopiesIdx) {
      paramVal = orig_params[paramIdx*num_beads + ibd] + delta[i*num_beads + ibd];
      clamp_streaming(paramVal, CP_MULTIFLOWFIT[sId].beadParamsMinConstraints.Copies, CP_MULTIFLOWFIT[sId].beadParamsMaxConstraints.Copies);
    }
    if (paramIdx == DmultIdx) {
      paramVal = orig_params[paramIdx*num_beads + ibd] + delta[i*num_beads + ibd];
      clamp_streaming(paramVal, CP_MULTIFLOWFIT[sId].beadParamsMinConstraints.dmult, CP_MULTIFLOWFIT[sId].beadParamsMaxConstraints.dmult);
    }
    if (paramIdx >= AmplIdx && paramIdx <= (AmplIdx + NUMFB - 1)) {
      paramVal = orig_params[paramIdx*num_beads + ibd] + delta[i*num_beads + ibd];
      clamp_streaming(paramVal, CP_MULTIFLOWFIT[sId].beadParamsMinConstraints.Ampl, CP_MULTIFLOWFIT[sId].beadParamsMaxConstraints.Ampl);
    }
    //printf("old: %f new: %f pIdx: %d\n", params[paramIdx*num_beads + ibd], paramVal, paramIdx);
    new_params[paramIdx*num_beads + ibd] = paramVal;
  }
}

__device__ void UpdateBeadParams_dev(
  float* orig_params,
  float* new_params,
  unsigned int* paramIdxMap,
  int ibd,
  int num_params,
  int num_beads 
)
{
  unsigned int paramIdx;
  //printf("Updated Params in Lev Mar Iter\n");
  for (int i=0; i<num_params; ++i)
  {
    paramIdx = paramIdxMap[i];
    //printf("new: %f pIdx: %d\n", new_params[paramIdx*num_beads + ibd], paramIdx);
    orig_params[paramIdx*num_beads + ibd] = new_params[paramIdx*num_beads + ibd];
  }  
}

__device__ void CalculateMultiFlowFitResidual_dev(
  float& residual,
  float* pObservedTrace,
  float* pModelTrace,
  float* pEmphasisVec,
  int fnum,
  int num_beads,
  int num_frames
)
{
  float eval;
  pObservedTrace += fnum*num_beads*num_frames;
  for (int j=0; j<num_frames; ++j)
  {
    eval = (*pObservedTrace - *pModelTrace)*pEmphasisVec[j];
    residual += eval*eval;
    pObservedTrace += num_beads;
    pModelTrace += num_beads;
  }
}

__device__ void DecideOnEmphasisVectorsForInterpolation(
  float** emLeft,
  float** emRight,
  float& frac,
  float Ampl,
  float* emphasis,
  int num_frames
)
{
  if (Ampl < MAX_HPLEN) {
    int left = (int) Ampl;
    frac = (left + 1.0f - Ampl);
    if (left < 0) {
      left = 0;
      frac = 1.0f;
    }
#ifndef TRANSPOSED_EMPH
    *emLeft = &emphasis[left*num_frames];
    *emRight = &emphasis[(left + 1)*num_frames];
  }else{
    *emLeft = &emphasis[MAX_HPLEN*num_frames];
#else
    *emLeft = &emphasis[left];
    *emRight = &emphasis[left + 1];
  }else{
    *emLeft = &emphasis[MAX_HPLEN]; 
#endif
    *emRight = NULL;
    frac = 1.0f;
  }

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

/*****************************************************************************

              SINGLE FLOW FIT KERNELS 

*****************************************************************************/


// Let number of beads be N and frames be F. The size for each input argument in
// comments is in bytes.
__global__ void 
PerFlowLevMarFit_k(
  // inputs
  float* fg_buffers, // NxF
  float* emphasisVec, 
  float* nucRise, 
  // bead params
  //float* pAmpl, // NxNUMFB
  //float* pKmult, // NxNUMFB
  //float* pdmult, // N
  //float* pR, // N
  //float* pgain, // N
//  float* pSP, // N
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
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId
) 
{
  extern __shared__ float emphasis[];

#ifndef TRANSPOSED_EMPH
  for (int i=0; i<(MAX_HPLEN + 1)*num_frames; i+=num_frames)
  {
    if (threadIdx.x < num_frames)
      emphasis[i + threadIdx.x] = emphasisVec[i + threadIdx.x];
  }
#else

  
/*  for(int i=0; i<num_frames; i ++)
  {
     if (threadIdx.x < MAX_HPLEN+1)
      emphasis[(MAX_HPLEN+1)*i + threadIdx.x] = emphasisVec[num_frames*threadIdx.x + i ];
  }
*/
  int numWarps = blockDim.x/32;
  int threadWarpIdx = threadIdx.x%32;
  int warpIdx = threadIdx.x/32; 
  for(int i=warpIdx; i<num_frames; i += numWarps)
  {
     if (threadWarpIdx < MAX_HPLEN+1)
      emphasis[(MAX_HPLEN+1)*i + threadWarpIdx ] = emphasisVec[num_frames*threadWarpIdx + i ];
  }
  
#endif

  __syncthreads();

  int ibd = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(ibd >= num_beads) return;

  num_beads = ((num_beads+31)/32) * 32;

/*
  pAmpl += ibd;
  pKmult += ibd;
  pR += ibd;
  //pSP += ibd;  
  //float2 * pTauBSP = (float2*) ptauB;
  //pTauBSP += ibd;
  pCopies += ibd;
  pgain += ibd;
  pdmult += ibd;
*/

  pBeadParamsBase += ibd;
  pState += ibd; 
  

  float * pCopies = &pBeadParamsBase[BEAD_COPIES(Copies)*num_beads];
  float *pAmpl = &pBeadParamsBase[BEAD_AMPL(Ampl[0])*num_beads];

   
  err += ibd;
  fval += ibd;
  tmp_fval += ibd;
  jac += ibd;
  meanErr += ibd;
  
  fg_buffers += ibd;


  if (pState->corrupt || !pState->clonal_read) return;

  float avg_err;
  for(int fnum=0; fnum<NUMFB; fnum++){

    int nucid = CP_SINGLEFLOWFIT[sId].flowIdxMap[fnum];
    float sens = CP_SINGLEFLOWFIT[sId].sens*SENSMULTIPLIER; 

    float copies = *pCopies;
    float R = *(pCopies + num_beads);
    float d = *(pCopies + 2*num_beads);
    float gain = *(pCopies + 3 * num_beads) ;
    
    float *pKmult = pAmpl + num_beads*NUMFB;


    d *= CP_SINGLEFLOWFIT[sId].d[nucid];


    //offset for next value gets added to address at end of fnum loop
    
//    float krate = *(pAmpl + num_beads*NUMFB); // *pKmult;
    float krate = *pKmult;

    float Ampl = *pAmpl;
    //float2 tmp = *pTauBSP;
    float etbR;
    float tauB; // = tmp.x; // *ptauB;
    float SP; //= tmp.y; //*pSP;  
 
 
    ComputeEtbR_dev(etbR, &CP_SINGLEFLOWFIT[sId], R, sId, nucid, realFnum+fnum);
    ComputeTauB_dev(tauB, &CP_SINGLEFLOWFIT[sId], etbR, sId);
    ComputeSP_dev(SP, &CP_SINGLEFLOWFIT[sId], copies, realFnum+fnum, sId);
  
    bool twoParamFit = ( copies * Ampl > 2.0f  );
 
    float residual, newresidual; // lambdaThreshold;
    int i;
    // These values before start are always zero since there is no nucrise yet. Don't need to
    // zero it out. Have to change the residual calculation accordingly for the frames before the
    // start.
    for (i =0; i < CP_SINGLEFLOWFIT[sId].start[fnum]; i++) {
      fval[num_beads*i] = 0;
      tmp_fval[num_beads*i] = 0;
    }

    // first step
    // Evaluate model function using input Ampl and Krate and get starting residual
    ModelFuncEvaluationForSingleFlowFit(twoParamFit,sId, fnum, nucid, nucRise, 
        Ampl, krate*CP_SINGLEFLOWFIT[sId].krate[nucid], tauB, gain, SP, d, 
        sens, ISIG_SUB_STEPS_SINGLE_FLOW* CP_SINGLEFLOWFIT[sId].start[fnum],
        num_frames, num_beads, fval);

    float *emLeft, *emRight;
    float frac;

    // calculating weighted sum of square residuals for the convergence test
    DecideOnEmphasisVectorsForInterpolation(&emLeft,&emRight,frac,Ampl,emphasis, num_frames);
    ResidualCalculationPerFlow(fg_buffers, fval, emLeft, emRight, frac, err, residual,  
      num_beads, num_frames);
  
    // new Ampl and Krate generated from the Lev mar Fit
    float newAmpl, newKrate;

    // convergence test variables 
    float delta0 = 0, delta1 = 0;

    // Lev Mar Fit Outer Loop
    int iter;
    for (iter = 0; iter < ITER; ++iter) {

      // new Ampl and krate by adding delta to existing values
      newAmpl = Ampl + 0.001f;
      newKrate = (twoParamFit)?(krate + 0.001f):(krate);
 
      // Evaluate model function for new Ampl keeping Krate constant
      float aa = 0, akr= 0, krkr = 0, rhs0 = 0, rhs1 = 0;

      ModelFuncEvaluationForSingleFlowFit(twoParamFit,sId, fnum, nucid, nucRise,
          newAmpl, krate*CP_SINGLEFLOWFIT[sId].krate[nucid], tauB, gain, SP, d, 
          sens, CP_SINGLEFLOWFIT[sId].start[fnum]*ISIG_SUB_STEPS_SINGLE_FLOW, 
          num_frames, num_beads, tmp_fval, 1, jac, emLeft, emRight, frac, fval, 
          err, &aa, &rhs0, &krkr, &rhs1, &akr);


      if (twoParamFit) 
        ModelFuncEvaluationForSingleFlowFit(twoParamFit,sId, fnum, nucid, nucRise, 
            Ampl, newKrate*CP_SINGLEFLOWFIT[sId].krate[nucid], tauB, gain, SP, d, 
            sens, CP_SINGLEFLOWFIT[sId].start[fnum]*ISIG_SUB_STEPS_SINGLE_FLOW, 
            num_frames, num_beads, tmp_fval,2, jac, emLeft, emRight, frac, fval, 
            err, &aa, &rhs0, &krkr, &rhs1, &akr);

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

        clamp_streaming(newAmpl, minAmpl, (float)MAXAMPL);
        if(twoParamFit)clamp_streaming(newKrate, minKmult, maxKmult);

        // Evaluate using new params
        ModelFuncEvaluationForSingleFlowFit(twoParamFit,sId, fnum, nucid, nucRise, 
            newAmpl, newKrate*CP_SINGLEFLOWFIT[sId].krate[nucid], tauB, gain, SP, 
            d, sens, CP_SINGLEFLOWFIT[sId].start[fnum]*ISIG_SUB_STEPS_SINGLE_FLOW, 
            num_frames, num_beads, tmp_fval);
        // residual calculation using new parameters
        if (useDynamicEmphasis)
          DecideOnEmphasisVectorsForInterpolation(&emLeft,&emRight,frac,newAmpl,emphasis, num_frames);
        ResidualCalculationPerFlow(fg_buffers, tmp_fval, emLeft, emRight, frac, err, newresidual, 
           num_beads, num_frames);
        
        if (newresidual < residual) {
          Ampl = newAmpl;
          if(twoParamFit)krate = newKrate;
          // copy new function val to fval
          for (i=CP_SINGLEFLOWFIT[sId].start[fnum]; i<num_frames; ++i)
            fval[num_beads*i] = tmp_fval[num_beads*i];

          residual = newresidual;
        }
        else {
          if (useDynamicEmphasis) {
            DecideOnEmphasisVectorsForInterpolation(&emLeft,&emRight,frac,Ampl,emphasis, 
              num_frames);
          }
        }
      }

      if ((delta0*delta0) < 0.0000025f)
        break;

    } // end ITER loop

    if(fnum==0) avg_err = pState->avg_err * realFnum;  

    if(twoParamFit) *pKmult = krate;//*(pAmpl + num_beads*NUMFB)= krate;
    *pAmpl= Ampl;
 
    //CalculateMeanResidualErrorPerFlow(fg_buffers, fval, emphasis+MAX_HPLEN*num_frames, residual, 
    //  num_beads, num_frames); 
    CalculateMeanResidualErrorPerFlow(fg_buffers, fval, emphasis+MAX_HPLEN, residual, 
      num_beads, num_frames); 
  
    avg_err += residual;
    meanErr[num_beads * fnum] = residual;

    //pKmult += num_beads;  //[offset];
    pAmpl += num_beads; //[offset];
    //ptauB += num_beads; //[offset];
    //pSP += num_beads; //[offset];  
    //pTauBSP += num_beads;

    fg_buffers += num_frames*num_beads;
  } // end fnum loop

  avg_err /= (realFnum + NUMFB);
  pState->avg_err = avg_err;
  int high_err_cnt = 0;
  avg_err *= WASHOUT_THRESHOLD;
  for (int fnum = NUMFB - 1; fnum >= 0 
                           && (meanErr[num_beads* fnum] > avg_err); fnum--)
    high_err_cnt++;

  if (high_err_cnt > WASHOUT_FLOW_DETECTION)
    pState->corrupt = true;

}


///////// Pre-processing kernel (bkg correct and well params calculation)
__global__ void PreSingleFitProcessing_k(// Here FL stands for flows
  // inputs from data reorganization
  float* pCopies, // N
  float* pR, // N
  float* pdmult, // N
  float* pgain, // N
  float* pAmpl, // FLxN
  float* pkmult, // FLxN
  float* sbg, // FLxF 
  float* dark_matter, // FLxF
  float* fgbuffers, // FLxFxN

  // other inputs 
  int flowNum, // starting flow number to calculate absolute flow num
  int num_beads, // 4
  int num_frames, // 4
  int numEv, // 4

  // outputs
//  float* pSP, // FLxN
//  float* ptauB, 
  bool alternatingFit,
  int sId
)
{
  int ibd = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(ibd >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;

  int NucId, i;
  float Rval, tau, SP;
  float darkness = CP_SINGLEFLOWFIT[sId].darkness[0];
  float gain = pgain[ibd];
  float *fval, *sbgPtr;
  for (int fnum=0; fnum < NUMFB; ++fnum) {
  
    sbgPtr = sbg + fnum*num_frames; // may shift to constant memory
    NucId = CP_SINGLEFLOWFIT[sId].flowIdxMap[fnum]; 
    SP = (float) (COPYMULTIPLIER * pCopies[ibd]) * 
      pow(CP_SINGLEFLOWFIT[sId].CopyDrift, flowNum + fnum);

//    pSP[fnum*num_beads + ibd] = SP; 

    if (CP_SINGLEFLOWFIT[sId].fit_taue) {
      Rval = pR[ibd];
      if (Rval)
 	 Rval = CP_SINGLEFLOWFIT[sId].NucModifyRatio[NucId] /(CP_SINGLEFLOWFIT[sId].NucModifyRatio[NucId] +
 	 	 (1.0f - (CP_SINGLEFLOWFIT[sId].RatioDrift * (flowNum + fnum)/SCALEOFBUFFERINGCHANGE))*
 	 	  (1.0f / Rval - 1.0f));
      // tau b calculation..Will go into a device function
      tau = Rval ? (CP_SINGLEFLOWFIT[sId].tauE / Rval) : MINTAUB;
    }
    else {
      Rval = pR[ibd] * CP_SINGLEFLOWFIT[sId].NucModifyRatio[NucId] + CP_SINGLEFLOWFIT[sId].RatioDrift * (flowNum + fnum) *
 	 	(1.0f - pR[ibd] * CP_SINGLEFLOWFIT[sId].NucModifyRatio[NucId]) / SCALEOFBUFFERINGCHANGE;
      tau = CP_SINGLEFLOWFIT[sId].tau_R_m * Rval + CP_SINGLEFLOWFIT[sId].tau_R_o;
    }
    clamp_streaming(tau, (float)MINTAUB, (float)MAXTAUB); 

//    ptauB[fnum*num_beads + ibd] = tau;

    Rval -= 1.0f;
    float dv = 0.0f;
    float dv_rs = 0.0f;
    float dvn = 0.0f;
    float curSbgVal;
    float aval;

    // need to go in constant memory since same word access for each thread in the warp
    float* et = &dark_matter[NucId*num_frames]; 

    fval = &fgbuffers[fnum*num_beads*num_frames];
    for (i=0; i<num_frames; i++)
    {
      aval = CP_SINGLEFLOWFIT[sId].deltaFrames[i]/(2.0f * tau);

      // calculate new dv
      curSbgVal = sbgPtr[i];
      dvn = (Rval*curSbgVal - dv_rs/tau - dv*aval) / (1.0f + aval);
      dv_rs += (dv+dvn) * CP_SINGLEFLOWFIT[sId].deltaFrames[i] * 0.5f;
      dv = dvn;

      fval[i*num_beads + ibd] = fval[i*num_beads + ibd]
                    -  ((dv+curSbgVal)*gain + darkness*et[i]);
    }

    // Guess amplitude for alternating fit
    if (alternatingFit)
    {
      int start_frame = CP_SINGLEFLOWFIT[sId].start[fnum]; 
      aval = CP_SINGLEFLOWFIT[sId].deltaFrames[start_frame]/(2.0f * tau);
      float sig_val_at_start =  fval[start_frame*num_beads + ibd] 
                                    * (1.0f + aval);  
      int cutoff_frame = start_frame + 
                     CP_SINGLEFLOWFIT[sId].nuc_shape.nuc_flow_span*0.75f;
      if (cutoff_frame > (num_frames -1))
        cutoff_frame = num_frames - 1;
      float ampl_guess = sig_val_at_start;
      for (i=(start_frame + 1); i<=cutoff_frame; ++i)
      {
        aval = CP_SINGLEFLOWFIT[sId].deltaFrames[i]/(2.0f * tau);
        ampl_guess = (1.0f + aval)*fval[i*num_beads + ibd] - 
                     (1.0f - aval)*fval[(i-1)*num_beads + ibd] + ampl_guess;
      }
      pAmpl[fnum*num_beads + ibd] = (ampl_guess - sig_val_at_start) / (SP*CP_MULTIFLOWFIT[fnum].sens*SENSMULTIPLIER);
    }

  }
}
/*
__global__ void 
PerFlowAlternatingFit_k(
  // inputs
  float* fg_buffers_base, // NxF
  float* emphasisVec, 
  float* nucRise, 
  // bead params
  float* pAmpl, // N
  float* pKmult, // N
  float* pdmult, // N
  float* ptauB, // N
  float* pgain, // N
  float* pSP, // N
  float * pCopies, //N
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
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  int sId
) 
{
  extern __shared__ float emphasis[];

  for (int i=0; i<(MAX_HPLEN + 1)*num_frames; i+=num_frames)
  {
    if (threadIdx.x < num_frames)
      emphasis[i + threadIdx.x] = emphasisVec[i + threadIdx.x];
  }
  __syncthreads();

  int ibd = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(ibd >= num_beads) return;

  pState += ibd; 

  num_beads = ((num_beads+32-1)/32) * 32;

  float avg_err;

  if (pState->corrupt || !pState->clonal_read) return;

  float gain = pgain[ibd];
  float sens = CP_SINGLEFLOWFIT[sId].sens*SENSMULTIPLIER; 
  float d = pdmult[ibd];
  float copies = pCopies[ibd];
  fval += ibd;
  tmp_fval += ibd;
  err += ibd;
  meanErr += ibd;
  pKmult += ibd;
  pAmpl += ibd;
  ptauB += ibd;
  pSP += ibd;
  
  for(int fnum=0; fnum<NUMFB; fnum++){

    int nucid = CP_SINGLEFLOWFIT[sId].flowIdxMap[fnum];
    d *= CP_SINGLEFLOWFIT[sId].d[nucid];

    float krate = pKmult[fnum*num_beads];
    float Ampl = pAmpl[fnum*num_beads];
    float tauB = ptauB[fnum*num_beads];
    float SP = pSP[fnum*num_beads];  
 
  
    float * fg_buffers = fg_buffers_base + num_frames*fnum*num_beads + ibd;
    bool twoParamFit = false;  //TODO: whatever.. just addd false to get rid of warning 
    float residual, newresidual, start_res; 
    int i, iter;

    // These values before start are always zero since there is no nucrise yet. Don't need to
    // zero it out. Have to change the residual calculation accordingly for the frames before the
    // start.
    for (i=0; i < CP_SINGLEFLOWFIT[sId].start[fnum]; i++) {
      fval[num_beads*i] = 0;
      tmp_fval[num_beads*i] = 0;
    }

    ModelFuncEvaluationForSingleFlowFit(twoParamFit,sId, fnum, nucid, nucRise, 
        Ampl, krate*CP_SINGLEFLOWFIT[sId].krate[nucid], tauB, gain, SP, d, 
        sens, ISIG_SUB_STEPS_SINGLE_FLOW* CP_SINGLEFLOWFIT[sId].start[fnum],
        num_frames, num_beads, fval);

    float *emLeft, *emRight;
    float frac;

    // calculating weighted sum of square residuals for the convergence test
    DecideOnEmphasisVectorsForInterpolation(&emLeft,&emRight,frac,Ampl,emphasis, num_frames);
    ResidualForAlternatingFit(fg_buffers, fval, emLeft, emRight, frac, err, start_res,  
      num_beads, num_frames);
  
    iter = 0;
    while (iter < ITER) 
    {
      {
        // try change in amplitude
        residual = start_res;
        float epsilon = 0.01f;
        if ((Ampl + epsilon) > (float)MAXAMPL)
          epsilon = -1.0f*epsilon;
     
        float newAmpl = Ampl + epsilon;
    
        float delta;
        ModelFuncEvaluationAndProjectiontForSingleFlowFit(sId, fnum, nucid, 
          nucRise, newAmpl, krate*CP_SINGLEFLOWFIT[sId].krate[nucid], tauB, 
          gain, SP, d, sens, 
          ISIG_SUB_STEPS_SINGLE_FLOW* CP_SINGLEFLOWFIT[sId].start[fnum],
          num_frames, num_beads, epsilon, fval, emLeft, emRight, 
          frac, err, delta);
      
        newAmpl = Ampl + delta;
        clamp_streaming(newAmpl, minAmpl, (float)MAXAMPL);
        DecideOnEmphasisVectorsForInterpolation(&emLeft,&emRight,frac,newAmpl,emphasis, num_frames);
        DynamicConstraintKrate(copies, newAmpl, krate, twoParamFit);
    
        ModelFuncEvaluationForSingleFlowFit(twoParamFit,sId, fnum, nucid, nucRise, 
          newAmpl, krate*CP_SINGLEFLOWFIT[sId].krate[nucid], tauB, gain, SP, d, 
          sens, ISIG_SUB_STEPS_SINGLE_FLOW* CP_SINGLEFLOWFIT[sId].start[fnum],
          num_frames, num_beads, tmp_fval);
        ResidualForAlternatingFit(fg_buffers, tmp_fval, emLeft, emRight, frac, err, newresidual, 
          num_beads, num_frames);
        if (newresidual < residual)
        {
          Ampl = newAmpl;
          residual = newresidual;
          for (i=CP_SINGLEFLOWFIT[sId].start[fnum]; i<num_frames; ++i)
            fval[num_beads*i] = tmp_fval[num_beads*i];
        }
        else 
        {
          DynamicConstraintKrate(copies, Ampl, krate, twoParamFit);
          DecideOnEmphasisVectorsForInterpolation(&emLeft,&emRight,frac,Ampl,emphasis, num_frames);
          ResidualForAlternatingFit(fg_buffers, fval, emLeft, emRight, frac, err, residual,  
            num_beads, num_frames);        
        }
      }
      // end of amplitude fit

      if (twoParamFit)
      {
        float epsilon = 0.01f;
        if ((krate + epsilon) > maxKmult)
          epsilon = -1.0f*epsilon;
     
        float newKrate = krate + epsilon;
   
        float delta; 
        ModelFuncEvaluationAndProjectiontForSingleFlowFit(sId, fnum, nucid, nucRise, 
          Ampl, newKrate*CP_SINGLEFLOWFIT[sId].krate[nucid], tauB, gain, SP, d, 
          sens, ISIG_SUB_STEPS_SINGLE_FLOW* CP_SINGLEFLOWFIT[sId].start[fnum],
          num_frames, num_beads, epsilon, fval, emLeft, emRight, frac, err, 
          delta);
      
        newKrate = krate + delta;
        clamp_streaming(newKrate, minKmult, maxKmult);
    
        ModelFuncEvaluationForSingleFlowFit(twoParamFit,sId, fnum, nucid, nucRise, 
          Ampl, newKrate*CP_SINGLEFLOWFIT[sId].krate[nucid], tauB, gain, SP, d, 
          sens, ISIG_SUB_STEPS_SINGLE_FLOW* CP_SINGLEFLOWFIT[sId].start[fnum], 
          num_frames, num_beads, tmp_fval);
        ResidualForAlternatingFit(fg_buffers, tmp_fval, emLeft, emRight, frac, err, 
          newresidual, num_beads, num_frames);
        if (newresidual < residual)
        {
          krate = newKrate;
          residual = newresidual;
          for (i=CP_SINGLEFLOWFIT[sId].start[fnum]; i<num_frames; ++i)
            fval[num_beads*i] = tmp_fval[num_beads*i];
          //float* temp = tmp_fval;
          //tmp_fval = fval;
          //fval = temp;
        }
        else 
        {
          ResidualForAlternatingFit(fg_buffers, fval, emLeft, emRight, frac, err, residual,  
            num_beads, num_frames);        
        }

      }
      // end of krate fit
   
      if ((residual*1.01f) > start_res)
      {
        break;
      }
      else 
      {
        start_res = residual;
        iter++;
      }
    }

    if(fnum==0) avg_err = pState->avg_err * realFnum;  

    pKmult[fnum*num_beads]= krate;
    pAmpl[fnum*num_beads]= Ampl;
 
    CalculateMeanResidualErrorPerFlowForAlternatingFit(err, emphasis+MAX_HPLEN*num_frames, residual, 
      num_beads, num_frames); 
  
    avg_err += residual;
    meanErr[num_beads * fnum] = residual;

  } // end fnum loop
  avg_err /= (realFnum + NUMFB);
  pState->avg_err = avg_err;
  int high_err_cnt = 0;
  avg_err *= WASHOUT_THRESHOLD;
  for (int fnum = NUMFB - 1; fnum >= 0 
                           && (meanErr[num_beads* fnum] > avg_err); fnum--)
    high_err_cnt++;

  if (high_err_cnt > WASHOUT_FLOW_DETECTION)
    pState->corrupt = true;

}
*/

/*****************************************************************************

              MULTI FLOW FIT KERNELS 

*****************************************************************************/

//////// Computing Partial Derivatives
__global__ void ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_k (
  // inputs
  int maxEmphasis,
  float restrict_clonal,
  float* pobservedTrace, 
  float* pival, // FLxNxF   //scatch
  float* pscratch_ival, // FLxNxF
  float* pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* psbg, // FLxF
  float* pemphasis, // MAX_HPLEN+1 xF 
  float* pnon_integer_penalty, // MAX_HPLEN
  float* pdarkMatterComp, // NUMNUC * F  
  float* pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep_t
  CpuStep_t* psteps, // we need a specific struct describing this config for this well fit for GPU
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
  int sId
) 
{
  extern __shared__ float emphasisVec[];

  for (int i=0; i<(MAX_HPLEN + 1)*num_frames; i+=num_frames)
  {
    if (threadIdx.x < num_frames)
      emphasisVec[i + threadIdx.x] = pemphasis[i + threadIdx.x];
  }
  __syncthreads();


  int ibd = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(ibd >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;
  //num_beads += 32 - (num_beads%32);

  int i, j, fnum;
  int stepOffset = num_beads*num_frames;
  float* ptemp, *pfval;

  float kmult, Ampl, tauB, etbR, SP;
  float gain = pbeadParamsTranspose[(BEAD_GAIN(gain))*num_beads + ibd];
  float dmult = pbeadParamsTranspose[(BEAD_DMULT(dmult))*num_beads + ibd];
  float R = pbeadParamsTranspose[(BEAD_R(R))*num_beads + ibd];
  float Copies = pbeadParamsTranspose[(BEAD_COPIES(Copies))*num_beads + ibd];

  //pfval = poutput;
  pfval = poutput + ibd;

  float tot_err = 0.0f;
  pobservedTrace += ibd;
  pival += ibd;
  pscratch_ival += ibd;
  for (fnum=0; fnum<NUMFB; ++fnum) {
    // calculate emphasis vector index
    Ampl = pbeadParamsTranspose[(BEAD_AMPL(Ampl[0]) + fnum)*num_beads + ibd];
    kmult = pbeadParamsTranspose[(BEAD_KMULT(kmult[0]) + fnum)*num_beads + ibd];

    int emphasisIdx = (int)(Ampl) > maxEmphasis ? maxEmphasis : (int)Ampl;
    int nucid = CP_MULTIFLOWFIT[sId].flowIdxMap[fnum];
    for (i=0; i<num_steps; ++i) {
      ptemp = poutput + i*stepOffset + ibd;
      switch (psteps[i].PartialDerivMask) {
	case YERR:
	{
          float eval;
	  for (j=0; j<num_frames; ++j) {
	    eval = (pobservedTrace[j*num_beads] - 
                        pfval[j*num_beads]) * 
		            emphasisVec[emphasisIdx*num_frames + j];
            //ptemp[j*num_beads + ibd] = eval;
            //ptemp[j*num_beads] = eval;
            *ptemp = eval;
            tot_err += eval*eval;
            ptemp += num_beads;
	  }
	}
	break;
	case FVAL:
	{
          ComputeEtbR_dev(etbR, &CP_MULTIFLOWFIT[sId], R, sId, nucid, 0+fnum);
          ComputeTauB_dev(tauB, &CP_MULTIFLOWFIT[sId] ,etbR, sId);
          ComputeSP_dev(SP,  &CP_MULTIFLOWFIT[sId], Copies, fnum, sId);
	  ComputeHydrogenForMultiFlowFit_dev(sId, fnum, nucid, pnucRise, Ampl, 
                          kmult*CP_MULTIFLOWFIT[sId].krate[nucid], gain, SP, 
                          dmult*CP_MULTIFLOWFIT[sId].d[nucid], 
			  ISIG_SUB_STEPS_MULTI_FLOW*CP_MULTIFLOWFIT[sId].start[fnum], 
                          num_frames, num_beads, pival);
	  ComputeSignalForMultiFlowFit_dev(restrict_clonal, sId, fnum, Ampl, tauB,
	  		  etbR, gain, num_frames, num_beads, pnon_integer_penalty,
	  		  pdarkMatterComp + num_frames*nucid, psbg, pival, pfval);
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
            ComputeSP_dev(SP, &CP_MULTIFLOWFIT[sId], Copies, fnum, sId);
	    ComputeHydrogenForMultiFlowFit_dev(sId, fnum, nucid, pnucRise, Ampl, 
                kmult*CP_MULTIFLOWFIT[sId].krate[nucid], gain, SP, 
                dmult*CP_MULTIFLOWFIT[sId].d[nucid], 
                ISIG_SUB_STEPS_MULTI_FLOW*CP_MULTIFLOWFIT[sId].start[fnum], 
                num_frames, num_beads, pivtemp);
          }
          ComputeEtbR_dev(etbR, &CP_MULTIFLOWFIT[sId], R, sId, nucid, 0+fnum);
          ComputeTauB_dev(tauB, &CP_MULTIFLOWFIT[sId], etbR, sId);
	  ComputeSignalForMultiFlowFit_dev(restrict_clonal, sId, fnum, Ampl, tauB,
		etbR, gain, num_frames, num_beads, pnon_integer_penalty,
		pdarkMatterComp + num_frames*nucid, psbg, pivtemp, ptemp, true, 
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
    ptemp = pJTJ + ibd;
    for(int row=0;row<num_params;row++) {
      for(int col = 0; col <= row; col++) {
        unsigned int mask = pDotProdMasks[row*num_params+col];
        if ((mask >> fnum) & 1) {
          unsigned int stepIdx1 = mask >> PARAM1_STEPIDX_SHIFT;
          unsigned int stepIdx2 = (mask >> PARAM2_STEPIDX_SHIFT) & 63;
          *ptemp += dotProduct(poutput + stepIdx1*stepOffset + ibd,
                               poutput + stepIdx2*stepOffset + ibd,
                               num_frames,
                               num_beads);
        }
        ptemp += num_beads;
      }
    }
    
    ptemp = pRHS + ibd;
    for(int row=0;row<num_params;row++){
      unsigned int mask = pDotProdMasks[row*num_params+row];
      unsigned int stepIdx1 = mask >> PARAM1_STEPIDX_SHIFT;
      if ((mask >> fnum) & 1) {
        *ptemp += dotProduct(poutput + stepIdx1*stepOffset + ibd,
			     poutput + (num_steps - 1)*stepOffset + ibd,
			     num_frames,
			     num_beads);
      }
      ptemp += num_beads;
    }
  }
  residual[ibd] = sqrtf(tot_err / (NUMFB*num_frames));
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
  float* pemphasis, // MAX_HPLEN+1 xF // needs precomputation
  float* pnon_integer_penalty, // MAX_HPLEN
  float* pdarkMatterComp, // NUMNUC * F  
  float* pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep_t
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
  int sId
)
{
  extern __shared__ float emphasisVec[];

  for (int i=0; i<(MAX_HPLEN + 1)*num_frames; i+=num_frames)
  {
    if (threadIdx.x < num_frames)
      emphasisVec[i + threadIdx.x] = pemphasis[i + threadIdx.x];
  }
  __syncthreads();


  int ibd = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(ibd >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;
  //num_beads += 32 - (num_beads%32);

  float lambda = plambda[ibd];
  float oldResidual = presidual[ibd];
  bool done = false;

  pival += ibd;
  pfval += ibd;
  pobservedTrace += ibd;
  while(!done) {
    // solve for delta in params
    CholeskySolve_dev(lambda, pjtj, pltr, pb, pdelta, ibd, num_params, num_beads);
    // calculate new beadparams
    CalculateNewBeadParams_dev(pbeadParamsTranspose, pevalBeadParams, pdelta, 
      paramIdxMap, ibd, num_params, num_beads, sId);

    // calculate residual and decide whether to perform further lamda tuning and run cholesky again
    float newResidual = 0; 
    float kmult, Ampl, tauB, etbR, SP;
    float gain = pevalBeadParams[(BEAD_GAIN(gain))*num_beads + ibd];
    float dmult = pevalBeadParams[(BEAD_DMULT(dmult))*num_beads + ibd];
    float R = pevalBeadParams[(BEAD_R(R))*num_beads + ibd];
    float Copies = pevalBeadParams[(BEAD_COPIES(Copies))*num_beads + ibd];
    for (int fnum=0; fnum<NUMFB; ++fnum)
    {
      // calculate emphasis vector index
      Ampl = pevalBeadParams[(BEAD_AMPL(Ampl[0]) + fnum)*num_beads + ibd];
      kmult = pevalBeadParams[(BEAD_KMULT(kmult[0]) + fnum)*num_beads + ibd];

      int emphasisIdx = (int)(Ampl) > maxEmphasis ? maxEmphasis : (int)Ampl;
      int nucid = CP_MULTIFLOWFIT[sId].flowIdxMap[fnum];
  
      ComputeEtbR_dev(etbR, &CP_MULTIFLOWFIT[sId], R, sId, nucid, 0+fnum);
      ComputeTauB_dev(tauB, &CP_MULTIFLOWFIT[sId], etbR, sId);
      ComputeSP_dev(SP, &CP_MULTIFLOWFIT[sId], Copies, fnum, sId);
      ComputeHydrogenForMultiFlowFit_dev(sId, fnum, nucid, pnucRise, Ampl, 
        kmult*CP_MULTIFLOWFIT[sId].krate[nucid], gain, SP, 
        dmult*CP_MULTIFLOWFIT[sId].d[nucid], 
        ISIG_SUB_STEPS_MULTI_FLOW*CP_MULTIFLOWFIT[sId].start[fnum], 
        num_frames, num_beads, pival);
      ComputeSignalForMultiFlowFit_dev(restrict_clonal, sId, fnum, Ampl, tauB,
	etbR, gain, num_frames, num_beads, pnon_integer_penalty,
	pdarkMatterComp + num_frames*nucid, psbg, pival, pfval);    
      CalculateMultiFlowFitResidual_dev(newResidual, pobservedTrace, pfval, 
        emphasisVec + num_frames*emphasisIdx, fnum, num_beads, num_frames);
     } 
   
     newResidual = sqrtf(newResidual/(NUMFB*num_frames));

     if (newResidual < oldResidual)
     {
       // TODO change wrt to ampl*copies
       UpdateBeadParams_dev(pbeadParamsTranspose, pevalBeadParams, paramIdxMap, ibd, num_params, num_beads);
       lambda /= 30.0f; // it is LAMBDA_STEP in LevMarState.cpp
       if (lambda < FLT_MIN)
         lambda = FLT_MIN;
       plambda[ibd] = lambda;
       presidual[ibd] = newResidual;      
       done = true;
     }
     else
     {
       lambda *= 30.0f;
     }
     if (lambda >= 1E+10f)
     {
       done = true;
       plambda[ibd] = lambda;
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
  float* pRHS // pxN  
  )
{

  int ibd = blockIdx.x * blockDim.x + threadIdx.x; 
  if(ibd >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;
  pJTJ += ibd;
  for(int row=0;row<num_params;row++){
    for(int col = 0; col <= row; col++){

      *pJTJ =  CalculateJTJEntry( pDotProdMasks[row*num_params+col], 
                                  pPartialDeriv,  
                                  ibd,
                                  num_beads,
                                  num_frames );
      pJTJ += num_beads;
    }
  }
  pRHS += ibd;
  for(int row=0;row<num_params;row++){
    *pRHS = CalculateRHSEntry( pDotProdMasks[row*num_params+row], 
                                     pPartialDeriv,  
                                     ibd,
                                     num_steps,
                                     num_beads,
                                     num_frames );
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
  float* pRHS // pxN  
  )
{
  int ibd = blockIdx.x * (blockDim.x*2) + threadIdx.x*2; 
  if(ibd >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;

  unsigned int * masks = pDotProdMasks; 
  pJTJ += ibd;
  for(int row=0;row<num_params;row++){
    for(int col = 0; col <= row; col++){
      *((float2*)pJTJ) =  CalculateJTJEntryVec2(  masks[row*num_params+col], 
                                  pPartialDeriv,  
                                  ibd,
                                  num_beads,
                                  num_frames);
      pJTJ += num_beads;
    }
  }
  pRHS += ibd;
  for(int row=0;row<num_params;row++){
    *((float2*)pRHS) = CalculateRHSEntryVec2( masks[row*num_params+row], 
                                     pPartialDeriv,  
                                     ibd,
                                     num_steps,
                                     num_beads,
                                     num_frames );
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
  float* pRHS // pxN  
  )
{
  int ibd = blockIdx.x * (blockDim.x*4) + threadIdx.x*4; 

  
  extern  __shared__ unsigned int masks[];

  // load dotproduct masks to shared memory
  int i=threadIdx.x;
  while(i < num_params*num_params)
  {
    masks[i] = pDotProdMasks[i];
    i += blockDim.x;
  }
  __syncthreads(); 


  if(ibd >= num_beads) return;

  num_beads = ((num_beads+32-1)/32) * 32;
  //num_beads += 32 - (num_beads%32);

//  unsigned int * masks = pDotProdMasks;


  pJTJ += ibd;
  for(int row=0;row<num_params;row++){
    for(int col = 0; col <= row; col++){
      *((float4*)pJTJ) =  CalculateJTJEntryVec4( masks[row*num_params+col], 
                                  pPartialDeriv,  
                                  ibd,
                                  num_beads,
                                  num_frames);
      pJTJ += num_beads;
    }
  }
  pRHS += ibd;
  for(int row=0;row<num_params;row++){
    *((float4*)pRHS) = CalculateRHSEntryVec4( masks[row*num_params+row], 
                                     pPartialDeriv,  
                                     ibd,
                                     num_steps,
                                     num_beads,
                                     num_frames );
    pRHS += num_beads;
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


extern "C" void copySingleFlowFitConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream)
{
  cudaMemcpyToSymbolAsync ( CP_SINGLEFLOWFIT, ptr, sizeof(ConstParams), offset*sizeof(ConstParams),cudaMemcpyHostToDevice, stream);
}

extern "C" void copyMultiFlowFitConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream)
{
  cudaMemcpyToSymbolAsync ( CP_MULTIFLOWFIT, ptr, sizeof(ConstParams), offset*sizeof(ConstParams),cudaMemcpyHostToDevice, stream);
}






extern "C"
void  PerFlowLevMarFit_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  float* fg_buffers_base, // NxF
  float* emphasis, // F
  float* nucRise, 
  // bead params
//  float* pAmpl, // N
//  float* pKmult, // N
//  float* pdmult, // N
//  float* pR, // N
//  float* pgain, // N
//  float* pSP, // N
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
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId
) 
{
  cudaFuncSetCacheConfig(PerFlowLevMarFit_k, cudaFuncCachePreferL1);
  PerFlowLevMarFit_k<<< grid, block, smem, stream >>> (
    fg_buffers_base, // NxF
    emphasis,
    nucRise, 
//    pAmpl, // N
//    pKmult, // N
//    pdmult, // N
//    pR, // N
//    pgain, // N
    //pSP, // N
    pBeadParamsBase, //N
    pState,
    err, // NxF
    fval, // NxF
    tmp_fval, // NxF
    jac, // NxF 
    meanErr,
    minAmpl,
    maxKmult,
    minKmult,
    realFnum,
    num_beads, // 4
    num_frames, // 4
    useDynamicEmphasis,
//    pMonitor,
    sId);
}

///////// Pre-processing kernel (bkg correct and well params calculation);
extern "C"
void PreSingleFitProcessing_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,// Here FL stands for flows
  // inputs from data reorganization
  float* pCopies, // N
  float* pR, // N
  float* pdmult, // N
  float* pgain, // N
  float* pAmpl, // FLxN
  float* pkmult, // FLxN
  float* sbg, // FLxF 
  float* dark_matter, // FLxF
  float* fgbuffers, // FLxFxN
  // other inputs 
  int flowNum, // starting flow number to calculate absolute flow num
  int num_beads, // 4
  int num_frames, // 4
  int numEv, // 4
  // outputs
//  float* pSP, // FLxN
//  float* ptauB, 
  bool alternatingFit,
  int sId 
)
{
  PreSingleFitProcessing_k<<< grid, block, smem, stream >>>(
    pCopies, // N
    pR, // N
    pdmult, // N
    pgain, // N
    pAmpl, // FLxN
    pkmult, // FLxN
    sbg, // FLxF 
    dark_matter, // FLxF
    fgbuffers, // FLxFxN
    flowNum, // starting flow number to calculate absolute flow num
    num_beads, // 4
    num_frames, // 4
    numEv, // 4
    //pSP, // FLxN
    //ptauB, 
    alternatingFit,
    sId 
    );
}
/*
extern "C"
void  PerFlowAlternatingFit_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  float* fg_buffers_base, // NxF
  float* emphasis, // (MAX_HPLEN+1)xF
  float* nucRise, 
  // bead params
  float* pAmpl, // N
  float* pKmult, // N
  float* pdmult, // N
  float* ptauB, // N
  float* pgain, // N
  float* pSP, // N
  float * pCopies, //N
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
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  int sId
) 
{
  cudaFuncSetCacheConfig(PerFlowAlternatingFit_k, cudaFuncCachePreferL1);
  PerFlowAlternatingFit_k<<< grid, block, smem, stream >>> (
    fg_buffers_base, // NxF
    emphasis, 
    nucRise, 
    pAmpl, // N
    pKmult, // N
    pdmult, // N
    ptauB, // N
    pgain, // N
    pSP, // N
    pCopies, //N
    pState,
    err, // NxF
    fval, // NxF
    tmp_fval, // NxF
    meanErr,
    minAmpl,
    maxKmult,
    minKmult,
    realFnum,
    num_beads, // 4
    num_frames, // 4
    sId);
}
*/





//////// Computing Partial Derivatives
extern "C"
void ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_Wrapper(
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
  float* pemphasis, // MAX_HPLEN+1 xF // needs precomputation
  float* pnon_integer_penalty, // MAX_HPLEN
  float* pdarkMatterComp, // NUMNUC * F  
  float* pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep_t
  CpuStep_t* psteps, // we need a specific struct describing this config for this well fit for GPU
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
  int sId
)
{
  cudaFuncSetCacheConfig(ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_k, cudaFuncCachePreferL1);
  ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_k<<<grid,block,smem,stream>>>(
  // inputs
  maxEmphasis,
  restrict_clonal,
  pobservedTrace, 
  pival, // FLxNxF   //scatch
  pscratch_ival, // FLxNxF
  pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  psbg, // FLxF
  pemphasis, // MAX_HPLEN+1 xF // needs precomputation
  pnon_integer_penalty, // MAX_HPLEN
  pdarkMatterComp, // NUMNUC * F  
  pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep_t
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
  sId); 
} 



extern "C"
void BuildMatrix_Wrapper( dim3 grid, dim3 block, int smem, cudaStream_t stream, 
  float* pPartialDeriv, // S*FLxNxF   //scatch
  unsigned int * pDotProdMasks, // pxp
  int num_steps,
  int num_params,
  int num_beads,
  int num_frames,
  // outputs
  float* pJTJ, // pxpxN
  float* pRHS, // pxN  
  int vec
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
                  pRHS // pxN  
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
                  pRHS // pxN  
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
                  pRHS // pxN  
                  );


    }
}

extern "C"
void MultiFlowLevMarFit_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  int maxEmphasis,
  float restrict_clonal,
  float* pobservedTrace, 
  float* pival,
  float* pfval, // FLxNxFx2  //scratch for both ival and fval
  float* pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* psbg, // FLxF
  float* pemphasis, // MAX_HPLEN+1 xF // needs precomputation
  float* pnon_integer_penalty, // MAX_HPLEN
  float* pdarkMatterComp, // NUMNUC * F  
  float* pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep_t
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
  int sId
  )
{
  cudaFuncSetCacheConfig(MultiFlowLevMarFit_k, cudaFuncCachePreferL1);
  MultiFlowLevMarFit_k<<< grid ,block ,  smem, stream >>>(
    maxEmphasis,
    restrict_clonal,
    pobservedTrace,
    pival,
    pfval, // FLxNxFx2  //scratch for both ival and fval
    pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
    psbg, // FLxF
    pemphasis, // MAX_HPLEN+1 xF // needs precomputation
    pnon_integer_penalty, // MAX_HPLEN
    pdarkMatterComp, // NUMNUC * F  
    pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep_t
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
    sId);
}



extern "C"
void transposeData_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, float *source, int width, int height)
{
  transposeData_k<<< grid, block, smem, stream >>>( dest, source, width, height);
}

///////// Transpose Kernel
extern "C"
void transposeDataToFloat_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, FG_BUFFER_TYPE *source, int width, int height)
{
  transposeDataToFloat_k<<< grid, block, smem, stream >>>( dest,source,width,height);
}



extern "C" 
void initPoissonTables(int device, float ** poiss_cdf)
{

  cudaSetDevice(device);

  ///////// regular float version
  int poissTableSize = (MAX_HPLEN + 1) * MAX_POISSON_TABLE_ROW * sizeof(float);
  float * devPtr =NULL;
  cudaMalloc(&devPtr, poissTableSize); CUDA_ALLOC_CHECK(devPtr);
  cudaMemcpyToSymbol(POISS_APPROX_TABLE_CUDA_BASE , &devPtr  , sizeof (float*)); CUDA_ERROR_CHECK();
  for(int i = 0; i< (MAX_HPLEN+1); i++)
  {
    cudaMemcpy(devPtr, poiss_cdf[i], sizeof(float)*MAX_POISSON_TABLE_ROW, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    devPtr += MAX_POISSON_TABLE_ROW;
  }

#ifndef USE_CUDA_ERF
    cudaMemcpyToSymbol (ERF_APPROX_TABLE_CUDA, ERF_APPROX_TABLE, sizeof (ERF_APPROX_TABLE)); CUDA_ERROR_CHECK();
#endif


}


extern "C" 
void initPoissonTablesLUT(int device, void ** poissLUT)
{

  cudaSetDevice(device);
////////// float4/avx version
//  float4 ** pPoissLUT = (float4**) poissLUT;
#ifdef POISS_FLOAT4

  int poissTableSize =  (MAX_HPLEN + 2) * MAX_POISSON_TABLE_ROW * sizeof(float4);
  float4 * devPtrLUT = NULL;  
  cudaMalloc(&devPtrLUT, poissTableSize); CUDA_ALLOC_CHECK(devPtrLUT);
  cudaMemcpyToSymbol(POISS_APPROX_LUT_CUDA_BASE, &devPtrLUT  , sizeof (float4*)); CUDA_ERROR_CHECK();

#ifdef CREATE_POISSON_LUT_ON_DEVICE
  // run kernel to create LUT table from CDF tables on device
  dim3 block(512,1);
  dim3 grid (MAX_HPLEN+1, 1);
  build_poiss_LUT_k<<<grid, block >>>( ); 
  CUDA_ERROR_CHECK();
#else  
  // cast and copy host side __m128 SSE/AVX data to float4
  float4** pPoissLUT =(float4**)poissLUT;
  for(int i = 0; i< (MAX_HPLEN+2); i++)
  {
    cudaMemcpy(devPtrLUT, (float*)&pPoissLUT[i][0], sizeof(float4)*MAX_POISSON_TABLE_ROW, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    devPtrLUT += MAX_POISSON_TABLE_ROW;
  }
#endif
#endif

}


extern "C" 
void destroyPoissonTables(int device)
{
  cudaSetDevice(device);

  float * basepointer;

  cudaMemcpyFromSymbol (&basepointer,  POISS_APPROX_TABLE_CUDA_BASE , sizeof (float*)); CUDA_ERROR_CHECK();

  if(basepointer != NULL){
    cudaFree(basepointer); CUDA_ERROR_CHECK();
  }
}

extern "C"
void destroyPoissonTablesLUT(int device)
{
  cudaSetDevice(device);

  float * basepointerLUT;

  cudaMemcpyFromSymbol (&basepointerLUT,  POISS_APPROX_LUT_CUDA_BASE , sizeof (float*)); CUDA_ERROR_CHECK();

  if(basepointerLUT != NULL){
    cudaFree(basepointerLUT); CUDA_ERROR_CHECK();
  }
}















