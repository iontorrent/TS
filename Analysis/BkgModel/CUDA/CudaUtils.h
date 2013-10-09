/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CUDAUTILS_H
#define CUDAUTILS_H

// patch for CUDA5.0/GCC4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include "CudaConstDeclare.h"
#include "StreamingKernels.h"

__device__
float erf_approx_streaming (float x);

__device__
float poiss_cdf_approx_streaming (float x, const float* ptr);

__device__
float poiss_cdf_approx_float4 (float x, const float4* ptr, float occ_l, float occ_r);


extern __device__
const float*  precompute_pois_params_streaming (int n);

extern __device__
const float4*  precompute_pois_LUT_params_streaming (int il, int ir);


__device__
void clamp_streaming ( int &x, int a, int b);


__device__
void clamp_streaming ( double &x, double a, double b);


__device__
void clamp_streaming ( float &x, float a, float b);


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

  int max_dim_minus_one = MAX_POISSON_TABLE_ROW - 1;

  float idelta = x-left;

  if (left > max_dim_minus_one ){
    left = max_dim_minus_one;
//    idelta = 1.0f;
  }
  float ifrac = 1.0f-idelta;
 
#if __CUDA_ARCH__ >= 350
  float4 mixLUT = __ldg(ptr + left);
#else
  float4 mixLUT = ptr[left];
#endif
/*
      x = ptrR[right];
      y = ptrL[right]; 

      z = ptrR[left]; 
      w = ptrL[left]; 
*/

#ifndef CREATE_POISSON_LUT_ON_DEVICE
  //using _mm128 type casted as float4 --> reverse order of x,y,z,w ---> w,z,y,x
   ret = ( ifrac * ( occ_l * mixLUT.w + occ_r * mixLUT.z ) + idelta * (occ_l * mixLUT.y + occ_r * mixLUT.x )); 

/*
    if ( (left >= 0) && (left < MAX_POISSON_TABLE_ROW-1))
    {
      ifrac = (float) (x - left);
      ret = (float) ( (1.0f - ifrac) * mixLUT.z + ifrac * mixLUT.x) * occ_r ;
      ret += (float) ( (1.0f - ifrac) * mixLUT.w + ifrac * mixLUT.y) * occ_l;
    }
    else
    {
      if (left < 0){
        ret = (float) mixLUT.z * occ_r;
        ret += (float) mixLUT.w * occ_l;
      }else{
        ret = (float) mixLUT.x * occ_r ;
        ret += (float) mixLUT.y * occ_l ;
      }
    }


*/


  #else
   ret = ( ifrac * ( occ_l * mixLUT.x + occ_r * mixLUT.y ) + idelta * (occ_l * mixLUT.z + occ_r * mixLUT.w )); 
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
      ret = (float) ( (1.0f - frac) * ptr[left] + frac * ptr[right]);
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

__device__ 
void compare_poisson_cdf_table(const float * ptrR, const float* ptrL, const float4 * ptrLUT, float x)
{

  int left, right;
  x *= 20.0f;
  left = (int) x; // left-most point in the lookup table
  right = left + 1; // right-most point in the lookup table
  float4 cdf;
    if ( (left >= 0) && (right < MAX_POISSON_TABLE_ROW))
    {
      cdf.x = ptrR[right];
      cdf.y = ptrL[right]; 
      cdf.z = ptrR[left]; 
      cdf.w = ptrL[left]; 
    }
    else
    {
      if (left < 0){
        cdf.x = ptrR[0];
        cdf.y = ptrL[0];
        cdf.z = ptrR[0]; 
        cdf.w = ptrL[0]; 
      }else{
        cdf.x = ptrR[MAX_POISSON_TABLE_ROW - 1];
        cdf.y = ptrL[MAX_POISSON_TABLE_ROW - 1];
        cdf.z = ptrR[MAX_POISSON_TABLE_ROW - 1];
        cdf.w = ptrL[MAX_POISSON_TABLE_ROW - 1];
   }
  }
  
  int max_dim_minus_one = MAX_POISSON_TABLE_ROW - 1;
  left = (left < max_dim_minus_one )?(left):(max_dim_minus_one);
#if __CUDA_ARCH__ >= 350
  float4 mixLUT = __ldg(ptrLUT + left);
#else
  float4 mixLUT = ptrLUT[left];
#endif
                    
if( cdf.x != mixLUT.x || cdf.y != mixLUT.y || cdf.z != mixLUT.z || cdf.w != mixLUT.w  )
   printf("ERROR %d %d: %f %f %f %f, %f %f %f %f\n", left, right, cdf.x,cdf.y,cdf.z,cdf.w, mixLUT.x,mixLUT.y,mixLUT.z,mixLUT.w); 

//if( cdf.x == mixLUT.x && cdf.y == mixLUT.y && cdf.z == mixLUT.z && cdf.w == mixLUT.w  )
//   printf("PASS %d %d: %f %f %f %f, %f %f %f %f\n", left, right, cdf.x,cdf.y,cdf.z,cdf.w, mixLUT.x,mixLUT.y,mixLUT.z,mixLUT.w); 


}


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

__device__
float ApplyDarkMatterToFrame(
    float * darkMatter,
    float * pca_val,
    int frame,
    int num_frames,
    int num_beads,  
    int sId 
  )
{
  if( !CP[sId].useDarkMatterPCA)
    return darkMatter[frame]*CP[sId].darkness[0]; //CP_MULTIFLOWFIT

  float val = 0;
  for(int i=0; i<NUM_DM_PCA; i++)
    val += darkMatter[i*num_frames+frame]*pca_val[i*num_beads];

  return val;
}

// compute tmid muc. This routine mimics CPU routine in BookKeeping/RegionaParams.cpp
__device__
void ComputeMidNucTime_dev(float& tmid, const ConstParams* pCP, int nucId, int fnum) {
  tmid = pCP->nuc_shape.t_mid_nuc[0];
  tmid +=  pCP->nuc_shape.t_mid_nuc_delay[nucId]*
          (pCP->nuc_shape.t_mid_nuc[0] -  pCP->nuc_shape.valve_open) /
          ( pCP->nuc_shape.magic_divisor_for_timing + SAFETYZERO);
  tmid +=  pCP->nuc_shape.t_mid_nuc_shift_per_flow[fnum];

}

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
  if (CP[sId].fit_taue) { //CP_MULTIFLOWFIT
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



#endif // CUDAUTILS_H
