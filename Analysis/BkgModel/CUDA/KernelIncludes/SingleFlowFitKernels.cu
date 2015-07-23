/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#include "SingleFlowFitKernels.h"
#include "MathModel/PoissonCdf.h"
#include "cuda_error.h"
#include "MathModel/PoissonCdf.h"
#include "ImgRegParams.h"
#include "UtilKernels.h"
#include "ConstantSymbolDeclare.h"

//#define __CUDA_ARCH__ 350

namespace { 
  enum ModelFuncEvaluationOutputMode { NoOutput, OneParam, TwoParams };
}


/*__device__
const float4*  precompute_pois_LUT_params_SingelFLowFit (int il, int ir)
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
float poiss_cdf_approx_float4_SingelFLowFit (float x, const float4* ptr, float occ_l, float occ_r)
{
  float ret;
  x *= 20.0f;
  int left = (int) x;

  int max_dim_minus_one = MAX_POISSON_TABLE_ROW - 1;

  float idelta = x-left;

  if (left > max_dim_minus_one ){
    left = max_dim_minus_one;

  }
  float ifrac = 1.0f-idelta;

  float4 mixLUT = LDG_ACCESS(ptr, left);

  ret = ( ifrac * ( occ_l * mixLUT.w + occ_r * mixLUT.z ) + idelta * (occ_l * mixLUT.y + occ_r * mixLUT.x ));

  return ret;
}
*/






// smoothing kernel to provide weights for smoothing exponential tail 
__device__
void GenerateSmoothingKernelForExponentialTailFit_dev(
    const float* frameNumber,
    const int size,
    const float taubInv,
    const int exp_start,
    float* kern
)
{
  float dt;
  for (int i=0; i<size; ++i) {
    dt = (frameNumber[i+exp_start] - frameNumber[exp_start + 3])*taubInv;
    kern[i] = __expf(dt);   
  }
}


/*
__device__
float ApplyDarkMatterToFrame(
    const float* beadParamCube,
    const float* regionFrameCube,
    const float darkness,
    const int frame,
    const int num_frames,
    const int frameStride,
    const int regionFrameStride)
{

  if( !ConfigP.UseDarkMatterPCA() )
    return ((*(regionFrameCube + (RfDarkMatter0 + ConstFlowP.getNucId())*regionFrameStride + frame))
        *darkness);

  float val = 0;

  regionFrameCube += RfDarkMatter0*regionFrameStride + frame;  //RfDarkMatter0
  beadParamCube += BpPCAval0*frameStride;  //BpPCAval0
  val += (*regionFrameCube) * (*beadParamCube);
  regionFrameCube += regionFrameStride; //RfDarkMatter1
  beadParamCube += frameStride; //BpPCAval1
  val += (*regionFrameCube) * (*beadParamCube);
  regionFrameCube += regionFrameStride; //RfDarkMatter2
  beadParamCube += frameStride; //BpPCAval2
  val += (*regionFrameCube) * (*beadParamCube);
  regionFrameCube += regionFrameStride; //RfDarkMatter3
  beadParamCube += frameStride; //BpPCAval3
  val += (*regionFrameCube) * (*beadParamCube);

  return val;
}

// compute tmid muc. This routine mimics CPU routine in BookKeeping/RegionaParams.cpp
__device__
float ComputeMidNucTime(
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP
)
{
  float tmid = perFlowRegP->getTMidNuc();
  tmid +=  perNucRegP->getTMidNucDelay()*
      (perFlowRegP->getTMidNuc() -  ConstGlobalP.getValveOpen()) /
      ( ConstGlobalP.getMagicDivisorForTiming() + SAFETYZERO);
  tmid +=  perFlowRegP->getTMidNucShift();

  return tmid;
}



__device__ 
float ComputeETBR(
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const float R,
    float copies
) {

  float etbR;

  if (ConfigP.FitTauE()) {
    etbR = R;
    if (etbR)
      etbR = perNucRegP->getNucModifyRatio() /(perNucRegP->getNucModifyRatio() +
          (1.0f - (perFlowRegP->getRatioDrift() * (ConstFlowP.getRealFnum())/SCALEOFBUFFERINGCHANGE))*
          (1.0f / etbR - 1.0f));
  }
  else {
    if ( !ConfigP.UseAlternativeEtbRequation()) {
      etbR = R*perNucRegP->getNucModifyRatio() +
          (1.0f - R*perNucRegP->getNucModifyRatio())*
          perFlowRegP->getRatioDrift()*(ConstFlowP.getRealFnum())/SCALEOFBUFFERINGCHANGE;
    }
    else {
      etbR = R*perNucRegP->getNucModifyRatio() +
          perFlowRegP->getRatioDrift()*copies*(ConstFlowP.getRealFnum())/(6.0*SCALEOFBUFFERINGCHANGE);
    }
  }
  return etbR;
}

__device__
float ComputeTauB( 
    const ConstantParamsRegion * constRegP,
    const float etbR) {

  float tauB;
  if (ConfigP.FitTauE()) {
    tauB = etbR  ? (constRegP->getTauE() / etbR) : ConstGlobalP.getMinTauB();
  }
  else {
    tauB = constRegP->getTauRM()*etbR + constRegP->getTauRO();
  }

  clampT(tauB, ConstGlobalP.getMinTauB(), ConstGlobalP.getMaxTauB());

  return tauB;
}

__device__
float ComputeSP(
    const PerFlowParamsRegion * perFlowRegP,
    const float copies
) {
  return ((float)(COPYMULTIPLIER * copies) * pow(perFlowRegP->getCopyDrift(), ConstFlowP.getRealFnum()));
}
*/

__device__ 
float BlockLevel_DecideOnEmphasisVectorsForInterpolation(
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

__device__ void
Fermi_ModelFuncEvaluationForSingleFlowFit(
    const ConstantParamsRegion * constRegP,
    const PerNucParamsRegion * perNucRegP,
    const int startFrame,
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
    const float * const deltaFrame,
    int endFrames
)
{
  float sens1 = sens_in;

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
  const float4 * LUTptr1 = precompute_pois_LUT_params_SingelFLowFit (ileft1, iright1);
#endif

#ifndef POISS_FLOAT4
  const float* rptr2 = precompute_pois_params_streaming (iright2);
  const float* lptr2 = precompute_pois_params_streaming (ileft2);
#else
  const float4 * LUTptr2 = precompute_pois_LUT_params_SingelFLowFit (ileft2, iright2);
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
  const float cp_sid_kmax_nucid = perNucRegP->getKmax();

  float c_dntp_sum1 = 0.0;
  float c_dntp_new_rate1 = 0;

  const float scaled_kr1 = Krate1*constRegP->getMoleculesToMicromolarConversion()/d; //CP_SINGLEFLOWFIT

  float red_hydro_prev1;

  float c_dntp_bot_plus_kmax1 = 1.0f/cp_sid_kmax_nucid; //CP_SINGLEFLOWFIT

  float c_dntp_sum2 = 0.0;
  float c_dntp_new_rate2 = 0;

  float fval_local1 = 0.f;
  float fval_local2 = 0.f;

  const float scaled_kr2 = Krate2*constRegP->getMoleculesToMicromolarConversion()/d; //CP_SINGLEFLOWFIT

  float red_hydro_prev2;

  float c_dntp_bot_plus_kmax2 = 1.0f/cp_sid_kmax_nucid; //CP_SINGLEFLOWFIT

  for (int i=startFrame;i < endFrames; i++) //CP_SINGLEFLOWFIT
  {
    float delta_frame = deltaFrame[i];

    float red_hydro1 = totocc1;
    float red_hydro2 = totocc2;

    // Move memory fetches well ahead of where they're used.    
    const float fval_in_i = fval_in[i];

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

          float pact_new1 = poiss_cdf_approx_float4_SingelFLowFit(c_dntp_sum1, LUTptr1, occ_l1, occ_r1);

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

          float pact_new2 = poiss_cdf_approx_float4_SingelFLowFit(c_dntp_sum2, LUTptr2, occ_l2, occ_r2);

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

    float err_bxi = err[i]; // Grab this early so that we only get it once.

    // calculate the 'background' part (the accumulation/decay of the protons in the well
    // normally accounted for by the background calc)

    red_hydro1 *= sens1;  

    // variables used for solving background signal shape
    const float one_over_two_tauB = 1.0f/ (2.0f*tau);
    const float aval = delta_frame*one_over_two_tauB; //CP_SINGLEFLOWFIT
    const float one_over_one_plus_aval = 1.0f/ (1.0f+aval);

    if( i == startFrame ) //CP_SINGLEFLOWFIT
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

      if( i == startFrame ) //CP_SINGLEFLOWFIT
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


__device__ void
Keplar_ModelFuncEvaluationForSingleFlowFit(
    const ConstantParamsRegion * constRegP,
    const PerNucParamsRegion * perNucRegP,
    const bool twoParamFit,
    const int startFrame,
    const float * nucRise,
    float A,
    const float Krate,
    const float tau,
    const float gain,
    const float SP,
    const float d,
    float sens,
    int c_dntp_top_ndx,
    float* fval,
    const float* deltaFrame,
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

  const float4 * LUTptr = precompute_pois_LUT_params_SingelFLowFit (ileft, iright);


  // We reuse this constant every loop...
  float cp_sid_kmax_nucid = perNucRegP->getKmax();


  float c_dntp_bot = 0.0; // concentration of dNTP in the well
  float c_dntp_sum = 0.0;
  float c_dntp_old_rate = 0;
  float c_dntp_new_rate = 0;

  float scaled_kr = Krate*constRegP->getMoleculesToMicromolarConversion()/d; //CP_SINGLEFLOWFIT
  float half_kr = Krate*0.5f;

  // variables used for solving background signal shape
  float aval = 0.0f;

  //new Solve HydrogenFlowInWell

  float one_over_two_tauB = 1.0f;
  float one_over_one_plus_aval = 1.0f/ (1.0f+aval);
  float red_hydro_prev; 
  float fval_local  = 0.0f;

  float red_hydro;

  float c_dntp_bot_plus_kmax = 1.0f/cp_sid_kmax_nucid; //CP_SINGLEFLOWFIT

  for (i=startFrame;i < endFrames; i++) //CP_SINGLEFLOWFIT
  {
    if (totgen > 0.0f)
    {
      ldt = (deltaFrame[i]/( ISIG_SUB_STEPS_SINGLE_FLOW * FRAMESPERSEC)) * half_kr; //CP_SINGLEFLOWFIT
      for (st=1; (st <= ISIG_SUB_STEPS_SINGLE_FLOW) && (totgen > 0.0f);st++)
      {
        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;

        // All the threads should be grabbing from the same nucRise location.
        c_dntp_bot = LDG_ACCESS(nucRise, c_dntp_top_ndx++) / (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + cp_sid_kmax_nucid); //CP_SINGLEFLOWFIT

        c_dntp_new_rate = c_dntp_bot*c_dntp_bot_plus_kmax;
        c_dntp_int = ldt* (c_dntp_new_rate+c_dntp_old_rate);
        c_dntp_sum += c_dntp_int;

        // calculate new number of active polymerase

        pact_new = poiss_cdf_approx_float4_SingelFLowFit(c_dntp_sum, LUTptr, occ_l, occ_r);

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
    aval = LDG_ACCESS(deltaFrame, i)*one_over_two_tauB;
    one_over_one_plus_aval = 1.0f/ (1.0f+aval);

    if(i==startFrame) //CP_SINGLEFLOWFIT
      fval_local  = red_hydro; // *one_over_one_plus_aval;
    else
      fval_local = red_hydro - red_hydro_prev + (1.0f-aval)*fval_local; // *one_over_one_plus_aval;

    red_hydro_prev = red_hydro;

    fval_local *=  one_over_one_plus_aval;

    switch( flag ) {
      case NoOutput:
        fval[i] = fval_local * gain;
        break;

      case OneParam:
      case TwoParams:

        float weight = emRight != NULL ? frac*emLeft[i*(MAX_POISSON_TABLE_COL)] + (1.0f - frac)*emRight[i*(MAX_POISSON_TABLE_COL)] : emLeft[i*(MAX_POISSON_TABLE_COL)];

        float err_bxi = err[i]; // Grab this early so that we only get it once.
        float jac_tmp =  weight * (fval_local*gain - fval_in[i]) * 1000.0f;
        if(flag==OneParam){
          jac_out[i] = jac_tmp;
          *aa += jac_tmp * jac_tmp;
          if (!twoParamFit)
            *rhs0 += (jac_tmp * err_bxi);
        }
        else {            // Two params.
          float my_jac_out = jac_out[i];           // Only grab it from memory once.
          *akr +=  my_jac_out * jac_tmp;
          *rhs0 += my_jac_out * err_bxi;
          *rhs1 += jac_tmp * err_bxi;
          *krkr += jac_tmp * jac_tmp;
        }
        break;
    }
  }
}

/*
__device__ void
BkgModelRedTraceCalculation(
    const ConstantParamsRegion * constRegP,
    const PerNucParamsRegion * perNucRegP,
    const int startFrame,
    const float * nucRise,
    float A,
    const float Krate,
    const float tau,
    const float gain,
    const float SP,
    const float d,
    float sens,
    int c_dntp_top_ndx,
    float * fval,
    const float* deltaFrame,
    int endFrames
)
{
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


  const float4 * LUTptr = precompute_pois_LUT_params_SingelFLowFit (ileft, iright);

  float totocc = SP*A;
  float totgen = totocc;

  // We reuse this constant every loop...
  float cp_sid_kmax_nucid = perNucRegP->getKmax();

  float c_dntp_sum = 0.0;
  float c_dntp_old_rate = 0;
  float c_dntp_new_rate = 0;

  float scaled_kr = Krate*constRegP->getMoleculesToMicromolarConversion()/d;
  float half_kr = Krate*0.5f;

  // variables used for solving background signal shape
  float aval = 0.0f;

  //new Solve HydrogenFlowInWell

  float one_over_two_tauB = 1.0f;
  float one_over_one_plus_aval = 1.0f/ (1.0f+aval);
  float red_hydro_prev; 
  float fval_local  = 0.0f;

  float red_hydro;

  float c_dntp_bot_plus_kmax = 1.0f/cp_sid_kmax_nucid;

  bool start_frame = true;
  for (int i=startFrame;i < endFrames;i++)
  {
    if (totgen > 0.0f)
    {
      float ldt = (deltaFrame[i]/( ISIG_SUB_STEPS_SINGLE_FLOW * FRAMESPERSEC)) * half_kr;
      for (int st=1; (st <= ISIG_SUB_STEPS_SINGLE_FLOW) && (totgen > 0.0f);st++)
      {
        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;

        // All the threads should be grabbing from the same nucRise location.
        // c_dntp_bot is the concentration of dNTP in the well
        float c_dntp_bot = LDG_ACCESS(nucRise, c_dntp_top_ndx++) / (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + cp_sid_kmax_nucid);

        c_dntp_new_rate = c_dntp_bot*c_dntp_bot_plus_kmax;
        float c_dntp_int = ldt* (c_dntp_new_rate+c_dntp_old_rate);
        c_dntp_sum += c_dntp_int;

        // calculate new number of active polymerase
        float pact_new = poiss_cdf_approx_float4_SingelFLowFit(c_dntp_sum, LUTptr, occ_l, occ_r);


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
    aval = deltaFrame[i]*one_over_two_tauB; //CP_SINGLEFLOWFIT
    one_over_one_plus_aval = 1.0f/ (1.0f+aval);

    if(start_frame) { //CP_SINGLEFLOWFIT
      fval_local = red_hydro; // *one_over_one_plus_aval;
      start_frame = false;
    } else {
      fval_local = red_hydro - red_hydro_prev + (1.0f-aval)*fval_local; // *one_over_one_plus_aval;
    }

    red_hydro_prev = red_hydro;

    fval_local *=  one_over_one_plus_aval;

    fval[i] = fval_local * gain;  
  }
}
*/

__device__ 
void ZeromerCorrectionFromRawTrace(
    const float* bkgTrace,
    const short* rawTrace,
    const float* beadParamCube,
    const float* regionFrameCube,
    const float* deltaFrames,
#if FG_TRACES_REZERO
    const float dcOffset,
#endif
    const float darkness,
    const float etbR,
    const float gain,
    const float tauB,
    const int num_frames,
    const int frameStride,
    const int regionFrameStride,
    float* correctedTrace
)
{
  float R = etbR - 1.0f;
  float dv = 0.0f;
  float dv_rs = 0.0f;
  float dvn = 0.0f;
  float aval;
  float curSbgVal, deltaFrameVal;
  //  printf("fg after PreSingleFit\n"); //T*** REMOVE!!  DEBUG ONLY
  for (int i=0; i<num_frames; ++i) {
    deltaFrameVal = LDG_ACCESS(deltaFrames, i);
#ifdef EMPTY_TRACES_REZERO_SHARED
    curSbgVal = bkgTrace[i];
#else
    curSbgVal = LDG_ACCESS(bkgTrace, i);
#endif
    aval = deltaFrameVal/(2.0f * tauB);
    dvn = (R*curSbgVal - dv_rs/tauB - dv*aval) / (1.0f + aval);
    dv_rs += (dv+dvn) * deltaFrameVal * 0.5f;
    dv = dvn;
    correctedTrace[i] = (float)(*rawTrace)
#if FG_TRACES_REZERO
                            - dcOffset
#endif
                            - ((dv+curSbgVal)*gain
                                + ApplyDarkMatterToFrame( beadParamCube,
                                    regionFrameCube,
                                    darkness,
                                    i,
                                    num_frames,
                                    frameStride,
                                    regionFrameStride
                                )

                            );
    rawTrace += frameStride;
    //    printf("%f ",correctedTrace[i] ); //T*** REMOVE!!  DEBUG ONLY
  }
  //  printf("\n"); //T*** REMOVE!!  DEBUG ONLY
}


__device__ 
void ExponentialTailFitCorrection(
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const float* bkgTrace,
    const float* frameNumber,
    const float Ampl,
    const float adjustedTauB,
    const int num_frames,
    float* correctedTrace
)
{
  float kern[7];

  if (adjustedTauB > 0.0f) { 

    float tmid = ComputeMidNucTime(perFlowRegP->getTMidNuc(), perFlowRegP, perNucRegP);

    // set up start and end point for exponential tail
    float tail_start = tmid + 6.0f + 1.75f * Ampl;
    int tail_start_idx = -1, tail_end_idx = -1;
    for (int i=0; i<num_frames; ++i) {
      if ((tail_start_idx == -1) && frameNumber[i] >= tail_start)
        tail_start_idx = i;
      if ((tail_end_idx == -1) && frameNumber[i] >= (tail_start + 60.0f))
        tail_end_idx = i;
    }

    if (tail_start_idx == -1)
      return;

    if (tail_end_idx == -1)
      tail_end_idx = num_frames;

    // too few points
    int tailLen = tail_end_idx - tail_start_idx;
    if (tailLen >= 5) {

      // Generate smoothing kernel vector. Distance from the point is +/- 3 so need
      // 7 weights
      int exp_kern_start = tailLen < 7 ? (tail_end_idx - 7) : tail_start_idx;
      float taubInv = 1.0f / adjustedTauB;
      GenerateSmoothingKernelForExponentialTailFit_dev(
          frameNumber,
          7, 
          taubInv, 
          exp_kern_start, 
          kern); //CP_SINGLEFLOWFIT

      // perform kernel smoothing on exponential tail
      float avg_bkg_amp_tail = 0;
      float lhs_01=0,lhs_11=0, rhs_0=0, rhs_1=0;
      for (int i=tail_start_idx; i<tail_end_idx; ++i) {
        float sum=0,scale=0;
        for (int j=i-3, k=0; j <= (i+3); ++j, ++k) {
          if (j >= 0 && j < num_frames) {
            sum += (kern[k] * correctedTrace[j]);
            scale += kern[k];
          }
        }
        float tmp_fval = sum / scale;
#ifdef EMPTY_TRACES_REZERO_SHARED
        avg_bkg_amp_tail += bkgTrace[i];
#else
        avg_bkg_amp_tail += LDG_ACCESS(bkgTrace, i);
#endif



        // linear regression to calculate A and C in Aexp(-(t-t0)/taub) + C
        // First calculate lhs and rhs matrix entries which are obtained by taking
        // derivative of the squared residual (y - (Aexp(-(t-t0)/taub) + C))^2 w.r.t
        // A and C to 0 which gives two linear equations in A and C
        float expval = __expf((-frameNumber[i] +
            frameNumber[tail_start_idx])*taubInv);
        lhs_01 += expval;
        lhs_11 += expval*expval;
        rhs_0 += tmp_fval;
        rhs_1 += tmp_fval*expval;  
      }
      float detInv = 1.0f / (tailLen*lhs_11 - lhs_01*lhs_01);
      float C = (lhs_11*rhs_0 - lhs_01*rhs_1) * detInv;
      float A = (-lhs_01*rhs_0 + tailLen*rhs_1) * detInv;

      // if negative  then no incorporation
      if (A < -20.0f) {
        C = rhs_0 / tailLen;
      }

      avg_bkg_amp_tail /= tailLen;
      if (avg_bkg_amp_tail) 
        C /= avg_bkg_amp_tail;

      //      printf("fg after exptail: \n");//T*** REMOVE!!  DEBUG ONLY
      for (int i=0; i<num_frames; ++i) {
#ifdef EMPTY_TRACES_REZERO_SHARED
        correctedTrace[i] -= C*bkgTrace[i];
#else
        correctedTrace[i] -= C*LDG_ACCESS(bkgTrace, i);
#endif
        //        printf("%f ", correctedTrace[i]); //T*** REMOVE!!  DEBUG ONLY
      }
      //     printf("\n"); //T*** REMOVE!!  DEBUG ONLY
    }
  }
}

/*__device__
float ProjectionSearch(
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const float* observedTrace,
    const float* emphasisVec,
    const int * nonZeroEmphFrames,
    const float* nucRise,
    const float* deltaFrames,
    const float kmult,
    const float d,
    const float tauB,
    const float gain,
    const float SP,
    float* tmp_fval
    //bool print
)
{
  float Ampl = 1.0f;

  for (int i=0; i<2; ++i) { //TODO invariant code motion?


    BkgModelRedTraceCalculation(
        constRegP,
        perNucRegP,
        perFlowRegP->getStart(),
        nucRise, 
        Ampl, 
        kmult*perNucRegP->getKrate(),
        tauB, 
        gain, 
        SP, 
        d, 
        constRegP->getSens()*SENSMULTIPLIER,
        ISIG_SUB_STEPS_SINGLE_FLOW * perFlowRegP->getStart(),
        tmp_fval, 
        deltaFrames, 
        nonZeroEmphFrames[0]);


    float num = 0, den = 0.0001f;
    float emphasisVal;
    for (int j=perFlowRegP->getStart(); j<nonZeroEmphFrames[0]; ++j) {
      emphasisVal = emphasisVec[j*MAX_POISSON_TABLE_COL] * emphasisVec[j*MAX_POISSON_TABLE_COL];
      num += tmp_fval[j]*observedTrace[j]*emphasisVal; // multiply by emphasis vectors
      den += tmp_fval[j]*tmp_fval[j]*emphasisVal;
    }
    Ampl *= (num/den);
    if (isnan(Ampl))
      Ampl = 1.0f;
    else
      clampT(Ampl, 0.001f, (float)LAST_POISSON_TABLE_COL);
  }
  return Ampl;
}*/

__device__ float 
ResidualCalculationPerFlow(
    const int startFrame,
    const float* rawTrace,
    const float* fval,
    const float* emLeft,
    const float* emRight,
    const float frac,
    float* err,
    const int nonZeroEmpFrames) {

  float e;  
  float weight;
  float wtScale = 0;
  float residual = 0;
  int i;

  for (i=0; i<nonZeroEmpFrames; ++i) {
    weight = (emRight != NULL) ?( frac* (*emLeft) + (1.0f - frac)*emRight[i*(MAX_POISSON_TABLE_COL)]) :( (*emLeft));

    if (i < startFrame)
      e = weight * rawTrace[i];
    else
      err[i] = e = weight * (rawTrace[i] - fval[i]);

    residual += e*e;
    wtScale += weight*weight;
    emLeft += (MAX_POISSON_TABLE_COL);
  }
  residual /= wtScale;

  return residual;
}

__device__ 
float CalculateMeanResidualErrorPerFlow(
    const int startFrame,
    const float* rawTrace,
    const float* fval,
    const float* weight, // highest hp weighting emphasis vector
    const int num_frames)
{
  float wtScale = 0.0f;
  float residual = 0;
  float e;
  float wt;
  for (int i=0; i<num_frames; ++i) {

    wt = *weight;
    wtScale += wt * wt;

    if (i < startFrame)
      e = wt * rawTrace[i];
    else
      e = wt * (rawTrace[i] - fval[i]);

    residual += e*e;

    weight += MAX_POISSON_TABLE_COL;
  }
  residual = sqrtf(residual/wtScale);

  return residual;
}



//global PLimits containing limits
//global Image and Region params in constant as ImgRegP
//global config flags are available in constant mem symbol: ConfigP

__device__ void
SingleFlowFitUsingRelaxKmultGaussNewton(
    float* ResultCube, //Ampl, kmult, avg_error, points to correct Ampl value, stride == frameStride
    //per bead
    //in parameters
    const unsigned short * BStateMask, //needed to update corrupt state
    const short * RawTraces, // imgW*imgHxF
    const float * BeadParamCube, //Copies, R, dmult, gain, tau_adj, phi, stride == frameStride
    const float* emphasisVec, //(MAX_POISSON_TABLE_COL)*F
    const int * nonZeroEmphFrames,
    const float* nucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
    //per region
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const float * RegionFrameCube,  //bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
    const float * EmptyTraceAvg,  //bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
#if FG_TRACES_REZERO
    const float dcOffset,
#endif
    // other inputs
    const size_t num_frames, // 4
    const size_t frameStride, //stride from one CUBE plane to the next for the Per Well Cubes
    const size_t regionFrameStride,//, //stride in Region Frame Cube to get to next parameter
    const size_t emphStride
    //bool print
    //int * maxIterWarp = NULL
    //   float * fgBufferFloat
)
{
  float correctedTrace[MAX_COMPRESSED_FRAMES_GPU];
  float fval[MAX_COMPRESSED_FRAMES_GPU];
  float tmp_fval[MAX_COMPRESSED_FRAMES_GPU];
  float err[MAX_COMPRESSED_FRAMES_GPU];
#if __CUDA_ARCH__ >= 350
  float jac[MAX_COMPRESSED_FRAMES_GPU];
#endif

  // right now getting bead params in the order they were in bead_params struct
  const float copies = *(BeadParamCube + BpCopies*frameStride);
  const float R = *(BeadParamCube + BpR*frameStride);
  const float d = (*(BeadParamCube + BpDmult*frameStride)) * perNucRegP->getD(); // effective diffusion
  const float gain = *(BeadParamCube + BpGain*frameStride);


  *(ResultCube + ResultKmult*frameStride) = 1.0f;
  float kmult = 1.0f; //*(ResultCube + ResultKmult*frameStride);

  // calculate empty to bead ratio and buffering 
  const float etbR = ComputeETBR(perNucRegP, perFlowRegP->getRatioDrift(), R, copies);
  const float tauB = ComputeTauB(constRegP, etbR);
  const float SP = ComputeSP(perFlowRegP->getCopyDrift(), copies);


  const float* bkgTrace = EmptyTraceAvg;//RegionFrameCube + RfBkgTraces*regionFrameStride;
  //const float* bkgTrace = RegionFrameCube + RfBkgTraces*regionFrameStride;
  const float* deltaFrames = RegionFrameCube + RfDeltaFrames*regionFrameStride;

  // zeromer correction
  ZeromerCorrectionFromRawTrace(
      bkgTrace,
      RawTraces,
      BeadParamCube,
      RegionFrameCube,
      deltaFrames,
#if FG_TRACES_REZERO
      dcOffset,
#endif
      perFlowRegP->getDarkness(),
      etbR,
      gain,
      tauB,
      num_frames,
      frameStride,
      regionFrameStride,
      correctedTrace);



  // projection search for initial ampl estimates
  float Ampl = ProjectionSearch(
      constRegP,
      perFlowRegP,
      perNucRegP,
      correctedTrace,
      emphasisVec,
      nonZeroEmphFrames[0],
      nucRise,
      deltaFrames,
      kmult,
      d,
      tauB,
      gain,
      SP,
      tmp_fval,
      perFlowRegP->getStart(),
      frameStride,
      emphStride,
      ISIG_SUB_STEPS_SINGLE_FLOW
      //print
  );


#if !PROJECTION_ONLY
  if( Match(BStateMask,BkgMaskPolyClonal)){
#endif
    *(ResultCube + ResultAmpl*frameStride) = Ampl;
    return;
#if !PROJECTION_ONLY
  }
#endif



  // exponential tail fit
  if(true){ //ConfigP.PerformExpTailFitting()){

    const float adjustTauB = tauB * (*(BeadParamCube + BpTauAdj*frameStride));
    const float* frameNumber = RegionFrameCube + RfFrameNumber*regionFrameStride;
    ExponentialTailFitCorrection(
        perFlowRegP,
        perNucRegP,
        bkgTrace,
        frameNumber,
        Ampl,
        adjustTauB,
        num_frames,
        correctedTrace);


    // TODO
    // recompress tail of raw trace
  }


  //used and done with
  // copies
  // R
  // etbR
  // adjustTauB
  //


  //used and used again

  // tmp_fval  (gets overwritten in modelfunction dump would not be needed)
  // correctedTrace,

  // d
  // gain
  // kmult
  // tauB
  // SP
  // Ampl

  // not used yet
  // fval
  // err
  // jac


  // perform gauss newton fit
  deltaFrames = ConfigP.PerformRecompressTailRawTrace() ?
      RegionFrameCube + RfDeltaFramesStd*regionFrameStride :
      RegionFrameCube + RfDeltaFrames*regionFrameStride;
  const int* nonZeroEmpFramesVec = (ConfigP.PerformRecompressTailRawTrace())?
      (nonZeroEmphFrames + ImgRegP.getNumRegions()*MAX_POISSON_TABLE_COL):(nonZeroEmphFrames);

  float localMinKmult = ConstGlobalP.getMinKmult();
  float localMaxKmult= ConstGlobalP.getMaxKmult();

  const bool twoParamFit = ConfigP.FitKmult() || ( copies * Ampl > ConstGlobalP.getAdjKmult() );

  float residual, newresidual;
  // These values before start are always zero since there is no nucrise yet. Don't need to
  // zero it out. Have to change the residual calculation accordingly for the frames before the
  // start.

  float sens = constRegP->getSens() * SENSMULTIPLIER;
  int relax_kmult_pass = 0;
  int startFrame = perFlowRegP->getStart();


  while (relax_kmult_pass < 2)
  {
    // first step
    // Evaluate model function using input Ampl and Krate and get starting residual
#if __CUDA_ARCH__ >= 350
    Keplar_ModelFuncEvaluationForSingleFlowFit(
        constRegP,
        perNucRegP,
        twoParamFit,
        startFrame,
        nucRise, 
        Ampl, 
        kmult*perNucRegP->getKrate(),
        tauB, 
        gain, 
        SP, 
        d,
        sens, 
        ISIG_SUB_STEPS_SINGLE_FLOW*startFrame,
        fval, 
        deltaFrames, 
        num_frames, 
        NoOutput);
#else
    BkgModelRedTraceCalculation(
        constRegP,
        perNucRegP,
        startFrame,
        nucRise, 
        Ampl, 
        kmult*perNucRegP->getKrate(),
        tauB, 
        gain, 
        SP, 
        d,
        sens, 
        ISIG_SUB_STEPS_SINGLE_FLOW * startFrame,
        fval, 
        deltaFrames, 
        ISIG_SUB_STEPS_SINGLE_FLOW,
        num_frames);
#endif
    const float *emLeft, *emRight;

    // calculating weighted sum of square residuals for the convergence test
    const float EmphSel = (relax_kmult_pass == 1) ? (Ampl + 2.0f) : Ampl;
    int nonZeroEmpFrames;
    float frac = BlockLevel_DecideOnEmphasisVectorsForInterpolation(
        nonZeroEmpFramesVec,
        &emLeft,
        &emRight,
        EmphSel,
        emphasisVec,
        num_frames,
        nonZeroEmpFrames);

    residual = ResidualCalculationPerFlow(
        startFrame,
        correctedTrace,
        fval,
        emLeft,
        emRight,
        frac,
        err,
        nonZeroEmpFrames);

    //    printf("DEBUG: start residual %f\n",residual); //T*** REMOVE!!  DEBUG ONLY
    // new Ampl and Krate generated from the Lev mar Fit
    float newAmpl, newKmult;
    float delta0 = 0, delta1 = 0;
    int iter;
    for (iter = 0; iter < ITER; ++iter) {

      // new Ampl and krate by adding delta to existing values
      newAmpl = Ampl + 0.001f;
      newKmult = (twoParamFit)?(kmult + 0.001f):(kmult);

      // Evaluate model function for new Ampl keeping Krate constant
      float aa = 0, akr= 0, krkr = 0, rhs0 = 0, rhs1 = 0;



#if __CUDA_ARCH__ >= 350

      Keplar_ModelFuncEvaluationForSingleFlowFit(
          constRegP,
          perNucRegP,
          twoParamFit, 
          startFrame,
          nucRise,
          newAmpl, 
          kmult*perNucRegP->getKrate(),
          tauB, 
          gain, 
          SP, 
          d, 
          sens, 
          startFrame*ISIG_SUB_STEPS_SINGLE_FLOW, 
          tmp_fval, 
          deltaFrames, 
          nonZeroEmpFrames, 
          OneParam, 
          jac, 
          emLeft, 
          emRight, 
          frac, 
          fval, 
          err,
          &aa, 
          &rhs0, 
          &krkr, 
          &rhs1, 
          &akr);

      if (twoParamFit) 
        Keplar_ModelFuncEvaluationForSingleFlowFit(
            constRegP,
            perNucRegP,
            twoParamFit,
            startFrame,
            nucRise, 
            Ampl,
            newKmult*perNucRegP->getKrate(),
            tauB, 
            gain, 
            SP, 
            d, 
            sens, 
            startFrame*ISIG_SUB_STEPS_SINGLE_FLOW,
            tmp_fval, 
            deltaFrames, 
            nonZeroEmpFrames, 
            TwoParams, 
            jac, 
            emLeft, 
            emRight, 
            frac, 
            fval, 
            err, 
            &aa, 
            &rhs0, 
            &krkr, 
            &rhs1, 
            &akr);
#else
      Fermi_ModelFuncEvaluationForSingleFlowFit(
          constRegP,
          perNucRegP,
          startFrame,
          nucRise,
          newAmpl,
          Ampl,
          kmult*perNucRegP->getKrate(),
          newKmult*perNucRegP->getKrate(),
          tauB,
          gain,
          SP,
          d,
          sens,
          startFrame*ISIG_SUB_STEPS_SINGLE_FLOW,
          twoParamFit ? TwoParams : OneParam,
              emLeft,
              emRight,
              frac,
              fval,
              err,
              &aa,
              &rhs0,
              &krkr,
              &rhs1,
              &akr,
              deltaFrames,
              nonZeroEmpFrames);
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
        if(twoParamFit)
          newKmult = kmult + delta1;

        clampT(newAmpl, ConstGlobalP.getMinAmpl(), (float)LAST_POISSON_TABLE_COL);
        if(twoParamFit)clampT(newKmult, localMinKmult, localMaxKmult);

        //         printf("DEBUG: %d newAmpl %f\n", iter, newAmpl); //T*** REMOVE!!  DEBUG ONLY

        // Evaluate using new params
        if (ConfigP.UseDynamicEmphasis())
          frac = BlockLevel_DecideOnEmphasisVectorsForInterpolation(
              nonZeroEmpFramesVec,
              &emLeft,
              &emRight,
              newAmpl,
              emphasisVec,
              num_frames,
              nonZeroEmpFrames);

#if __CUDA_ARCH__ >= 350
        Keplar_ModelFuncEvaluationForSingleFlowFit(
            constRegP,
            perNucRegP,
            twoParamFit,
            startFrame,
            nucRise,
            newAmpl,
            newKmult*perNucRegP->getKrate(),
            tauB,
            gain,
            SP,
            d,
            sens,
            startFrame*ISIG_SUB_STEPS_SINGLE_FLOW,
            tmp_fval,
            deltaFrames,
            num_frames,
            NoOutput);
#else
        BkgModelRedTraceCalculation(
            constRegP,
            perNucRegP,
            startFrame,
            nucRise,
            newAmpl, 
            newKmult*perNucRegP->getKrate(),
            tauB, 
            gain, 
            SP,
            d, 
            sens, 
            startFrame*ISIG_SUB_STEPS_SINGLE_FLOW,
            tmp_fval,
            deltaFrames,
            ISIG_SUB_STEPS_SINGLE_FLOW,
            num_frames);
#endif
        // residual calculation using new parameters
        newresidual = ResidualCalculationPerFlow(
            startFrame,
            correctedTrace,
            tmp_fval,
            emLeft, 
            emRight, 
            frac, 
            err, 
            nonZeroEmpFrames);
        //          printf("DEBUG: %d residual %f\n", iter, newresidual); //T*** REMOVE!!  DEBUG ONLY
        if (newresidual < residual) {
          Ampl = newAmpl;
          if(twoParamFit)kmult = newKmult;
          // copy new function val to fval
          for (int i=startFrame; i<num_frames; ++i) {
            fval[i] = tmp_fval[i];
          }
          residual = newresidual;
        }
        else {
          if (ConfigP.UseDynamicEmphasis()) {
            frac = BlockLevel_DecideOnEmphasisVectorsForInterpolation(
                nonZeroEmpFramesVec,
                &emLeft,
                &emRight,
                Ampl,
                emphasisVec,
                num_frames, 
                nonZeroEmpFrames);
          }
        }
      }
      //       printf("DEBUG: %d delta %f\n", iter, delta0); //T*** REMOVE!!  DEBUG ONLY
      if ((delta0*delta0) < 0.0000025f){
        iter++;
        break;
      }

    } // end ITER loop

    //DEBUG (rawtrase const?)
    //RawTraces[(ConstFrmP.getRawFrames() - 2 + relax_kmult_pass)*frameStride] = iter; //threadIdx.x%9;
   // if(relax_kmult_pass == 0) atomicMax(maxIterWarp,iter);
    //      printf("DEBUG: done in pass %d at iter %d\n",relax_kmult_pass, iter-1); //T*** REMOVE!!  DEBUG ONLY

    // probably slower incorporation
    if ((kmult - localMinKmult) < 0.01f) {
      if (sqrtf(residual) > 20.0f) {
        localMaxKmult = localMinKmult;
        kmult = 0.3f;
        localMinKmult = 0.3f;
        relax_kmult_pass++;
        continue;
      }
    }

    relax_kmult_pass = 2;
  }// end relax_kmult_pass loop

  if(twoParamFit)
    *(ResultCube + ResultKmult*frameStride) = kmult;
  *(ResultCube + ResultAmpl*frameStride) = Ampl;
  //*(ResultCube + ResultAmpl*frameStride) = (Ampl * pow(perFlowRegP->getCopyDrift(), ConstFlowP.getRealFnum()) * copies);


  residual = CalculateMeanResidualErrorPerFlow(
      startFrame,
      correctedTrace,
      fval,
      emphasisVec+LAST_POISSON_TABLE_COL,
      num_frames);

  //   printf("DEBUG: final residual %f\n",residual); //T*** REMOVE!!  DEBUG ONLY

  float avg_err = *(ResultCube + ResultAvgErr*frameStride) * ConstFlowP.getRealFnum();
  avg_err = (avg_err + residual) / (ConstFlowP.getRealFnum() + 1);
  *(ResultCube + ResultAvgErr*frameStride) = avg_err;

  //int high_err_cnt = 0;
  //avg_err *= WASHOUT_THRESHOLD;
  //for (int flow_ndx = flow_end - 1; flow_ndx >= 0
  //                       && (meanErr[num_beads* flow_ndx] > avg_err); flow_ndx--)
  //  high_err_cnt++;

  //if (high_err_cnt > WASHOUT_FLOW_DETECTION)
  //  pState->corrupt = true;

}


// execute with one warp per row and 2D thread blocks of width warp length
// each warp will slide across one row of the region

// kernel parameters:
// thread block dimensions (WARPSIZE,n,1)  //n = number of warps per block)
// grid dimension ( numRegions.x, (imgH + n-1)/n, 1) // one block per region in x direction and one per n img rows in y direction
// const execParams ep, moved to constant memory as ExecP
// const ImgRegParams moved to constant memory as ImgRegP
__global__
void ExecuteThreadBlockPerRegion2DBlocks(
    const unsigned short * RegionMask,
    const unsigned short  * bfMask,
    unsigned short  * bstateMask,
    //per bead
    //in parameters
    const short * RawTraces, // NxF
    const float * BeadParamCube,
    const float* emphasisVec, //(MAX_POISSON_TABLE_COL)*F
    const int * nonZeroEmphFrames,
    const float* nucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
    //in out parameters
    float* ResultCube,
    const size_t * numFramesRegion,  //constant memory?
    const int * numLBeadsRegion,  //constant memory?
    //per region
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const float * RegionFrameCube,  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
    const float * EmptyTraceRegion  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
    //DEBUG buffer
    //int * numLBeads//, //ToDo only for debuging
    // float * fgBufferFloat
)
{
  extern __shared__ float emphasis[];

#if EMPTY_IN_SHARED
  float * smEmptyTrace = emphasis + MAX_POISSON_TABLE_COL * ConstFrmP.getMaxCompFrames();
#endif

  //region on chip

  //determine region location
  const size_t regionCol = blockIdx.x;
  const size_t regionRow = (blockIdx.y*blockDim.y)/ImgRegP.getRegH();

  //image coordinates
  size_t ix = regionCol * ImgRegP.getRegW() + threadIdx.x;
  const size_t iy = (blockIdx.y*blockDim.y) + threadIdx.y;
  size_t idx = ImgRegP.getWellIdx(ix,iy);
  //region coordinates
  int rx = threadIdx.x;
  //  int leftshift = (idx-threadIdx.x)%32;


  //if(idx == imgRegP.getWellIdx(5,149) print = true;

  //Shift block to the left so thread 0 of the block aligns with
  //a 128 byte or at least 64 byte (for short) alignment boundary
  //  rx -= leftshift;
  //  idx -= leftshift;



  //region index to address region specific parameters
  //does not work if any blockDim > RegionDim
  //const size_t regId = ImgRegP.getRegId(ix,iy);
  //use regionCol and Row instead
  const size_t regId = regionRow*ImgRegP.getGridDimX()+regionCol;
  size_t numf = numFramesRegion[regId];
  ///////////////////////////////////////////////////
  //If the Region does not have any useful data frames will be 0 or numLBeads will be 0, so nothing has to be done
  if( LDG_ACCESS(RegionMask,regId) != RegionMaskLive) return;
  //if( LDG_ACCESS(numLBeadsRegion,regId) == 0) return;  // no live beads in region, no more work for this thread block
  if (numf == 0) return;

  //strides
  const size_t BeadFrameStride = ImgRegP.getPlaneStride();
  const size_t RegionFrameStride = ConstFrmP.getMaxCompFrames() * ImgRegP.getNumRegions();
  const size_t windowSize = blockDim.x;

  //if EmptyTraces from GenerateBeadTrace Kernel padding is uncompressed frames
  const float * emptyTraceAvg = EmptyTraceRegion + regId*ConstFrmP.getUncompFrames();
  RegionFrameCube += regId*ConstFrmP.getMaxCompFrames();  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber

  ////////////////////////////////////////////////////////////
  // setup code that needs to be done by all threads
  emphasisVec += regId * MAX_POISSON_TABLE_COL * ConstFrmP.getMaxCompFrames();

  //
  //careful when accessing pointers since threads that would work out of bounds
  //are not yet filtered at this point
  if(blockDim.x == 32){
    const int numWarps = blockDim.y;
    const int threadWarpIdx = threadIdx.x;
    const int warpIdx = threadIdx.y;
    for(int i=warpIdx; i<numf; i += numWarps)
    {
      if (threadWarpIdx < MAX_POISSON_TABLE_COL)
        emphasis[(MAX_POISSON_TABLE_COL)*i + threadWarpIdx ] = emphasisVec[numf*threadWarpIdx + i ];
    }
  }else{
    const int numthreads = blockDim.x*blockDim.y;
    const int numWarps = numthreads/32;
    const int absThreadIdx = threadIdx.y*blockDim.x+threadIdx.x;
    const int threadWarpIdx = absThreadIdx%32;
    const int warpIdx = absThreadIdx/32;
    for(int i=warpIdx; i<numf; i += numWarps)
    {
      if (threadWarpIdx < MAX_POISSON_TABLE_COL)
        emphasis[(MAX_POISSON_TABLE_COL)*i + threadWarpIdx ] = emphasisVec[numf*threadWarpIdx + i ];
    }
  }

  size_t emphStride = MAX_POISSON_TABLE_COL;

  //update per region pointers
  constRegP += regId;
  perFlowRegP += regId;

  //point to correct nuc
  perNucRegP +=  ImgRegP.getNumRegions() * ConstFlowP.getNucId() + regId;


  nonZeroEmphFrames += regId*MAX_POISSON_TABLE_COL;
  nucRise += regId *  ISIG_SUB_STEPS_SINGLE_FLOW * ConstFrmP.getMaxCompFrames() ;

  float rezero_t_start =  perFlowRegP->getTMidNuc()+perFlowRegP->getTMidNucShift();


#if EMPTY_TRACES_REZERO_SHARED_UNCOMPRESSED_INPUT


  if( threadIdx.y == 0){  // only first warp
    float * sm = smEmptyTrace;
    const float * frameNumber = (RegionFrameCube+RegionFrameStride*RfFrameNumber);
    float dcoffset = ComputeDcOffsetForUncompressedTrace(emptyTraceAvg,ConstFrmP.getUncompFrames(),constRegP->getTimeStart(), rezero_t_start-MAGIC_OFFSET_FOR_EMPTY_TRACE);
    for(int fn = threadIdx.x; fn < numf; fn+=blockDim.x){
        TShiftAndPseudoCompressionOneFrame(sm,emptyTraceAvg,frameNumber, perFlowRegP->getTshift(), fn, ConstFrmP.getUncompFrames(),dcoffset);
    }
    __syncthreads(); // guarantee sm writes are completed and visiable within block
  }
  emptyTraceAvg = smEmptyTrace;

#endif
#if EMPTY_TRACES_REZERO_SHARED_COMPRESSED_INPUT && !EMPTY_TRACES_REZERO_SHARED_UNCOMPRESSED_INPUT


  if( threadIdx.y == 0){  // only first warp
    volatile float * sm = smEmptyTrace;
    const float * frameNumber = (RegionFrameCube+RegionFrameStride*RfFrameNumber);
    for(int fn = threadIdx.x; fn < numf; fn+=blockDim.x){
          sm[fn] = LDG_ACCESS(emptyTraceAvg, fn);
    }
    //__syncthreads(); // guarantee sm writes are completed and visiable within block

    float dcoffset = ComputeDcOffsetForCompressedTrace(smEmptyTrace,1,frameNumber,constRegP->getTimeStart(),rezero_t_start-MAGIC_OFFSET_FOR_EMPTY_TRACE, numf);
    for(int fn = threadIdx.x; fn < numf; fn+=blockDim.x){
      sm[fn] -= dcoffset;
    }
    //__syncthreads();
  }
  emptyTraceAvg = smEmptyTrace;
#endif

  __syncthreads();


  //end all thread setup code, now excess threads can drop out
  ////////////////////////////////////////////////////////////
  /*if(idx == 20){
    printf("input trace \n");
    for( size_t i = 0 ; i < *numFrames; i++){
      printf("%d, ", RawTraces[i * ImgRegP.getPlaneStride()]);
    }
    printf("\n");
}*/
  //filter blocks that are outside the region in y-direction (ToDO: when 2d blocks this has to be done per warp after the all threads tasks are completed)
  const size_t ry = iy%ImgRegP.getRegH();
  if( ! ImgRegP.isValidIdx(idx) || ry >= ImgRegP.getRegH(regId)) return;

  //get actual region Width
  const size_t regionWidth = ImgRegP.getRegW(regId);


  //update bead pointers to base for
  bfMask += idx;
  bstateMask += idx;

  RawTraces += idx;
  BeadParamCube += idx;
  ResultCube += idx;


  //fgBufferFloat += idx;
  // if (threadIdx.x == 0) printf(" %d %d %lu %lf %x %lf \n", rx, leftshift, idx, idx/32.0, RawTraces, ((size_t)(RawTraces))/128.0);


  //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0){
  //printf("Flow %d (%d) NucId from Kernel: %d \n ",ConstFlowP.getRealFnum(), ConstFlowP.getFlowIdx(), ConstFlowP.getNucId());
  //  printf("Flow %d nucrise: ",ConstFlowP.getRealFnum() );
  //  for(int i=0; i<ISIG_SUB_STEPS_SINGLE_FLOW*numf;i++)
  //             printf("%f,",nucRise[i]);
  //           printf("\n");
  // }

  //sliding window if thread block is too small to handle a whole row in a region
  while(rx < regionWidth){ //while thread inside region


    //do not do work if thread points to a well left of the region boundary
    //  if(rx >= 0){

    //  if (threadIdx.x < 4  && blockIdx.y%224 == 0) printf("x %d bx %d by %d rx %d idx %lu %lf %x %lf \n",threadIdx.x, blockIdx.x, blockIdx.y, rx, idx, idx/32.0, RawTraces, ((size_t)(RawTraces))/128.0);


    if(Match(bfMask,MaskLive)){
      //printf( "x: %d y: %lu\n",rx, ry) ;
      //atomicAdd(numLBeads,1);
      if(!Match(bstateMask,BkgMaskCorrupt)){
        //here we know all the coordinates and if the bead is live and not corrupted...
        //so lets do some work!
        //printf("%d, %d in block %d, %d I am alive!\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

        // bool print = false;
        // if(rx ==191 && ry == 43) print = true;
#if FG_TRACES_REZERO
        float dcOffset = ComputeDcOffsetForCompressedTrace ( RawTraces, BeadFrameStride, RegionFrameCube + RegionFrameStride * RfFrameNumber,
            constRegP->getTimeStart(), rezero_t_start - perFlowRegP->getSigma(),numf );
#endif



        SingleFlowFitUsingRelaxKmultGaussNewton(
            //per Bead
            ResultCube, //Ampl, kmult, avg_error, points to correct Ampl value, stride == frameStride
            bstateMask,
            RawTraces, // imgW*imgHxF
            BeadParamCube, //Copies, R, dmult, gain, tau_adj, phi, stride == frameStride
            //per region
            emphasis, //(MAX_POISSON_TABLE_COL)*F
            nonZeroEmphFrames,
            nucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
            constRegP,
            perFlowRegP,
            perNucRegP,
            RegionFrameCube,  //bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
            emptyTraceAvg, //smEmptyTrace  //EmptyTraceRegion
#if FG_TRACES_REZERO
            dcOffset,
#endif
            // other scalar inputs
            numf, // 4
            //strides
            BeadFrameStride,
            RegionFrameStride,
            emphStride
            //print
        );
      }
    } //end work for active bead move to next beads
    //   }

    rx += windowSize;
    //   idx += windowSize;

    bfMask += windowSize;
    bstateMask += windowSize;

    RawTraces += windowSize;
    BeadParamCube += windowSize;
    ResultCube += windowSize;
  }
}




// execute with one warp per row and 2D thread blocks of width warp length
// each warp will slide across one row of the region

// kernel parameters:
// thread block dimensions (WARPSIZE,n,1)  //n = number of warps per block)
// grid dimension ( numRegions.x, (imgH + n-1)/n, 1) // one block per region in x direction and one per n img rows in y direction
// const execParams ep, moved to constant memory as ExecP
// const ImgRegParams moved to constant memory as ImgRegP
// in this implementation the warp (sliding window) will not contain any non-live beads.
// before assigning the beads/wells to the threads a single pass over the masks is performed and all non-live beads are discarded.
// this improves the overall parallelism during the execution. ~ 17% speedup

//launch bounds:
//K20
//regs per SM: 65536
//

#if __CUDA_ARCH__ >= 300
    #define SINGLEFLOW_MAX_THREADS  128
    #define SINGLEFLOW_MIN_BLOCKS   8
#else
    #define SINGLEFLOW_MAX_THREADS  128
    #define SINGLEFLOW_MIN_BLOCKS   5
#endif


__global__
__launch_bounds__(SINGLEFLOW_MAX_THREADS,SINGLEFLOW_MIN_BLOCKS)
void ExecuteThreadBlockPerRegion2DBlocksDense(
    const unsigned short * RegionMask,
    const unsigned short  * bfMask,
    unsigned short  * bstateMask,
    //per bead
    //in parameters
    const short * RawTraces, // NxF
    const float * BeadParamCube,
    const float* emphasisVec, //(MAX_POISSON_TABLE_COL)*F
    const int * nonZeroEmphFrames,
    const float* nucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
    //in out parameters
    float* ResultCube,
    const size_t * numFramesRegion,  //constant memory?
    const int * numLBeadsRegion,  //constant memory?
    //per region
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const float * RegionFrameCube,  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
    const float * EmptyTraceRegion  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
    //DEBUG buffer
    //int * numLBeads//, //ToDo only for debuging
    // float * fgBufferFloat
)
{
  extern __shared__ float emphasis[];

#if EMPTY_IN_SHARED
  float * smEmptyTrace = emphasis + MAX_POISSON_TABLE_COL * ConstFrmP.getMaxCompFrames();
#endif

  //region on chip

  //determine region location
  const size_t regionCol = blockIdx.x;
  const size_t regionRow = (blockIdx.y*blockDim.y)/ImgRegP.getRegH();

  //image coordinates
  size_t ix = regionCol * ImgRegP.getRegW(); // + threadIdx.x;
  const size_t iy = (blockIdx.y*blockDim.y) + threadIdx.y;
  size_t idx = ImgRegP.getWellIdx(ix,iy);

  //region index to address region specific parameters
  //does not work if any blockDim > RegionDim
  //const size_t regId = ImgRegP.getRegId(ix,iy);
  //use regionCol and Row instead
  const size_t regId = ImgRegP.getRegIdFromGrid(regionCol,regionRow);
  size_t numf = numFramesRegion[regId];
  ///////////////////////////////////////////////////
  //If the Region does not have any useful data frames will be 0 or numLBeads will be 0, so nothing has to be done

  if( LDG_ACCESS(RegionMask,regId) != RegionMaskLive) return;
  //if( LDG_ACCESS(numLBeadsRegion,regId) == 0) return;  // no live beads in region, no more work for this thread block
  if (numf == 0) return;

  //strides
  const size_t BeadFrameStride = ImgRegP.getPlaneStride();
  const size_t RegionFrameStride = ConstFrmP.getMaxCompFrames() * ImgRegP.getNumRegions();
  //const size_t windowSize = blockDim.x;

  //if EmptyTraces from GenerateBeadTrace Kernel padding is uncompressed frames
  const float * emptyTraceAvg = EmptyTraceRegion + regId*ConstFrmP.getUncompFrames();
  RegionFrameCube += regId*ConstFrmP.getMaxCompFrames();  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber

  ////////////////////////////////////////////////////////////
  // setup code that needs to be done by all threads
  emphasisVec += regId * MAX_POISSON_TABLE_COL * ConstFrmP.getMaxCompFrames();

  //
  //careful when accessing pointers since threads that would work out of bounds
  //are not yet filtered at this point
  if(blockDim.x == 32){
    const int numWarps = blockDim.y;
    const int threadWarpIdx = threadIdx.x;
    const int warpIdx = threadIdx.y;
    for(int i=warpIdx; i<numf; i += numWarps)
    {
      if (threadWarpIdx < MAX_POISSON_TABLE_COL)
        emphasis[(MAX_POISSON_TABLE_COL)*i + threadWarpIdx ] = emphasisVec[numf*threadWarpIdx + i ];
    }
  }else{
    const int numthreads = blockDim.x*blockDim.y;
    const int numWarps = numthreads/32;
    const int absThreadIdx = threadIdx.y*blockDim.x+threadIdx.x;
    const int threadWarpIdx = absThreadIdx%32;
    const int warpIdx = absThreadIdx/32;
    for(int i=warpIdx; i<numf; i += numWarps)
    {
      if (threadWarpIdx < MAX_POISSON_TABLE_COL)
        emphasis[(MAX_POISSON_TABLE_COL)*i + threadWarpIdx ] = emphasisVec[numf*threadWarpIdx + i ];
    }
  }

  size_t emphStride = MAX_POISSON_TABLE_COL;

  //update per region pointers
  constRegP += regId;
  perFlowRegP += regId;

  //point to correct nuc
  perNucRegP +=  ImgRegP.getNumRegions() * ConstFlowP.getNucId() + regId;


  nonZeroEmphFrames += regId*MAX_POISSON_TABLE_COL;
  nucRise += regId *  ISIG_SUB_STEPS_SINGLE_FLOW * ConstFrmP.getMaxCompFrames() ;

  float rezero_t_start =  perFlowRegP->getTMidNuc()+perFlowRegP->getTMidNucShift();


#if EMPTY_TRACES_REZERO_SHARED_UNCOMPRESSED_INPUT


  if( threadIdx.y == 0){  // only first warp
    float * sm = smEmptyTrace;
    const float * frameNumber = (RegionFrameCube+RegionFrameStride*RfFrameNumber);
    float dcoffset = ComputeDcOffsetForUncompressedTrace(emptyTraceAvg,ConstFrmP.getUncompFrames(),constRegP->getTimeStart(), rezero_t_start-MAGIC_OFFSET_FOR_EMPTY_TRACE);
    for(int fn = threadIdx.x; fn < numf; fn+=blockDim.x){
        TShiftAndPseudoCompressionOneFrame(sm,emptyTraceAvg,frameNumber, perFlowRegP->getTshift(), fn, ConstFrmP.getUncompFrames(),dcoffset);
    }
    __syncthreads(); // guarantee sm writes are completed and visiable within block
  }
  emptyTraceAvg = smEmptyTrace;

#endif
#if EMPTY_TRACES_REZERO_SHARED_COMPRESSED_INPUT && !EMPTY_TRACES_REZERO_SHARED_UNCOMPRESSED_INPUT


  if( threadIdx.y == 0){  // only first warp
    volatile float * sm = smEmptyTrace;
    const float * frameNumber = (RegionFrameCube+RegionFrameStride*RfFrameNumber);
    for(int fn = threadIdx.x; fn < numf; fn+=blockDim.x){
          sm[fn] = LDG_ACCESS(emptyTraceAvg, fn);
    }
    //__syncthreads(); // guarantee sm writes are completed and visiable within block

    float dcoffset = ComputeDcOffsetForCompressedTrace(smEmptyTrace,1,frameNumber,constRegP->getTimeStart(),rezero_t_start-MAGIC_OFFSET_FOR_EMPTY_TRACE, numf);
    for(int fn = threadIdx.x; fn < numf; fn+=blockDim.x){
      sm[fn] -= dcoffset;
    }
    //__syncthreads();
  }
  emptyTraceAvg = smEmptyTrace;
#endif

  __syncthreads();


  //end all thread setup code, now excess threads can drop out
  ////////////////////////////////////////////////////////////

  //filter blocks that are outside the region in y-direction (ToDO: when 2d blocks this has to be done per warp after the all threads tasks are completed)
  const size_t ry = iy%ImgRegP.getRegH();
  if( ry >= ImgRegP.getRegH(regId)) return;

  //get actual region Width
  const size_t regionWidth = ImgRegP.getRegW(regId);

  //update bead pointers to point to first well in row ry of region regId
  bfMask += idx;
  bstateMask += idx;

  //int * maxIterWarp = (int*)(RawTraces + (ConstFrmP.getRawFrames() - 3 ) * BeadFrameStride);
  // one value per warp per region row
  // warps per row = (ImgPregW+31/32)
  //int warpsPerRegionRow = (ImgRegP.getRegW()+31)/32;
  //int warpsPerImgRow = warpsPerRegionRow * ImgRegP.getGridDimX();
  //int regionRowWarpsStride = ImgRegP.getRegH()* warpsPerImgRow;
  //maxIterWarp += regionRow * regionRowWarpsStride + ry * warpsPerImgRow + regionCol * warpsPerRegionRow;

  RawTraces += idx;
  BeadParamCube += idx;
  ResultCube += idx;

  //sliding window if thread block is too small to handle a whole row in a region
  int rx = 0;

  while(rx < regionWidth){ //while thread inside region

    size_t liveBeadCountWarp = 0; // how many live beads are found in this warp
    size_t myOffset = regionWidth; // init to regionWidth so tha tif no bead is found for a thread it will drop out of the execution
    while(rx < regionWidth && liveBeadCountWarp < blockDim.x) //stop search when end of row reached or warp is full
    {
      if( Match(bfMask + rx, MaskLive) && (!Match(bstateMask + rx,BkgMaskCorrupt))){  // if live bead
        if(liveBeadCountWarp == threadIdx.x) //assign to correct thread (n-th live bead found handled by n-th thread in warp)
          myOffset = rx; //offset is the actual x-coordinate within the region of the well we are looking at in the current row
        liveBeadCountWarp++; //keep track of howe many live beads are already found for this warp
      }
      rx++; //move to next bead in the row
    }


    if(myOffset < regionWidth){ //filter out threads that do not have a correct live bead to work on

      //update local offsets (region regId does not change since warp only works on beads of one row in one region)
      unsigned short  * lbstateMask = bstateMask + myOffset;
      const short * lRawTraces = RawTraces + myOffset;
      const float * lBeadParamCube = BeadParamCube+ myOffset;
      float* lResultCube = ResultCube + myOffset;

#if FG_TRACES_REZERO
        // Compute the per flow dc offset for the bead traces (maybe can be done later when we touch the raw traces the first time.
        float dcOffset = ComputeDcOffsetForCompressedTrace ( lRawTraces, BeadFrameStride, RegionFrameCube + RegionFrameStride * RfFrameNumber,
            constRegP->getTimeStart(), rezero_t_start - perFlowRegP->getSigma(),numf );
#endif



        SingleFlowFitUsingRelaxKmultGaussNewton(
            //per Bead
            lResultCube, //Ampl, kmult, avg_error, points to correct Ampl value, stride == frameStride
            lbstateMask,
            lRawTraces, // imgW*imgHxF
            lBeadParamCube, //Copies, R, dmult, gain, tau_adj, phi, stride == frameStride
            //per region
            emphasis, //(MAX_POISSON_TABLE_COL)*F
            nonZeroEmphFrames,
            nucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
            constRegP,
            perFlowRegP,
            perNucRegP,
            RegionFrameCube,  //bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
            emptyTraceAvg, //smEmptyTrace  //EmptyTraceRegion
#if FG_TRACES_REZERO
            dcOffset,
#endif
            // other scalar inputs
            numf, // 4
            //strides
            BeadFrameStride,
            RegionFrameStride,
            emphStride
            //maxIterWarp
            //print
        );
        //maxIterWarp++;
    }
    //end work for active bead move to next beads
  }
}


