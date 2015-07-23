
#include "FittingHelpers.h"

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
    const float tmidNuc,
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP
)
{
  float tmid = tmidNuc;
  tmid +=  perNucRegP->getTMidNucDelay()*
      (tmidNuc -  ConstGlobalP.getValveOpen()) /
      ( ConstGlobalP.getMagicDivisorForTiming() + SAFETYZERO);
  tmid +=  perFlowRegP->getTMidNucShift();

  return tmid;
}



__device__ 
float ComputeETBR(
    //const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const float RatioDrift,
    const float R,
    float copies
) {

  float etbR;

  if (ConfigP.FitTauE()) {
    etbR = R;
    if (etbR)
      etbR = perNucRegP->getNucModifyRatio() /(perNucRegP->getNucModifyRatio() +
          (1.0f - (RatioDrift * (ConstFlowP.getRealFnum())/SCALEOFBUFFERINGCHANGE))*
          (1.0f / etbR - 1.0f));
  }
  else {
    if ( !ConfigP.UseAlternativeEtbRequation()) {
      etbR = R*perNucRegP->getNucModifyRatio() +
          (1.0f - R*perNucRegP->getNucModifyRatio())*
          RatioDrift*(ConstFlowP.getRealFnum())/SCALEOFBUFFERINGCHANGE;
    }
    else {
      etbR = R*perNucRegP->getNucModifyRatio() +
          RatioDrift*copies*(ConstFlowP.getRealFnum())/(6.0*SCALEOFBUFFERINGCHANGE);
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
    const float copyDrift,
    const float copies
) {
  return ((float)(COPYMULTIPLIER * copies) * powf(copyDrift, ConstFlowP.getRealFnum()));
}


__device__ 
float ComputeSigma(
    const PerFlowParamsRegion *perFlowRegP,
    const PerNucParamsRegion *perNucRegP)
{
  return (perFlowRegP->getSigma() * perNucRegP->getSigmaMult());
}

__device__
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

//Provide emphasis stride to projection search
__device__
float ProjectionSearch(
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const float* observedTrace,
    const float* emphasisVec,
    const int frames,
    const float* nucRise,
    const float* deltaFrames,
    const float kmult,
    const float d,
    const float tauB,
    const float gain,
    const float SP,
    float* tmp_fval,
    int nucStart,
    int beadFrameStride,
    int emphStride,
    int nucIntLoopSteps
)
{
  float Ampl = 1.0f;

  for (int i=0; i<2; ++i) { //TODO invariant code motion?


    BkgModelRedTraceCalculation(
        constRegP,
        perNucRegP,
        nucStart,
        nucRise, 
        Ampl, 
        kmult*perNucRegP->getKrate(),
        tauB, 
        gain, 
        SP, 
        d, 
        constRegP->getSens()*SENSMULTIPLIER,
        nucIntLoopSteps * nucStart,
        tmp_fval, 
        deltaFrames, 
        nucIntLoopSteps,
        frames);

    /*if (threadIdx.x == 0) {
      for (int i=0; i<frames; ++i) {
        printf("%f,", tmp_fval[i]);
      } 
      printf("\n");
    }
    __syncthreads();*/

    float num = 0, den = 0.0001f;
    float emphasisVal;
    for (int j=nucStart; j<frames; ++j) {
      emphasisVal = emphasisVec[j*emphStride] * emphasisVec[j*emphStride];
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
}

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
    const int nucIntLoopSteps,
    const int endFrames
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

  // zero out frames before nuc start
  for (int i=0; i<startFrame; ++i) {
    fval[i] = 0;
  }
  for (int i=startFrame;i < endFrames;i++)
  {
    if (totgen > 0.0f)
    {
      float ldt = (deltaFrame[i]/( nucIntLoopSteps * FRAMESPERSEC)) * half_kr;
      for (int st=1; (st <= nucIntLoopSteps) && (totgen > 0.0f);st++)
      {
        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;

        // TODO Nucrise can be in shared memory...Need to fix it
        // All the threads should be grabbing from the same nucRise location.
        // c_dntp_bot is the concentration of dNTP in the well
        //float c_dntp_bot = LDG_ACCESS(nucRise, c_dntp_top_ndx++) / (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        float c_dntp_bot = nucRise[c_dntp_top_ndx++] / (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + cp_sid_kmax_nucid);

        c_dntp_new_rate = c_dntp_bot*c_dntp_bot_plus_kmax;
        float c_dntp_int = ldt* (c_dntp_new_rate+c_dntp_old_rate);
        c_dntp_sum += c_dntp_int;

        /*      if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 2)
            printf ("%d top_ndx %d nucrise %f skr %f pact %f dntps+kmax %f\n", i, c_dntp_top_ndx-1, nucRise[c_dntp_top_ndx-1],scaled_kr, pact,c_dntp_bot_plus_kmax  );
         */
        // calculate new number of active polymerase
        float pact_new = poiss_cdf_approx_float4_SingelFLowFit(c_dntp_sum, LUTptr, occ_l, occ_r);

        /*
                       if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 2)
                         printf ("%d pn: %f cds: %f LU: %f %f %f %f ocl: %f ocr: %f\n", i, pact_new, c_dntp_sum, (*LUTptr).x,(*LUTptr).y,(*LUTptr).z,(*LUTptr).w, occ_l, occ_r );
         */

        totgen -= ( (pact+pact_new) * 0.5f) * c_dntp_int;
        pact = pact_new;
      }

      if (totgen < 0.0f) totgen = 0.0f;
      red_hydro = (totocc-totgen);
    }else{

      red_hydro = totocc;
    }

    /*
                if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 2)
                  printf ("%d df: %f rh: %f\n", i, deltaFrame[i], red_hydro);
     */
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

__device__ void
IncorporationSignalCalculation(
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
    const int nucIntLoopSteps,
    const int endFrames
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

  float red_hydro;

  float c_dntp_bot_plus_kmax = 1.0f/cp_sid_kmax_nucid;

  // zero out frames before nuc start
  for (int i=0; i<startFrame; ++i) {
    fval[i] = 0;
  }
  for (int i=startFrame;i < endFrames;i++)
  {
    if (totgen > 0.0f)
    {
      float ldt = (deltaFrame[i]/( nucIntLoopSteps * FRAMESPERSEC)) * half_kr;
      for (int st=1; (st <= nucIntLoopSteps) && (totgen > 0.0f);st++)
      {
        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;

        // TODO Nucrise can be in shared memory...Need to fix it
        // All the threads should be grabbing from the same nucRise location.
        // c_dntp_bot is the concentration of dNTP in the well
        //float c_dntp_bot = LDG_ACCESS(nucRise, c_dntp_top_ndx++) / (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        float c_dntp_bot = nucRise[c_dntp_top_ndx++] / (1.0f + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + cp_sid_kmax_nucid);

        c_dntp_new_rate = c_dntp_bot*c_dntp_bot_plus_kmax;
        float c_dntp_int = ldt* (c_dntp_new_rate+c_dntp_old_rate);
        c_dntp_sum += c_dntp_int;

        /*      if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 2)
            printf ("%d top_ndx %d nucrise %f skr %f pact %f dntps+kmax %f\n", i, c_dntp_top_ndx-1, nucRise[c_dntp_top_ndx-1],scaled_kr, pact,c_dntp_bot_plus_kmax  );
         */
        // calculate new number of active polymerase
        float pact_new = poiss_cdf_approx_float4_SingelFLowFit(c_dntp_sum, LUTptr, occ_l, occ_r);

        /*
                       if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 2)
                         printf ("%d pn: %f cds: %f LU: %f %f %f %f ocl: %f ocr: %f\n", i, pact_new, c_dntp_sum, (*LUTptr).x,(*LUTptr).y,(*LUTptr).z,(*LUTptr).w, occ_l, occ_r );
         */

        totgen -= ( (pact+pact_new) * 0.5f) * c_dntp_int;
        pact = pact_new;
      }

      if (totgen < 0.0f) totgen = 0.0f;
      red_hydro = (totocc-totgen);
    }else{

      red_hydro = totocc;
    }

    fval[i] = red_hydro * sens;

  }
}


__device__ void
BkgModelPurpleTraceCalculation(
    const float *beadParamCube,
    const float *regionFrameCube,
    const float darkness,
    const float etbR,
    const float tauB,
    const float gain,
    const float *bkgTrace,
    const float* deltaFrame,
    const int num_frames,
    const int beadFrameStride,
    const int regionFrameStride,
    float *trace // both input and output
)
{
  float one_over_two_tauB = 1.0f / (2.0f * tauB);
  float one_over_one_plus_aval = 0.0f;
  
  int i=0;
  float xt = LDG_ACCESS(deltaFrame, i) * one_over_two_tauB; // decay term
  one_over_one_plus_aval = 1.0f/(1.0f+xt);

  float red_prev = trace[i];
  float purple_hdr = (red_prev + (etbR + xt)*LDG_ACCESS(bkgTrace,i)) * one_over_one_plus_aval;
  trace[i]  = purple_hdr*gain + ApplyDarkMatterToFrame(beadParamCube, 
                                                regionFrameCube, 
                                                darkness, 
                                                i, 
                                                num_frames, 
                                                beadFrameStride, 
                                                regionFrameStride);
  i++;

  for (; i<num_frames; i++) {
    xt = LDG_ACCESS(deltaFrame, i) * one_over_two_tauB;
    one_over_one_plus_aval = 1.0f / (1.0f + xt);
    purple_hdr = ((trace[i] - red_prev) + (etbR + xt) * LDG_ACCESS(bkgTrace,i) - (etbR - xt) * LDG_ACCESS(bkgTrace , (i-1)) + (1.0f - xt) * purple_hdr) * one_over_one_plus_aval;

    red_prev = trace[i];
    trace[i] = purple_hdr*gain + ApplyDarkMatterToFrame(beadParamCube,
                                                   regionFrameCube,
                                                   darkness,
                                                   i,
                                                   num_frames,
                                                   beadFrameStride,
                                                   regionFrameStride);

  }
  
}

__device__
void ComputeModelBasedTrace(
  const float *bkgTrace,
  const float* deltaFrame,
  const ConstantParamsRegion * constRegP,
  const PerNucParamsRegion * perNucRegP,
  const float *beadParamCube,
  const float *regionFrameCube,
  const float *nucRise,
  const int startFrame,
  const float Krate,
  const float tauB,
  const float gain,
  const float SP,
  const float d,
  const float darkness,
  const float etbR,
  float sens,
  float A,
  int c_dntp_top_ndx,
  const int nucIntLoopSteps,
  const int num_frames,
  const int beadFrameStride, 
  const int regionFrameStride,
  float *trace)
{
  IncorporationSignalCalculation(
    constRegP,
    perNucRegP,
    startFrame,
    nucRise,
    A,
    Krate,
    tauB,
    gain,
    SP,
    d,
    sens,
    c_dntp_top_ndx,
    trace,
    deltaFrame,
    nucIntLoopSteps,
    num_frames);

    BkgModelPurpleTraceCalculation(
      beadParamCube,
      regionFrameCube,
      darkness,
      etbR,
      tauB,
      gain,
      bkgTrace,
      deltaFrame,
      num_frames,
      beadFrameStride,
      regionFrameStride,
      trace);
}


__device__ 
float GenerateStratifiedEmphasis_Dev(
  const int hpNum,
  const int frame,
  const float tcenter,
  const int *framesPerPoint,
  const float *frameNumber)
{
  const float *emp = ConstGlobalP.getEmphParams();
  float na = emp[0] + hpNum*emp[1];
  float nb = emp[2] + hpNum*emp[3];
  float db = emp[4] + hpNum*emp[5];
 
  float deltat = frameNumber[frame] - tcenter;
  float EmphasisOffsetB = (deltat - nb) / ( ConstGlobalP.getEmphWidth()*db );
  float tmp = ConstGlobalP.getEmphAmpl() * expf(-EmphasisOffsetB*EmphasisOffsetB);
  float EmphasisOffsetA = (deltat - na) / ConstGlobalP.getEmphWidth(); 

  float empVal = framesPerPoint[frame];
  if ((EmphasisOffsetA < 0.0f) && (deltat >= -3.0f))
     empVal *= tmp;
  else if (EmphasisOffsetA >= 0.0f)
    empVal *= tmp * expf(-EmphasisOffsetA*EmphasisOffsetA);
  
  return empVal;
}

__device__ 
float GenerateBlankEmphasis_Dev(
  const int frame,
  const float tcenter,
  const int *framesPerPoint,
  const float *frameNumber)
{
  const float *emp = ConstGlobalP.getEmphParams();
  float deltat = frameNumber[frame] - tcenter;
  float EmphasisOffsetC = (deltat - emp[6]) / emp[7];
  
  return (framesPerPoint[frame] * expf(-EmphasisOffsetC*EmphasisOffsetC));
}


// generate emphasis vectors for all regions of a proton block
// need frames per point
// emphasis constants from gopt
__global__
void GenerateEmphasis(
  const int numEv,
  const float amult,
  const PerFlowParamsRegion *perFlowRegP,
  const int *framePerPoint,
  const float *RegionFrameCube,
  const size_t *numFramesRegion,
  float *emphasisVec,
  int *nonZeroEmphFrames) 
{
  extern __shared__ float smBuffer[];
  __shared__ float empScale[MAX_POISSON_TABLE_COL];

  int regId = blockIdx.x;
  int num_frames = numFramesRegion[regId];

  const float *frameNumber = RegionFrameCube +  RfFrameNumber*ConstFrmP.getMaxCompFrames()*ImgRegP.getNumRegions() + 
                     regId*ConstFrmP.getMaxCompFrames();
  perFlowRegP += regId;
  nonZeroEmphFrames += regId*numEv;
  emphasisVec += regId * numEv * ConstFrmP.getMaxCompFrames();
  framePerPoint += regId * ConstFrmP.getMaxCompFrames();

  float tmidNuc = perFlowRegP->getTMidNuc();
  int empVecSize = numEv * num_frames;
  for (int i=0; i<empVecSize; i+=blockDim.x) {

    int serializedFrame = i + threadIdx.x;
    int hpNum = serializedFrame / num_frames;

    if (hpNum < numEv) {
      int frameToCompute = serializedFrame - (hpNum*num_frames);

      // generate different emphasis based on amult*empAmpl
      if ((amult * ConstGlobalP.getEmphAmpl()) > 0.0f)
        smBuffer[serializedFrame] = GenerateStratifiedEmphasis_Dev(
            hpNum,
            frameToCompute,
            tmidNuc,
            framePerPoint,
            frameNumber);
      else 
        smBuffer[serializedFrame] = GenerateBlankEmphasis_Dev(
            frameToCompute,
            tmidNuc,
            framePerPoint,
            frameNumber);
    }
  }
  __syncthreads();

  if (threadIdx.x < numEv) {
    float *myEmp = smBuffer + threadIdx.x * num_frames;
    float tmpScale = 0;
    for (int i=0; i<num_frames; ++i) {
      if (myEmp[i] < 0.0f ) 
        myEmp[i] = 0.0f;

      tmpScale += myEmp[i];  
    }
    empScale[threadIdx.x] = (float)num_frames / tmpScale;
  }
  __syncthreads();

  for (int frm=0; frm<empVecSize; frm += blockDim.x) {
    int serializedFrame = frm + threadIdx.x;
    int hpNum = serializedFrame / num_frames;
    if (hpNum < numEv) {
      emphasisVec[serializedFrame] = smBuffer[serializedFrame] * empScale[hpNum];
    }
  }  
  __syncthreads();

  if (threadIdx.x < numEv) {
    int zeroCnt = 0;
    float *myEmp = emphasisVec + threadIdx.x*num_frames;
    for (int i=num_frames-1; i>=0; i--) {
      if (myEmp[i] <= CENSOR_THRESHOLD) 
        zeroCnt++;
      else 
        break;
    }
    nonZeroEmphFrames[threadIdx.x] = num_frames - zeroCnt;
  }
  __syncthreads();
}


__device__
float instantSplineVal(float scaled_dt)
{
  float last_nuc_value = 0.0f;
  if ((scaled_dt>0.0f))
  {
    float scaled_dt_square = scaled_dt*scaled_dt;
    last_nuc_value = scaled_dt_square*(3.0f-2.0f*scaled_dt); //spline! with zero tangents at start and end points

    if (scaled_dt>1.0f)
      last_nuc_value =1.0f;
  }
  return(last_nuc_value);
}


__device__ 
int CalculateNucRise(
  const float tmidnuc,
  const float sigma,
  const float C,
  const float nuc_time_offset,
  const float *frame_times,
  const size_t numf,
  const int subSteps,
  float *nucRise
  )
{
   float tlast = 0.0f;
   float last_nuc_value = 0.0f;
   float my_inv_sigma = 1.0f/(3.0f*sigma); // bring back into range for ERF
   float scaled_dt = -1.0f;
   float scaled_end_dt = -1.0f;
   float scaled_end_nuc_time = nuc_time_offset*my_inv_sigma;
   float my_inv_sub_steps = 1.0f/((float)subSteps);
   bool start_uninitialized = true;
   int start = 0;

   for (int i=0; i < numf; i++)
   {
     // get the frame number of this data point (might be fractional because this point could be
     // the average of several frames of data.  This number is the average time of all the averaged
     // data points
     float t=frame_times[i];

     for (int st=1;st <= subSteps;st++)
     {
       float tnew = tlast + (t - tlast) * (float)st*my_inv_sub_steps;
       scaled_dt = (tnew - tmidnuc) * my_inv_sigma + 0.5f;
       scaled_end_dt = scaled_dt - scaled_end_nuc_time;

       last_nuc_value = instantSplineVal(scaled_dt);
       last_nuc_value -= instantSplineVal(scaled_end_dt);

       last_nuc_value *= C;
       *nucRise = last_nuc_value;
       nucRise++;
     }

     // first time point where we have a nonzero time
     if (start_uninitialized && (scaled_dt>0.0f))
     {
       start = i;
       start_uninitialized=false;
     }

     tlast = t;
   }

   return start;
}


