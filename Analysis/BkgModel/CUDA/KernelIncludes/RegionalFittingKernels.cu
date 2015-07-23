/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */


#include "RegionalFittingKernels.h"
#include "GpuPipelineDefines.h"

#define DEBUG_REG_FITTING 0
#define TMIDNUC_REG_STEP 0.01f
#define RDR_REG_STEP 0.01f
#define PDR_REG_STEP 0.001f

template<typename T>
__device__
double CalculateDotProduct(
  T *src1,
  T *src2,
  int len)
{
  double sum = 0;
  for (int i=0; i<len; ++i) {
    sum += src1[i]*src2[i];
  }
  return sum;
}


__device__ 
void BkgCorrectedRawTrace(
    const float *bkgTrace,
    const float *rawTrace,
    const float *beadParamCube,
    const float *regionFrameCube,
    const float *deltaFrames,
    const float darkness,
    const float etbR,
    const float gain,
    const float tauB,
    const int num_frames,
    const int beadFrameStride,
    const int regionFrameStride,
    float *correctedTrace
)
{
  float R = etbR - 1.0f;
  float dv = 0.0f;
  float dv_rs = 0.0f;
  float dvn = 0.0f;
  float aval;
  float curSbgVal, deltaFrameVal;
  for (int i=0; i<num_frames; ++i) {
    deltaFrameVal = LDG_ACCESS(deltaFrames, i);
    curSbgVal = LDG_ACCESS(bkgTrace, i);
    aval = deltaFrameVal/(2.0f * tauB);
    dvn = (R*curSbgVal - dv_rs/tauB - dv*aval) / (1.0f + aval);
    dv_rs += (dv+dvn) * deltaFrameVal * 0.5f;
    dv = dvn;
    correctedTrace[i] = rawTrace[i]
                            - ((dv+curSbgVal)*gain
                                + ApplyDarkMatterToFrame( beadParamCube,
                                    regionFrameCube,
                                    darkness,
                                    i,
                                    num_frames,
                                    beadFrameStride,
                                    regionFrameStride));
  }
}

__device__
const float* setAdaptiveEmphasis(
  float ampl,
  const float *emphasis, 
  int emphVecLen,
  int emphMax)
{
  int emAmp = (int)ampl;
  
  return ((emAmp > emphMax) ? emphasis + emphVecLen*emphMax : emphasis + emphVecLen*emAmp);
}

__device__ 
float CalculateBeadResidualError(
   float *observedTrace,
   float *modelTrace,
   const float *emphasis,
   int num_frames)
{
  float res = 0;
    for (int frm=0; frm < num_frames; ++frm) {
      float frm_res = (observedTrace[frm] - modelTrace[frm])*emphasis[frm];
      res += (frm_res * frm_res);
    }
  return sqrtf(res/num_frames);
}


__device__
void CalculatePartialDerivative(
  float *pdTmidNuc, 
  float *newTrace, 
  float *purpleTrace, 
  const float *emphForFitting, 
  const float stepSize,
  const int frames)
{
  for (int frm=0; frm<frames; ++frm) {
    pdTmidNuc[frm] = ((newTrace[frm] - purpleTrace[frm])*emphForFitting[frm]) 
                        / stepSize;
  }
}
 
__device__
void AccumulatePartialDerivatives(
  const float *pd,
  float *smReadBuffer,
  float *smWriteBuffer,
  const int samples,
  const int frames)
{
  for (int j=0; j<frames; ++j) {
    smReadBuffer[threadIdx.x] = pd[j]; // not coalsced..need to write to global mem cubes
    __syncthreads();

    SimplestReductionAndAverage(smReadBuffer, samples, false);

    if (threadIdx.x == 0) 
      smWriteBuffer[j] = smReadBuffer[0]; // reduce one syncthread here
    __syncthreads(); 
  }
}


// TODO Need to apply nonzero emphasis frames optimization
// TODO transpose emphasis
// TODO dense layout should benefit here with lots of reductions

__device__ 
void MultiFlowRegionalLevMarFit(
  float *scratchSpace,
  const short *observedTrace, // NUM_SAMPLES_RF x F
  const float *BeadParamCube, //Copies, R, dmult, gain, tau_adj, phi, stride == beadFrameStride
  const unsigned short *BeadStateCube, //key_norm,  ppf, ssq
  const float *emphasisVec, //(MAX_POISSON_TABLE_COL)*F
  const int *nonZeroEmphFrames,
  float *nucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
  const ConstantParamsRegion *constRegP,
  PerFlowParamsRegion *perFlowRegP,
  const PerNucParamsRegion *perNucRegP,
  const float *RegionFrameCube,  //bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
  const float *EmptyTraceAvg,  //bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
  const size_t beadFrameStride, //stride from one CUBE plane to the next for the Per Well Cubes
  const size_t regionFrameStride, //, //stride in Region Frame Cube to get to next parameter
  const size_t num_frames, // 4
  const size_t numFlows,
  const size_t samples)
{
  //__shared__ float smBuffer[256];
  __shared__ double smBuffer[256];
  __shared__ float smNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU]; 
  __shared__ float tmpNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU];
  //__shared__ float deltas[3]; // NUM_PARAMS=3
  __shared__ double deltas[3]; // NUM_PARAMS=3
  __shared__ bool cont_lambda_itr;
  __shared__ bool nan_detected;
  __shared__ bool solved;


  // zero out shared memory for reductions
  for (size_t i=threadIdx.x; i<256; i+=blockDim.x) {
    smBuffer[i] = 0;
  }

  if (threadIdx.x == 0) {
    perFlowRegP->setTMidNucShift(0);
  }

  __syncthreads();

  float correctedTrace[MAX_COMPRESSED_FRAMES_GPU];
  float obsTrace[MAX_COMPRESSED_FRAMES_GPU]; // raw traces being written to
  float tmpTrace[MAX_COMPRESSED_FRAMES_GPU]; // raw traces being written to
  float purpleTrace[MAX_COMPRESSED_FRAMES_GPU];
  float pdTmidNuc[MAX_COMPRESSED_FRAMES_GPU];
  float pdRDR[MAX_COMPRESSED_FRAMES_GPU];
  float pdPDR[MAX_COMPRESSED_FRAMES_GPU];
  float yerr[MAX_COMPRESSED_FRAMES_GPU];
  
  // right now getting bead params in the order they were in bead_params struct
  const float copies = *(BeadParamCube + BpCopies*beadFrameStride);
  const float R = *(BeadParamCube + BpR*beadFrameStride);
  const float d = (*(BeadParamCube + BpDmult*beadFrameStride)) * perNucRegP->getD(); // effective diffusion
  const float gain = *(BeadParamCube + BpGain*beadFrameStride);

  // calculate empty to bead ratio, buffering and copies
  const float C = perNucRegP->getC();
  const float nuc_flow_span = ConstGlobalP.getNucFlowSpan();
  const float sigma = ComputeSigma(perFlowRegP, perNucRegP);

  float tmidNuc = ComputeMidNucTime(perFlowRegP->getTMidNuc(), perFlowRegP, perNucRegP); 
  float etbR = ComputeETBR(perNucRegP, perFlowRegP->getRatioDrift(), R, copies);
  float tauB = ComputeTauB(constRegP, etbR);
  float SP = ComputeSP(perFlowRegP->getCopyDrift(), copies);
 
  // Need shifted background  
  const float* bkgTrace = EmptyTraceAvg;//RegionFrameCube + RfBkgTraces*regionFrameStride;
  const float* deltaFrames = RegionFrameCube + RfDeltaFrames*regionFrameStride;
  const float* frameNumber = RegionFrameCube + RfFrameNumber*regionFrameStride;

  // background subtracted trace for amplitude estimation

  // calculate initial nucRise here
  if (threadIdx.x == 0) {
#if DEBUG_REG_FITTING
   printf("C: %f sigma: %f, tmidNuc: %f\n", C, sigma, tmidNuc);
   printf("copies: %f R: %f d: %f gain: %f, etbR: %f tauB: %f\n", copies, R, d, gain, etbR, tauB);
#endif
    smBuffer[0] = CalculateNucRise(
        tmidNuc,
        sigma,
	C,
	nuc_flow_span,
	frameNumber,
	num_frames,
        ISIG_SUB_STEPS_MULTI_FLOW,
	smNucRise);

  }
  __syncthreads();
  
  int nucStart = smBuffer[0];

  __syncthreads();

  // DEBUG
#if DEBUG_REG_FITTING
  if (threadIdx.x == 1) {
    printf("GPU before fitting...start: %d, tmidnuc: %f rdr: %f pdr: %f\n",
        nucStart,perFlowRegP->getTMidNuc(), perFlowRegP->getRatioDrift(), perFlowRegP->getCopyDrift());

    printf("Nucrise\n");
    for (int i=0; i<(ISIG_SUB_STEPS_MULTI_FLOW*num_frames); ++i) {
      printf("%f ",smNucRise[i]);
    }
    printf("\n");
    printf("Emphasis\n");
    for (int i=0; i<num_frames; ++i) {
      printf("%f ",emphasisVec[i]);
    }
    printf("\n");
    printf("Shifted Bkg\n");
    for (int i=0; i<num_frames; ++i) {
      printf("%f ",bkgTrace[i]);
    }
    printf("\n");
  }
  __syncthreads();
#endif
  // END DEBUG

  // read raw traces
  for (int i=0; i<num_frames; ++i) {
    obsTrace[i] = (float)(*observedTrace);
    observedTrace += beadFrameStride;
  }

  // START AMPLITUDE ESTIMATION
  BkgCorrectedRawTrace(
      bkgTrace,
      obsTrace,
      BeadParamCube,
      RegionFrameCube,
      deltaFrames,
      perFlowRegP->getDarkness(),
      etbR,
      gain,
      tauB,
      num_frames,
      beadFrameStride,
      regionFrameStride,
      correctedTrace);
 
#if DEBUG_REG_FITTING
  if (threadIdx.x == 0) {
    printf("====>tid: %d,", threadIdx.x);
    for (int i=0; i<num_frames; ++i) {
      printf("%f ", correctedTrace[i]);
    } 
    printf("\n");
  }
  __syncthreads();
#endif

  //Provide emphasis stride to projection search...the transposed layout is 
  // used in the below function because of single flow fit
  float ampl  = ProjectionSearch(
      constRegP,
      perFlowRegP,
      perNucRegP,
      correctedTrace,
      emphasisVec,
      num_frames,
      smNucRise,
      deltaFrames,
      1.0f,
      d,
      tauB,
      gain,
      SP,
      tmpTrace,
      nucStart,
      beadFrameStride,
      1, // emphasis stride
      ISIG_SUB_STEPS_MULTI_FLOW
      );

#if DEBUG_REG_FITTING
    printf("====> GPU....tid: %d Ampl: %f\n", threadIdx.x, ampl);
#endif

  // END AMPLITUDE ESTIMATION 

  // select emphasis now based on Ampl
  // TODO check if max emphasis is correct
  const float *emphForFitting = setAdaptiveEmphasis(ampl, emphasisVec, num_frames, MAX_POISSON_TABLE_COL - 1);

  // calculate starting regional residual
  bool goodBead = true;
  ComputeModelBasedTrace(
      bkgTrace,
      deltaFrames,
      constRegP,
      perNucRegP,
      BeadParamCube,
      RegionFrameCube,
      smNucRise,
      nucStart,
      perNucRegP->getKrate(),
      tauB,
      gain,
      SP,
      d,
      perFlowRegP->getDarkness(),
      etbR,
      constRegP->getSens()*SENSMULTIPLIER,
      ampl,
      ISIG_SUB_STEPS_MULTI_FLOW * nucStart,
      ISIG_SUB_STEPS_MULTI_FLOW,
      num_frames,
      beadFrameStride,
      regionFrameStride,
      purpleTrace);

  float beadRes = CalculateBeadResidualError(
      obsTrace,
      purpleTrace,
      emphForFitting,
      num_frames);
  smBuffer[threadIdx.x] = beadRes;
  __syncthreads();

  // reduce here for average residual value
  // reduction has slightly wrong logic currently with assuming 256 threads being spawned
  SimplestReductionAndAverage(smBuffer, samples, false);
  float curRes = smBuffer[0];
  float curAvgRes = curRes / (float)(samples);
  
  __syncthreads();

#if DEBUG_REG_FITTING
  if (threadIdx.x == 0)
    printf("====> GPU....Before fitting residual: %f\n", curAvgRes);
#endif
  
  // Lev mar iterations loop
  float ratioDrift, copyDrift;
  float new_tmidnuc, new_ratiodrift, new_copydrift;
  //float lambda = 0.0001f;
  double lambda = 0.0001;
  float *oldTrace = purpleTrace;
  float *newTrace = tmpTrace;
  for (int iter=0; iter<4; ++iter) {

    // possibly filter beads at this point
    // residual not changing or corrupt or ...
    goodBead = beadRes < 4.0f*curAvgRes;

    tmidNuc = perFlowRegP->getTMidNuc();
    ratioDrift = perFlowRegP->getRatioDrift();
    copyDrift = perFlowRegP->getCopyDrift(); 

#if DEBUG_REG_FITTING
    if (threadIdx.x == 0) {
      printf("====GPU REG Fitting...iter:%d, tmidNuc:%f, rdr:%f, pdr:%f\n", iter, tmidNuc, 
          ratioDrift, copyDrift);
    }
    __syncthreads();
#endif

    // START YERR
    CalculatePartialDerivative(
      yerr, 
      obsTrace, 
      oldTrace, 
      emphForFitting, 
      1.0f,
      num_frames);
    // END YERR

    // START TMIDNUC PARTIAL DERIVATIVE
    new_tmidnuc = ComputeMidNucTime(
        tmidNuc + TMIDNUC_REG_STEP, 
        perFlowRegP, 
        perNucRegP);
    if (threadIdx.x == 0) {
      smBuffer[0] = CalculateNucRise(
          new_tmidnuc,
          sigma,
	  C,
	  nuc_flow_span,
	  frameNumber,
	  num_frames,
          ISIG_SUB_STEPS_MULTI_FLOW,
	  tmpNucRise);
    }
    __syncthreads();
    int tmp_nucStart = smBuffer[0];
    __syncthreads();

    ComputeModelBasedTrace(
      bkgTrace,
      deltaFrames,
      constRegP,
      perNucRegP,
      BeadParamCube,
      RegionFrameCube,
      tmpNucRise,
      tmp_nucStart,
      perNucRegP->getKrate(),
      tauB,
      gain,
      SP,
      d,
      perFlowRegP->getDarkness(),
      etbR,
      constRegP->getSens()*SENSMULTIPLIER,
      ampl,
      ISIG_SUB_STEPS_MULTI_FLOW * tmp_nucStart,
      ISIG_SUB_STEPS_MULTI_FLOW,
      num_frames,
      beadFrameStride,
      regionFrameStride,
      newTrace);


    CalculatePartialDerivative(
      pdTmidNuc, 
      newTrace, 
      oldTrace, 
      emphForFitting, 
      TMIDNUC_REG_STEP,
      num_frames);

    // END TMIDNUC PARTIAL DERIVATIVE
   
    
    // START RATODRIFT DERIVATIVE
      
    new_ratiodrift = ratioDrift + RDR_REG_STEP; 
    etbR = ComputeETBR(perNucRegP, new_ratiodrift, R, copies);
    tauB = ComputeTauB(constRegP, etbR);
    
    ComputeModelBasedTrace(
      bkgTrace,
      deltaFrames,
      constRegP,
      perNucRegP,
      BeadParamCube,
      RegionFrameCube,
      smNucRise,
      nucStart,
      perNucRegP->getKrate(),
      tauB,
      gain,
      SP,
      d,
      perFlowRegP->getDarkness(),
      etbR,
      constRegP->getSens()*SENSMULTIPLIER,
      ampl,
      ISIG_SUB_STEPS_MULTI_FLOW * nucStart,
      ISIG_SUB_STEPS_MULTI_FLOW,
      num_frames,
      beadFrameStride,
      regionFrameStride,
      newTrace);

    CalculatePartialDerivative(
      pdRDR, 
      newTrace, 
      oldTrace, 
      emphForFitting, 
      RDR_REG_STEP,
      num_frames);

    // END RATODRIFT DERIVATIVE

    // START COPYDRIFT DERIVATIVE
    new_copydrift = copyDrift + PDR_REG_STEP;
    etbR = ComputeETBR(perNucRegP, ratioDrift, R, copies);
    tauB = ComputeTauB(constRegP, etbR);
    SP = ComputeSP(new_copydrift, copies);

    ComputeModelBasedTrace(
      bkgTrace,
      deltaFrames,
      constRegP,
      perNucRegP,
      BeadParamCube,
      RegionFrameCube,
      smNucRise,
      nucStart,
      perNucRegP->getKrate(),
      tauB,
      gain,
      SP,
      d,
      perFlowRegP->getDarkness(),
      etbR,
      constRegP->getSens()*SENSMULTIPLIER,
      ampl,
      ISIG_SUB_STEPS_MULTI_FLOW * nucStart,
      ISIG_SUB_STEPS_MULTI_FLOW,
      num_frames,
      beadFrameStride,
      regionFrameStride,
      newTrace);

    CalculatePartialDerivative(
      pdPDR, 
      newTrace, 
      oldTrace, 
      emphForFitting, 
      PDR_REG_STEP,
      num_frames);

    // END COPYDRIFT DERIVATIVE

      
#if DEBUG_REG_FITTING
      // DEBUG
      if (threadIdx.x == 0) {
        for (int i=0; i<num_frames; ++i) {
          printf("%f,",oldTrace[i]);
        }
        printf("\n");
        for (int i=0; i<num_frames; ++i) {
          printf("%f,",pdTmidNuc[i]);
        }
        printf("\n");
        for (int i=0; i<num_frames; ++i) {
          printf("%f,",pdRDR[i]);
        }
        printf("\n");
        for (int i=0; i<num_frames; ++i) {
          printf("%f,",pdPDR[i]);
        }
        printf("\n");
        for (int i=0; i<num_frames; ++i) {
          printf("%f,",yerr[i]);
        }
        printf("\n");
      }
      __syncthreads();
#endif
     
      // Calculate JTJ matrix entries
      //float lhs_00=0, lhs_01=0, lhs_02=0, lhs_11=0, lhs_12=0, lhs_22=0;
      double lhs_00=0, lhs_01=0, lhs_02=0, lhs_11=0, lhs_12=0, lhs_22=0;
      //float rhs_0=0, rhs_1=0, rhs_2=0;
      double rhs_0=0, rhs_1=0, rhs_2=0;
      smBuffer[threadIdx.x] = goodBead ? CalculateDotProduct(
          pdTmidNuc,
          pdTmidNuc,
          num_frames) : 0;
      __syncthreads();

#if DEBUG_REG_FITTING
      if (threadIdx.x == 0) {
        printf("sum:%f\n", smBuffer[0]);
      }
      __syncthreads();
#endif

      SimplestReductionAndAverage(smBuffer, samples, false);
      if (threadIdx.x == 0)
        lhs_00 = smBuffer[0];
      __syncthreads();

      smBuffer[threadIdx.x] = goodBead ? CalculateDotProduct(
          pdTmidNuc,
          pdRDR,
          num_frames) : 0;
      __syncthreads();

#if DEBUG_REG_FITTING
      if (threadIdx.x == 0) {
        printf("sum:%f\n", smBuffer[0]);
      }
      __syncthreads();
#endif

      SimplestReductionAndAverage(smBuffer, samples, false);
      if (threadIdx.x == 0)
        lhs_01 = smBuffer[0];
      __syncthreads();

      smBuffer[threadIdx.x] = goodBead ? CalculateDotProduct(
          pdTmidNuc,
          pdPDR,
          num_frames) : 0;
      __syncthreads();
#if DEBUG_REG_FITTING
      if (threadIdx.x == 0) {
        printf("sum:%f\n", smBuffer[0]);
      }
      __syncthreads();
#endif
      SimplestReductionAndAverage(smBuffer, samples, false);
      if (threadIdx.x == 0)
        lhs_02 = smBuffer[0];
      __syncthreads();

      smBuffer[threadIdx.x] = goodBead ? CalculateDotProduct(
          pdRDR,
          pdRDR,
          num_frames) : 0;
      __syncthreads();
#if DEBUG_REG_FITTING
      if (threadIdx.x == 0) {
        printf("sum:%f\n", smBuffer[0]);
      }
      __syncthreads();
#endif
      SimplestReductionAndAverage(smBuffer, samples, false);
      if (threadIdx.x == 0)
        lhs_11 = smBuffer[0];
      __syncthreads();

      smBuffer[threadIdx.x] = goodBead ? CalculateDotProduct(
          pdRDR,
          pdPDR,
          num_frames) : 0;
      __syncthreads();
#if DEBUG_REG_FITTING
      if (threadIdx.x == 0) {
        printf("sum:%f\n", smBuffer[0]);
      }
      __syncthreads();
#endif
      SimplestReductionAndAverage(smBuffer, samples, false);
      if (threadIdx.x == 0)
        lhs_12 = smBuffer[0];
      __syncthreads();

      smBuffer[threadIdx.x] = goodBead ? CalculateDotProduct(
          pdPDR,
          pdPDR,
          num_frames) : 0;
      __syncthreads();
#if DEBUG_REG_FITTING
      if (threadIdx.x == 0) {
        printf("sum:%f\n", smBuffer[0]);
      }
      __syncthreads();
#endif
      SimplestReductionAndAverage(smBuffer, samples, false);
      if (threadIdx.x == 0)
        lhs_22 = smBuffer[0];
      __syncthreads();

      smBuffer[threadIdx.x] = goodBead ? CalculateDotProduct(
          yerr,
          pdTmidNuc,
          num_frames) : 0;
      __syncthreads();
#if DEBUG_REG_FITTING
      if (threadIdx.x == 0) {
        printf("sum:%f\n", smBuffer[0]);
      }
      __syncthreads();
#endif
      SimplestReductionAndAverage(smBuffer, samples, false);
      if (threadIdx.x == 0)
        rhs_0 = smBuffer[0];
      __syncthreads();
    
      smBuffer[threadIdx.x] = goodBead ? CalculateDotProduct(
          yerr,
          pdRDR,
          num_frames) : 0;
      __syncthreads();
#if DEBUG_REG_FITTING
      if (threadIdx.x == 0) {
        printf("sum:%f\n", smBuffer[0]);
      }
      __syncthreads();
#endif
      SimplestReductionAndAverage(smBuffer, samples, false);
      if (threadIdx.x == 0)
        rhs_1 = smBuffer[0];
      __syncthreads();

      smBuffer[threadIdx.x] = goodBead ? CalculateDotProduct(
          yerr,
          pdPDR,
          num_frames) : 0;
      __syncthreads();
#if DEBUG_REG_FITTING
      if (threadIdx.x == 0) {
        printf("sum:%f\n", smBuffer[0]);
      }
      __syncthreads();
#endif
      SimplestReductionAndAverage(smBuffer, samples, false);
      if (threadIdx.x == 0)
        rhs_2 = smBuffer[0];
      __syncthreads();


      if (threadIdx.x == 0)
      {
        cont_lambda_itr = true;
        nan_detected = false;
        solved = false;
      }
      __syncthreads();

      // multiply jtj matrix by lambda
      // solve for delta change in parameters
      // calculate new parameters and clamp them within boundaries
      // find new residual and compare to current one
      // if residual decreases this iteration is done else increase/decrease lambda accordingly and 
      // back to lambda multiply step

      // since solve in three variable, inlining matrix inverse 
      // rather than going for LUT decomposition

      float new_residual = 0;
      float newBeadRes = 0;
      while (cont_lambda_itr) {
        if (threadIdx.x == 0) {
          /*float new_lhs00 = lhs_00 * (1.0f + lambda);
          float new_lhs11 = lhs_11 * (1.0f + lambda);
          float new_lhs22 = lhs_22 * (1.0f + lambda);*/
#if DEBUG_REG_FITTING
          printf("jtj: %f,%f,%f,%f,%f,%f,%f,%f,%f\n", lhs_00,lhs_01,lhs_02,lhs_11,lhs_12,lhs_22,rhs_0,rhs_1,rhs_2);
#endif
          double new_lhs00 = lhs_00 * (1.0 + lambda);
          double new_lhs11 = lhs_11 * (1.0 + lambda);
          double new_lhs22 = lhs_22 * (1.0 + lambda);
        
          // calculate determinant
          /*float det = new_lhs00*(new_lhs11*new_lhs22 - lhs_12*lhs_12) - 
                lhs_01*(lhs_01*new_lhs22 - lhs_12*lhs_02) +
                lhs_02*(lhs_01*lhs_12 - new_lhs11*lhs_02);
          det = 1.0f/det;*/
          double det = new_lhs00*(new_lhs11*new_lhs22 - lhs_12*lhs_12) - 
                lhs_01*(lhs_01*new_lhs22 - lhs_12*lhs_02) +
                lhs_02*(lhs_01*lhs_12 - new_lhs11*lhs_02);
          det = 1.0/det;
         
          deltas[0] = det*(rhs_0*(new_lhs11*new_lhs22 - lhs_12*lhs_12) +
                   rhs_1*(lhs_02*lhs_12 - lhs_01*new_lhs22) +
                   rhs_2*(lhs_01*lhs_12 - lhs_02*new_lhs11));
          deltas[1] = det*(rhs_0*(lhs_12*lhs_02 - lhs_01*new_lhs22) +
                   rhs_1*(new_lhs00*new_lhs22 - lhs_02*lhs_02) +
                   rhs_2*(lhs_01*lhs_02 - new_lhs00*lhs_12));
          deltas[2] = det*(rhs_0*(lhs_01*lhs_12 - lhs_02*new_lhs11) +
                   rhs_1*(lhs_01*lhs_02 - new_lhs00*lhs_12) +
                   rhs_2*(new_lhs00*new_lhs11 - lhs_01*lhs_01));
      
           if (isnan(deltas[0]) || isnan(deltas[1]) || isnan(deltas[2]))
             nan_detected = true;
#if DEBUG_REG_FITTING
          printf("===GPU REG Params...iter:%d,delta0:%f,delta1:%f,delta2:%f\n", iter, deltas[0], deltas[1], deltas[2]);
#endif
        }  
        __syncthreads();

        if (!nan_detected) {
          new_tmidnuc = tmidNuc + deltas[0];
          new_ratiodrift = ratioDrift + deltas[1];  
          new_copydrift = copyDrift + deltas[2];

          // clamp the parameters here
          clampT(new_tmidnuc, constRegP->getMinTmidNuc(), constRegP->getMaxTmidNuc());
          clampT(new_ratiodrift, constRegP->getMinRatioDrift(), constRegP->getMaxRatioDrift());
          clampT(new_copydrift, constRegP->getMinCopyDrift(), constRegP->getMaxCopyDrift());

          // compute residual
          if (threadIdx.x == 0) {
            smBuffer[0] = CalculateNucRise(
                ComputeMidNucTime(new_tmidnuc, perFlowRegP, perNucRegP),
		sigma,
		C,
		nuc_flow_span,
		frameNumber,
		num_frames,
                ISIG_SUB_STEPS_MULTI_FLOW,
		tmpNucRise);
	  }
          __syncthreads();
          tmp_nucStart = smBuffer[0];
          __syncthreads();
          etbR = ComputeETBR(perNucRegP, new_ratiodrift, R, copies);
          tauB = ComputeTauB(constRegP, etbR);
          SP = ComputeSP(new_copydrift, copies);
        
          ComputeModelBasedTrace(
            bkgTrace,
	    deltaFrames,
	    constRegP,
	    perNucRegP,
	    BeadParamCube,
	    RegionFrameCube,
	    tmpNucRise,
	    tmp_nucStart,
	    perNucRegP->getKrate(),
	    tauB,
	    gain,
	    SP,
	    d,
	    perFlowRegP->getDarkness(),
	    etbR,
	    constRegP->getSens()*SENSMULTIPLIER,
	    ampl,
	    ISIG_SUB_STEPS_MULTI_FLOW * tmp_nucStart,
	    ISIG_SUB_STEPS_MULTI_FLOW,
	    num_frames,
	    beadFrameStride,
	    regionFrameStride,
	    newTrace);

            newBeadRes = CalculateBeadResidualError(
                obsTrace,
		newTrace,
		emphForFitting,
		num_frames);
            smBuffer[threadIdx.x] = newBeadRes;
	    __syncthreads();

            // reduce here for average residual value
            SimplestReductionAndAverage(smBuffer, samples, false);
            //curAvgRes = smBuffer[0] / (float)(samples);
            new_residual = smBuffer[0];
            __syncthreads();
        }
            
        if (threadIdx.x == 0) {
          //new_residual = smBuffer[0];
            
          // DEBUG 
#if DEBUG_REG_FITTING
          printf("===GPU REG Params...iter:%d,tmidnuc:%f,rdr:%f,pdr:%f,old_residual:%f,new_residual:%f\n", iter, new_tmidnuc, new_ratiodrift, new_copydrift, curRes, new_residual);
#endif

          if (!nan_detected && new_residual < curRes) {
            solved = true;
            //curRes = new_residual;
            lambda /= 30.0; // use correct lambda step from bkgmodel
            if (lambda < FLT_MIN)
              lambda = FLT_MIN;

            // update parameters
            perFlowRegP->setTMidNuc(new_tmidnuc); 
            perFlowRegP->setRatioDrift(new_ratiodrift);
            perFlowRegP->setCopyDrift(new_copydrift);

            // update nucrise
            smBuffer[0] = tmp_nucStart;
            for (int i=0; i<num_frames; ++i) {
              smNucRise[i] = tmpNucRise[i];
            }

            cont_lambda_itr = false;
          }
          else {
            lambda *= 30.0;
            if (lambda > 1E+9f) {
              cont_lambda_itr = false;
              smBuffer[0] = nucStart;
              solved = false;
            }  
          }
          nan_detected = false;
        }
        __syncthreads();
      }   
      nucStart = smBuffer[0];
      if (solved) {
        float *tmp = oldTrace;
        oldTrace = newTrace;
        newTrace = tmp;
  
        // update residuals for next iteration
        beadRes = newBeadRes;
        curRes = new_residual;
        curAvgRes = curRes / (float)(samples);
      }
      __syncthreads();

   }   
} 


__device__ 
void UpdateNucRiseForSingleFlowFit(
  const ConstantParamsRegion * constRegP,
  const PerNucParamsRegion * perNucRegP,
  PerFlowParamsRegion * perFlowRegP,
  const float * RegionFrameCube,
  const int RegionFrameStride,
  const int num_frames,
  float *nucRise)
{
  perFlowRegP->setStart(CalculateNucRise(
      ComputeMidNucTime(perFlowRegP->getTMidNuc(), perFlowRegP, perNucRegP),
      ComputeSigma(perFlowRegP, perNucRegP),
      perNucRegP->getC(),
      ConstGlobalP.getNucFlowSpan(),
      RegionFrameCube + RfFrameNumber*RegionFrameStride,
      num_frames,
      ISIG_SUB_STEPS_SINGLE_FLOW,
      nucRise));
}


__global__ 
void PerformMultiFlowRegionalFitting(
  const unsigned short * RegionMask,
  const short *observedTrace, // REGIONS X NUM_SAMPLES_RF x F
  const float *beadParamCube,
  const unsigned short *beadStateCube,
  const float *emphasisVec, //(MAX_POISSON_TABLE_COL)*F
  const int *nonZeroEmphFrames, 
  float *nucRise,
  float *scratchSpace,
  const size_t *numFramesRegion,
  const ConstantParamsRegion * constRegP,
  PerFlowParamsRegion * perFlowRegP,
  const PerNucParamsRegion * perNucRegP,
  const float * RegionFrameCube,
  const float * EmptyTraceRegion,
  const int *NumSamples,
  const size_t numFlows
)
{
  // each region is fitted by one thread block
  
  const size_t regId = blockIdx.x;
  const size_t beadId = threadIdx.x;

  if( LDG_ACCESS(RegionMask,regId) != RegionMaskLive) return;
  if (beadId >= NumSamples[regId])
    return;
  //strides
  const size_t BeadFrameStride = ( ImgRegP.getGridParam( NUM_SAMPLES_RF )).getPlaneStride();
  const size_t RegionFrameStride = ConstFrmP.getMaxCompFrames() * ImgRegP.getNumRegions();

  //if EmptyTraces from GenerateBeadTrace Kernel padding is uncompressed frames
  const float * emptyTraceAvg = EmptyTraceRegion + regId*ConstFrmP.getUncompFrames();
  RegionFrameCube += regId*ConstFrmP.getMaxCompFrames();  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber

  //update per region pointers
  constRegP += regId;
  perFlowRegP += regId;

  // update per region pointer depending on nuc id
  perNucRegP +=  ImgRegP.getNumRegions() * ConstFlowP.getNucId() + regId;


  nonZeroEmphFrames += regId*MAX_POISSON_TABLE_COL;
  emphasisVec += regId * MAX_POISSON_TABLE_COL * ConstFrmP.getMaxCompFrames();

  const size_t numf = numFramesRegion[regId];

  // bead specific pointers
  beadStateCube += NUM_SAMPLES_RF*regId + threadIdx.x;
  beadParamCube += NUM_SAMPLES_RF*regId + threadIdx.x;
  observedTrace += NUM_SAMPLES_RF*regId + threadIdx.x;

  float *multiFlowNucRise = nucRise + regId *  ISIG_SUB_STEPS_MULTI_FLOW * ConstFrmP.getMaxCompFrames() ;
  MultiFlowRegionalLevMarFit(
    scratchSpace,
    observedTrace,
    beadParamCube,
    beadStateCube,
    emphasisVec,
    nonZeroEmphFrames,
    multiFlowNucRise,
    constRegP,
    perFlowRegP,
    perNucRegP,
    RegionFrameCube,
    emptyTraceAvg,
    BeadFrameStride,
    RegionFrameStride,
    numf,
    numFlows,
    NumSamples[regId]);

  if (beadId == 0) {
    UpdateNucRiseForSingleFlowFit(
      constRegP,
      perNucRegP,
      perFlowRegP,
      RegionFrameCube,
      RegionFrameStride,
      numf,
      nucRise + regId *  ISIG_SUB_STEPS_SINGLE_FLOW * ConstFrmP.getMaxCompFrames());  
  };
}

