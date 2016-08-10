/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */


#include "RegionalFittingKernels.h"
#include "GpuPipelineDefines.h"
#include "UtilKernels.h"

#define FIT_COPYDRIFT 0
#define NUM_TIME_VARYING_PARAMS 3
#define DEBUG_REG_FITTING 0
#define TMIDNUC_REG_STEP 0.01f
#define RDR_REG_STEP 0.01f
#define PDR_REG_STEP 0.001f
#define REGLEVMARMAT_LHS_LEN (((NUM_TIME_VARYING_PARAMS)*(NUM_TIME_VARYING_PARAMS + 1)) / 2)
#define REGLEVMARMAT_RHS_LEN (NUM_TIME_VARYING_PARAMS)
#define LEVMARITERS 4
#define REG_FIT_SM_ACCUM_BUFFERSIZE 256

template<typename T, const size_t len = 6>
struct OneDimVec {
  T val[len];

  __device__ 
  OneDimVec() {
    for (size_t i=0; i<len; ++i)
      val[i] = 0;
  }
  
  __device__ __inline__
  void clear() {
    for (size_t i=0; i<len; ++i)
      val[i] = 0;
  }
};

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
float CalculateTmidNucShiftFitResidual(
  float *obsTrace,
  float *modelTrace,
  float *err,
  int num_frames
)
{
  float residual = 0;
  for (int i=0; i<num_frames; ++i) {
    float e = obsTrace[i] - modelTrace[i];
    residual += e*e;
    err[i] = e;
  }

  return (residual / (float)(num_frames));
}

__device__ 
void BkgCorrectedRawTrace(
    const float *bkgTrace,
    const short *rawTrace,
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
    correctedTrace[i] = (float)(*rawTrace)
                            - ((dv+curSbgVal)*gain
                                + ApplyDarkMatterToFrame( beadParamCube,
                                    regionFrameCube,
                                    darkness,
                                    i,
                                    num_frames,
                                    beadFrameStride,
                                    regionFrameStride));
    rawTrace += beadFrameStride;
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
  return res;
}

__device__ 
float CalculateBeadResidualError(
   const short *observedTrace,
   const float *modelTrace,
   const float *emphasis,
   const size_t obsTrStride,
   const size_t num_frames)
{
  float res = 0;
    for (size_t frm=0; frm < num_frames; ++frm) {
      float frm_res = ((float)(*observedTrace) - modelTrace[frm])*emphasis[frm];
      res += (frm_res * frm_res);
      observedTrace += obsTrStride;
    }
  return res;
}

__device__
void CalculateYerr(
  float *pd, 
  const short *obsTrace, 
  const float *oldTrace, 
  const float *emphForFitting, 
  const float stepSize,
  const size_t stride,
  const size_t frames)
{
  for (size_t frm=0; frm<frames; ++frm) {
    pd[frm] = (((float)(*obsTrace) - oldTrace[frm])*emphForFitting[frm]) 
                        / stepSize;
    obsTrace += stride;
  }
}
 
__device__
void CalculatePartialDerivative(
  float *pd, 
  const float *newTrace, 
  const float *oldTrace, 
  const float *emphForFitting, 
  const float stepSize,
  const size_t frames)
{
  if (emphForFitting) {
    for (size_t frm=0; frm<frames; ++frm) 
      pd[frm] = ((newTrace[frm] - oldTrace[frm])*emphForFitting[frm]) 
                        / stepSize;
  }
  else {
    for (size_t frm=0; frm<frames; ++frm) 
      pd[frm] = (newTrace[frm] - oldTrace[frm]) 
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

__device__
void FitTmidNucShiftPerFlow(
  const int realFnum,
  const float ampl,
  const short *obsTrace,
  const float *beadParamCube,
  const unsigned short *beadStateCube,
  const float *emptyTraceAvg,
  const float *emphasisVec,
  const PerNucParamsRegion *perNucRegP,
  const ConstantParamsRegion *constRegP,
  const float *regionFrameCube,
  const size_t beadFrameStride,
  const size_t regionFrameStride,
  const size_t num_frames,
  const size_t samples,
  PerFlowParamsRegion *perFlowRegP)
{
  __shared__ float smAvgCopies[512];
  __shared__ float smAvgR[512];
  __shared__ int smOnemerCount[512];

  float correctedTrace[MAX_COMPRESSED_FRAMES_GPU];
  
  // right now getting bead params in the order they were in bead_params struct
  const float copies = *(beadParamCube + BpCopies*beadFrameStride);
  const float R = *(beadParamCube + BpR*beadFrameStride);
  const float gain = *(beadParamCube + BpGain*beadFrameStride);

  // calculate empty to bead ratio, buffering and copies
  const float C = perNucRegP->getC();
  const float nuc_flow_span = ConstGlobalP.getNucFlowSpan();
  const float sigma = ComputeSigma(perFlowRegP, perNucRegP);

  float tmidNuc = ComputeMidNucTime(perFlowRegP->getTMidNuc(), perFlowRegP, perNucRegP); 
  float etbR = ComputeETBR(perNucRegP, perFlowRegP->getRatioDrift(), R, copies, realFnum);
  float tauB = ComputeTauB(constRegP, etbR);
 
  // Need shifted background  
  const float* deltaFrames = regionFrameCube + RfDeltaFrames*regionFrameStride;
  const float* frameNumber = regionFrameCube + RfFrameNumber*regionFrameStride;

  BkgCorrectedRawTrace(
    emptyTraceAvg,
    obsTrace,
    beadParamCube,
    regionFrameCube,
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
    printf("Unaverage corrected trace\n");
    for (int i=0; i<num_frames; ++i) {
      printf(" %f", correctedTrace[i]);  
    }
    printf("Unaverage raw trace\n");
    for (int i=0; i<num_frames; ++i) {
      printf(" %f", (float)(*(obsTrace + i*beadFrameStride)));  
    }
    printf("Empty trace\n");
    for (int i=0; i<num_frames; ++i) {
      printf(" %f", emptyTraceAvg[i]);  
    }
    printf("\n");
  }
  __syncthreads();
#endif

  // calculate avergae bead params and average coorected trace

  bool accum = false;
  if (ampl > 0.5f && ampl < 1.5f) {
    accum = true;
    smOnemerCount[threadIdx.x] = 1;
    smAvgCopies[threadIdx.x] = copies;
    smAvgR[threadIdx.x] = R;
  }
  else {
    smOnemerCount[threadIdx.x] = 0;
    smAvgCopies[threadIdx.x] = 0;
    smAvgR[threadIdx.x] = 0;
  }

  __syncthreads();

  SimplestReductionAndAverage(smOnemerCount, samples, false);
  SimplestReductionAndAverage(smAvgCopies, samples, false);
  SimplestReductionAndAverage(smAvgR, samples, false);

  int oneMerCount = smOnemerCount[0];
  float avgCopies = smAvgCopies[0] / oneMerCount;
  float avgR = smAvgR[0] / oneMerCount;

  float *smFrameAvg = &smAvgR[0];
  for (size_t i =0; i< num_frames; ++i) {
    if (accum)
      smFrameAvg[threadIdx.x] = correctedTrace[i];
    else
      smFrameAvg[threadIdx.x] = 0;

    SimplestReductionAndAverage(smFrameAvg, samples, false);
    correctedTrace[i] = smFrameAvg[0] / oneMerCount;
  }

  if (threadIdx.x == 0) {

#if DEBUG_REG_FITTING
    printf("===> count:%d, copies:%f, R:%f \n", oneMerCount, avgCopies,avgR);
    printf("====> Tmid nuc corrected trace\n");
    for (size_t i =0; i< num_frames; ++i)
      printf("%f ", correctedTrace[i]);
    printf("\n");
#endif

    // corrected trace is average trace now
    // perform lev mar on this average 1-mer bead

    float fineNucRise[ISIG_SUB_STEPS_SINGLE_FLOW * MAX_COMPRESSED_FRAMES_GPU];
    float modelTrace[MAX_COMPRESSED_FRAMES_GPU];
    float err[MAX_COMPRESSED_FRAMES_GPU];

    // using average dmult of 1
    const float d = 1.0f * perNucRegP->getD(); // effective diffusion

    float avgAmpl = 1.0f;
    float avgKmult = 1.0f;
    float deltaTmidNuc = 0.0f;

    // fine nuc trace 
    int nucStart = CalculateNucRise(
        tmidNuc,
        sigma,
	C,
	nuc_flow_span,
	frameNumber,
	num_frames,
        ISIG_SUB_STEPS_SINGLE_FLOW,
	fineNucRise);


    etbR = ComputeETBR(perNucRegP, perFlowRegP->getRatioDrift(), avgR, avgCopies, realFnum);
    tauB = ComputeTauB(constRegP, etbR);
    float SP = ComputeSP(perFlowRegP->getCopyDrift(), avgCopies, realFnum);

#if DEBUG_REG_FITTING
    printf("====> etbr:%f, taub:%f, SP:%f tmidNuc:%f \n", etbR, tauB, SP, tmidNuc);
#endif
    // calculate model trace (red trace) 
    BkgModelRedTraceCalculation(
        constRegP,
        perNucRegP,
        nucStart,
        fineNucRise, 
        avgAmpl, 
        avgKmult*perNucRegP->getKrate(),
        tauB, 
        gain, 
        SP, 
        d, 
        constRegP->getSens()*SENSMULTIPLIER,
        ISIG_SUB_STEPS_SINGLE_FLOW * nucStart,
        modelTrace, 
        deltaFrames, 
        ISIG_SUB_STEPS_SINGLE_FLOW,
        num_frames);

   
    // calculate residual
    float oldResidual, newResidual;
    oldResidual = CalculateTmidNucShiftFitResidual(
                      correctedTrace,
                      modelTrace,
                      err,
                      num_frames);

    // run lev mar iterations
    const float amplMin = 0.5f;
    const float amplMax = 1.5f;
    const float delta_tmid_nuc_min = -3.0f;
    const float delta_tmid_nuc_max = 3.0f;
    const float kmultMin = 0.2f;
    const float kmultMax = 2.0f;
    const int maxIters = 100;
    const float lambda_max = 1E+10;

    float delta0 = 0, delta1 = 0, delta2 = 0;
    int done = 0;
    float lambda = 1;
    

    float pdA[MAX_COMPRESSED_FRAMES_GPU];
    float pdKmult[MAX_COMPRESSED_FRAMES_GPU];
    float pdDeltaTmidNuc[MAX_COMPRESSED_FRAMES_GPU];

    float newA, newKmult, newDeltaTmidNuc;
    for (int iter=0; iter<maxIters; ++iter) {
      if (delta0*delta0 < 0.0000025f)
        done++;
      else 
        done = 0;

      if (done >=5)
        break;

      // calculate partial derivatives using pertubed parameters
      newA = avgAmpl + 0.001f;
      newKmult = avgKmult + 0.001f;
      newDeltaTmidNuc = deltaTmidNuc + 0.001f;

      // partial derivative w.r.t A
      BkgModelRedTraceCalculation(
          constRegP,
	  perNucRegP,
	  nucStart,
	  fineNucRise, 
	  newA, 
	  avgKmult*perNucRegP->getKrate(),
	  tauB, 
	  gain, 
	  SP, 
	  d, 
	  constRegP->getSens()*SENSMULTIPLIER,
	  ISIG_SUB_STEPS_SINGLE_FLOW * nucStart,
	  pdA,
	  deltaFrames, 
	  ISIG_SUB_STEPS_SINGLE_FLOW,
	  num_frames);

     CalculatePartialDerivative(
       pdA,
       pdA,
       modelTrace,
       NULL,
       0.001f,
       num_frames);

      // partial derivative w.r.t kmult
      BkgModelRedTraceCalculation(
          constRegP,
	  perNucRegP,
	  nucStart,
	  fineNucRise, 
	  avgAmpl, 
	  newKmult*perNucRegP->getKrate(),
	  tauB, 
	  gain, 
	  SP, 
	  d, 
	  constRegP->getSens()*SENSMULTIPLIER,
	  ISIG_SUB_STEPS_SINGLE_FLOW * nucStart,
	  pdKmult, 
	  deltaFrames, 
	  ISIG_SUB_STEPS_SINGLE_FLOW,
	  num_frames);

      CalculatePartialDerivative(
        pdKmult,
	pdKmult,
	modelTrace,
	NULL,
	0.001f,
	num_frames);

      // partial derivative w.r.t deltaTmidNuc
      nucStart = CalculateNucRise(
        tmidNuc + newDeltaTmidNuc,
        sigma,
	C,
	nuc_flow_span,
	frameNumber,
	num_frames,
        ISIG_SUB_STEPS_SINGLE_FLOW,
	fineNucRise);

      BkgModelRedTraceCalculation(
          constRegP,
	  perNucRegP,
	  nucStart,
	  fineNucRise, 
	  avgAmpl, 
	  avgKmult*perNucRegP->getKrate(),
	  tauB, 
	  gain, 
	  SP, 
	  d, 
	  constRegP->getSens()*SENSMULTIPLIER,
	  ISIG_SUB_STEPS_SINGLE_FLOW * nucStart,
	  pdDeltaTmidNuc, 
	  deltaFrames, 
	  ISIG_SUB_STEPS_SINGLE_FLOW,
	  num_frames);

      CalculatePartialDerivative(
        pdDeltaTmidNuc,
	pdDeltaTmidNuc,
	modelTrace,
	NULL,
	0.001f,
	num_frames);

      // jacobian matrix members
      float lhs_00=0, lhs_01=0, lhs_02=0, lhs_11=0, lhs_12=0, lhs_22=0;
      float rhs_0=0, rhs_1=0, rhs_2=0, det;
   
      // calculate jtj matrix entries
      for (int i=0; i<num_frames; ++i) {
        lhs_00 += pdA[i]*pdA[i];
        lhs_01 += pdA[i]*pdKmult[i];
        lhs_02 += pdA[i]*pdDeltaTmidNuc[i];
        lhs_22 += pdDeltaTmidNuc[i]*pdDeltaTmidNuc[i];
        lhs_12 += pdKmult[i]*pdDeltaTmidNuc[i];
        lhs_11 += pdKmult[i]*pdKmult[i];
        rhs_0 += pdA[i]*err[i];
        rhs_1 += pdKmult[i]*err[i];
        rhs_2 += pdDeltaTmidNuc[i]*err[i];
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

        if (!::isnan(delta0) && !::isnan(delta1) && !::isnan(delta2)) {
          newA = avgAmpl + delta0;
          newKmult = avgKmult + delta1;
          newDeltaTmidNuc = deltaTmidNuc + delta2;
            
          clampT(newA, amplMin, amplMax);
          clampT(newKmult, kmultMin, kmultMax);
          clampT(newDeltaTmidNuc, delta_tmid_nuc_min, delta_tmid_nuc_max);
     
#if DEBUG_REG_FITTING
          printf("Not NAN newA:%f, newKmult:%f, newDeltaTmidNuc:%f, iter:%d, lambda:%f, delta0:%f, delta1:%f, delta2:%f\n",newA, newKmult, newDeltaTmidNuc, iter, lambda, delta0, delta1, delta2); 
#endif

          nucStart = CalculateNucRise(
			  tmidNuc + newDeltaTmidNuc,
			  sigma,
			  C,
			  nuc_flow_span,
			  frameNumber,
			  num_frames,
			  ISIG_SUB_STEPS_SINGLE_FLOW,
			  fineNucRise);


          BkgModelRedTraceCalculation(
            constRegP,
            perNucRegP,
            nucStart,
	    fineNucRise, 
	    newA, 
	    newKmult*perNucRegP->getKrate(),
	    tauB, 
	    gain, 
	    SP, 
	    d, 
	    constRegP->getSens()*SENSMULTIPLIER,
	    ISIG_SUB_STEPS_SINGLE_FLOW * nucStart,
	    modelTrace, 
	    deltaFrames, 
	    ISIG_SUB_STEPS_SINGLE_FLOW,
	    num_frames);

            newResidual = CalculateTmidNucShiftFitResidual(
                              correctedTrace,
                              modelTrace,
                              err,
                              num_frames);

            nan_detected = false;
        }

        if (!nan_detected && newResidual < oldResidual) {
          lambda /= 10.0f;
          if (lambda < FLT_MIN)
            lambda = FLT_MIN;
          
          avgAmpl = newA;
          avgKmult = newKmult;
          deltaTmidNuc = newDeltaTmidNuc;

#if DEBUG_REG_FITTING
          printf("iter:%d, avgAmpl:%f, avgKmult:%f, deltaTmidNuc:%f, oldResidual:%f, newResidual:%f\n",iter, avgAmpl, avgKmult, deltaTmidNuc, oldResidual, newResidual);
#endif
          //if (bead_ndx == 0)
          //  printf("===> iter: %d Tau: %.2f residual: %.2f newresidual: %.2f\n", iter, taub, residual, newresidual);

          oldResidual = newResidual;
          cont_proc = true;
        }
        else {
          lambda *= 10.0f;
        }

        if (lambda > lambda_max)
          cont_proc = true;
        
      }

      if (lambda > lambda_max)
        break;
    }
    
    // update the tmidnuc shift
#if DEBUG_REG_FITTING
    printf("===> Fitted Tmidnuc Shift : %f, avgAmpl: %f, avgKmult: %f\n", deltaTmidNuc);
#endif

    perFlowRegP->setTMidNucShift(deltaTmidNuc);
  }
  __syncthreads();
}


// TODO Need to apply nonzero emphasis frames optimization
// TODO transpose emphasis
// TODO dense layout should benefit here with lots of reductions

__device__ 
void SingleFlowRegionalLevMarFit(
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
  const size_t samples)
{
  __shared__ double smBuffer[REG_FIT_SM_ACCUM_BUFFERSIZE];
  __shared__ float smNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU]; 
  __shared__ float tmpNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU];
  __shared__ double deltas[3]; // NUM_PARAMS=3
  __shared__ bool cont_lambda_itr;
  __shared__ bool nan_detected;
  __shared__ bool solved;


  // zero out shared memory for reductions
  for (size_t i=threadIdx.x; i<REG_FIT_SM_ACCUM_BUFFERSIZE; i+=blockDim.x) {
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

  // START AMPLITUDE ESTIMATION
  BkgCorrectedRawTrace(
      bkgTrace,
      observedTrace,
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
  const float *emphForFitting = setAdaptiveEmphasis(ampl, emphasisVec, num_frames, MAX_HPXLEN);

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
  beadRes = sqrtf(beadRes/num_frames);
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
      
           if (::isnan(deltas[0]) || ::isnan(deltas[1]) || ::isnan(deltas[2]))
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
            newBeadRes = sqrtf(newBeadRes/num_frames);
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
void UpdateFineNucRiseForSingleFlowFit(
  const ConstantParamsRegion * constRegP,
  const PerNucParamsRegion * perNucRegP,
  PerFlowParamsRegion * perFlowRegP,
  const float * RegionFrameCube,
  const int RegionFrameStride,
  const int num_frames,
  float *nucRise)
{
  perFlowRegP->setFineStart(CalculateNucRise(
      ComputeMidNucTime(perFlowRegP->getTMidNuc(), perFlowRegP, perNucRegP),
      ComputeSigma(perFlowRegP, perNucRegP),
      perNucRegP->getC(),
      ConstGlobalP.getNucFlowSpan(),
      RegionFrameCube + RfFrameNumber*RegionFrameStride,
      num_frames,
      ISIG_SUB_STEPS_SINGLE_FLOW,
      nucRise));
}

__device__ 
void UpdateCoarseNucRiseForSingleFlowFit(
  const ConstantParamsRegion * constRegP,
  const PerNucParamsRegion * perNucRegP,
  PerFlowParamsRegion * perFlowRegP,
  const float * RegionFrameCube,
  const int RegionFrameStride,
  const int num_frames,
  float *nucRise)
{
  perFlowRegP->setCoarseStart(CalculateNucRise(
      ComputeMidNucTime(perFlowRegP->getTMidNuc(), perFlowRegP, perNucRegP),
      ComputeSigma(perFlowRegP, perNucRegP),
      perNucRegP->getC(),
      ConstGlobalP.getNucFlowSpan(),
      RegionFrameCube + RfFrameNumber*RegionFrameStride,
      num_frames,
      ISIG_SUB_STEPS_MULTI_FLOW,
      nucRise));
}


__device__
float ComputeMultiFlowBeadResidual(
  const float *Ampl,
  const short **observedTrace,
  const float **emptyTrace,
  const PerNucParamsRegion **nucRegParams,
  const float *frameNumber,
  const float *deltaFrames,
  const float *emphasisVec,
  const ConstantParamsRegion *constRegP,
  const float *BeadParamCube,
  const float *RegionFrameCube,
  const PerFlowParamsRegion *perFlowRegP,
  const size_t num_frames,
  const size_t beadFrameStride,
  const size_t regionFrameStride)
{
  __shared__ float smNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU];
  __shared__ int smNucStart;

  float purpleTrace[MAX_COMPRESSED_FRAMES_GPU];

  // right now getting bead params in the order they were in bead_params struct
  const float copies = *(BeadParamCube + BpCopies*beadFrameStride);
  const float R = *(BeadParamCube + BpR*beadFrameStride);
  const float gain = *(BeadParamCube + BpGain*beadFrameStride);

  double beadRes = 0;
    
  for(int histFlowIdx = 0; histFlowIdx < ConstHistCol.getNumHistoryFlows(); histFlowIdx++) {
    const PerNucParamsRegion *histFlowNucParams = nucRegParams[histFlowIdx];
    const float d = (*(BeadParamCube + BpDmult*beadFrameStride)) * histFlowNucParams->getD(); // effective diffusion

    // calculate empty to bead ratio, buffering and copies
    const float C = histFlowNucParams->getC();
    const float nuc_flow_span = ConstGlobalP.getNucFlowSpan();
    const float sigma = ComputeSigma(perFlowRegP, histFlowNucParams);

    float tmidNuc = ComputeMidNucTime(perFlowRegP->getTMidNuc(), perFlowRegP, histFlowNucParams); 

    int realFlowNum = ConstFlowP.getRealFnum() - (ConstHistCol.getNumHistoryFlows() - 1 - histFlowIdx);

    float etbR = ComputeETBR(histFlowNucParams, perFlowRegP->getRatioDrift(), R, copies, realFlowNum);

    float tauB = ComputeTauB(constRegP, etbR);

    float SP = ComputeSP(perFlowRegP->getCopyDrift(), copies, realFlowNum);

    // Compute nucrise
    if (threadIdx.x == 0) {
      smNucStart = CalculateNucRise(
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


    ComputeModelBasedTrace(
      emptyTrace[histFlowIdx],
      deltaFrames,
      constRegP,
      histFlowNucParams,
      BeadParamCube,
      RegionFrameCube,
      smNucRise,
      smNucStart,
      histFlowNucParams->getKrate(),
      tauB,
      gain,
      SP,
      d,
      perFlowRegP->getDarkness(),
      etbR,
      constRegP->getSens()*SENSMULTIPLIER,
      Ampl[histFlowIdx],
      ISIG_SUB_STEPS_MULTI_FLOW * smNucStart,
      ISIG_SUB_STEPS_MULTI_FLOW,
      num_frames,
      beadFrameStride,
      regionFrameStride,
      purpleTrace);

    const float *emphForFitting = setAdaptiveEmphasis(Ampl[histFlowIdx], emphasisVec, num_frames, MAX_HPXLEN);

    beadRes += CalculateBeadResidualError(
                  observedTrace[histFlowIdx],
		  purpleTrace,
		  emphForFitting,
		  beadFrameStride,
		  num_frames);
  }

  beadRes = sqrtf(beadRes/(float)(num_frames * ConstHistCol.getNumHistoryFlows()));
  return beadRes;
}

__device__
void BuildMatrices(
  double *smBuffer, // shared memory buffer
  OneDimVec<double,REGLEVMARMAT_LHS_LEN> *jtj,
  OneDimVec<double,REGLEVMARMAT_RHS_LEN> *rhs,
  const bool goodBead,
  const float *Ampl,
  const short **observedTrace,
  const float **emptyTrace,
  const PerNucParamsRegion **nucRegParams,
  const float *frameNumber,
  const float *deltaFrames,
  const float *emphasisVec,
  const ConstantParamsRegion *constRegP,
  const float *BeadParamCube,
  const float *RegionFrameCube,
  const PerFlowParamsRegion *perFlowRegP,
  const size_t samples,
  const size_t num_frames,
  const size_t beadFrameStride,
  const size_t regionFrameStride)
{
  __shared__ float smNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU];
  __shared__ float smTmpNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU];
  __shared__ int smNucStart;
  __shared__ int smTmpNucStart;

  
  float pdTmidNuc[MAX_COMPRESSED_FRAMES_GPU];
  float pdRDR[MAX_COMPRESSED_FRAMES_GPU];
#if FIT_COPYDRIFT
  float pdPDR[MAX_COMPRESSED_FRAMES_GPU];
#endif
  float yerr[MAX_COMPRESSED_FRAMES_GPU];
  float oldTrace[MAX_COMPRESSED_FRAMES_GPU];
  float newTrace[MAX_COMPRESSED_FRAMES_GPU];

  // right now getting bead params in the order they were in bead_params struct
  const float copies = *(BeadParamCube + BpCopies*beadFrameStride);
  const float R = *(BeadParamCube + BpR*beadFrameStride);
  const float gain = *(BeadParamCube + BpGain*beadFrameStride);

  for(int histFlowIdx = 0; histFlowIdx < ConstHistCol.getNumHistoryFlows(); histFlowIdx++) {
    const PerNucParamsRegion *histFlowNucParams = nucRegParams[histFlowIdx];
    const float d = (*(BeadParamCube + BpDmult*beadFrameStride)) * histFlowNucParams->getD(); // effective diffusion

    // calculate empty to bead ratio, buffering and copies
    const float C = histFlowNucParams->getC();
    const float nuc_flow_span = ConstGlobalP.getNucFlowSpan();
    const float sigma = ComputeSigma(perFlowRegP, histFlowNucParams);

    int realFlowNum = ConstFlowP.getRealFnum() - (ConstHistCol.getNumHistoryFlows() - 1 - histFlowIdx);

    float tmidNuc = ComputeMidNucTime(perFlowRegP->getTMidNuc(), perFlowRegP, histFlowNucParams); 

    float etbR = ComputeETBR(histFlowNucParams, perFlowRegP->getRatioDrift(), R, copies, realFlowNum);

    float tauB = ComputeTauB(constRegP, etbR);

    float SP = ComputeSP(perFlowRegP->getCopyDrift(), copies, realFlowNum);
      
    const float *emphForFitting = setAdaptiveEmphasis(Ampl[histFlowIdx], emphasisVec, num_frames, MAX_HPXLEN);

    // Compute nucrise
    if (threadIdx.x == 0) {
      smNucStart = CalculateNucRise(
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


    ComputeModelBasedTrace(
      emptyTrace[histFlowIdx],
      deltaFrames,
      constRegP,
      histFlowNucParams,
      BeadParamCube,
      RegionFrameCube,
      smNucRise,
      smNucStart,
      histFlowNucParams->getKrate(),
      tauB,
      gain,
      SP,
      d,
      perFlowRegP->getDarkness(),
      etbR,
      constRegP->getSens()*SENSMULTIPLIER,
      Ampl[histFlowIdx],
      ISIG_SUB_STEPS_MULTI_FLOW * smNucStart,
      ISIG_SUB_STEPS_MULTI_FLOW,
      num_frames,
      beadFrameStride,
      regionFrameStride,
      oldTrace);

    // START YERR
    CalculateYerr(
      yerr, 
      observedTrace[histFlowIdx], 
      oldTrace, 
      emphForFitting, 
      1.0f,
      beadFrameStride,
      num_frames);
    // END YERR

    // START TMIDNUC PARTIAL DERIVATIVE
    float new_tmidnuc = ComputeMidNucTime(
        perFlowRegP->getTMidNuc() + TMIDNUC_REG_STEP, 
        perFlowRegP, 
        histFlowNucParams);
    if (threadIdx.x == 0) {
      smTmpNucStart = CalculateNucRise(
          new_tmidnuc,
          sigma,
	  C,
	  nuc_flow_span,
	  frameNumber,
	  num_frames,
          ISIG_SUB_STEPS_MULTI_FLOW,
	  smTmpNucRise);
    }
    __syncthreads();

    ComputeModelBasedTrace(
      emptyTrace[histFlowIdx],
      deltaFrames,
      constRegP,
      histFlowNucParams,
      BeadParamCube,
      RegionFrameCube,
      smTmpNucRise,
      smTmpNucStart,
      histFlowNucParams->getKrate(),
      tauB,
      gain,
      SP,
      d,
      perFlowRegP->getDarkness(),
      etbR,
      constRegP->getSens()*SENSMULTIPLIER,
      Ampl[histFlowIdx],
      ISIG_SUB_STEPS_MULTI_FLOW * smTmpNucStart,
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
      
    float new_ratiodrift = perFlowRegP->getRatioDrift() + RDR_REG_STEP; 
    etbR = ComputeETBR(histFlowNucParams, new_ratiodrift, R, copies, realFlowNum);
    tauB = ComputeTauB(constRegP, etbR);
    
    ComputeModelBasedTrace(
      emptyTrace[histFlowIdx],
      deltaFrames,
      constRegP,
      histFlowNucParams,
      BeadParamCube,
      RegionFrameCube,
      smNucRise,
      smNucStart,
      histFlowNucParams->getKrate(),
      tauB,
      gain,
      SP,
      d,
      perFlowRegP->getDarkness(),
      etbR,
      constRegP->getSens()*SENSMULTIPLIER,
      Ampl[histFlowIdx],
      ISIG_SUB_STEPS_MULTI_FLOW * smNucStart,
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
#if FIT_COPYDRIFT
    float new_copydrift = perFlowRegP->getCopyDrift() + PDR_REG_STEP;
    etbR = ComputeETBR(histFlowNucParams, perFlowRegP->getRatioDrift(), R, copies, realFlowNum);
    tauB = ComputeTauB(constRegP, etbR);
    SP = ComputeSP(new_copydrift, copies, realFlowNum);

    ComputeModelBasedTrace(
      emptyTrace[histFlowIdx],
      deltaFrames,
      constRegP,
      histFlowNucParams,
      BeadParamCube,
      RegionFrameCube,
      smNucRise,
      smNucStart,
      histFlowNucParams->getKrate(),
      tauB,
      gain,
      SP,
      d,
      perFlowRegP->getDarkness(),
      etbR,
      constRegP->getSens()*SENSMULTIPLIER,
      Ampl[histFlowIdx],
      ISIG_SUB_STEPS_MULTI_FLOW * smNucStart,
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
#endif
      
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
        jtj->val[0] += smBuffer[0];
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
        jtj->val[1] += smBuffer[0];
      __syncthreads();

#if FIT_COPYDRIFT
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
        jtj->val[2] += smBuffer[0];
      __syncthreads();
#endif

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
#if FIT_COPYDRIFT
        jtj->val[3] += smBuffer[0];
#else
        jtj->val[2] += smBuffer[0];
#endif
      __syncthreads();
     
#if FIT_COPYDRIFT
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
        jtj->val[4] += smBuffer[0];
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
        jtj->val[5] += smBuffer[0];
      __syncthreads();
#endif

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
        rhs->val[0] += smBuffer[0];
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
        rhs->val[1] += smBuffer[0];
      __syncthreads();

#if FIT_COPYDRIFT
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
        rhs->val[2] += smBuffer[0];
      __syncthreads();
#endif
  }
 
}

__device__ 
void MultiFlowRegionalLevMarFit(
  const float *Ampl,
  const short **obsTrace,
  const float **emptyTrace,
  const PerNucParamsRegion **nucRegParams,
  const float *BeadParamCube, //Copies, R, dmult, gain, tau_adj, phi, stride == beadFrameStride
  const unsigned short *BeadStateCube, //key_norm,  ppf, ssq
  const float *emphasisVec, //(MAX_POISSON_TABLE_COL)*F
  const ConstantParamsRegion *constRegP,
  const float *RegionFrameCube,  //bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
  const size_t beadFrameStride, //stride from one CUBE plane to the next for the Per Well Cubes
  const size_t regionFrameStride, //, //stride in Region Frame Cube to get to next parameter
  const size_t num_frames, // 4
  const size_t samples,
  float *scratchSpace,
  float *nucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
  PerFlowParamsRegion *perFlowRegP)
{
  __shared__ double smBuffer[REG_FIT_SM_ACCUM_BUFFERSIZE];
  __shared__ double deltas[NUM_TIME_VARYING_PARAMS]; // NUM_PARAMS=3
  __shared__ bool cont_lambda_itr;
  __shared__ bool nan_detected;
  __shared__ bool solved;
 
  const float* deltaFrames = RegionFrameCube + RfDeltaFrames*regionFrameStride;
  const float* frameNumber = RegionFrameCube + RfFrameNumber*regionFrameStride;


  // Calculate starting residual over block of flows (history of flows here)
  float beadRes = 0;
  beadRes = ComputeMultiFlowBeadResidual(
                   Ampl,
		   obsTrace,
		   emptyTrace,
		   nucRegParams,
                   frameNumber,
		   deltaFrames,
		   emphasisVec,
		   constRegP,
		   BeadParamCube,
		   RegionFrameCube,
		   perFlowRegP,
		   num_frames,
		   beadFrameStride,
		   regionFrameStride);

  smBuffer[threadIdx.x] = beadRes;
  __syncthreads(); 

  SimplestReductionAndAverage(smBuffer, samples, false);
  float curAvgRes = smBuffer[0] / (float)(samples);
  
  __syncthreads();

  // calculate partial derivatives and build matrix
  // solve for parameters
  // iterate
 
  // Lev mar iterations loop
  double lambda = 0.0001;
  bool goodBead = true;
  OneDimVec<double, REGLEVMARMAT_LHS_LEN> jtj;
  OneDimVec<double, REGLEVMARMAT_RHS_LEN> rhs;
  for (int iter=0; iter<(LEVMARITERS); ++iter) {

    if (iter > 0 && !solved) {
      if (threadIdx.x == 0)
        printf("max lambda reached: %f\n", lambda);
      return;
    }
    // zero out shared memory for reductions
    for (size_t i=threadIdx.x; i<REG_FIT_SM_ACCUM_BUFFERSIZE; i+=blockDim.x) {
      smBuffer[i] = 0;
    }

    __syncthreads();

  

    float tmidNuc = perFlowRegP->getTMidNuc();
    float ratioDrift = perFlowRegP->getRatioDrift();;
    float copyDrift = perFlowRegP->getCopyDrift();
    float new_tmidnuc, new_ratiodrift;
#if FIT_COPYDRIFT
    new_copydrift;
#endif
    // possibly filter beads at this point
    // residual not changing or corrupt or ...
    goodBead = beadRes < 4.0f*curAvgRes;
    //goodBead = true;

    jtj.clear();
    rhs.clear();
    BuildMatrices(
        smBuffer,
        &jtj,
        &rhs,
        goodBead,
        Ampl,
	obsTrace,
	emptyTrace,
	nucRegParams,
	frameNumber,
	deltaFrames,
	emphasisVec,
	constRegP,
	BeadParamCube,
	RegionFrameCube,
	perFlowRegP,
        samples,
	num_frames,
	beadFrameStride,
	regionFrameStride);
  
    // Solve
    // Compute new residual error nad iterate
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

      float newBeadRes = 0;
      float newAvgRes = 0;
      while (cont_lambda_itr) {
        if (threadIdx.x == 0) {
          OneDimVec<double,REGLEVMARMAT_RHS_LEN> newJTJDiag;

#if FIT_COPYDRIFT
          newJTJDiag.val[0] = jtj.val[0] * (1.0 + lambda);
          newJTJDiag.val[1] = jtj.val[3] * (1.0 + lambda);
          newJTJDiag.val[2] = jtj.val[5] * (1.0 + lambda);
        
          // calculate determinant
          double det = newJTJDiag.val[0]*(newJTJDiag.val[1]*newJTJDiag.val[2] - jtj.val[4]*jtj.val[4]) - 
                jtj.val[1]*(jtj.val[1]*newJTJDiag.val[2] - jtj.val[4]*jtj.val[2]) +
                jtj.val[2]*(jtj.val[1]*jtj.val[4] - newJTJDiag.val[1]*jtj.val[2]);
          det = 1.0/det;
         
          deltas[0] = det*(rhs.val[0]*(newJTJDiag.val[1]*newJTJDiag.val[2] - jtj.val[4]*jtj.val[4]) +
                   rhs.val[1]*(jtj.val[2]*jtj.val[4] - jtj.val[1]*newJTJDiag.val[2]) +
                   rhs.val[2]*(jtj.val[1]*jtj.val[4] - jtj.val[2]*newJTJDiag.val[1]));
          deltas[1] = det*(rhs.val[0]*(jtj.val[4]*jtj.val[2] - jtj.val[1]*newJTJDiag.val[2]) +
                   rhs.val[1]*(newJTJDiag.val[0]*newJTJDiag.val[2] - jtj.val[2]*jtj.val[2]) +
                   rhs.val[2]*(jtj.val[1]*jtj.val[2] - newJTJDiag.val[0]*jtj.val[4]));
          deltas[2] = det*(rhs.val[0]*(jtj.val[1]*jtj.val[4] - jtj.val[2]*newJTJDiag.val[1]) +
                   rhs.val[1]*(jtj.val[1]*jtj.val[2] - newJTJDiag.val[0]*jtj.val[4]) +
                   rhs.val[2]*(newJTJDiag.val[0]*newJTJDiag.val[1] - jtj.val[1]*jtj.val[1]));
      
           if (isnan(deltas[0]) || isnan(deltas[1]) || isnan(deltas[2]))
             nan_detected = true;
#else
          newJTJDiag.val[0] = jtj.val[0] * (1.0 + lambda);
          newJTJDiag.val[1] = jtj.val[2] * (1.0 + lambda);
        
          // calculate determinant
          double det = (newJTJDiag.val[0]*newJTJDiag.val[1]) - (jtj.val[1] * jtj.val[1]);
          det = 1.0/det;
         
          deltas[0] = det*(newJTJDiag.val[1]*rhs.val[0] - jtj.val[1]*rhs.val[1]);
          deltas[1] = det*(-jtj.val[1]*rhs.val[0] + newJTJDiag.val[0]*rhs.val[1]);
      
           if (::isnan(deltas[0]) || ::isnan(deltas[1]))
             nan_detected = true;
#endif

#if DEBUG_REG_FITTING
          printf("===GPU REG Params...iter:%d,delta0:%f,delta1:%f,delta2:%f,lambda:%f\n", iter, deltas[0], deltas[1], deltas[2],lambda);
#endif
        }  
        __syncthreads();

        if (!nan_detected) {
          new_tmidnuc = tmidNuc + deltas[0];
          clampT(new_tmidnuc, constRegP->getMinTmidNuc(), constRegP->getMaxTmidNuc());
          perFlowRegP->setTMidNuc(new_tmidnuc);

          new_ratiodrift = ratioDrift + deltas[1];  
          clampT(new_ratiodrift, constRegP->getMinRatioDrift(), constRegP->getMaxRatioDrift());
          perFlowRegP->setRatioDrift(new_ratiodrift); 

#if FIT_COPYDRIFT
          new_copydrift = copyDrift + deltas[2];
          clampT(new_copydrift, constRegP->getMinCopyDrift(), constRegP->getMaxCopyDrift());
          perFlowRegP->setCopyDrift(new_copydrift);
#endif

          // Calculate residual
          newBeadRes = ComputeMultiFlowBeadResidual(
                           Ampl,
		           obsTrace,
			   emptyTrace,
			   nucRegParams,
			   frameNumber,
			   deltaFrames,
			   emphasisVec,
			   constRegP,
			   BeadParamCube,
			   RegionFrameCube,
			   perFlowRegP,
			   num_frames,
			   beadFrameStride,
			   regionFrameStride);

          smBuffer[threadIdx.x] = newBeadRes;
          __syncthreads(); 


          // reduce here for average residual value
          SimplestReductionAndAverage(smBuffer, samples, false);
          newAvgRes = smBuffer[0] / (float)(samples);
          __syncthreads();
        }
            
        if (threadIdx.x == 0) {
          //new_residual = smBuffer[0];
            
          // DEBUG 
#if DEBUG_REG_FITTING
          printf("===GPU REG Params...iter:%d,tmidnuc:%f,rdr:%f,pdr:%f,old_residual:%f,new_residual:%f\n", iter, new_tmidnuc, new_ratiodrift, new_copydrift, curAvgRes, newAvgRes);
#endif

          if (!nan_detected && newAvgRes < curAvgRes) {
            solved = true;
            lambda /= 30.0; // use correct lambda step from bkgmodel
            if (lambda < FLT_MIN)
              lambda = FLT_MIN;

            cont_lambda_itr = false;
          }
          else {
            lambda *= 30.0;
            if (lambda > 1E+9f) {
              cont_lambda_itr = false;
              solved = false;
            }  
            perFlowRegP->setTMidNuc(tmidNuc); 
            perFlowRegP->setRatioDrift(ratioDrift);
#if FIT_COPYDRIFT
            perFlowRegP->setCopyDrift(copyDrift);
#endif
          }
          nan_detected = false;
        }
        __syncthreads();
      }   
      if (solved) {
        // update residuals for next iteration
        beadRes = newBeadRes;
        curAvgRes = newAvgRes;
      }
      __syncthreads();
  }

}


__device__
float EstimateAmplForMultiFlowRegionalLevMarFit(
  const int realFnum,
  const short *observedTrace, // NUM_SAMPLES_RF x F
  const float *BeadParamCube, //Copies, R, dmult, gain, tau_adj, phi, stride == beadFrameStride
  const float *emphasisVec, //(MAX_POISSON_TABLE_COL)*F
  float *nucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
  const ConstantParamsRegion *constRegP,
  PerFlowParamsRegion *perFlowRegP,
  const PerNucParamsRegion *perNucRegP,
  const float *RegionFrameCube,  //bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
  const float *EmptyTraceAvg,  //bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
  const size_t beadFrameStride, //stride from one CUBE plane to the next for the Per Well Cubes
  const size_t regionFrameStride, //, //stride in Region Frame Cube to get to next parameter
  const size_t num_frames)
{
  __shared__ float smNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU]; 
  __shared__ int tStart;


  if (threadIdx.x == 0) {
    perFlowRegP->setTMidNucShift(0);
  }

  __syncthreads();

  float correctedTrace[MAX_COMPRESSED_FRAMES_GPU];
  float obsTrace[MAX_COMPRESSED_FRAMES_GPU]; // raw traces being written to
  
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
  float etbR = ComputeETBR(perNucRegP, perFlowRegP->getRatioDrift(), R, copies, realFnum);
  float tauB = ComputeTauB(constRegP, etbR);
  float SP = ComputeSP(perFlowRegP->getCopyDrift(), copies, realFnum);
 
  // Need shifted background  
  const float* bkgTrace = EmptyTraceAvg;
  const float* deltaFrames = RegionFrameCube + RfDeltaFrames*regionFrameStride;
  const float* frameNumber = RegionFrameCube + RfFrameNumber*regionFrameStride;

  // background subtracted trace for amplitude estimation

  // calculate initial nucRise here
  if (threadIdx.x == 0) {
#if DEBUG_REG_FITTING
   printf("C: %f sigma: %f, tmidNuc: %f\n", C, sigma, tmidNuc);
   printf("copies: %f R: %f d: %f gain: %f, etbR: %f tauB: %f\n", copies, R, d, gain, etbR, tauB);
#endif
    tStart = CalculateNucRise(
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
  
  // DEBUG
#if DEBUG_REG_FITTING
  if (threadIdx.x == 0) {
    printf("GPU before fitting...start: %d, tmidnuc: %f rdr: %f pdr: %f\n",
        tStart,perFlowRegP->getTMidNuc(), perFlowRegP->getRatioDrift(), perFlowRegP->getCopyDrift());

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

  // START AMPLITUDE ESTIMATION
  BkgCorrectedRawTrace(
      bkgTrace,
      observedTrace,
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
      obsTrace,
      tStart,
      beadFrameStride,
      1, // emphasis stride
      ISIG_SUB_STEPS_MULTI_FLOW
      );

#if DEBUG_REG_FITTING
    printf("====> GPU....tid: %d Ampl: %f\n", threadIdx.x, ampl);
#endif

   return ampl;
}


// Fit time varying parameters on a collection of flows. For flow by
// flow pipeline it means fitting on a history of flows and recycling that
// history as we advance in the flows
__global__ 
void PerformMultiFlowRegionalFitting(
  const unsigned short * RegionMask,
  const float *beadParamCube,
  const unsigned short *beadStateCube,
  const float *emphasisVec, //(MAX_POISSON_TABLE_COL)*F
  const int *nonZeroEmphFrames, 
  float *finenucRise,
  float *coarsenucRise,
  float *scratchSpace,
  const size_t *numFramesRegion,
  const ConstantParamsRegion * constRegP,
  PerFlowParamsRegion * perFlowRegP,
  const PerNucParamsRegion * perNucRegP,
  const float * RegionFrameCube,
  const int *NumSamples
)
{
  // each region is fitted by one thread block
  
  const size_t regId = blockIdx.x;
  const size_t beadId = threadIdx.x;

  if( LDG_ACCESS(RegionMask,regId) != RegionMaskLive) return;
  if (beadId >= NumSamples[regId])
    return;
  //strides
  //const size_t BeadFrameStride = ( ImgRegP.getGridParam( NUM_SAMPLES_RF )).getPlaneStride();
  const size_t BeadFrameStride = ( ImgRegParams::getGridParam( ImgRegP, NUM_SAMPLES_RF )).getPlaneStride();
  const size_t RegionFrameStride = ConstFrmP.getMaxCompFrames() * ImgRegP.getNumRegions();

  RegionFrameCube += regId*ConstFrmP.getMaxCompFrames();  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber

  //update per region pointers
  constRegP += regId;
  perFlowRegP += regId;

  // update per region pointer depending on nuc id
  //perNucRegP +=  ImgRegP.getNumRegions() * ConstFlowP.getNucId() + regId;


  //nonZeroEmphFrames += regId*MAX_POISSON_TABLE_COL;
  emphasisVec += regId * MAX_POISSON_TABLE_COL * ConstFrmP.getMaxCompFrames();

  const size_t numf = numFramesRegion[regId];

  /////////////////////////////////////
  //Observed Sample traces now come from a Sample collection:
  // if we have a sample history this needs to be replaced by the following:
  // n sample buffers starting with the oldest: idx = 0 to the latest: idx= numSampleFlows-1
    
  beadStateCube += NUM_SAMPLES_RF*regId + threadIdx.x;
  beadParamCube += NUM_SAMPLES_RF*regId + threadIdx.x;
  float *multiFlowNucRise =coarsenucRise + regId *  ISIG_SUB_STEPS_MULTI_FLOW * ConstFrmP.getMaxCompFrames() ;
  float AmplEst[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  const short* obsTracePtr[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  const float* emptyTracePtr[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  const PerNucParamsRegion* nucRegParamsPtr[MAX_NUM_FLOWS_IN_BLOCK_GPU];

  for(int histFlowIdx = 0; histFlowIdx < ConstHistCol.getNumHistoryFlows(); histFlowIdx++) {
    
    const short *observedTrace = ConstHistCol.getSampleTraces(histFlowIdx) + NUM_SAMPLES_RF*regId + threadIdx.x;
    obsTracePtr[histFlowIdx] = observedTrace;


    const float *emptyTraceAvg = ConstHistCol.getEmptyTraces(histFlowIdx) + regId*ConstFrmP.getUncompFrames();
    emptyTracePtr[histFlowIdx] = emptyTraceAvg;
    const PerNucParamsRegion *histNucRegParams = perNucRegP +  ImgRegP.getNumRegions() * ConstHistCol.getNucId(histFlowIdx) + regId;
    nucRegParamsPtr[histFlowIdx] = histNucRegParams;
  
    int realFlowNum = ConstFlowP.getRealFnum() - (ConstHistCol.getNumHistoryFlows() - 1 - histFlowIdx);
    AmplEst[histFlowIdx] = EstimateAmplForMultiFlowRegionalLevMarFit(
      realFlowNum,
      observedTrace,
      beadParamCube,
      emphasisVec,
      multiFlowNucRise,
      constRegP,
      perFlowRegP,
      histNucRegParams,
      RegionFrameCube,
      emptyTraceAvg,
      BeadFrameStride,
      RegionFrameStride,
      numf);
  }

  MultiFlowRegionalLevMarFit(
    AmplEst,
    obsTracePtr,
    emptyTracePtr,
    nucRegParamsPtr,
    beadParamCube,
    beadStateCube,
    emphasisVec,
    constRegP,
    RegionFrameCube,
    BeadFrameStride,
    RegionFrameStride,
    numf,
    NumSamples[regId],
    scratchSpace,
    multiFlowNucRise,
    perFlowRegP);

  __syncthreads();

  if (ConfigP.FitTmidNucShift())
    FitTmidNucShiftPerFlow(
      ConstFlowP.getRealFnum(),
      AmplEst[ConstHistCol.getNumHistoryFlows() - 1],
      obsTracePtr[ConstHistCol.getNumHistoryFlows() - 1],
      beadParamCube,
      beadStateCube,
      emptyTracePtr[ConstHistCol.getNumHistoryFlows() - 1],
      emphasisVec,
      nucRegParamsPtr[ConstHistCol.getNumHistoryFlows() - 1],
      constRegP,
      RegionFrameCube,
      BeadFrameStride,
      RegionFrameStride,
      numf,
      NumSamples[regId],
      perFlowRegP);

  if (beadId == 0) {
    UpdateFineNucRiseForSingleFlowFit(
      constRegP,
      nucRegParamsPtr[ConstHistCol.getNumHistoryFlows() - 1],
      perFlowRegP,
      RegionFrameCube,
      RegionFrameStride,
      numf,
      finenucRise + regId *  ISIG_SUB_STEPS_SINGLE_FLOW * ConstFrmP.getMaxCompFrames());  
    UpdateCoarseNucRiseForSingleFlowFit(
      constRegP,
      nucRegParamsPtr[ConstHistCol.getNumHistoryFlows() - 1],
      perFlowRegP,
      RegionFrameCube,
      RegionFrameStride,
      numf,
      coarsenucRise + regId *  ISIG_SUB_STEPS_MULTI_FLOW * ConstFrmP.getMaxCompFrames());
  };
}

