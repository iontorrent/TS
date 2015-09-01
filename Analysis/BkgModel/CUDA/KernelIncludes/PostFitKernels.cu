/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * XTalkKernels.cu
 *
 *  Created on: Oct 1, 2014
 *      Author: jakob
 */


#include "DeviceParamDefines.h"
#include "ConstantSymbolDeclare.h"
#include "FittingHelpers.h"
#include "PostFitKernels.h"
#include "EnumDefines.h"

#define XTALK_CONST_FRAC              0.25f
#define XTALK_CONST_FRAC_REF          0.86f
#define XTALK_CONST_FRAC_REF_COMP     0.14f
#define XTALK_MAGIC_FLOW              32.0f
#define XTALK_MAGIC_LAMBDA            1.425f
#define XTALK_MAGIC_DISCOUNT          0.33f

#define POLYCLONAL_PPF_CUTOFF         0.84f
#define POLYCLONAL_POS_THRESHOLD      0.25f
#define POLYCLONAL_BAD_READ_THRESHOLD 100.0f

//////////////////////////////////////////////
//XTALK MAGIC AND DEVICE FUNCTIONS

class MagicXTalkForFLow{

  float cscale;
  float cscale_ref;
  float magic_hplus_ref;

protected:


  __device__ inline float modulateEffectByFlow(const float start_frac, const float flow_num, const float offset) const
  {
    float approach_one_rate = flow_num/(flow_num+offset);
    return ( (1.0f-start_frac) * approach_one_rate + start_frac);
  }

  __device__ inline void setCscaleRefForFlow(const int flow) {
    cscale_ref = modulateEffectByFlow( XTALK_CONST_FRAC_REF, flow, XTALK_MAGIC_FLOW);
  }
  __device__ inline void setCscaleForFlow(const  int flow ) {
    this->cscale = modulateEffectByFlow( XTALK_CONST_FRAC, flow, XTALK_MAGIC_FLOW);
  }
  __device__ inline void setMagicHplusRef( const float regionMeanSignal){
    this->magic_hplus_ref =  getMagicRefConstant() * regionMeanSignal;
  }


public:

  __device__
  MagicXTalkForFLow(const int flow, const float regionMeanSignal){
    setCscaleForFlow(flow);
    setCscaleRefForFlow(flow);
    setMagicHplusRef(regionMeanSignal);
  }

  __device__ inline float getCscale() const {
    return cscale;
  }
  __device__ inline float getCscaleRef() const {
    return cscale_ref;
  }

  __device__ inline float getMagicRefConstant() const {
    return (XTALK_CONST_FRAC_REF_COMP/XTALK_CONST_FRAC_REF) * getCscaleRef();
  }

  __device__ inline float getMagicCscaleConstant() const{
    return XTALK_MAGIC_DISCOUNT * getCscale();
  }
  __device__ inline float  getMagicHplusRef() const {
    return this->magic_hplus_ref;
  }


  __device__ inline float getMagicHPlusCorrector(const float etbR, const float bead_corrector) const {
    float magic_bead_corrector = (getMagicCscaleConstant()*etbR*bead_corrector);
    float magic_hplus_corrector = XTALK_MAGIC_LAMBDA* ( magic_bead_corrector - getMagicHplusRef());
    return magic_hplus_corrector;
  }

};


//Ampl pointsd to Amplitude of current bead
//rx ry are coordinates of current bead in region
//regId id of current region
__device__ inline
float UnweaveMap(
    const float * AmplCpy,
    const float default_signal,
    const int rx,
    const int ry,
    const size_t regId  )
{
  float sum = 0.0f;
  int phase = ConstXTalkP.getPhase(rx);
  int ly = 0;
  for (int r=(-ConstXTalkP.getSpanY()); r<=(ConstXTalkP.getSpanY()); r++, ly++)
  {
    int lx=0;
    for (int c=(-ConstXTalkP.getSpanX()); c<=(ConstXTalkP.getSpanX()); c++, lx++)
    {
      int tx = c + rx;
      int ty = r + ry;
      if ((tx < 0) || tx>= (ImgRegP.getRegW(regId)) || (ty < 0) || ty>=(ImgRegP.getRegH(regId)))
      {
        // if we are on the edge of the region...as a stand-in for actual data use the region average signal
        //ToDo: instead of region average we can use actual values from neighboring regions if available
        sum += default_signal * (ConstXTalkP.coeff(lx,ly,phase));
        //if(rx == 4 && ry  == 0 )
         //printf("GPU %d %d, %f ,%d %d, %d %d, %f, %d %d, %d %f %f\n", rx,ry, *AmplCpy, c, r, tx, ty, default_signal, lx, ly, phase, ConstXTalkP.coeff(lx,ly,phase), sum);
      }else{
        float amplCopy = LDG_ACCESS(AmplCpy,r*ImgRegP.getImgW()+c);
        sum += amplCopy * (ConstXTalkP.coeff(lx,ly,phase));
        //if(rx == 4 && ry == 0 )
          //printf("GPU %d %d, %f ,%d %d, %d %d, %f, %d %d, %d %f %f\n",rx,ry, *AmplCpy, c, r, tx,ty , amplCopy, lx, ly, phase, ConstXTalkP.coeff(lx,ly,phase), sum);
      }
    }
  }
  return sum;
}

//__device__ inline
//float EmptyCorrector(float default_signal)
//{
//  float sum = 0.0f;
//  for (int mx=0; mx< XTALK_MAP; mx++){
//    sum += default_signal*ConstXTalkP.odd(mx);
//  }
//  return sum;
//}

__device__ inline
void DoXTalk(
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const MagicXTalkForFLow & MagicStuff,
    const float* AmplIn,
    const float * AmpCpy,
    float *  AmplOut,
    const float regAvgSig,
    const float valCopies,
    const float valR,
    const size_t regId,
    const size_t rx,
    const size_t ry
)
{
  float Ampl = *AmplIn;
  float etbR = ComputeETBR(perNucRegP,perFlowRegP->getRatioDrift(),valR, valCopies);
  float bead_corrector  = UnweaveMap(AmpCpy,regAvgSig, rx,ry,regId);
  float hplus_corrector = MagicStuff.getMagicHPlusCorrector(etbR,bead_corrector);  //bead_corrector - empty_corrector;
  Ampl -= hplus_corrector/(valCopies);
  if(Ampl != Ampl) Ampl = 0.0f;  // NaN check
  *AmplOut = Ampl;
}

// END XTALK DEVICE FUNCTIONS
//////////////////////////////////////////////



//////////////////////////////////////////////
// POLYCLONAL DEVICE FUNCTIONS


__device__ inline
void ClonalFilterUpdate(  float * PolyClonalCube,
    unsigned short * BeadStateMask,
    const float * Ampl,
    const size_t planeStride,
    float const CopyDrift,
    const float copies
)
{

  float keynorm = *(PolyClonalCube + PolyKeyNorm * planeStride);
  float tmpAmpl =  ((*Ampl) * copies)/keynorm;
  float * pPPF = PolyClonalCube + PolyPpf * planeStride;
  float * pSSQ = PolyClonalCube + PolySsq * planeStride;

  if((tmpAmpl >  POLYCLONAL_POS_THRESHOLD) || (ConstGlobalP.isLastClonalUpdateFlow(ConstFlowP.getRealFnum())))
  {
    float ppf;
    ppf = (tmpAmpl >  POLYCLONAL_POS_THRESHOLD)?(*pPPF + 1):(*pPPF);
    if(ConstGlobalP.isLastClonalUpdateFlow(ConstFlowP.getRealFnum()))
        ppf /= ConstGlobalP.getClonalFilterNumFlows(); //average across training flows after last training flow collected
    *pPPF = ppf;
  }

  float x = tmpAmpl - round(tmpAmpl);
  *pSSQ  = *pSSQ + x * x;

  if(tmpAmpl >POLYCLONAL_BAD_READ_THRESHOLD)
    *BeadStateMask = ((*BeadStateMask) | BkgMaskBadRead);

}

// END POLYCLONAL DEVICE FUNCTIONS
//////////////////////////////////////////////



__device__ inline
void EffectiveAmplitudeForRawWells(
    const float *AmplIn,
    float *AmpOut,
    float copies,
    float copyDrift)
{
  float copyMultiplier = pow(copyDrift, ConstFlowP.getRealFnum());
  //*AmpOut = (*AmplIn) * copies * copyMultiplier;

  // Copies are written separately in the new format
  *AmpOut = (*AmplIn) * copyMultiplier;
}


//////////////////////////////////////////////
//KERNELS

//one threadblock per region
//calculates region mean signal,
//updates signal for non live beads
//sm layout: numwarps * uncompressed frames + numwaprs integers to sum up count
//reduce empty average
__global__
__launch_bounds__(128, 16)
void UpdateSignalMap_k(
    const unsigned short * RegionMask,
    const unsigned short  * bfMask,
    const float* BeadParamCube,
    float * ResultCube,
    float * regionAvg  // has to be inited to 0.0f for all regions
    //  float * beadAvg     // has to be inited to 0.0f for all regions
)
{
  extern __shared__ float smemSum[];  // uncompressed frames per warp

  float * sm_base = smemSum;
  float * sm_warp_base = sm_base + threadIdx.y*blockDim.x;
  float * sm = sm_warp_base + threadIdx.x;

  //same for all warps within block
  const size_t regId = blockIdx.x + blockIdx.y * gridDim.x;
  const size_t windowSize = blockDim.x; //window size to slide accross row
  const size_t nextWorkRowStride = ImgRegP.getImgW() * blockDim.y;  //stride to get to next row to work on
  const size_t regWidth =  ImgRegP.getRegW(regId);
  const size_t regHeight = ImgRegP.getRegH(regId);

  //already filters out regions with 0 live beads. so no more checks are needed
  if( LDG_ACCESS(RegionMask,regId) != RegionMaskLive) return;

  //starting coordinates for each thread within region
  size_t rx = threadIdx.x;  //region x to work on
  size_t ry = threadIdx.y;
  //starting offset for each thread within image
  size_t idx = ImgRegP.getWellIdx(regId,rx,ry);

  bfMask += idx;

  float * AmpCopy = ResultCube + ImgRegP.getImgSize() * ResultAmplCopyMapXTalk + idx;
  const float * Ampl = ResultCube + ImgRegP.getImgSize() * ResultAmpl + idx;
  const float * copies = BeadParamCube + ImgRegP.getImgSize() * BpCopies + idx;
  regionAvg += regId;
  //beadAvg += regId;

  *sm = 0;
  //float Cnt = 0;

  while(ry < regHeight){

    size_t windowStart = 0;
    const unsigned short* bfmaskRow = bfMask;
    const float* AmplRow = Ampl;
    const float* copiesRow = copies;
    float* AmpCopyRow = AmpCopy;

    //slide warp/window across row and create sum for of num live beads for each warp
    while(windowStart < regWidth){
      if(rx < regWidth){ //if bead still in reagion set sm according to mask
        if(Match(bfmaskRow,(MaskType)MaskLive)){
          float ampcpy = (*AmplRow) * (*copiesRow);
          *AmpCopyRow = ampcpy;
          *sm +=  ampcpy;
          //Cnt += 1.0f; //increase t0 count to calculate average.
        }
      }
      //slide window
      rx += windowSize;
      windowStart += windowSize;
      bfmaskRow += windowSize;
      AmplRow += windowSize;
      AmpCopyRow += windowSize;
      copiesRow += windowSize;

    } //row done

    rx = threadIdx.x;
    ry += blockDim.y;
    bfMask += nextWorkRowStride;
    Ampl += nextWorkRowStride;
    AmpCopy += nextWorkRowStride;
    copies += nextWorkRowStride;
  }
  float sigSum = ReduceSharedMemory(sm_base,sm);
  __syncthreads();
  //*sm = Cnt;
  //Cnt = ReduceSharedMemory(sm_base,sm);
  //__syncthreads();
  if(threadIdx.x == 0 && threadIdx.y == 0){
   // printf("regionSum GPU: %f / %d\n", sigSum, ImgRegP.getRegSize(regId) );
    *regionAvg = sigSum/ImgRegP.getRegSize(regId);
  }
  //*beadAvg = sigSum/Cnt;  // Cnt always > 0 otherwise region will not even be handles by the kernel

}

// execute with one warp per row and 2D thread blocks of width warp length
// each warp will slide across one row of the region
// kernel parameters:
// thread block dimensions (WARPSIZE,n,1)  //n = number of warps per block)
// grid dimension ( numRegions.x, (imgH + n-1)/n, 1) // one block per region in x direction and one per n img rows in y direction
__global__
__launch_bounds__(128, 16)
void PostProcessingCorrections_k(
    const unsigned short * RegionMask,
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const unsigned short  * bfMask,
    const float* BeadParamCube,
    unsigned short * BeadStateMask,
    float* PolyClonalCube,
    float* ResultCube,
    float * regionAvgSignal  // has to be inited to 0.0f for all regions
)
{

  const size_t regionCol = blockIdx.x;
  const size_t regionRow = (blockIdx.y*blockDim.y)/ImgRegP.getRegH();
  const size_t regId = ImgRegP.getRegIdFromGrid(regionCol,regionRow);

  if( LDG_ACCESS(RegionMask,regId) != RegionMaskLive) return;

  const size_t regionWidth = ImgRegP.getRegW(regId);
  const size_t windowWidth = blockDim.x;
  const size_t ix = regionCol * ImgRegP.getRegW()+ threadIdx.x;
  const size_t iy = (blockIdx.y*blockDim.y) + threadIdx.y;
  const size_t idx = ImgRegP.getWellIdx(ix,iy);
  const size_t ry = iy%ImgRegP.getRegH();
  size_t rx = threadIdx.x;

  const size_t planeStride = ImgRegP.getImgSize();

  //per bead
  const float * copies = BeadParamCube + BpCopies*planeStride + idx;
  const float * R = BeadParamCube + BpR*planeStride + idx;
  const float * AmpCpy = ResultCube + planeStride * ResultAmplCopyMapXTalk + idx;

  //when no longer using Xtlak on host one of these buffers can be removed
  const float * AmplIn = ResultCube + planeStride * ResultAmpl + idx;
  float * AmplOut = ResultCube + planeStride * ResultAmpl + idx;

  BeadStateMask += idx;
  PolyClonalCube += idx;

  bfMask += idx;

  //per region
  perFlowRegP += regId;
  perNucRegP +=  ImgRegP.getNumRegions() * ConstFlowP.getNucId() + regId;
  const float regAvgSig = LDG_LOAD(regionAvgSignal+regId);

  MagicXTalkForFLow MagicStuff(ConstFlowP.getRealFnum(), regAvgSig);

  while(rx < regionWidth ){ //while thread inside region

    if(Match(bfMask,MaskLive)){
      if(ConfigP.PerformWellsLevelXTalk())
        DoXTalk(perFlowRegP,perNucRegP,MagicStuff,AmplIn,AmpCpy,AmplOut,regAvgSig,*copies,*R,regId,rx,ry);
      //use Xtalk corrected amplitude

      // Prepare wells amplitude for raw wells writing by multiplicative scaling with copy multiplier
      EffectiveAmplitudeForRawWells(AmplOut, AmplOut, *copies, perFlowRegP->getCopyDrift());

      if( ConfigP.PerformPolyClonalFilter() && ConstGlobalP.isClonalUpdateFlow(ConstFlowP.getRealFnum()))
        ClonalFilterUpdate(PolyClonalCube,BeadStateMask,AmplOut,planeStride,perFlowRegP->getCopyDrift(),*copies);
//        ClonalFilterUpdate(PolyClonalCube,BeadStateMask,AmplOut,planeStride,perFlowRegP->getCopyDrift(),*copies,ry);

    }

    rx += windowWidth;
    copies += windowWidth;
    R += windowWidth;
    bfMask += windowWidth;
    AmplIn += windowWidth;
    AmpCpy += windowWidth;
    AmplOut += windowWidth;
    BeadStateMask += windowWidth;
    PolyClonalCube += windowWidth;
  }
}


/*
__device__ inline
void loadSharedMemoryAll( float * AmplBase,
                          size_t regId,
                          size_t startRy,
                          float * sm
)
{
  size_t regBaseIdx = ImgRegP.getRegBaseIdx(regId);
  size_t globalx = ImgRegP.getXFromIdx(regBaseIdx);
  szie_t globaly = ImgRegP.getYFromIdx(regBaseIdx);
  size_t tIdx = threadIdx.y * blockDim.x + threadIdx.x;
  int rx = tIdx - ConstXTalkP.getSpanX();
  int ry = startRy - ConstXTalkP.getSpanY();

  int windowDimX = blokcDim.x + 2 * ConstXTalkP.getSpanX();
  int windowDimY = blokcDim.y + 2 * ConstXTalkP.getSpanY();

  for (int r= ry; r<= ry+windowSimY; r++)
    {
      for (int c= rx; c<=(rx+windowDimX); c++)
      {
        if ((r < 0) || (r>= ImgRegP.getRegH(regId)) || (c < 0) || (c>=ImgRegP.getRegH(regId)))
        {
          sum += default_signal*ConstXTalkP.coeff(lx,ly,phase);
        }else{
          if(!useSharedMemory)
            sum += LDG_ACCESS(Ampl,r*AmplWidth+c) * ConstXTalkP.coeff(lx,ly,phase);
          else
            sum += Ampl[r*AmplWidth+c] * ConstXTalkP.coeff(lx,ly,phase);
        }
      }
    }

}

//__device__ inline

// execute with one warp per row and 2D thread blocks of width warp length
// each warp will slide across one row of the region
// kernel parameters:
// thread block dimensions (8,8)
// grid dimension ( numRegions.x, (imgH + n-1)/n, 1) // one block per region in x direction and one per n img rows in y direction
__global__
__launch_bounds__(128, 32)
void ProtonXTalkShared_k(
    const unsigned short * RegionMask,
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,

    const unsigned short  * bfMask,
    const float* BeadParamCube,
    unsigned short * BeadStateMask,
    float* PolyClonalCube,
    float* ResultCube,

    float * regionAvgSignal  // has to be inited to 0.0f for all regions
)
{

  extern __shared__ float smem[];


  const size_t regionCol = blockIdx.x;
  const size_t regionRow = (blockIdx.y*blockDim.y)/ImgRegP.getRegH();
  const size_t regId = ImgRegP.getRegIdFromGrid(regionCol,regionRow);

  if( LDG_ACCESS(RegionMask,regId) != RegionMaskLive) return;

  const size_t regionWidth = ImgRegP.getRegW(regId);
  const size_t windowWidth = blockDim.x;
  const size_t ix = regionCol * ImgRegP.getRegW()+ threadIdx.x;
  const size_t iy = (blockIdx.y*blockDim.y) + threadIdx.y;
  const size_t idx = ImgRegP.getWellIdx(ix,iy);
  const size_t ry = iy%ImgRegP.getRegH();
  size_t rx = threadIdx.x;

  const size_t planeStride = ImgRegP.getImgSize();

  //per bead
  const float * copies = BeadParamCube + BpCopies*planeStride + idx;
  const float * R = BeadParamCube + BpR*planeStride + idx;
  const float * AmplIn = ResultCube + planeStride * ResultAmpl + idx;
  const size_t AmplInWidth = ImgRegP.getImgW();
  float * AmplOut = ResultCube + planeStride * ResultAmplXTalk + idx;
  BeadStateMask += idx;
  PolyClonalCube += idx;

  bfMask += idx;

  //per region
  perFlowRegP += regId;
  perNucRegP +=  ImgRegP.getNumRegions() * ConstFlowP.getNucId() + regId;
  const float regAvgSig = LDG_LOAD(regionAvgSignal+regId);

  MagicXTalkForFLow MagicStuff(ConstFlowP.getRealFnum(), regAvgSig);

  while(rx < regionWidth ){ //while thread inside region

    if(Match(bfMask,MaskLive)){
      DoXTalk(perFlowRegP,perNucRegP,MagicStuff,AmplIn,AmplInWidth,AmplOut,regAvgSig,*copies,*R,regId,rx,ry);
      ClonalFilterUpdate(PolyClonalCube,BeadStateMask,AmplOut,planeStride,perFlowRegP->getCopyDrift(),*copies);

    }

    rx += windowWidth;
    copies += windowWidth;
    R += windowWidth;
    bfMask += windowWidth;
    AmplIn += windowWidth;
    AmplOut += windowWidth;
    BeadStateMask += windowWidth;
    PolyClonalCube += windowWidth;
  }

}

*/


