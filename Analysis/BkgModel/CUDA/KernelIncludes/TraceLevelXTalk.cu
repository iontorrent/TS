/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#include "SingleFlowFitKernels.h"
#include "MathModel/PoissonCdf.h"
#include "cuda_error.h"
#include "MathModel/PoissonCdf.h"
#include "ImgRegParams.h"
#include "UtilKernels.h"
#include "TraceLevelXTalk.h"
#include "ConstantSymbolDeclare.h"




#define XTALK_REGION_ONLY


/*

// xtalk calculation from excess hydrogen by neighbours
__global__ void SimpleXTalkNeighbourContributionAndAccumulation_LocalMem(// Here FL stands for flows
    const unsigned short * RegionMask, //per Region
    const unsigned short  * bfMask, // per Bead
    const unsigned short  * bstateMask, //per Bead

    float * xTalkContribution,  // buffer XTalk contribution to this well NxF
    float * genericXTalkTracesRegion, // one trace of max compressed frames per thread block or per region (atomicAdd)
    int * numGenericXTalkTracesRegion, //one int per region to average after accumulation
    const short* RawTraces,  //NxF
    const float * EmptyTraceRegion, //FxR
    const float* BeadParamCube, //NxP
    const float* RegionFrameCube, //FxRxT bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
    const ConstantParamsRegion * constRegP, // R
    const PerFlowParamsRegion * perFlowRegP, // R
    const PerNucParamsRegion * perNucRegP, //RxNuc
    const size_t * numFramesRegion, // R
    const bool * TestingGenericXtakSampleMask //ToDo: remove whne testing done
)
{

  extern __shared__ float smBaseimpleXTalkNeighbour[];



  /////////////////////////
  //coordinates and indices

  const size_t regionCol = blockIdx.x;
  const size_t regionRow = (blockIdx.y*blockDim.y)/ImgRegP.getRegH();

  //image coordinates
  const int ix = regionCol * ImgRegP.getRegW() + threadIdx.x;
  const int iy = (blockIdx.y*blockDim.y) + threadIdx.y;
  size_t idx = ImgRegP.getWellIdx(ix,iy);


  //region index to address region specific parameters
  const size_t regId = ImgRegP.getRegIdFromGrid(regionCol,regionRow);


  ////////////////////////
  // region specifics

  //set offset to first trace for this region
  float * genericXTalkTraceGlobal = genericXTalkTracesRegion + regId*ConstFrmP.getMaxCompFrames();
  int * numGenericXTalkTracesGlobal = numGenericXTalkTracesRegion + regId;

  //exit if no work for whole region
  if( LDG_ACCESS(RegionMask,regId) != RegionMaskLive) return;

  size_t nframes = numFramesRegion[regId];
  if (nframes == 0) return;


  //Shared Memory pointers
  float * sm_base = smBaseimpleXTalkNeighbour;
  float * sm_warp_base = sm_base + threadIdx.y*blockDim.x;
  float * sm = sm_warp_base + threadIdx.x;
  float * sm_warpTrace_base = sm_base + blockDim.x * blockDim.y;
  float * sm_warpTrace = sm_warpTrace_base + threadIdx.y * ConstFrmP.getMaxCompFrames();

  int t=threadIdx.x;
  //set shared mem warp trace buffer to 0
  while( t < ConstFrmP.getMaxCompFrames() ){
    sm_warpTrace[t] = 0.0f;
    t += blockDim.x;
  }

  //stride from one per bead plane to the next
  const size_t BeadPlaneStride = ImgRegP.getPlaneStride();
  //stride from one regions*frames plane to the next
  const size_t RegionFramesPlaneStride = ConstFrmP.getMaxCompFrames() * ImgRegP.getNumRegions();

  //update base pointer to data for this region in region frame cube
  RegionFrameCube += regId*ConstFrmP.getMaxCompFrames();  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
  const float * emptyTrace = EmptyTraceRegion + regId*ConstFrmP.getUncompFrames();
  const float* deltaFrames = RegionFrameCube + RfDeltaFrames*RegionFramesPlaneStride;


  //update per region pointers
  constRegP += regId;
  perFlowRegP += regId;

  //point to correct nuc
  perNucRegP +=  ImgRegP.getNumRegions() * ConstFlowP.getNucId() + regId;


  /////////////////////////////////
  // bead specific pointer updates
  //bfMask += idx;
  //bstateMask += idx;
  //RawTraces += idx;
  //BeadParamCube += idx;

  //sliding window
  int rx = threadIdx.x;
  int ry = iy%ImgRegP.getRegH();




  const int windowWidth = blockDim.x;
  const int rowsPerStride = blockDim.y;
  const size_t rowStride = rowsPerStride * ImgRegP.getImgW();

  int windowOffset = 0;

  //|regId 0      |regId 1      |
  //|b0w0-------> |b2w0-->      | each warp slides across region
  //|b0w1---->    |b2w1----->   | independent from other warps.
  //|b1w0         |...          | multiple blocks work on one region
  //|b1w1         |             | all the threads within a block work
  //|_____________|_____________| on wells of the same region
  //|regId 2      |regId 3      |
  //|             |             |

  //////////////////////////////////
  // local memory and variables
  //can probably be partially removed and reused instead
  float incorp_rise[MAX_COMPRESSED_FRAMES_GPU];
  float lost_hydrogen[MAX_COMPRESSED_FRAMES_GPU];
  float bulk_signal[MAX_COMPRESSED_FRAMES_GPU];
  float xtalk[MAX_COMPRESSED_FRAMES_GPU];
  //volatile float * xtalk = xtalkBuff;




  int numGenericXTalkTraces = 0;


  if(ry < ImgRegP.getRegH(regId)) {
    // warp divergent code
    // DO NOT call syncthread within this branch!!!

    while(windowOffset < ImgRegP.getRegW(regId)){

      //update coordinates and offsets for well we are accumulating for
      const int lrx = rx + windowOffset;
      const int lix = ix + windowOffset;
      const int lidx = idx + windowOffset;
      float * lxTalkContribution = xTalkContribution + lidx;
      //const unsigned short  * lBfMask = bfMask + lidx;
      bool useForGenericXTalk = false;

      // zeroing has to be done before next if statement for later warp level accumulation
      for (int f=0; f<nframes; ++f) {
        xtalk[f] = 0;
      }


      //ony threads that work on well within the reagion actually do this work here:
      if(lrx < ImgRegP.getRegW(regId)){

        useForGenericXTalk = TestingGenericXtakSampleMask[lidx]; //ToDo: remove after testing
        //useForGenericXTalk =  useForEmpty(lBfMask);
        //useForGenericXTalk = Match(lBfMask, MaskLive);
        //useForGenericXTalk = (Match(lBfMask, MaskLive) || useForEmpty(lBfMask));


        for (int nid=0; nid<ConstTraceXTalkP.getNumNeighbours(); ++nid){

          //neighbor global coordinates
          int nix;
          int niy;

          //get coordinates for neighbor we are workign on
          ConstTraceXTalkP.getBlockCoord(nix,niy,nid,lix,iy);

          if( ImgRegP.getRegBaseX(regId) <= nix && nix < ImgRegP.getRegUpperX(regId))
          {
            if( ImgRegP.getRegBaseY(regId) <= niy && niy < ImgRegP.getRegUpperY(regId))
            {

              //update local mask offsets for current neighbor for filtering
              size_t nIdx = ImgRegP.getWellIdx(nix,niy);
              const unsigned short  * nBfMask = bfMask +nIdx;
              const unsigned short  * nBstateMask = bstateMask + nIdx;
              //filter non-live, pinned or corrupt neighbors
              if( Match(nBfMask, MaskLive) && !( Match(nBstateMask,BkgMaskPinned) || Match(nBstateMask,BkgMaskCorrupt)) ){
                //update local buffer offsets for current neighbor
                const short* nRawTraces = RawTraces + nIdx;
                const float* nBeadParamCube = BeadParamCube + nIdx;
                const float copies = *(nBeadParamCube + BpCopies*BeadPlaneStride);
                const float R = *(nBeadParamCube + BpR*BeadPlaneStride);


                //float Rval, tau;
                const float etbR = ComputeETBR(perNucRegP, perFlowRegP->getRatioDrift(), R, copies);
                const float tauB = ComputeTauB(constRegP, etbR);
                //const float SP = ComputeSP(perFlowRegP->getCopyDrift(), copies);

                // Calculate approximate incorporation signal
                float one_over_two_taub = 1.0f / (2.0f*tauB);
                int f = 0;

                float xt = deltaFrames[f]*one_over_two_taub;

                incorp_rise[f] = (1.0f+xt)*nRawTraces[f*BeadPlaneStride] - (etbR+xt)*emptyTrace[f];
                f++;

                for (;f<nframes; ++f) {
                  xt = deltaFrames[f]*one_over_two_taub;
                  incorp_rise[f] = (1.0+xt)*nRawTraces[f*BeadPlaneStride] - (1.0f-xt)*nRawTraces[(f-1)*BeadPlaneStride] - ((etbR+xt)*emptyTrace[f]-(etbR-xt)*emptyTrace[f-1]) + incorp_rise[(f-1)];
                }

                // Calculate lost hydrogen
                f = perFlowRegP->getStart();
                xt = 1.0f/(1.0f + (deltaFrames[f]*one_over_two_taub));
                lost_hydrogen[f] = incorp_rise[f]*xt;
                f++;
                for (;f<nframes; ++f) {
                  xt = 1.0f/(1.0f + (deltaFrames[f]*one_over_two_taub));
                  lost_hydrogen[f] = (incorp_rise[f] - incorp_rise[(f-1)] + (1.0f-(deltaFrames[f]*one_over_two_taub))*lost_hydrogen[(f-1)])*xt;
                }


                for (f = perFlowRegP->getStart();f<nframes; ++f) {
                  lost_hydrogen[f] = incorp_rise[f] - lost_hydrogen[f];
                }

                // Calculate ions from bulk
                float taue = etbR * tauB;
                f = perFlowRegP->getStart();
                one_over_two_taub = 1.0f / (2.0f*taue);
                xt = 1.0f/(1.0f + (deltaFrames[f]*one_over_two_taub));
                bulk_signal[f] = lost_hydrogen[f]*xt;
                f++;
                for (;f<nframes; ++f) {
                  xt = 1.0f/(1.0f + (deltaFrames[f]*one_over_two_taub));
                  bulk_signal[f] = (lost_hydrogen[f] - lost_hydrogen[(f-1)] + (1.0f-(deltaFrames[f]*one_over_two_taub))*bulk_signal[(f-1)])*xt;
                }

                // Scale down the ion by neighbor multiplier

                for (f=perFlowRegP->getStart(); f<nframes; ++f){
                  xtalk[f] += bulk_signal[f] * ConstTraceXTalkP.getMultiplier(nid);
                }

              }
            }
          }
        }
      } //if rx < regiond width

      //now this is where the warp level magic gets thrown in

      int genericXTalkTracesThisWarp = 0;
      // how many traces are we actually working on in this window
      int * ismWB = (int*)sm_warp_base;
      int * ism = ismWB + threadIdx.x;
 *ism = (useForGenericXTalk)?(1):(0);
      WarpSumNoSync(ism);
      genericXTalkTracesThisWarp = *ismWB;
      numGenericXTalkTraces += genericXTalkTracesThisWarp; //count how many XTalks traces for empties got already handled by this warp

      for (int f=0; f<nframes; ++f){
        float thisFrame = xtalk[f] ;

        if(lrx < ImgRegP.getRegW(regId)) // let only threads write to global that actually have data from within the region (all others have 0)
 *lxTalkContribution = thisFrame ; //store xtalk for this single well frame by frame
        //more warp level magic:
        //accumulate generic Xtalk for this window and add to xtalk already accumulated by this warp
        //WarpTraceAccumSingleFrame(sm_warpTrace,f,sm_warp_base,thisFrame,useForGenericXTalk);//thisFrame,useForGenericXTalk);
 *sm = (useForGenericXTalk)?(thisFrame):(0.0f);
           WarpSumNoSync(sm);
           float genXtalkFrame = *sm_warp_base; //WarpTraceAccumSingleFrame(sm_warpTrace,f,sm_warp_base,thisFrame,useForGenericXTalk);//thisFrame,useForGenericXTalk);

           if(threadIdx.x == 0)
               sm_warpTrace[f] += genXtalkFrame;

        lxTalkContribution+=BeadPlaneStride; // one frame per plane per bead
      }
      windowOffset += windowWidth;

    } // while windowOffset < region Width.

  }// if ry < region Height
  // END of warp divergent code from here on we can syncthread again!!
  __syncthreads();
  //block level reduction and global accumulation with atomics
  BlockTraceAccumfromWarpsInplaceToGlobal(genericXTalkTraceGlobal,1,sm_warpTrace_base,nframes, ConstFrmP.getMaxCompFrames(), true);
  __syncthreads();
  int * ism = (int*)sm;
 *ism = numGenericXTalkTraces;
  BlockAccumValuePerWarpToGlobal(numGenericXTalkTracesGlobal,(int*)sm_base,true);

}

 */




//the next two kernels calculate the same as above but use 1/neighbours the calculations
// xtalk calculation from excess hydrogen by neighbours
__global__ void SimpleXTalkNeighbourContribution(// Here FL stands for flows
    const unsigned short * RegionMask, //per Region
    const unsigned short  * bfMask, // per Bead
    const unsigned short  * bstateMask, //per Bead

    float * myBaseXTalkContribution,   // buffer XTalk contribution of this well NxF

    const short* RawTraces,  //NxF
    const float * EmptyTraceRegion, //FxR
    const float* BeadParamCube, //NxP
    const float* RegionFrameCube, //FxRxT bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
    const ConstantParamsRegion * constRegP, // R
    const PerFlowParamsRegion * perFlowRegP, // R
    const PerNucParamsRegion * perNucRegP, //RxNuc
    const size_t * numFramesRegion // R
)
{

  /////////////////////////
  //coordinates and indices

  const size_t regionCol = blockIdx.x;
  const size_t regionRow = (blockIdx.y*blockDim.y)/ImgRegP.getRegH();

  //image coordinates
  const int ix = regionCol * ImgRegP.getRegW() + threadIdx.x;
  const int iy = (blockIdx.y*blockDim.y) + threadIdx.y;
  size_t idx = ImgRegP.getWellIdx(ix,iy);


  //region index to address region specific parameters
  const size_t regId = ImgRegP.getRegIdFromGrid(regionCol,regionRow);


  ////////////////////////
  // region specifics

  //exit if no work for whole region
  if( LDG_ACCESS(RegionMask,regId) != RegionMaskLive) return;

  size_t nframes = numFramesRegion[regId];
  if (nframes == 0) return;

  //stride from one per bead plane to the next
  const size_t BeadPlaneStride = ImgRegP.getPlaneStride();
  //stride from one regions*frames plane to the next
  const size_t RegionFramesPlaneStride = ConstFrmP.getMaxCompFrames() * ImgRegP.getNumRegions();

  //update base pointer to data for this region in region frame cube
  RegionFrameCube += regId*ConstFrmP.getMaxCompFrames();  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
  const float * emptyTrace = EmptyTraceRegion + regId*ConstFrmP.getUncompFrames();
  const float* deltaFrames = RegionFrameCube + RfDeltaFrames*RegionFramesPlaneStride;


  //update per region pointers
  constRegP += regId;
  perFlowRegP += regId;

  //point to correct nuc
  perNucRegP +=  ImgRegP.getNumRegions() * ConstFlowP.getNucId() + regId;

  //////////////////////////
  // sliding window base region coordinates
  //|regId 0      |regId 1      |
  //|b0w0-------> |b2w0-->      | each warp slides across region
  //|b0w1---->    |b2w1----->   | independent from other warps.
  //|b1w0         |...          | multiple blocks work on one region
  //|b1w1         |             | all the threads within a block work
  //|_____________|_____________| on wells of the same region
  //|regId 2      |regId 3      |
  //|             |             |

  int rx = threadIdx.x;
  int ry = iy%ImgRegP.getRegH();

  const int windowWidth = blockDim.x;

  int windowOffset = 0;

  //////////////////////////////////
  // local memory and variables
  float incorp_rise[MAX_COMPRESSED_FRAMES_GPU];
  float lost_hydrogen[MAX_COMPRESSED_FRAMES_GPU];
  float bulk_signal[MAX_COMPRESSED_FRAMES_GPU];

  if(ry < ImgRegP.getRegH(regId)) {
    // warp divergent code
    // DO NOT call syncthread within this branch it can lead to undefined state

    while(windowOffset < ImgRegP.getRegW(regId)){

      //update coordinates and offsets for well we are accumulating for
      const int lrx = rx + windowOffset;
      //const int lix = ix + windowOffset;
      const int lidx = idx + windowOffset;
      float * lmyBaseXTalkContribution = myBaseXTalkContribution + lidx;

      //ony threads that work on well within the reagion actually do this work here:
      if(lrx < ImgRegP.getRegW(regId)){


        //Acquire well specific pointers:
        const unsigned short  * lBfMask = bfMask +lidx;
        const unsigned short  * lBstateMask = bstateMask + lidx;
        //filter non-live, pinned or corrupt neighbors
        const short* lRawTraces = RawTraces + lidx;
        const float* lBeadParamCube = BeadParamCube + lidx;
        const float copies = *(lBeadParamCube + BpCopies*BeadPlaneStride);
        const float R = *(lBeadParamCube + BpR*BeadPlaneStride);
        //am I a worthy contributor to xtalk? only then perform calculation. if not store 0 at the end
        bool contributor =  Match(lBfMask, MaskLive) && !( Match(lBstateMask,BkgMaskPinned) || Match(lBstateMask,BkgMaskCorrupt));

        if(contributor){

          //float Rval, tau;
          const float etbR = ComputeETBR(perNucRegP, perFlowRegP->getRatioDrift(), R, copies);
          const float tauB = ComputeTauB(constRegP, etbR);
          //const float SP = ComputeSP(perFlowRegP->getCopyDrift(), copies);

          // Calculate approximate incorporation signal
          float one_over_two_taub = 1.0f / (2.0f*tauB);
          int f = 0;

          float xt = deltaFrames[f]*one_over_two_taub;

          incorp_rise[f] = (1.0f+xt)*lRawTraces[f*BeadPlaneStride] - (etbR+xt)*emptyTrace[f];
          f++;

          for (;f<nframes; ++f) {
            xt = deltaFrames[f]*one_over_two_taub;
            incorp_rise[f] = (1.0+xt)*lRawTraces[f*BeadPlaneStride] - (1.0f-xt)*lRawTraces[(f-1)*BeadPlaneStride] - ((etbR+xt)*emptyTrace[f]-(etbR-xt)*emptyTrace[f-1]) + incorp_rise[(f-1)];
          }

          // Calculate lost hydrogen
          f = perFlowRegP->getStart();
          xt = 1.0f/(1.0f + (deltaFrames[f]*one_over_two_taub));
          lost_hydrogen[f] = incorp_rise[f]*xt;
          f++;
          for (;f<nframes; ++f) {
            xt = 1.0f/(1.0f + (deltaFrames[f]*one_over_two_taub));
            lost_hydrogen[f] = (incorp_rise[f] - incorp_rise[(f-1)] + (1.0f-(deltaFrames[f]*one_over_two_taub))*lost_hydrogen[(f-1)])*xt;
          }

          for (f = perFlowRegP->getStart();f<nframes; ++f) {
            lost_hydrogen[f] = incorp_rise[f] - lost_hydrogen[f];
          }

          // Calculate ions from bulk
          float taue = etbR * tauB;
          f = perFlowRegP->getStart();
          one_over_two_taub = 1.0f / (2.0f*taue);
          xt = 1.0f/(1.0f + (deltaFrames[f]*one_over_two_taub));
          bulk_signal[f] = lost_hydrogen[f]*xt;
          f++;
          for (;f<nframes; ++f) {
            xt = 1.0f/(1.0f + (deltaFrames[f]*one_over_two_taub));
            bulk_signal[f] = (lost_hydrogen[f] - lost_hydrogen[(f-1)] + (1.0f-(deltaFrames[f]*one_over_two_taub))*bulk_signal[(f-1)])*xt;
          }
        }// if contributor

        //contributors store bulk_signal to global all other store 0 values
        for(int f=perFlowRegP->getStart(); f<nframes; f++)
          lmyBaseXTalkContribution[f*BeadPlaneStride] = (contributor)?(bulk_signal[f]):(0);

      } //if lrx < regiond width

      windowOffset += windowWidth;
    } // while windowOffset < regW
  } //if ry < regH
}



// xtalk calculation from excess hydrogen by neighbours
__global__ void GenericXTalkAndNeighbourAccumulation(// Here FL stands for flows
    const unsigned short * RegionMask, //per Region
    const unsigned short  * bfMask, // per Bead
    const unsigned short  * bstateMask, //per Bead
    float * BaseXTalkContribution,  // XTalk of each single well
    float * xTalkContribution,  // buffer XTalk to store accumulated xtalk at each well
    float * genericXTalkTracesperBlock, // one trace of max compressed frames per thread block or per region (atomicAdd)
    int * numGenericXTalkTracesRegion, //one int per region to average after accumulation
    const PerFlowParamsRegion * perFlowRegP, // R
    const size_t * numFramesRegion, // R
    const bool * TestingGenericXtakSampleMask //ToDo: remove whne testing done
)
{

  extern __shared__ float smBaseimpleXTalkNeighbour[];


  /////////////////////////
  //coordinates and indices

  const size_t regionCol = blockIdx.x;
  const size_t regionRow = (blockIdx.y*blockDim.y)/ImgRegP.getRegH();

  //image coordinates
  const int ix = regionCol * ImgRegP.getRegW() + threadIdx.x;
  const int iy = (blockIdx.y*blockDim.y) + threadIdx.y;
  size_t idx = ImgRegP.getWellIdx(ix,iy);


  //region index to address region specific parameters
  const size_t regId = ImgRegP.getRegIdFromGrid(regionCol,regionRow);


  ////////////////////////
  // region specifics
  perFlowRegP += regId;

  //set offset to first trace for this region
  int blocksPerRegion = (ImgRegP.getRegH() + blockDim.y -1)/blockDim.y;
  float * genericXTalkRegionBase = genericXTalkTracesperBlock + regId * ConstFrmP.getMaxCompFrames() * blocksPerRegion;

  int * numGenericXTalkTracesGlobal = numGenericXTalkTracesRegion + regId;

  //exit if no work for whole region
  if( LDG_ACCESS(RegionMask,regId) != RegionMaskLive) return;

  size_t nframes = numFramesRegion[regId];
  if (nframes == 0) return;

  ////////////////////////
  //Shared Memory pointers
  float * sm_base = smBaseimpleXTalkNeighbour;
  float * sm_warp_base = sm_base + threadIdx.y*blockDim.x;
  float * sm = sm_warp_base + threadIdx.x;
  float * sm_warpTrace_base = sm_base + blockDim.x * blockDim.y;
  float * sm_warpTrace = sm_warpTrace_base + threadIdx.y * ConstFrmP.getMaxCompFrames();

  int t=threadIdx.x;
  //set shared mem warp trace buffer to 0
  while( t < ConstFrmP.getMaxCompFrames() ){
    sm_warpTrace[t] = 0.0f;
    t += blockDim.x;
  }


  //////////////////////////
  // sliding window base region coordinates
  int rx = threadIdx.x;
  int ry = iy%ImgRegP.getRegH();

  //find offset for generic xtalk trace in global for this threadBlock
  int blockInRegion = ry/blockDim.y;
  float * genericXTalkTraceGlobal = genericXTalkRegionBase + blockInRegion*ConstFrmP.getMaxCompFrames();


  const int windowWidth = blockDim.x;
  const int rowsPerStride = blockDim.y;
  const size_t rowStride = rowsPerStride * ImgRegP.getImgW();
  //stride from one per bead plane to the next:
  const size_t BeadPlaneStride = ImgRegP.getPlaneStride();
  int windowOffset = 0;
  int numGenericXTalkTraces = 0;

  // local memory and variables
  float xtalk[MAX_COMPRESSED_FRAMES_GPU];

  if(ry < ImgRegP.getRegH(regId)) {
    // warp divergent code
    // DO NOT call syncthread within this branch!!!

    while(windowOffset < ImgRegP.getRegW(regId)){

      //update coordinates and offsets for well we are accumulating for
      const int lrx = rx + windowOffset;
      const int lix = ix + windowOffset;
      const int lidx = idx + windowOffset;
      float * lxTalkContribution = xTalkContribution + lidx;
      const unsigned short  * lBfMask = bfMask + lidx;
      bool useForGenericXTalk = false;

      // zeroing has to be done before next if statement for later warp level accumulation
      for (int f=0; f<nframes; ++f) {
        xtalk[f] = 0;
      }


      //only threads that work on well within the region actually do this work here:
      if(lrx < ImgRegP.getRegW(regId)){

        //useForGenericXTalk = TestingGenericXtakSampleMask[lidx]; //ToDo: remove after testing
        useForGenericXTalk =  useForEmpty(lBfMask);
        //useForGenericXTalk = Match(lBfMask, MaskLive);
        //useForGenericXTalk = (Match(lBfMask, MaskLive) || useForEmpty(lBfMask));


        for (int nid=0; nid<ConstTraceXTalkP.getNumNeighbours(); ++nid){

          //neighbor global coordinates
          int nix;
          int niy;

          //get coordinates for neighbor we are working on
          ConstTraceXTalkP.getBlockCoord(nix,niy,nid,lix,iy);

#ifdef XTALK_REGION_ONLY
          if( ImgRegP.getRegBaseX(regId) <= nix && nix < ImgRegP.getRegUpperX(regId))
          {
            if( ImgRegP.getRegBaseY(regId) <= niy && niy < ImgRegP.getRegUpperY(regId))
            {
#else
          if( 0 <= nix && nix < ImgRegP.getImgW())
          {
             if(  0 <= niy && niy < ImgRegP.getImgH())
             {
#endif

//update local mask offsets for current neighbor for filtering
                  size_t nIdx = ImgRegP.getWellIdx(nix,niy);
                  const unsigned short  * nBfMask = bfMask +nIdx;
                  const unsigned short  * nBstateMask = bstateMask + nIdx;
                  const float * nBaseXTalkContribution = BaseXTalkContribution + nIdx;
                  //filter non-live, pinned or corrupt neighbors
                  //if(lrx == 41 && ry == 55) printf("%lu,%d,%d,",nIdx, nix, niy);
                  if( Match(nBfMask, MaskLive) && !( Match(nBstateMask,BkgMaskPinned) || Match(nBstateMask,BkgMaskCorrupt)) ){
                    //update local buffer offsets for current neighbor

                    //for (int f=perFlowRegP->getStart(); f<nframes; ++f){
                    for (int f=0; f<nframes; ++f){
                                         xtalk[f] += nBaseXTalkContribution[f*BeadPlaneStride] * ConstTraceXTalkP.getMultiplier(nid);
                    //  if(lrx == 41 && ry == 55) printf("%f,",nBaseXTalkContribution[f*BeadPlaneStride] * ConstTraceXTalkP.getMultiplier(nid));

                    }

                  }
                  //if(lrx == 41 && ry == 55) printf("\n");
                }
              }
            }// neighbour loop
          } //if rx < region width

          //now this is where the warp level magic gets thrown in

          int genericXTalkTracesThisWarp = 0;
          // how many traces are we actually working on in this window
          int * ismWB = (int*)sm_warp_base;
          int * ism = ismWB + threadIdx.x;
          *ism = (useForGenericXTalk)?(1):(0);
          WarpSumNoSync(ism);
          genericXTalkTracesThisWarp = *ismWB;
          numGenericXTalkTraces += genericXTalkTracesThisWarp; //count how many XTalks traces for empties got already handled by this warp

          for (int f=0; f<nframes; ++f){
            float thisFrame = xtalk[f] ;

            if(lrx < ImgRegP.getRegW(regId)) // let only threads write to global that actually have data from within the region (all others have 0)
              *lxTalkContribution = thisFrame ; //store xtalk for this single well frame by frame
            //more warp level magic:
            //accumulate generic Xtalk for this window and add to xtalk already accumulated by this warp
            *sm = (useForGenericXTalk)?(thisFrame):(0.0f);
            WarpSumNoSync(sm);
            float genXtalkFrame = *sm_warp_base + sm_warpTrace[f] ; //WarpTraceAccumSingleFrame(sm_warpTrace,f,sm_warp_base,thisFrame,useForGenericXTalk);//thisFrame,useForGenericXTalk);

            if(threadIdx.x == 0)
              sm_warpTrace[f] = genXtalkFrame;

            lxTalkContribution+=BeadPlaneStride; // one frame per plane per bead
          }
          windowOffset += windowWidth;

        } // while windowOffset < region Width.

      }// if ry < region Height
      // END of warp divergent code from here on we can syncthread again!!

      //block level reduction and global accumulation with atomics
      BlockTraceAccumfromWarpsInplaceToGlobal(genericXTalkTraceGlobal,1,sm_warpTrace_base,nframes, ConstFrmP.getMaxCompFrames(), false);
      __syncthreads();
      int * ism = (int*)sm;
      *ism = numGenericXTalkTraces;
      BlockAccumValuePerWarpToGlobal(numGenericXTalkTracesGlobal,(int*)sm_base,true);

    }


    //one 1D block per region

    __global__ void GenericXTalkAccumulation(// Here FL stands for flows
        float * genericXTalkTracesRegion, // one trace of max compressed frames per thread block or per region (atomicAdd)
        const float * genericXTalkTracesPerBlock,
        const int * numGenericXTalkTracesRegion, //one int per region to average after accumulation
        const size_t * numFrames,
        const int blocksPerRegion
    )
    {

      //one block per region
      const size_t regId = blockIdx.x + blockIdx.y * gridDim.x;

      //per region trace output
      genericXTalkTracesRegion += regId*ConstFrmP.getMaxCompFrames();
      //pointer to first partial trace for accumulation
      const float * genericXTalkRegionBase = genericXTalkTracesPerBlock + regId * ConstFrmP.getMaxCompFrames() * blocksPerRegion;
      const int numtraces=numGenericXTalkTracesRegion[regId];


      const int tIdx = threadIdx.x + threadIdx.y*blockDim.x;
      const int windowSize = blockDim.x*blockDim.y;
      const size_t numf = numFrames[regId];

      //if more frames than threads in block (schould never happen)
      for(size_t f=tIdx; f<numf;f+=windowSize){

        float myframe=0;
        //pointer to frame f in first partial trace
        const float * PerBlockThisFrame = genericXTalkRegionBase + tIdx;

        for(int block=0; block<blocksPerRegion;block++){

          if(f<numf){
            myframe += *PerBlockThisFrame;
          }
          //move to frame f for next block in partial traces
          PerBlockThisFrame += ConstFrmP.getMaxCompFrames(); // move to next trace;
        }

        genericXTalkTracesRegion[f] = (numtraces>0)?(myframe/numtraces):(0.0f);

      }


    }












