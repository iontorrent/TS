/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "GenerateBeadTraceKernels.h"
//#include "cuda_runtime.h"
#include "cuda_error.h"
#include "dumper.h"
#include "Mask.h"
#include "Image.h"
#include "TimeCompression.h"
#include "Region.h"
#include "BeadTracker.h"
#include "BkgTrace.h"
#include "EnumDefines.h"
#include "SingleFlowFitKernels.h"
#include "UtilKernels.h"
#include "ConstantSymbolDeclare.h"

using namespace std;


#define COLLECT_SAMPLES
#define SAMPLES_IN_ORDER

template<typename T>
__device__ inline
float ComputeDcOffsetForCompressedTrace_v2 ( const T * fgBufferLocal,
    const size_t frameStride,
    const float* frameNumber,
    const float t_start,
    const float t_end,
    const int numFrames)//int flow_max)
{

  // re-zero the traces
  //
  // Identical in output to above function, just extracts logic
  //   from the inner loop to make it faster
  int pt;
  int start_pt=0;
  int end_pt=0;
  float cnt;
  float dc_zero=0.0f;// = ComputeDcOffset(fgPtr, t_start, t_end);
  float overhang_start=0.0f;
  float overhang_end=0.0f;
  int overhang_start_pt=1;
  int overhang_end_pt=1;


  //    Timer tmr;

  /*ComputeDcOffset_params( frameNumber, numFrames, t_start, t_end, start_pt, end_pt, cnt,
      overhang_start, overhang_end); *///only needs to be done once maybe global object with locking mechanism?
  cnt = 0.0001f;
  start_pt = -1;
  end_pt = 0;
  overhang_start = 0.0f;
  overhang_end = 0.0f;

  // TODO: is this really "rezero frames before pH step start?"
  // this should be compatible with i_start from the nuc rise - which may change if we change the shape???
  for (pt = 0; frameNumber[pt] < t_end; pt++)
  {
    end_pt = pt + 1;
    if (frameNumber[pt] > t_start)
    {
      if (start_pt == -1)
        start_pt = pt; // set to first point above t_start
      cnt += 1.0f; // should this be frames_per_point????
    }
  }

  if (start_pt < 0)
    start_pt = 0; // make sure we don't index incorrectly
  // include values surrounding t_start & t_end weighted by overhang
  else
  {
    // This part is really broken.  Fixing it makes things worse??
    //   the fraction overhang_start is > 1

    int ohstart_pt = start_pt ? start_pt : 1;
    float den = (frameNumber[ohstart_pt] - frameNumber[ohstart_pt - 1]);
    if (den > 0)
    {
      overhang_start = (frameNumber[ohstart_pt] - t_start) / den;
      cnt += overhang_start;
    }
  }

  if ((end_pt < numFrames) && (end_pt > 0))
  {
    // timecp->frameNumber[end_pt-1] <= t_end < timecp->frameNumber[end_pt]
    // normalize to a fraction in the spirit of "this somehow makes it worse
    float den = (frameNumber[end_pt] - frameNumber[end_pt - 1]);
    if (den > 0)
    {
      overhang_end = (t_end - frameNumber[end_pt - 1]) / den;
      cnt += overhang_end;
    }
  }


  if(start_pt > 0)
    overhang_start_pt = start_pt-1;
  else
    overhang_start_pt = 0;

  if(end_pt > 0 && end_pt < numFrames)
    overhang_end_pt = end_pt;
  else
    overhang_end_pt=0;


  dc_zero=0;

  for (pt = start_pt; pt < end_pt; pt++)
    dc_zero += (fgBufferLocal[pt]);

  // add end interpolation parts
  dc_zero += overhang_start*(fgBufferLocal[overhang_start_pt]);
  dc_zero += overhang_end  *(fgBufferLocal[overhang_end_pt]);

  // make it into an average
  dc_zero /= cnt;




  return dc_zero;
  /*
    // now, subtract the dc offset from all the points
    FG_BUFFER_TYPE dc_zero_s = dc_zero;
    for (int pt = 0;pt < numFrames;pt++){   // over real data
   *fgBuffer = fgBufferLocal[pt] - dc_zero_s;
      fgBuffer += frameStride;
    }

   */

}


//shifts values from in to out. shift by determines direction and shift amoount
// shiftBy < 0 shift left, shiftBy > 0 shifts right.
__host__ __device__ inline
void ShiftUncompressedTraceBy(float * out, const float * in, const float shift, int frames)
{

  if(shift != 0){
    int shiftwhole = (int)shift;  //Truncate to get number of whole frames to shift.

    int nearFrame = -shiftwhole;   //determine the closer of the two frames to interpolate in-between
    int farFrame = nearFrame + ((shift < 0)?(1):(-1));  //determine the frame further away. interpolate between near and far

    float farFrac =  abs(shift-(float)shiftwhole);  //determine fraction of far frame
    float nearFrac = 1.0f - farFrac;  //and fraction of near frame used for interpolation

    //  cout << "nearFrame "<< nearFrame <<" nearFrac "<< nearFrac <<" farFrame " << farFrame <<" farFrac "<<farFrac <<endl;

    int lastframe = frames-1;  // useful input frames range from 0 to frames-1

    for(int i=0; i<frames; i++){

      int nframe = nearFrame;
      int fframe = farFrame;

      if(nframe < 0 || fframe < 0)
        nframe = fframe = 0;

      if(nframe > lastframe || fframe > lastframe) //handle right boundary, use last frame for left and right when right is out of bounds
        nframe = fframe = lastframe;

      out[i] =in[nframe]*nearFrac + in[fframe]*farFrac;

      nearFrame++;
      farFrame++;

    }
  }else{
    for(int i=0; i<frames; i++){
      out[i] = in[i];
    }
  }

}


template <typename T>
__device__ inline
void SmoothWeightedNeighborAverage(T * out, T * in, int n, float weight)
{
  if(in == out){ //in place
    for(int i = 1; i < n-1; i++)
    {
      T last = in[0];
      T current = in[1];
      T next;
      for(int i = 1; i < n-1; i++)
      {
        next = in[i+1];
        T val = (current + (last + next)*weight)/(1+2*weight);
        if(i == 1) out[0] = val;

        out[i] = val;

        if(i == n-2)
          out[i+1] = val;
        else{
          last = current;
          current = next;
        }
      }
    }



  }else{
    for(int i = 1; i < n-1; i++)
    {
      T val = (in[i] + (in[i-1] + in[i+1])*weight)/(1+2*weight);
      if(i == 1) out[0] = val;
      out[i] = val;
      if(i == n-2) out[i+1] = val;
    }

  }

}


__device__ inline
void ShiftUncompressedTraceAndAccum(float * out, const float * in, const float shift, int frames, float * accum_out = NULL)
{

  if(shift != 0){
    int shiftwhole = (int)shift;  //Truncate to get number of whole frames to shift.

    int nearFrame = -shiftwhole;   //determine the closer of the two frames to interpolate in-between
    int farFrame = nearFrame + ((shift < 0)?(1):(-1));  //determine the frame further away. interpolate between near and far

    float farFrac =  abs(shift-(float)shiftwhole);  //determine fraction of near frame
    float nearFrac = 1.0f - farFrac;  //and fraction of farm frame used for interpolation

    //  cout << "nearFrame "<< nearFrame <<" nearFrac "<< nearFrac <<" farFrame " << farFrame <<" farFrac "<<farFrac <<endl;

    int lastframe = frames-1;  // useful input frames range from 0 to frames-1

    for(int i=0; i<frames; i++){

      int nframe = nearFrame;
      int fframe = farFrame;

      if(nframe < 0 || fframe < 0)
        nframe = fframe = 0;

      if(nframe > lastframe || fframe > lastframe) //handle right boundary, use last frame for left and right when right is out of bounds
        nframe = fframe = lastframe;

      float tmp = in[nframe]*nearFrac + in[fframe]*farFrac;

      out[i] = tmp;

      if(accum_out != NULL){
        accum_out[i] += tmp;
      }

      nearFrame++;
      farFrame++;

    }
  }else{
    for(int i=0; i<frames; i++){
      out[i] = in[i];
      if(accum_out != NULL){
        accum_out[i] += in[i];
      }
    }
  }

}



//Uncompresses the VFC image, shifts it by t0 and re-compresses according to the
//the frames per point passed in framesPerPoint
//works on one single well in the raw.image at position l_coord
//all other passed pointers and values are specific to the well
// const symbols:
// ConstFrmP constant symbol needs to be initialized for this kernel
__device__ inline
int LoadImgWOffset(
    FG_BUFFER_TYPE * fgptr, //now points to frame 0 of current bead in image
    short * image,
    const float * frameNumber,
    const int * framesPerPoint,
    int nfrms,
    int frameStride,
    float t0Shift,
    bool isEmpty,
    float * emptyTraceSum,
    const float time_start,
    const PerFlowParamsRegion & perFlowRegP,
    size_t idx
)
{

  int my_frame = 0,compFrm,curFrms,curCompFrms;

  float tmpAdder;

  //int interf,lastInterf=-1;
  FG_BUFFER_TYPE fgTmp[MAX_COMPRESSED_FRAMES_GPU];
  float traceTmp[MAX_UNCOMPRESSED_FRAMES_GPU+4];
  FG_BUFFER_TYPE lastVal;




  GetUncompressTrace(traceTmp+4, ConstFrmP,  image, frameStride );

  float * emptyAccumPtr = NULL;
  if(isEmpty) emptyAccumPtr = emptyTraceSum;
  //accumulate empty traces in this function so we do not have to touch all the frames again
  ShiftUncompressedTraceAndAccum(traceTmp,traceTmp+4,-t0Shift,ConstFrmP.getUncompFrames(),emptyAccumPtr);

#if UNCOMPRESSED_REZERO
  float toffset;
  if(isEmpty)
    toffset = MAGIC_OFFSET_FOR_EMPTY_TRACE;
  else
    toffset = perFlowRegP.getSigma();

  float dcOffset = ComputeDcOffsetForUncompressedTrace(traceTmp,ConstFrmP.getUncompFrames(),time_start, perFlowRegP.getTMidNuc()-toffset);
#endif
  ;
  my_frame = 0;//ConstFrmP.interpolatedFrames[startFrame]-1;
  compFrm = 0;
  tmpAdder=0.0f;
  curFrms=0;
  curCompFrms=framesPerPoint[compFrm];


  while ((my_frame < ConstFrmP.getUncompFrames()) && ((compFrm < nfrms)))
  {

    tmpAdder += traceTmp[my_frame];

    if(++curFrms >= curCompFrms)
    {
      tmpAdder /= curCompFrms;

#if UNCOMPRESSED_REZERO
      tmpAdder -= dcOffset;
#endif

      lastVal = (FG_BUFFER_TYPE)(tmpAdder); //Maybe use rintf or round to get more precision
      fgTmp[compFrm] = lastVal;
      compFrm++;
      curCompFrms = framesPerPoint[compFrm];
      curFrms=0;
      tmpAdder= 0.0f;
    }
    my_frame++;
  }


  if(compFrm > 0 )
  {
#if !UNCOMPRESSED_REZERO
    //this will produce garbage or crash if frameNumber is not passed to kernel
    float toffset;
    if(isEmpty)
      toffset = MAGIC_OFFSET_FOR_EMPTY_TRACE;
    else
      toffset = perFlowRegP.getSigma();

    float dcOffset =  ComputeDcOffsetForCompressedTrace(fgTmp,1,frameNumber,time_start, perFlowRegP.getTMidNuc()-toffset, nfrms );

    for(int i=0; i < nfrms ;i++){
      fgptr[frameStride*i] =  ((i<compFrm)?(fgTmp[i]):(lastVal)) - dcOffset;
    }
#else
    for(int i=0; i < nfrms ;i++){
      fgptr[frameStride*i] =  (i<compFrm)?(fgTmp[i]):(lastVal);
    }
#endif
  }

  return ConstFrmP.getUncompFrames();

}

__device__ inline
int LoadImgWOffset_WithRegionalSampleExtraction(
    FG_BUFFER_TYPE * fgptr, //now points to frame 0 of current bead in image
    short * image,
    const float * frameNumber,
    const int * framesPerPoint,
    int nfrms,
    int frameStride,
    float t0Shift,
    bool isEmpty,
    float * emptyTraceSum,
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    size_t idx,
    //inputs
    const float * BeadParamCube,
    const unsigned short * BeadStateMask,
    //meta data
    const int * SampleRowPtr,
    int * SampleRowCounter,
    //outputs
    unsigned short * SampleStateMask,
    short * SampleCompressedTraces, //SamplePlaneStride  = NUM_SAMPLES_RF * ImgRegP.getNumRegions()
    float * SampleParamCube,  //SamplePlaneStride  = NUM_SAMPLES_RF * ImgRegP.getNumRegions()
    SampleCoordPair * SampleCoord,
    SampleCoordPair myLocation
)
{

  int my_frame = 0,compFrm,curFrms,curCompFrms;

  float tmpAdder;
  float dcOffset = 0;
  //int interf,lastInterf=-1;
  FG_BUFFER_TYPE fgTmp[MAX_COMPRESSED_FRAMES_GPU];
  float traceTmp[MAX_UNCOMPRESSED_FRAMES_GPU+4];
  FG_BUFFER_TYPE lastVal;

  GetUncompressTrace(traceTmp+4, ConstFrmP,  image, frameStride );

  float * emptyAccumPtr = NULL;
  if(isEmpty) emptyAccumPtr = emptyTraceSum;
  //accumulate empty traces in this function so we do not have to touch all the frames again
  ShiftUncompressedTraceAndAccum(traceTmp,traceTmp+4,-t0Shift,ConstFrmP.getUncompFrames(),emptyAccumPtr);

#if UNCOMPRESSED_REZERO

  float tstart = constRegP->getTimeStart();
  float tend;
  if(isEmpty)
    tend = constRegP->getT0Frame() - MAGIC_OFFSET_FOR_EMPTY_TRACE;
  else
    tend = perFlowRegP->getTMidNuc()- perFlowRegP->getSigma();



  dcOffset = ComputeDcOffsetForUncompressedTrace(traceTmp,ConstFrmP.getUncompFrames(),tstart, tend);
#endif
  ;
  my_frame = 0;//ConstFrmP.interpolatedFrames[startFrame]-1;
  compFrm = 0;
  tmpAdder=0.0f;
  curFrms=0;
  curCompFrms=framesPerPoint[compFrm];


  while ((my_frame < ConstFrmP.getUncompFrames()) && ((compFrm < nfrms)))
  {

    tmpAdder += traceTmp[my_frame];

    if(++curFrms >= curCompFrms)
    {
      tmpAdder /= curCompFrms;

#if UNCOMPRESSED_REZERO
      tmpAdder -= dcOffset;
#endif

      lastVal = (FG_BUFFER_TYPE)(tmpAdder); //Maybe use rintf or round to get more precision
      fgTmp[compFrm] = lastVal;
      compFrm++;
      curCompFrms = framesPerPoint[compFrm];
      curFrms=0;
      tmpAdder= 0.0f;
    }
    my_frame++;
  }

  if(compFrm > 0 )
  {
#if !UNCOMPRESSED_REZERO
    //this will produce garbage or crash if frameNumber is not passed to kernel
    float tend;
    if(isEmpty)
      tend = constRegP->getT0Frame() - MAGIC_OFFSET_FOR_EMPTY_TRACE;
    else
      tend = perFlowRegP->getTMidNuc()- perFlowRegP->getSigma();


    dcOffset =  ComputeDcOffsetForCompressedTrace(fgTmp,1,frameNumber,constRegP->getTimeStart(), tend, nfrms );

    //for(int i=0; i < nfrms ;i++){
    //fgptr[frameStride*i] =  ((i<compFrm)?(fgTmp[i]):(lastVal)) - dcOffset;
    //}
#endif


    int writeOffset = 0;
    int SamplePlaneStride = NUM_SAMPLES_RF * ImgRegP.getNumRegions();

    bool isSample = false;
    if(Match(BeadStateMask,(BkgModelMaskType)(BkgMaskRegionalSampled|BkgMaskHighQaulity), true)){
      isSample = true;
      //determine offset for current bead in sample set
      writeOffset = (*SampleRowPtr) + atomicAdd(SampleRowCounter, 1);
      SampleStateMask += writeOffset;
      SampleCompressedTraces += writeOffset;
      SampleParamCube += writeOffset;
      SampleCoord += writeOffset;
      //  printf("tstart %f, tend %f, dcoffset %f, numframes %d, nframes %f,%f,%f,%f,%f,%f \n", constRegP->getTimeStart(), tend, dcOffset, nfrms,
      //      frameNumber[0],frameNumber[1],frameNumber[2],frameNumber[3],frameNumber[4],frameNumber[5]);
    }

    for(int i=0; i < nfrms ;i++){
#if !UNCOMPRESSED_REZERO
      float val = ((i<compFrm)?(fgTmp[i]):(lastVal)) - dcOffset;
#else
      float val = (i<compFrm)?(fgTmp[i]):(lastVal);
#endif
      fgptr[frameStride*i] =  val;
      if(isSample) SampleCompressedTraces[SamplePlaneStride*i] = val; //copy current bead information to sample set
    }

    if(isSample){ //copy current bead information to sample set
      *SampleCoord = myLocation;
      *SampleStateMask = *BeadStateMask;
      for(int i = 0; i < Bp_NUM_PARAMS; i++)
        SampleParamCube[SamplePlaneStride*i] = BeadParamCube[frameStride*i];
    }

  }

  return ConstFrmP.getUncompFrames();

}



__device__ inline
int LoadImgWOffset_WithRegionalSampleExtractionInOrder(
    FG_BUFFER_TYPE * fgptr, //now points to frame 0 of current bead in image
    short * image,
    const float * frameNumber,
    const int * framesPerPoint,
    int nfrms,
    int frameStride,
    float t0Shift,
    bool isEmpty,
    bool isSample,
    float * emptyTraceSum,
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    size_t idx,
    //inputs
    const float * BeadParamCube,
    const unsigned short * BeadStateMask,
    //meta data
    //const int * SampleRowPtr,
    //outputs
    unsigned short * SampleStateMask,
    short * SampleCompressedTraces, //SamplePlaneStride  = NUM_SAMPLES_RF * ImgRegP.getNumRegions()
    float * SampleParamCube,  //SamplePlaneStride  = NUM_SAMPLES_RF * ImgRegP.getNumRegions()
    SampleCoordPair * SampleCoord,
    SampleCoordPair myLocation
)
{

  int my_frame = 0,compFrm,curFrms,curCompFrms;

  float tmpAdder;
  float dcOffset = 0;
  //int interf,lastInterf=-1;
  FG_BUFFER_TYPE fgTmp[MAX_COMPRESSED_FRAMES_GPU];
  float traceTmp[MAX_UNCOMPRESSED_FRAMES_GPU+4];
  FG_BUFFER_TYPE lastVal;

  GetUncompressTrace(traceTmp+4, ConstFrmP,  image, frameStride );

  float * emptyAccumPtr = NULL;
  if(isEmpty) emptyAccumPtr = emptyTraceSum;
  //accumulate empty traces in this function so we do not have to touch all the frames again
  ShiftUncompressedTraceAndAccum(traceTmp,traceTmp+4,-t0Shift,ConstFrmP.getUncompFrames(),emptyAccumPtr);

#if UNCOMPRESSED_REZERO

  float tstart = constRegP->getTimeStart();
  float tend;
  if(isEmpty)
    tend = constRegP->getT0Frame() - MAGIC_OFFSET_FOR_EMPTY_TRACE;
  else
    tend = perFlowRegP->getTMidNuc()- perFlowRegP->getSigma();



  dcOffset = ComputeDcOffsetForUncompressedTrace(traceTmp,ConstFrmP.getUncompFrames(),tstart, tend);
#endif
  ;
  my_frame = 0;//ConstFrmP.interpolatedFrames[startFrame]-1;
  compFrm = 0;
  tmpAdder=0.0f;
  curFrms=0;
  curCompFrms=framesPerPoint[compFrm];


  while ((my_frame < ConstFrmP.getUncompFrames()) && ((compFrm < nfrms)))
  {

    tmpAdder += traceTmp[my_frame];

    if(++curFrms >= curCompFrms)
    {
      tmpAdder /= curCompFrms;

#if UNCOMPRESSED_REZERO
      tmpAdder -= dcOffset;
#endif

      lastVal = (FG_BUFFER_TYPE)(tmpAdder); //Maybe use rintf or round to get more precision
      fgTmp[compFrm] = lastVal;
      compFrm++;
      curCompFrms = framesPerPoint[compFrm];
      curFrms=0;
      tmpAdder= 0.0f;
    }
    my_frame++;
  }

  if(compFrm > 0 )
  {
#if !UNCOMPRESSED_REZERO
    //this will produce garbage or crash if frameNumber is not passed to kernel
    float tend;
    if(isEmpty)
      tend = constRegP->getT0Frame() - MAGIC_OFFSET_FOR_EMPTY_TRACE;
    else
      tend = perFlowRegP->getTMidNuc()- perFlowRegP->getSigma();


    dcOffset =  ComputeDcOffsetForCompressedTrace(fgTmp,1,frameNumber,constRegP->getTimeStart(), tend, nfrms );

    //for(int i=0; i < nfrms ;i++){
    //fgptr[frameStride*i] =  ((i<compFrm)?(fgTmp[i]):(lastVal)) - dcOffset;
    //}
#endif

    int SamplePlaneStride = NUM_SAMPLES_RF * ImgRegP.getNumRegions();

    for(int i=0; i < nfrms ;i++){
#if !UNCOMPRESSED_REZERO
      float val = ((i<compFrm)?(fgTmp[i]):(lastVal)) - dcOffset;
#else
      float val = (i<compFrm)?(fgTmp[i]):(lastVal);
#endif
      fgptr[frameStride*i] =  val;
      if(isSample) SampleCompressedTraces[SamplePlaneStride*i] = val; //copy current bead information to sample set
    }

    if(isSample){ //copy current bead information to sample set
      *SampleCoord = myLocation;
      *SampleStateMask = *BeadStateMask;
      for(int i = 0; i < Bp_NUM_PARAMS; i++)
        SampleParamCube[SamplePlaneStride*i] = BeadParamCube[frameStride*i];
    }

  }

  return ConstFrmP.getUncompFrames();

}

__device__ inline
int LoadImgWOffset_OnTheFlyCompressionWithRegionalSampleExtractionInOrder(
    FG_BUFFER_TYPE * fgptr, //now points to frame 0 of current bead in image
    short * image,
    const float * frameNumber,
    const int * framesPerPoint,
    int nfrms,
    int frameStride,
    float t0Shift,
    bool isEmpty,
    bool isSample,
    float * emptyTraceSum,
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    size_t idx,
    //inputs
    const float * BeadParamCube,
    const unsigned short * BeadStateMask,
    //meta data
    //const int * SampleRowPtr,
    //outputs
    unsigned short * SampleStateMask,
    short * SampleCompressedTraces, //SamplePlaneStride  = NUM_SAMPLES_RF * ImgRegP.getNumRegions()
    float * SampleParamCube,  //SamplePlaneStride  = NUM_SAMPLES_RF * ImgRegP.getNumRegions()
    SampleCoordPair * SampleCoord,
    SampleCoordPair myLocation
)
{


  int t0ShiftWhole;
  float t0ShiftFrac;
  int my_frame = 0,compFrm,curFrms,curCompFrms;
  double prev;
  double next;
  double tmpAdder;
  double mult;

  int interf,lastInterf=-1;

  //allow for negative t0Shift (faster traces)
  if(t0Shift < 0-(ConstFrmP.getUncompFrames()-2))
    t0Shift = 0-(ConstFrmP.getUncompFrames()-2);
  if(t0Shift > (ConstFrmP.getUncompFrames()-2))
    t0Shift = (ConstFrmP.getUncompFrames()-2);

  //by using floor() instead of (int) here
  //we now can allow for negative t0Shifts

  t0ShiftWhole=floor(t0Shift);
  t0ShiftFrac = t0Shift - (float)t0ShiftWhole;

  // skip t0ShiftWhole input frames,
  // if T0Shift whole < 0 start at frame 0;
  int StartAtFrame = (t0ShiftWhole < 0)?(0):(t0ShiftWhole);

  float dcOffset = 0;
  //int interf,lastInterf=-1;
  FG_BUFFER_TYPE fgTmp[MAX_COMPRESSED_FRAMES_GPU];
  float traceTmp[MAX_UNCOMPRESSED_FRAMES_GPU+4];
  FG_BUFFER_TYPE lastVal;



  float * emptyAccumPtr = NULL;
  if(isEmpty){
    emptyAccumPtr = emptyTraceSum;
    GetUncompressTrace(traceTmp+4, ConstFrmP,  image, frameStride );
    ShiftUncompressedTraceAndAccum(traceTmp,traceTmp+4,-t0Shift,ConstFrmP.getUncompFrames(),emptyAccumPtr);
  }

#if UNCOMPRESSED_REZERO

  float tstart = constRegP->getTimeStart();
  float tend;
  if(isEmpty)
    tend = constRegP->getT0Frame() - MAGIC_OFFSET_FOR_EMPTY_TRACE;
  else
    tend = perFlowRegP->getTMidNuc()- perFlowRegP->getSigma();



  dcOffset = ComputeDcOffsetForUncompressedTrace(traceTmp,ConstFrmP.getUncompFrames(),tstart, tend);
#endif
  ;
  my_frame = ConstFrmP.interpolatedFrames[StartAtFrame]-1;
  compFrm = 0;
  tmpAdder=0.0f;
  curFrms=0;
  curCompFrms=framesPerPoint[compFrm];

  interf= ConstFrmP.interpolatedFrames[my_frame];
  next = image[frameStride*interf];

  while ((my_frame < ConstFrmP.getUncompFrames()) && (compFrm < nfrms))
  {
    interf= ConstFrmP.interpolatedFrames[my_frame];

    if(interf != lastInterf) //always true
    {
      prev = next;
      next = image[frameStride*interf];
    }

    // interpolate
    mult = ConstFrmP.interpolatedMult[my_frame] - (t0ShiftFrac/ConstFrmP.interpolatedDiv[my_frame]);

    tmpAdder += ( (prev)-(next) ) * (mult) + (next);

    if(++curFrms >= curCompFrms)
    {
      tmpAdder /= curCompFrms;

      fgTmp[compFrm] = (FG_BUFFER_TYPE)(tmpAdder);
      compFrm++;

      if(compFrm < ConstFrmP.getMaxCompFrames())
        curCompFrms = framesPerPoint[compFrm];
     // if(idx == 0) printf( "my_frame %d  compFrm  %d curCompFrms %d \n", my_frame,compFrm,curCompFrms);
      curFrms=0;

      tmpAdder = 0.0f;
    }

    //reuse my_frame while not compensated for negative t0 shifts
    //T0ShiftWhole will be < 0 for negative t0
    if(t0ShiftWhole < 0)
      t0ShiftWhole++;
    else
      my_frame++;

  }//while

  if(compFrm > 0 )
  {
#if !UNCOMPRESSED_REZERO
    //this will produce garbage or crash if frameNumber is not passed to kernel
    float tend;
    if(isEmpty)
      tend = constRegP->getT0Frame() - MAGIC_OFFSET_FOR_EMPTY_TRACE;
    else
      tend = perFlowRegP->getTMidNuc()- perFlowRegP->getSigma();


    dcOffset =  ComputeDcOffsetForCompressedTrace(fgTmp,1,frameNumber,constRegP->getTimeStart(), tend, nfrms );
//    BeadParamCube[frameStride*BpDebugDCOffset] = dcOffset; //ToDo: remove DEBUG only
    //for(int i=0; i < nfrms ;i++){
    //fgptr[frameStride*i] =  ((i<compFrm)?(fgTmp[i]):(lastVal)) - dcOffset;
    //}
#endif

    int SamplePlaneStride = NUM_SAMPLES_RF * ImgRegP.getNumRegions();

    for(int i=0; i < nfrms ;i++){
#if !UNCOMPRESSED_REZERO
      float val = ((i<compFrm)?(fgTmp[i]):(lastVal)) - dcOffset;
#else
      float val = (i<compFrm)?(fgTmp[i]):(lastVal);
#endif
      fgptr[frameStride*i] =  val;
      if(isSample) SampleCompressedTraces[SamplePlaneStride*i] = val; //copy current bead information to sample set
    }

    if(isSample){ //copy current bead information to sample set
      *SampleCoord = myLocation;
      *SampleStateMask = *BeadStateMask;
      for(int i = 0; i < Bp_NUM_PARAMS; i++)
        SampleParamCube[SamplePlaneStride*i] = BeadParamCube[frameStride*i];
    }

  }

  return ConstFrmP.getUncompFrames();

}


//Uncompresses the VFC image, shifts it by t0 and re-compresses according to the
//the frames per point passed in compFrms
//works on one single well in the raw.image at position l_coord
//all other passed pointers and values are specific to the well
__device__ inline
void AverageRezeroTshiftAndCompressEmptyLocal(
    float * emptyTraceAvgGlobal,
    float * emptyAvgLocal,
    int count,
    const int * framesPerPoint,
    int nfrms,
    int usedUncomFrames,
    const float time_start,
    const PerFlowParamsRegion * perFlowRegP

)
{

  float tmpAdder = 0.0f;
  float lastVal = 0.0f;
  int my_frame = 0;
  int compFrm = 0;

  int curFrms=0;
  int curCompFrms=framesPerPoint[compFrm];

  float * compPtr = emptyTraceAvgGlobal;


  if(count != 0){

    if(count > 1){
      for(int f = 0; f < usedUncomFrames ; f++){
        emptyAvgLocal[f] /= count;
      }
    }
    // printf( "\n");

    float dcOffset = ComputeDcOffsetForUncompressedTrace(emptyAvgLocal, ConstFrmP.getUncompFrames() ,time_start, perFlowRegP->getTMidNuc() - MAGIC_OFFSET_FOR_EMPTY_TRACE);


    while ((my_frame < ConstFrmP.getUncompFrames()-1) && ((compFrm < nfrms)))
    {

      float t= my_frame;
      float fn=t- perFlowRegP->getTshift();
      if (fn < 0.0f) fn = 0.0f;
      if (fn > (ConstFrmP.getUncompFrames()-2)) fn = ConstFrmP.getUncompFrames()-2;
      int ifn= (int)fn;
      float frac = fn - ifn;

      //rezero and shift in one step
      tmpAdder += iPc(emptyAvgLocal[ifn], emptyAvgLocal[ifn+1],frac, dcOffset);
      //and then compress
      if(++curFrms >= curCompFrms)
      {
        tmpAdder /= curCompFrms;

        lastVal = tmpAdder;

        compPtr[compFrm] = lastVal;
        compFrm++;
        curCompFrms = framesPerPoint[compFrm];
        curFrms=0;
        tmpAdder= 0.0f;
      }

      my_frame++;
    }
  }

  if(compFrm >= 0 && compFrm < nfrms)
  {
    for(;compFrm < nfrms;compFrm++){
      compPtr[compFrm] = lastVal;
    }
  }
}


__device__ inline
float AverageRezeroEmptyNoCompression(
    float * emptyTraceAvgGlobal,
    float * emptyAvgLocal,
    const PerFlowParamsRegion * perFlowRegP,
    const ConstantParamsRegion * constRegP,
    const int count
)
{

  if(count != 0){

    if(count > 1){
      for(int f = 0; f < ConstFrmP.getUncompFrames() ; f++){
        emptyAvgLocal[f] /= count;
      }
    }
    // printf( "\n");

    float dcOffset = ComputeDcOffsetForUncompressedTrace(emptyAvgLocal, ConstFrmP.getUncompFrames() ,constRegP->getTimeStart(), perFlowRegP->getTMidNuc() - MAGIC_OFFSET_FOR_EMPTY_TRACE);

    for(int f = 0; f < ConstFrmP.getUncompFrames() ; f++){
      emptyTraceAvgGlobal[f] = emptyAvgLocal[f] - dcOffset;
    }
    return dcOffset;
  }else{
    for(int f = 0; f < ConstFrmP.getUncompFrames() ; f++){
      emptyTraceAvgGlobal[f] = 0.0f;
    }
  }
  return 0.0f;
}





// one block per region (e.g. for P1: regionX =6, regionY=6. => 36 regions)
// block width has to be a warp size or a 2^k fraction of it
// need one shared memory value per thread to calculate sum
// kernel creates meta data itself:
// number of life beads per region (and warp/thread block-row)
// t0 average gets calculated on the fly
// t0map not needed since t0map values directly calculated on the fly from t0est
// const symbols:
// ImgRegP constant symbol needs to be initialized for this kernel
__global__
void GenerateT0AvgAndNumLBeads_New(
    unsigned short * RegionMask,
    const unsigned short* bfmask,
    const unsigned short* BeadStateMask,
    const float* t0Est,
    int * SampleRowPtr,
    int * NumSamples,
    int * lBeadsRegion, //numLbeads of whole region
    float * T0avg // T0Avg per REgion //ToDo check if this is really needed of if updating the T0Est would be better
)
{

  //if(blockDim.x != warpsize) return; // block width needs to be warp size
  extern __shared__ int smBase[]; //one element per thread;

  int* sm_base = smBase;
  int* sm = sm_base + blockDim.x * threadIdx.y + threadIdx.x; //map sm to threads in threadblock
  int* sm_sample_base = sm_base + blockDim.x * blockDim.y;
  int* sm_sample = sm + blockDim.x * blockDim.y;
  //region id and base offset
  const size_t regId = blockIdx.y * gridDim.x + blockIdx.x;  // block x & y are a grid of regions

  //set inital coordinates in region to work on
  size_t rx = threadIdx.x;  //region x to work on
  size_t ry = threadIdx.y;  //region row to work on
  size_t idx = ImgRegP.getWellIdx(regId,rx,ry);

  //window size == block width ---> should be == warpsize or warphalf
  const size_t windowSize = blockDim.x; //window size to slide accross row
  const size_t nextWorkRowStride = ImgRegP.getImgW() * blockDim.y;  //stride to get to next row to work on


  //update pointers for first bead to work on
  bfmask += idx;
  BeadStateMask += idx;
  const float * t0EstPtr = t0Est + idx;

  //update region pointers to point to current region
  RegionMask += regId;
  lBeadsRegion += regId; // point to absolute num l beads for region
  T0avg += regId;
  SampleRowPtr += regId * ImgRegP.getRegH();
  NumSamples += regId;
  int * SamplesThisRow = SampleRowPtr + ry;

  //determine dimensions and handle border regions
  const size_t regWidth =  ImgRegP.getRegW(regId);
  const size_t regHeight = ImgRegP.getRegH(regId);

  // if(threadIdx.x == 0 && threadIdx.y == 0) printf ("RegId: %d dim: %dx%d, offsetsPerRow: %d\n", regId, regWidth, regHeight, offsetsPerRow);
  float t0Sum = 0.0f;
  int t0Cnt = 0;
  int sampleCnt = 0;
  *sm = 0;
  *RegionMask = (RegionStateMask)RegionMaskLive;
  // iterate over rows of region
  while(ry < regHeight){

    size_t windowStart = 0;
    const unsigned short* bfmaskRow = bfmask;
    const unsigned short* bsMaskRow = BeadStateMask;
    const float* t0EstRow = t0EstPtr;
    *sm_sample = 0;
    //slide warp/window across row and create sum for of num live beads for each warp
    while(windowStart < regWidth){

      if(rx < regWidth){ //if bead still in reagion set sm according to mask
        *sm += Match(bfmaskRow,(MaskType)MaskLive)?(1):(0);  //add one to sm if bead is live
        if(!Match(bfmaskRow,(MaskType) (MaskPinned | MaskIgnore | MaskExclude))){
          t0Sum += *t0EstRow;  //sum up T0 for all the valid beads this thread visits
          t0Cnt ++; //increase t0 count to calculate average.
        }
        if(Match(bsMaskRow, (BkgModelMaskType)(BkgMaskRegionalSampled|BkgMaskHighQaulity),true )){
          *sm_sample += 1;
          sampleCnt ++;
        }
      }
      //slide window
      rx += windowSize;
      windowStart += windowSize;
      bfmaskRow += windowSize;
      bsMaskRow += windowSize;
      t0EstRow += windowSize;
    } //row done

    WarpSumNoSync(sm_sample);

    if(threadIdx.x==0) *SamplesThisRow = *sm_sample; //store number of samples to global

    //move threads to first bead of next row to work on
    rx = threadIdx.x;
    ry += blockDim.y;
    SamplesThisRow += blockDim.y;
    bfmask += nextWorkRowStride;
    BeadStateMask += nextWorkRowStride;
    t0EstPtr += nextWorkRowStride;
  }//region done

  int numlBeads = ReduceSharedMemory(sm_base, sm);

  //thread 0 write number of live beads to global
  if(threadIdx.x==0 && threadIdx.y ==0){
    *lBeadsRegion = numlBeads;
    if(numlBeads == 0)
      *RegionMask = RegionMaskNoLiveBeads;
  }

  //if no live beads in region -> die
  if(numlBeads == 0)
    return;


  __syncthreads();
  //calculate t0 Sum for region
  float * smf_base = (float*)sm_base;
  float* smf = (float*)sm;

  *smf = t0Sum; //put t0 partial sums into shared
  //reduce partial sums inside each warp to one, sum only correct for threads with Idx.x == 0 in each warp
  t0Sum =  ReduceSharedMemory(smf_base, smf);

  __syncthreads();


  //calculate t0 cnt for region
  *sm = t0Cnt; //put t0 partial sums into shared
  //reduce partial sums inside each warp to one, sum only correct for threads with Idx.x == 0 in each warp
  t0Cnt = ReduceSharedMemory(sm_base, sm);

  unsigned short regMaskValue = *RegionMask;

  float t0avgRegion = t0Sum/t0Cnt;
  // each thread now has correct t0Avg for the region

  if(threadIdx.x == 0 && threadIdx.y == 0){
    //    printf("GPU regId %u t0_sum %f t0_cnt: %d t0 avg: %f \n" , regId, t0Sum, t0Cnt, t0avgRegion);
    *T0avg = t0avgRegion;
    //ToDo: determine what a minimum T0 average should be and what is definitely bogus
    if(t0avgRegion < THRESHOLD_T0_AVERAGE){
      regMaskValue |= RegionMaskT0AverageBelowThreshold;
      if(t0avgRegion == -1)
        regMaskValue |= RegionMaskNoT0Average;
    }
  }

  //calculate t0 cnt for region
  *sm_sample = sampleCnt; //put t0 partial sums into shared
  //reduce partial sums inside each warp to one, sum only correct for threads with Idx.x == 0 in each warp
  sampleCnt = ReduceSharedMemory(sm_sample_base, sm_sample);



  if(threadIdx.x == 0 && threadIdx.y == 0){
    //    printf("GPU regId %u t0_sum %f t0_cnt: %d t0 avg: %f \n" , regId, t0Sum, t0Cnt, t0avgRegion);
    *NumSamples = sampleCnt;
    //ToDo: determine minimum number of samples needed!
    if(sampleCnt < THRESHOLD_NUM_REGION_SAMPLE){

      regMaskValue |= RegionMaskNumRegionSamplesBelowThreshold;
      if(sampleCnt <= 0) regMaskValue |= RegionMaskNoRegionSamples;

    }
    //write updated region mask value back to global
    *RegionMask = regMaskValue;
  }


  //idx = threadIdx.x  + threadIdx.y * blockDim.x;
  //size_t blockStart = 0;
  //size_t blockSize = blockDim.x * blockDim.y;
  //size_t windowSize = blockDim.x*blockDim.y;
  int offset = 0;
  int next = 0;
  __threadfence_block();

  //not efficient at all, but kernel is executed only once and runs in < 1ms on a k20 for a whole block
  if(threadIdx.x == 0 && threadIdx.y == 0){
    for( idx =0; idx < ImgRegP.getRegH(); idx++){

      if(idx<ImgRegP.getRegH(regId))
        next += SampleRowPtr[idx];

      SampleRowPtr[idx] = offset;
      offset = next;

    }
  }

  //Todo let's see what else could be done here...

}








// grid size is grid.x = regionsX = 6;
// grid.y = imgHeight = 1288/blockDim.y
// block size block.x = warpsize = 32 (or maybe a warp half: 16, needs to be tested)
// block.y = TBD = probably 4 or so,
// one warp (fraction of a warp) works on on row of a region, where the warps slide
// across the row with a stride of blockdim.x*blockdim.y
// const symbols:
// ImgRegP constant symbol needs to be initialized for this kernel
// ConstFrmP constant symbol needs to be initialized for this kernel

//launch bounds:
//K20
//regs per SM: 65536
//blocks for 32 regs: 65536/(32 *128) = 16.25
__global__
__launch_bounds__(128, 16)
void GenerateAllBeadTraceEmptyFromMeta_k (
    unsigned short * RegionMask,
    short * img,  //perwell    input and output
    const unsigned short * bfmask, //per well
    const float* t0Est, //per well
    const float * frameNumberRegion, // from timing compression
    const int * framesPerPointRegion,
    const size_t * nptsRegion,  //per region
    const int * numlBeadsRegion,
    const float * T0avg,  // ToDo: try already subtract T0 after calculating the average so this would not be needed here anymore!
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    //one empty trace per thread block
    float * EmptyTraceAvgRegion, // contains result of emppty trace averaging per region
    float * EmptyTraceSumRegionTBlock, // has to be initialized to 0!! will contain sum of all empty trace frames for each row in a region
    int * EmptyTraceCountRegionTBlock, // has to be initialized to 0!! will contain number of empty traces summed up for each row in a region
    int * EmptyTraceComplete, //has to be initialized to 0!! completion counter per region for final sum ToDo: figure out if we can do without it
    //regional sample
    //inputs
    const float * BeadParamCube,
    const unsigned short * BeadStateMask,
    //meta data
    const int *SampleRowPtr,
    int *  SampleRowCounter,
    //outputs
    unsigned short * SampleStateMask,
    short * SampleCompressedTraces,
    float * SampleParamCube,
    SampleCoordPair * SampleCoord
)
{
  extern __shared__ int smemGenBeadTraces[];  //4 Byte per thread



  //determine shared memory pointers for each thread and the the base pointer for each warp (or warp half)
  int * sm_base = smemGenBeadTraces;
  int * sm_warp_base = sm_base + threadIdx.y*blockDim.x;
  int * sm = sm_warp_base + threadIdx.x;

  float * smf_base = (float*)sm_base;
  //float * smf_warp_base = (float*) sm_warp_base;
  float * smf = (float*) sm;

  //float * emptyTraceSum = (float*)(sm_base + (blockDim.x*blockDim.y));  //Todo see if summing up in for whole block inst4ead of in local per thread would  give a benefit here
  float emptyTraceSum[MAX_UNCOMPRESSED_FRAMES_GPU] = {0};


  //determine region location
  //const size_t regionCol = blockIdx.x;
  //const size_t regionRow = (blockIdx.y*blockDim.y)/ImgRegP.getRegH();
  size_t ix = blockIdx.x * ImgRegP.getRegW() + threadIdx.x; // x coordinate determined through the column index of he region, the region width and the therad idx in x
  const size_t iy = (blockIdx.y*blockDim.y) + threadIdx.y; // y coordinate defined by block y idx


  const size_t regId = ImgRegP.getRegId(ix,iy);  // regionRow*ImgRegP.getGridDimX()+regionCol;

  RegionMask += regId;
  int numLBeads = numlBeadsRegion[regId];
  if(numLBeads == 0 || *RegionMask != RegionMaskLive) return;

  //if(threadIdx.x==0 && blockIdx.y == 0 ) printf("region id: %d \n", regId );


  //image coordinates
  size_t rx = threadIdx.x;
  const size_t ry = iy%ImgRegP.getRegH();

  size_t idx = ImgRegP.getWellIdx(ix,iy);

  //per region values
  int numFrames = nptsRegion[regId];
  //if(threadIdx.x==0 && threadIdx.y == 0 && blockIdx.y == 0 ) printf("num frames: %d, region id: %u\n", numFrames, regId );
  float T0average = T0avg[regId];
  constRegP += regId;
  perFlowRegP += regId;
  EmptyTraceComplete += regId;

  //per region arrays
  if(frameNumberRegion != NULL)
    frameNumberRegion += regId* ConstFrmP.getMaxCompFrames();

  framesPerPointRegion += regId*ConstFrmP.getMaxCompFrames();

  size_t numTBlocksRegion = (ImgRegP.getRegH()+blockDim.y-1)/blockDim.y;
  size_t TBlockId = ry/blockDim.y;

  EmptyTraceAvgRegion +=regId* ConstFrmP.getUncompFrames();

  //offset to first buffer of region
  EmptyTraceCountRegionTBlock += regId * numTBlocksRegion;
  EmptyTraceSumRegionTBlock += regId*numTBlocksRegion * ConstFrmP.getUncompFrames();
  //offset to buffer for actual thread block
  int * EmptyTraceCountTBlock = EmptyTraceCountRegionTBlock + TBlockId;
  float * EmptyTraceSumTBlock = EmptyTraceSumRegionTBlock +  TBlockId * ConstFrmP.getUncompFrames();

  img += idx;
  FG_BUFFER_TYPE * fgPtr = (FG_BUFFER_TYPE *)img;
  bfmask += idx;
  t0Est += idx;

  //get actual region width
  size_t regWidth = ImgRegP.getRegW(regId);

  size_t warpOffset = 0;

  //slide warps accross row
  size_t windowSize = blockDim.x;

#ifdef COLLECT_SAMPLES
  //Sample Offset
  BeadParamCube += idx;
  BeadStateMask += idx;

  //meta data
  SampleRowPtr += regId * (ImgRegP.getRegH()) + ry;
  SampleRowCounter += regId * ImgRegP.getRegH() + ry;
  //outputs: write offset in region buffer determined by SampleRowPtr + atomicAdd(SampleRowCounter,1);
  SampleStateMask += regId * NUM_SAMPLES_RF;
  SampleCompressedTraces += regId * NUM_SAMPLES_RF;
  SampleParamCube += regId * NUM_SAMPLES_RF;
  SampleCoord += regId * NUM_SAMPLES_RF;
#endif
  // if(threadIdx.x == 0 && threadIdx.y == 0 && regId == 57) printf("KERNEL: reg 57, tb %lu, ix %lu, iy %lu, idx %lu, rx %lu, ry %lu\n", TBlockId, ix, iy, idx, rx, ry);

  //printf(" %p\n",(void*) );
  /*if( blockIdx.y == 1){
    if( threadIdx.x == 0 && threadIdx.y == 0){

      printf("RegionMask %p\n",(void*)RegionMask );
      printf("img %p\n",(void*)img );
      printf("bfmask %p\n",(void*)bfmask );
      printf("t0Est %p\n",(void*)t0Est );
      printf("frameNumberRegion %p\n",(void*)frameNumberRegion );
      printf("framesPerPointRegion %p\n",(void*)framesPerPointRegion );
      printf("nptsRegion %p\n",(void*)nptsRegion );
      printf("numlBeadsRegion %p\n",(void*)numlBeadsRegion );
      printf("T0avg %p\n",(void*)T0avg );
      printf("constRegP %p\n",(void*)constRegP );
      printf("perFlowRegP %p\n",(void*)perFlowRegP );
      printf("EmptyTraceAvgRegion %p\n",(void*)EmptyTraceAvgRegion );
      printf("EmptyTraceSumRegionTBlock %p\n",(void*)EmptyTraceSumRegionTBlock );
      printf("EmptyTraceCountRegionTBlock %p\n",(void*) EmptyTraceCountRegionTBlock);
      printf("EmptyTraceComplete %p\n",(void*)EmptyTraceComplete );
      printf("BeadParamCube %p\n",(void*)BeadParamCube );
      printf("BeadStateMask %p\n",(void*)BeadStateMask );
      printf("SampleRowPtr %p\n",(void*) SampleRowPtr);
      printf("SampleRowCounter %p\n",(void*)SampleRowCounter );
      printf("SampleStateMask %p\n",(void*)SampleStateMask );
      printf("SampleCompressedTraces %p\n",(void*) SampleCompressedTraces);
      printf("SampleParamCube %p\n",(void*) SampleParamCube);
      printf("SampleCoord %p\n",(void*)SampleCoord );
    }
  }
*/

  int emptyCnt = 0;
  int usedUncomFrames = 0; //how many of the uncompressed frames are really used for compression?

  if(ImgRegP.isValidCoord(ix,iy)  &&  ry < ImgRegP.getRegH(regId)) // skip work if warp outside of region
  {


#ifdef COLLECT_SAMPLES
#ifdef SAMPLES_IN_ORDER
    int SampleRowBaseOffset =  *SampleRowPtr;
#endif
#endif


    while(warpOffset < regWidth ){

      bool EmptyWell = false;
      int IamAlive = 0;
      *sm=0;

      //load mask for warp
      if(rx < regWidth){
        IamAlive = (Match(bfmask,(MaskType)MaskLive)) ?1:0; //added empty reference wells to local live beads so we will work on them
        EmptyWell = useForEmpty(bfmask);

        // if(EmptyWell && blockIdx.y == 0) printf("%u %u I am Empty!!!\n" ,ix, iy);
      }


      *sm  = IamAlive;
      WarpSumNoSync(sm); //sum up live beads in warp for local t0 average
      int numLBeadsWarp = *sm_warp_base;   //Also contains valid empty reference wells

      //Do Work if IamAlive!
      /////////////////////////////////
      //generate bead traces for live
      //beads in warp
      if(numLBeadsWarp > 0){ //if no live beads shift window right away //if there is no live bead but an empty in the warp the empty will be ignored

        //do t0 shift average over live wells in warp

        //DEBUG: local average only needed to match with vectorized version
        *sm = (IamAlive)?(1):(0);
        WarpSumNoSync(sm,8);  //sum up local t0 for all live beads
        int liveCount8 = *sm;

        float localT0 = (IamAlive)?(*t0Est - T0average):(0);
        *smf = localT0;
        WarpSumNoSync(smf,8);  //sum number of values to create local t0

        if(!EmptyWell) //do not use local t0 average for empty traces
          localT0 = (liveCount8 > 0)?((*smf)/(float)liveCount8):(localT0); //localT0 = (*smf_warp_base)/ (float)numLBeadsWarp; //average for all live beads
        else
          emptyCnt++; //count empties vor empty average,  this was moved up from after kernel execution


#ifndef COLLECT_SAMPLES
        if(IamAlive || EmptyWell){
          usedUncomFrames = LoadImgWOffset(
              fgPtr,
              img,
              frameNumberRegion,
              framesPerPointRegion,
              numFrames,
              ImgRegP.getPlaneStride(),
              localT0,
              EmptyWell,
              emptyTraceSum,
              constRegP->getTimeStart(),
              *perFlowRegP,
              idx
          );
          //moved up
          //if(EmptyWell) emptyCnt++;
        }

#else
#ifndef SAMPLES_IN_ORDER
        if(IamAlive || EmptyWell){
          SampleCoordPair myloc(rx,ry);
          usedUncomFrames = LoadImgWOffset_WithRegionalSampleExtraction(
              fgPtr,
              img,
              frameNumberRegion,
              framesPerPointRegion,
              numFrames,
              ImgRegP.getPlaneStride(),
              localT0,
              EmptyWell,
              emptyTraceSum,
              constRegP,
              perFlowRegP,
              idx,
              //sampling
              BeadParamCube,
              BeadStateMask,

              SampleRowPtr,
              SampleRowCounter,

              SampleStateMask,
              SampleCompressedTraces,
              SampleParamCube,
              SampleCoord,
              myloc
          );
          //moved up
          //if(EmptyWell) emptyCnt++;
        }
#endif
#endif

#ifdef COLLECT_SAMPLES
#ifdef SAMPLES_IN_ORDER

        //*sm = *SampleRowPtr; // init sm with the offset for the first sample in this row.

        int writeOffset = SampleRowBaseOffset;
        //int SamplePlaneStride = NUM_SAMPLES_RF * ImgRegP.getNumRegions();

        //if alive, sample and highquality mark as isSample
        bool isSample =  (IamAlive && Match(BeadStateMask,(BkgModelMaskType)(BkgMaskRegionalSampled|BkgMaskHighQaulity), true));

        *sm = (isSample)?(1):(0);

        int sum = 0;
        int myOffset = 0;
        for(int tid=0; tid < blockDim.x; tid++){
          if (tid == threadIdx.x) myOffset = sum; // running sum before adding local value is write offset for current thread in warp
          sum += sm_warp_base[tid]; //calculate final offset, which ill be added to row offset to generate new base offset for next window.
        }

        SampleRowBaseOffset += sum; //update rowOffset for sliding window
        writeOffset += myOffset; //update write offset for current thread.


        if(IamAlive || EmptyWell){

          SampleCoordPair myloc(rx,ry);
          usedUncomFrames = LoadImgWOffset_OnTheFlyCompressionWithRegionalSampleExtractionInOrder(
              fgPtr,
              img,
              frameNumberRegion,
              framesPerPointRegion,
              numFrames,
              ImgRegP.getPlaneStride(),
              localT0,
              EmptyWell,
              isSample,
              emptyTraceSum,
              constRegP,
              perFlowRegP,
              idx,
              BeadParamCube,
              BeadStateMask,
              SampleStateMask + writeOffset,
              SampleCompressedTraces+ writeOffset,
              SampleParamCube+ writeOffset,
              SampleCoord+ writeOffset,
              myloc
          );



        }
#endif
#endif

      }
      ////////////////////////////////

      //move window along row
      rx += windowSize;
      warpOffset += windowSize;
      idx += windowSize;

      bfmask += windowSize;
      t0Est += windowSize;
      img += windowSize;
      fgPtr += windowSize;


#ifdef COLLECT_SAMPLES
      BeadParamCube += windowSize;
      BeadStateMask += windowSize;
#endif

    }
  }

  //Todo: if too slow try different approach
  //maybe trade some memory to do this more efficient and use a reduce kernel afterwards
  //seperate non atomic stores per block or even per row and then sum them up with a micro kernel

  //not super efficient but needs to be done... empty trace average
  //too many syncs (numFrmaes*2 + 1) and and too many atomics per block (numframes + 2)...
  *sm = emptyCnt;
  //WarpSumNoSync(sm);
  //determine number of empty traces handled by this block
  int emptyInBlockCnt =  ReduceSharedMemory(sm_base, sm);  //*sm_warp_base;
  //for float operations on same shared memory as ints


  if(emptyInBlockCnt > 0){ //only do work if at least on empty reference well was found in warp.
    //iterate over all frames of the empty traces collected by each thread
    for(int f=0; f<usedUncomFrames;f++){  //ConstFrmP.getUncompFrames()
      __syncthreads(); //guarantee that all sm operatins are completed before overwrite
      if(emptyCnt > 0)
        *smf = emptyTraceSum[f];
      else
        *smf = 0.0f;
      //WarpSumNoSync(sm);
      float BlockFrameSum = ReduceSharedMemory(smf_base,smf);  //sum up frame f of all empty trace handled by block

      if(threadIdx.x == 0 && threadIdx.y == 0){
        EmptyTraceSumTBlock[f] = BlockFrameSum; //store trace for threadblock in global memory
      }
    }
    if(threadIdx.x == 0 && threadIdx.y == 0){
      //if(regId == 57) printf("KERNEL: regId 57, tb %lu, emptyCnt: %d\n", TBlockId,emptyInBlockCnt ) ;
      *EmptyTraceCountTBlock =  emptyInBlockCnt; // add count of empty traces to allow for later average calculation
    }
  }
#if EMPTY_AVERAGE_IN_GENTRACE

  __threadfence(); //guarantee global previous writes are visible to all threads
  if(threadIdx.x == 0 && threadIdx.y == 0){
    size_t done = atomicAdd(EmptyTraceComplete, 1); //increment complete counter
    done++; // inc return value to represent current value
    size_t numBlocks = (ImgRegP.getRegH(regId) + blockDim.y -1)/blockDim.y;
    //printf("block Idx %d in region %d done \n", (blockIdx.y/blockDim.y)%(ImgRegP.getRegH()/blockDim.y) , regId);
    if(done == numBlocks){ //if incremented return value == number of Blocks all blocks in region are completed so avg can be calculated
      size_t cnt = 0;
      //EmptyTraceCountTBlock = EmptyTraceCountRegionTBlock + TBlockId;
      //EmptyTraceSumTBlock = EmptyTraceSumRegionTBlock +  TBlockId * ConstFrmP.getUncompFrames();
      for(size_t b = 0; b <numTBlocksRegion; b++){
        cnt += *EmptyTraceCountRegionTBlock;

        for(size_t f=0; f<usedUncomFrames;f++){
          if(b ==0)
            emptyTraceSum[f] = EmptyTraceSumRegionTBlock[f];
          else
            emptyTraceSum[f] += EmptyTraceSumRegionTBlock[f];
        }
        EmptyTraceCountRegionTBlock++;
        EmptyTraceSumRegionTBlock += ConstFrmP.getUncompFrames();
      }
      //printf("************************************** count: %d\n ", cnt);
      //*EmptyTraceComplete = cnt;  // debug only //TODO: remove!!
#if STORE_EMPTY_UNCOMPRESSED
      float dco = AverageRezeroEmptyNoCompression(EmptyTraceAvgRegion,emptyTraceSum, perFlowRegP, constRegP, cnt );
      //if(regId ==27 && threadIdx.x == 0)
      //printf("GenBeadTraces reg 27 dcoffset t_Start: %f t_edn %f\n",constRegP->getTimeStart(),perFlowRegP->getTMidNuc()-MAGIC_OFFSET_FOR_EMPTY_TRACE);
#else
      AverageRezeroTshiftAndCompressEmptyLocal(EmptyTraceAvgRegion,emptyTraceSum, cnt,framesPerPointRegion, numFrames, usedUncomFrames+1, constRegP->getTimeStart(), perFlowRegP);
#endif
    }
  }
#endif
}




//one threadblock per region
//reduce numTBlocksPerReg partial sums and counts to empty average
//each warp will produce a partial sum
//sm layout: numwarps * uncompressed frames + numwaprs integers to sum up count
//reduce empty average
__global__
__launch_bounds__(128, 16)
void ReduceEmptyAverage_k(
    unsigned short * RegionMask,
    float * EmptyTraceAvgRegion,
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    const float * frameNumberRegion, // from timing compression
    const int * framesPerPointRegion,
    const size_t * nptsRegion,  //per region
    const float * EmptyTraceSumRegionTBlock, // has to be initialized to 0!! will contain sum of all empty trace frames for the row in a region
    const int * EmptyTraceCountRegionTBlock, // has to be initialized to 0!! will contain number of empty traces summed up for each warp in a region
    const size_t numTBlocksPerReg
    //float * dcOffset_debug
)
{
  extern __shared__ float smemTracePartialSum[];  // uncompressed frames per warp

  float * smemEmptyTraceAvg = smemTracePartialSum;
  float * smemFrameInTraceWarp = smemEmptyTraceAvg + (threadIdx.y * ConstFrmP.getUncompFrames()); // point to first frame of warp-trace buffer
  int * smemCount =  (int*)(smemEmptyTraceAvg + (blockDim.y * ConstFrmP.getUncompFrames())); // point to trace counter of warp 0

  //same for all warps within block
  size_t regId = blockIdx.x + blockIdx.y * gridDim.x;

  RegionMask += regId;
  if( *RegionMask != RegionMaskLive) return;


  EmptyTraceAvgRegion +=  regId * ConstFrmP.getUncompFrames();
  constRegP += regId;
  perFlowRegP += regId;
  //dcOffset_debug += regId;
  int numFrames = nptsRegion[regId];

  frameNumberRegion += regId* ConstFrmP.getMaxCompFrames();
  framesPerPointRegion += regId*ConstFrmP.getMaxCompFrames();

  //different for each warp in block
  EmptyTraceCountRegionTBlock += regId * numTBlocksPerReg + threadIdx.y;
  EmptyTraceSumRegionTBlock += (regId * numTBlocksPerReg + threadIdx.y) *  ConstFrmP.getUncompFrames();

  //sum up

  int count = 0;

  int psId = threadIdx.y; //partial sum Id

  while(psId < numTBlocksPerReg){  //iterate over partial sums (one per thread-block per region from the previous kernel
    int fx = threadIdx.x;  // frame of partial sum
    count +=  *EmptyTraceCountRegionTBlock;
    while(fx < ConstFrmP.getUncompFrames()){ //sliding window over frames
      if(psId < blockDim.y)
        smemFrameInTraceWarp[fx] = 0;
      //sum up each frame of the traces.
      smemFrameInTraceWarp[fx] +=  EmptyTraceSumRegionTBlock[fx];
      fx += blockDim.x; //slide window
    }
    //move to next partial sum
    psId += blockDim.y;
    EmptyTraceCountRegionTBlock += blockDim.y;
    EmptyTraceSumRegionTBlock += blockDim.y * ConstFrmP.getUncompFrames();
  }
  if(threadIdx.x == 0){
    smemCount[threadIdx.y] = count;  //thread 0 of each warp store trace count to sm
    //printf( "%lu %d %d \n", regId, threadIdx.y, smemCount[threadIdx.y] );

  }
  __syncthreads();
  //each of the blockDim.y warps now has a partial sum of the regional empty traces in shared memory
  //the number of empty traces per sum is stored in smemCount

  int idx = threadIdx.x + blockDim.x * threadIdx.y;
  int blockSize = blockDim.x * blockDim.y;
  count = 0;
  for(int i= 0 ; i<blockDim.y; i++)
    count += smemCount[i];

  while(idx < ConstFrmP.getUncompFrames()){
    float sum = 0;
    for(int i= 0 ; i<blockDim.y; i++){
      sum += smemEmptyTraceAvg[ ConstFrmP.getUncompFrames() * i + idx];
    }
    smemEmptyTraceAvg[idx] = (count > 0)?(sum/count):(0);
    //EmptyTraceAvgRegion[idx] = smemEmptyTraceAvg[idx];
    idx += blockSize;
  }

  __syncthreads();


  if(count < THRESHOLD_NUM_EMPTIES){
    unsigned short regMaskValue = *RegionMask;
    regMaskValue |= RegionMaskNumEmptiesBelowThreshold;
    if(count  <= 0)
      regMaskValue |= RegionMaskNoEmpties;
    if(threadIdx.x == 0 && threadIdx.y == 0)
      *RegionMask = regMaskValue;
  }

  //Calculation uncompressed average empty trace per region completed.
  //AverageRezeroTshiftAndCompressEmptyLocal(EmptyTraceAvgRegion,smemEmptyTraceAvg, count,framesPerPointRegion, numFrames, ConstFrmP.getUncompFrames(), constRegP->getTimeStart(), perFlowRegP);
  //dc offset correction
  float tstart = constRegP->getTimeStart();
  float tend =  constRegP->getT0Frame() - MAGIC_OFFSET_FOR_EMPTY_TRACE;
  float dcOffset = ComputeDcOffsetForUncompressedTrace(smemEmptyTraceAvg, ConstFrmP.getUncompFrames() ,tstart,tend);
  //float dcOffset = 0;
  idx = threadIdx.x + blockDim.x * threadIdx.y;
  //if(idx == 0) *dcOffset_debug = dcOffset;//printf(" regid,%lu,nucId,%d,dcoffset,%f\n", regId, ConstFlowP.getNucId(), dcOffset);
  while(idx < numFrames){
    TShiftAndPseudoCompressionOneFrame ( EmptyTraceAvgRegion , smemEmptyTraceAvg, frameNumberRegion, perFlowRegP->getTshift(), idx, dcOffset);
    idx += blockSize;
  }
}









