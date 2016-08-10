/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
 * 
 *
 *  Created on: Aug 20, 2014
 *  Author: Mohit
 */

#ifndef FITTINGHELPERS_H_
#define FITTINGHELPERS_H_

#include "MathModel/PoissonCdf.h"
#include "Mask.h"
#include "EnumDefines.h"
#include "ImgRegParams.h"
#include "DeviceParamDefines.h"
#include "ConstantSymbolDeclare.h"

__device__
float ApplyDarkMatterToFrame(
  const float* beadParamCube,
  const float* regionFrameCube,
  const float darkness,
  const int frame,
  const int num_frames,
  const int frameStride,
  const int regionFrameStride);


// compute tmid muc. This routine mimics CPU routine in BookKeeping/RegionaParams.cpp
__device__
float ComputeMidNucTime(
  const float tmidNuc,
  const PerFlowParamsRegion * perFlowRegP,
  const PerNucParamsRegion * perNucRegP);


__device__ 
float ComputeETBR(
  const PerNucParamsRegion * perNucRegP,
  const float ratioDrift,
  const float R,
  const float copies,
  const int flow = ConstFlowP.getRealFnum());


__device__
float ComputeTauB( 
  const ConstantParamsRegion * constRegP,
  const float etbR);


__device__
float ComputeSP(
  const float copyDrift,
  const float copies,
  const int flow = ConstFlowP.getRealFnum());

__device__ 
float ComputeSigma(
  const PerFlowParamsRegion *perFlowRegP,
  const PerNucParamsRegion *perNucRegP);

__device__
const float4*  precompute_pois_LUT_params_SingelFLowFit(
  int il, 
  int ir);

__device__
float poiss_cdf_approx_float4_SingelFLowFit(
  float x, 
  const float4* ptr, 
  float occ_l, 
  float occ_r);

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
  int frameStride,
  int emphStride,
  int nucIntLoopSteps);

__device__
void BkgModelRedTraceCalculation(
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
    const int endFrames);

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
);

__global__
void GenerateEmphasis(
  const unsigned short * RegionMask,
  const int numEv,
  const float amult,
  const PerFlowParamsRegion *perFlowRegP,
  const int *framePerPoint,
  const float *RegionFrameCube,
  const size_t *numFramesRegion,
  float *emphasisVec,
  int *nonZeroEmphFrames);

#endif // FITTINGHELPERS_H_
