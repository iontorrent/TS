/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */


#include "ConstantSymbolDeclare.h"
#include "DeviceSymbolCopy.h"
#include "MathModel/PoissonCdf.h"
#include "cuda_error.h"

void copySymbolsToDevice( const ConstantFrameParams & ciP){
  cudaMemcpyToSymbol( ConstFrmP, (void*) &ciP, sizeof(ConstantFrameParams), 0, cudaMemcpyHostToDevice);
}

void copySymbolsToDevice( const ImgRegParamsConst& irP ){
  cudaMemcpyToSymbol( ImgRegP, (void*) &irP, sizeof(ImgRegParamsConst), 0, cudaMemcpyHostToDevice);
}

void copySymbolsToDevice( const ConstantParamsGlobal& pl){
  cudaMemcpyToSymbol( ConstGlobalP, (void*) &pl, sizeof(ConstantParamsGlobal), 0, cudaMemcpyHostToDevice);
}
void copySymbolsToDevice(const PerFlowParamsGlobal& fp ){
  cudaMemcpyToSymbol( ConstFlowP, (void*) &fp, sizeof(PerFlowParamsGlobal), 0, cudaMemcpyHostToDevice);
}
void copySymbolsToDevice(const ConfigParams& cp){
  cudaMemcpyToSymbol( ConfigP, (void*) &cp, sizeof(ConfigParams), 0, cudaMemcpyHostToDevice);
}


void copySymbolsToDevice(const WellsLevelXTalkParamsConst<MAX_WELL_XTALK_SPAN,MAX_WELL_XTALK_SPAN> & cXtP){
  cudaMemcpyToSymbol( ConstXTalkP, (void*) &cXtP, sizeof(WellsLevelXTalkParamsConst<MAX_WELL_XTALK_SPAN,MAX_WELL_XTALK_SPAN>), 0, cudaMemcpyHostToDevice);
}


void copySymbolsToDevice(const XTalkNeighbourStatsConst<MAX_XTALK_NEIGHBOURS> & cTlXtP){
  cudaMemcpyToSymbol( ConstTraceXTalkP, (void*) &cTlXtP, sizeof(XTalkNeighbourStatsConst<MAX_XTALK_NEIGHBOURS>), 0, cudaMemcpyHostToDevice);
}

void copySymbolsToDevice(const HistoryCollectionConst& histConst ){
  cudaMemcpyToSymbol(ConstHistCol, (void*) &histConst, sizeof(HistoryCollectionConst), 0, cudaMemcpyHostToDevice);
}


/*void copySymbolsToDevice(const ConstantRegParamBounds& cp){
  cudaMemcpyToSymbol( ConstBoundRegP, (void*) &cp, sizeof(ConstantRegParamBounds), 0, cudaMemcpyHostToDevice);
}*/


void initPoissonTablesLUT_SingleFlowFit(int device, void ** poissLUT)
{

  cudaSetDevice(device);
  ////////// float4/avx version
  //  float4 ** pPoissLUT = (float4**) poissLUT;

  int poissTableSize =  MAX_LUT_TABLE_COL * MAX_POISSON_TABLE_ROW * sizeof(float4);
  float4 * devPtrLUT = NULL;
  cudaMalloc(&devPtrLUT, poissTableSize); CUDA_ALLOC_CHECK(devPtrLUT);
  cudaMemset(devPtrLUT, 0, poissTableSize); CUDA_ERROR_CHECK();
  cudaMemcpyToSymbol(POISS_APPROX_LUT_CUDA_BASE, &devPtrLUT  , sizeof (float4*)); CUDA_ERROR_CHECK();

  // cast and copy host side __m128 SSE/AVX data to float4
  float4** pPoissLUT =(float4**)poissLUT;
  for(int i = 0; i< MAX_LUT_TABLE_COL; i++)
  {
    cudaMemcpy(devPtrLUT, &pPoissLUT[i][0], sizeof(float4)*MAX_POISSON_TABLE_ROW, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    devPtrLUT += MAX_POISSON_TABLE_ROW;
  }


}


void CreatePoissonApproxOnDevice(int device)
{
  //TODO only works on one device for testing
  PoissonCDFApproxMemo poiss_cache;
  poiss_cache.Allocate (MAX_POISSON_TABLE_COL,MAX_POISSON_TABLE_ROW,POISSON_TABLE_STEP);
  poiss_cache.GenerateValues(); // fill out my table

  initPoissonTablesLUT_SingleFlowFit(device, (void**)poiss_cache.poissLUT);


}
