/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * BkgGpuPipeline.h
 *
 *  Created on: Jul 7, 2014
 *      Author: jakob
 */

#ifndef BKGGPUPIPELINE_H_
#define BKGGPUPIPELINE_H_

#include "LayoutTranslator.h"
#include "DeviceParamDefines.h"
#include "SignalProcessingFitterQueue.h"
#include "GpuPipelineDefines.h"


class cudaComputeVersion{
  int major;
  int minor;
public:

  cudaComputeVersion(int Major, int Minor){
    major = Major;
    minor = Minor;
  };

//actual comparison
  bool operator== (const cudaComputeVersion &other) const {
    return (major == other.major && minor == other.minor);
  }
  bool operator> (const cudaComputeVersion &other) const {
    if(major > other.major) return true;
    if( major == other.major && minor > other.minor) return true;
    return false;
  }

//combinations of ! and == and >
  bool operator!= (const cudaComputeVersion &other) const {
      return !(*this == other);
    }
  bool operator<= (const cudaComputeVersion &other) const {
    return !(*this > other);
  }

  bool operator< (const cudaComputeVersion &other) const {
    return ( !(*this > other) && ! (*this == other));
  }
  bool operator>= (const cudaComputeVersion &other) const{
    return !(*this < other);
  }

  int getMajor() const { return major;}
  int getMinor() const { return minor;}


};



//collection of all the device buffers, just to make things a little more organized
class DeviceData
{
  MemoryResource _resource;
  vector<size_t> bufferSizes;

public:
  LayoutCubeWithRegions<float> T0; //(ImgP,1, DeviceGlobal);
  LayoutCubeWithRegions<float> RegionFrameCube; //(ImgP.getGridParam(ConstFrmP.getMaxCompFrames()), Rf_NUM_PARAMS, DeviceGlobal);
  LayoutCubeWithRegions<float> BeadParamCube; //(ImgP, Bp_NUM_PARAMS, DeviceGlobal);
  LayoutCubeWithRegions<float> PolyClonalCube; //(ImgP, Bs_NUM_PARAMS ,DeviceGlobal);
  LayoutCubeWithRegions<float> AverageSignalRegion; //  1 per region
  LayoutCubeWithRegions<int> RegionFramesPerPoint; //(ImgP.getGridParam(ConstFrmP.getMaxCompFrames()), 1, DeviceGlobal);
  LayoutCubeWithRegions<size_t> NumFrames; //(ImgP.getGridParam(),1, DeviceGlobal);
  LayoutCubeWithRegions<ConstantParamsRegion> ConstRegP; //(ImgP.getGridParam(),1,DeviceGlobal);
  LayoutCubeWithRegions<PerNucParamsRegion> PerNucRegP; //(ImgP.getGridParam(),NUMNUC,DeviceGlobal);

  //DeviceBuffer updated more than once
  LayoutCubeWithRegions<unsigned short> BeadStateMask; //(ImgP,1, DeviceGlobal);
  LayoutCubeWithRegions<unsigned short> RegionStateMask; //(ImgP,1, DeviceGlobal);

  //DeviceBuffer updated per flow
  LayoutCubeWithRegions<PerFlowParamsRegion> PerFlowRegionParams; //(ImgP.getGridParam(),1,DeviceGlobal);


  LayoutCubeWithRegions<unsigned short> BfMask; //(ImgP,1,DeviceGlobal);
  LayoutCubeWithRegions<short> RawTraces; //(ImgP,ConstFrmP.getRawFrames(), DeviceGlobal);
  LayoutCubeWithRegions<float> EmphasisVec; //(ImgP.getGridParam(ConstFrmP.getMaxCompFrames()*MAX_POISSON_TABLE_COL),1, DeviceGlobal);
  LayoutCubeWithRegions<int> NonZeroEmphasisFrames; //(ImgP.getGridParam(MAX_POISSON_TABLE_COL),Nz_NUM_PARAMS, DeviceGlobal);
  LayoutCubeWithRegions<float> NucRise; //(ImgP.getGridParam(ConstFrmP.getMaxCompFrames()*ISIG_SUB_STEPS_SINGLE_FLOW),1, DeviceGlobal);

  //Buffers that are written to only by the kernels, scratch
  LayoutCubeWithRegions<float> EmptyTraceAvg; //(ImgP.getGridParam(ConstFrmP.getUncompFrames()),1,DeviceGlobal);
  LayoutCubeWithRegions<int> NumLBeads; //(ImgP.getGridParam(),1,DeviceGlobal);
  LayoutCubeWithRegions<float> T0Avg; //(ImgP.getGridParam(),1,DeviceGlobal);
  LayoutCubeWithRegions<int> EmptyTraceComplete; //(ImgP.getGridParam(),1,DeviceGlobal);
  //Sample for regional fitting
  LayoutCubeWithRegions<unsigned short> SampleStateMask; //200 per region
  LayoutCubeWithRegions<short> SampleCompressedTraces; //200 per region
  LayoutCubeWithRegions<float> SampleParamCube; //200 per region
  LayoutCubeWithRegions<SampleCoordPair> SampleCoord; //200 per region
  LayoutCubeWithRegions<int> SampleRowPtr; // maxRegH + 1 * per region
  LayoutCubeWithRegions<int> NumSamples; //  1 per region
  LayoutCubeWithRegions<int> SampleRowCounter; // maxRegH  per region
  //LayoutCubeWithRegions<int> SampleColIdx; // 200 per region
  //Output Buffers
  LayoutCubeWithRegions<float> ResultCube; //(ImgP,Result_NUM_PARAMS,DeviceGlobal);

  DeviceData(const ImgRegParams & ImgP, const ConstantFrameParams & ConstFrmP):
    _resource(),
    bufferSizes(),
    T0(ImgP,1, DeviceGlobal, bufferSizes),
    RegionFrameCube(ImgP.getGridParam(ConstFrmP.getMaxCompFrames()), Rf_NUM_PARAMS, DeviceGlobal, bufferSizes),
    BeadParamCube(ImgP, Bp_NUM_PARAMS, DeviceGlobal, bufferSizes),
    PolyClonalCube(ImgP, Poly_NUM_PARAMS ,DeviceGlobal, bufferSizes),
    AverageSignalRegion(ImgP.getGridParam(),1,DeviceGlobal, bufferSizes),
    RegionFramesPerPoint(ImgP.getGridParam(ConstFrmP.getMaxCompFrames()), 1, DeviceGlobal, bufferSizes),
    NumFrames(ImgP.getGridParam(),1, DeviceGlobal, bufferSizes),
    ConstRegP(ImgP.getGridParam(),1,DeviceGlobal, bufferSizes),
    PerNucRegP(ImgP.getGridParam(),NUMNUC,DeviceGlobal, bufferSizes),
    //DeviceBuffer updated more than once
    BeadStateMask(ImgP,1, DeviceGlobal, bufferSizes),
    RegionStateMask(ImgP.getGridParam(),1,DeviceGlobal, bufferSizes),
    //DeviceBuffer updated per flow
    PerFlowRegionParams(ImgP.getGridParam(),1,DeviceGlobal, bufferSizes),
    BfMask(ImgP,1,DeviceGlobal, bufferSizes),
    RawTraces(ImgP,ConstFrmP.getImageAllocFrames(), DeviceGlobal, bufferSizes),
    EmphasisVec(ImgP.getGridParam(ConstFrmP.getMaxCompFrames()*MAX_POISSON_TABLE_COL),1, DeviceGlobal, bufferSizes),
    NonZeroEmphasisFrames(ImgP.getGridParam(MAX_POISSON_TABLE_COL),Nz_NUM_PARAMS, DeviceGlobal, bufferSizes),
    NucRise(ImgP.getGridParam(ConstFrmP.getMaxCompFrames()*ISIG_SUB_STEPS_SINGLE_FLOW),1, DeviceGlobal, bufferSizes),
    //Buffers that are written to only by the kernels, scratch
    EmptyTraceAvg(ImgP.getGridParam(ConstFrmP.getUncompFrames()),1,DeviceGlobal, bufferSizes),
    NumLBeads(ImgP.getGridParam(),1,DeviceGlobal, bufferSizes),
    T0Avg(ImgP.getGridParam(),1,DeviceGlobal, bufferSizes),
    EmptyTraceComplete(ImgP.getGridParam(),1,DeviceGlobal, bufferSizes),
    //Sample for regional fitting
    SampleStateMask(ImgP.getGridParam(NUM_SAMPLES_RF),1, DeviceGlobal, bufferSizes),
    SampleCompressedTraces(ImgP.getGridParam(NUM_SAMPLES_RF),ConstFrmP.getMaxCompFrames(), DeviceGlobal, bufferSizes), //200 per region
    SampleParamCube(ImgP.getGridParam(NUM_SAMPLES_RF), Bp_NUM_PARAMS, DeviceGlobal, bufferSizes), //200 per region
    SampleCoord(ImgP.getGridParam(NUM_SAMPLES_RF), 1, DeviceGlobal, bufferSizes),
    SampleRowPtr(ImgP.getGridParam(ImgP.getRegH()),1,DeviceGlobal, bufferSizes),
    NumSamples(ImgP.getGridParam(),1,DeviceGlobal, bufferSizes),
    SampleRowCounter(ImgP.getGridParam(ImgP.getRegH()),1,DeviceGlobal, bufferSizes),
    //SampleColIdx(ImgP.getGridParam(NUM_SAMPLES_RF),1, DeviceGlobal), // 200 per region
    //Output Buffers
    ResultCube(ImgP,Result_NUM_PARAMS,DeviceGlobal, bufferSizes)
  {

    size_t accumBytes = 0;
    for (std::vector<size_t>::iterator it = bufferSizes.begin() ; it != bufferSizes.end(); ++it)
      accumBytes += *it;

    cout << "CUDA: Device Memory allocated: " << accumBytes / (1024.0* 1024.0) << " MB " << endl;

  }


};



class SpecialDeviceData
{
  vector<size_t> bufferSizes;
public:
  //Special Buffers for this kernel dependent on launch configuration
  LayoutCubeWithRegions<float>EmptyTraceSumRegionTBlock;
  LayoutCubeWithRegions<int>EmptyTraceCountRegionTBlock;

  SpecialDeviceData(const ImgRegParams & ImgP, const ConstantFrameParams & ConstFrmP, size_t numTBlocksPerReg):
    bufferSizes(),
    EmptyTraceSumRegionTBlock( ImgP.getGridParam(numTBlocksPerReg*ConstFrmP.getUncompFrames()), 1,DeviceGlobal, bufferSizes),
    EmptyTraceCountRegionTBlock( ImgP.getGridParam(numTBlocksPerReg),1,DeviceGlobal, bufferSizes)
  {
    size_t accumBytes = 0;
    for (std::vector<size_t>::iterator it = bufferSizes.begin() ; it != bufferSizes.end(); ++it)
      accumBytes += *it;

    cout << "CUDA: Special Device Buffers Memory allocated: " << accumBytes / (1024.0* 1024.0) << " MB " << endl;

  }

};


class HostData
{
  vector<size_t> bufferSizes;
public:
  //Host Buffer

  //wrapper for actual host buffer
  LayoutCubeWithRegions<unsigned short> BfMask;
  LayoutCubeWithRegions<short> RawTraces;

  //actually allocated buffers
  LayoutCubeWithRegions<unsigned short> BeadStateMask;
  LayoutCubeWithRegions<float> ResultCube;
  LayoutCubeWithRegions<PerFlowParamsRegion> PerFlowRegionParams;
  //LayoutCubeWithRegions<PerFlowParamsRegion> NewPerFlowRegionParams;
  LayoutCubeWithRegions<float> EmphasisVec;
  LayoutCubeWithRegions<float> NucRise;
  LayoutCubeWithRegions<size_t> NumFrames;
  LayoutCubeWithRegions<ConstantParamsRegion> ConstRegP;
  // Regional Fit Read in From File
  //CubePerFlowDump<reg_params> RegionDump;

  HostData(const ImgRegParams & ImgP, const ConstantFrameParams & ConstFrmP):
    bufferSizes(),
    BfMask(NULL,ImgP,1,HostMem),
    RawTraces(NULL,ImgP,ConstFrmP.getRawFrames(), HostMem),

    BeadStateMask(ImgP,1, HostMem, bufferSizes),
    ResultCube(ImgP,Result_NUM_PARAMS,HostMem, bufferSizes),
    PerFlowRegionParams(ImgP.getGridParam(),1,HostMem, bufferSizes),
    EmphasisVec(ImgP.getGridParam(ConstFrmP.getMaxCompFrames()*MAX_POISSON_TABLE_COL),1, HostMem, bufferSizes),
    NucRise( ImgP.getGridParam(ConstFrmP.getMaxCompFrames()*ISIG_SUB_STEPS_SINGLE_FLOW),1,HostMem, bufferSizes),
    NumFrames(ImgP.getGridParam(),1, HostMem, bufferSizes),
    ConstRegP(ImgP.getGridParam(),1,HostMem, bufferSizes)//,
  // RegionDump(ImgP.getGridDimX(), ImgP.getGridDimY(),1,1,1,1)  // Regional Fit Read in From File
  {
    size_t accumBytes = 0;
    for (std::vector<size_t>::iterator it = bufferSizes.begin() ; it != bufferSizes.end(); ++it)
      accumBytes += *it;

    cout << "CUDA: Host Buffers Memory allocated: " << accumBytes / (1024.0* 1024.0) << " MB " << endl;
  }

};

class GPUResultsBuffer
{
public:
  int flowBuffers;
  int readFlowPos;
  int writeFlowPos;
  LayoutCubeWithRegions<float> PerFlowAmpl;

  GPUResultsBuffer(const ImgRegParams &ImgP, int numBuffers):
    flowBuffers(numBuffers),
    PerFlowAmpl(ImgP, flowBuffers, HostMem)
  {
    readFlowPos = 0;
    writeFlowPos = 0;
  }  
};

class BkgGpuPipeline
{

  // Constant or global Parameters
  ImgRegParams ImgP;
  ConstantParamsGlobal ConstGP;
  ConstantFrameParams ConstFrmP;
  ConfigParams ConfP;
  WellsLevelXTalkParams ConstXTP;

  PerFlowParamsGlobal GpFP;

  DeviceData  * Dev;  //collection of Device buffers
  HostData * Host; // collection of Host buffers
  SpecialDeviceData * SpDev; // Launch Configuration specific Device buffers;

  // collection of GPU results over several flows to be distributed as jobs to CPU queue by a separate thread
  GPUResultsBuffer *resultsHostBuf; 


  int flowBlockSize;
  int startFlowNum;
  BkgModelWorkInfo * bkinfo;

protected:

  size_t checkAvailableDevMem();


public:

  BkgGpuPipeline(BkgModelWorkInfo* bkinfo, int fbSize, int startingFlow, int numFlowBuffers, int deviceId);  //ToDO: add stream and device info
  ~BkgGpuPipeline();

  void PerFlowDataUpdate(BkgModelWorkInfo* bkinfo, int flowInBlock);
  void PrepareInputsForSetupKernel();
  void InitPersistentData();
  void HandleResults();
  void HandleRegionalFittingResults();
  void PrepareForRegionalFitting();
  void PrepareForSingleFlowFit();

  void ExecuteT0AvgNumLBeadKernel();
  void ExecuteGenerateBeadTrace();
  void ExecuteRegionalFitting();
  void ExecuteSingleFlowFit();
  void ExecuteCrudeEmphasisGeneration();
  void ExecuteFineEmphasisGeneration();
  void ExecutePostFitSteps();

  void ApplyClonalFilter();

  //helper
  const ImgRegParams getParams() const { return ImgP; }
  const PerFlowParamsGlobal getFlowP() const { return GpFP; }
  const ConstantFrameParams getFrameP() const { return ConstFrmP; }
  bool firstFlow();

  void InitRegionalParamsAtFirstFlow();
  //////////////////////////////////////////////////
  //TODO: remove when no longer needed (also remove buffers from collections
  //Functions to be used until REgional Fitting available on Device/per FLow
  //hast to be called only once

  void ReadRegionDataFromFileForBlockOf20();
  void UpdateRegionParamsAndCopyPerFlow();
  void CopyNewToOldRegParams();


  //void checkXTalkResults();
  //void checkPolyClonal();

private:
  void getDataForRawWells();
  void getDataForPostFitStepsOnHost();

};


#endif /* BKGGPUPIPELINE_H_ */
