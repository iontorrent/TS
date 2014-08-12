/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef GENERATEBEADTRACESTREAM_H
#define GENERATEBEADTRACESTREAM_H

#include "Image.h"
#include "TimeCompression.h"
#include "Region.h"
#include "BeadTracker.h"
#include "BkgTrace.h"
#include "StreamManager.h"

//forward deceleration of item type
class BkgModelImgToTraceInfoGPU;


class BlockPersistentData
{
  int _numOffsetsPerRow;
  int _imgW;
  int _imgH;
  int _regsX;
  int _regsY;
  int * _dOffsetWarp; //num beads per row, length == regHeight * (ceil(regMaxWidth/blockDim.x)+1)

  //copied in once
  float * _hT0est;
  const unsigned short * _hBfmask;
  float * _dT0est;
  unsigned short * _dBfmask;

  //created on device
  float * _dT0avgRegion;
  int * _dlBeadsRegion;
  int * _hlBeadsRegion;

  bool _init;

public:

  BlockPersistentData();
  ~BlockPersistentData();

  void Allocate( int numOffsetsPerRow, int imgW, int imgH, int regsX, int regsY);
  void PrepareInputs(BkgModelImgToTraceInfoGPU * info);
  void CopyInputs();
  void CopyOutputs(); // not sure if we actually need this

  //mark as initialized
  void MarkAsCreated()  { _init = true; }
  bool AlreadyCreated(){ return _init; }
  void DebugPrint(int startX=0,int  numX=0, int  startY=0,int  numY=0);



    //persistent block data
  //unsigned short * getHBfMask(){ return _hBfmask;}
  //float * getHT0est(){ return _hT0est;}
  unsigned short * getDBfMask(){ return _dBfmask;}
  float * getDT0est(){ return _dT0est;}

  //meta data
  int * getHLBeadsRegion(){ return _hlBeadsRegion; }
  int * getDOffsetWarp(){ return _dOffsetWarp;} //num beads per row, length == regHeight * (ceil(regMaxWidth/blockDim.x)+1)
  int * getDlBeadsRegion(){ return _dlBeadsRegion;}
  float * getDT0avgRegion() {return _dT0avgRegion;}


  int getNumOffsetsPerRow(){ return _numOffsetsPerRow; }
  int getNumRegions(){ return _regsX*_regsY; }
  int getImgSize(){ return _imgW*_imgH; }
  int getNumOffsets(){ return _imgH*_numOffsetsPerRow * getNumRegions(); }

};


class GenerateBeadTraceStream : public cudaSimpleStreamExecutionUnit
{
  //Execution config
  //static int _bpb;   // beads per block
  //static int _l1type; // 0:sm=l1, 1:sm>l1, 2:sm<l1


  int _imgWidth ;
  int _imgHeight;
  int _regMaxWidth;
  int _regMaxHeight;
  int _regionsX;
  int _regionsY;

  int _threadBlockX; // can NOT be larger than warp size (32) to guarantee sync free calculations
  int _threadBlockY;

  //Only members starting with _pHost or _pDev are
  //allocated. pointers with _ph_ or _pd_ are just pointers
  //referencing memory inside the _pDev/_pHost buffers!!!

  BkgModelImgToTraceInfoGPU * _info;

  //complex array copy
  //host memory
  FG_BUFFER_TYPE* _hFgBufferRegion_Base;
  FG_BUFFER_TYPE* _dFgBufferRegion_Base;  // host array to build array of device pointers
  FG_BUFFER_TYPE** _hFgBufferRegion;  //device array
  FG_BUFFER_TYPE** _dFgBufferRegion;  //device array


  int * _hFramesPerPointRegion_Base; // host buffer to create data for copy in

  // device memory
  int * _dFramesPerPointRegion_Base;
  int** _hFramesPerPointRegion; // host array to build array of device pointers
  int** _dFramesPerPointRegion;


  //simple copy
  //host memory
  int* _hNptsRegion;
  // device memory
  int* _dNptsRegion;

  RawImage _draw;

    //kinda hackish
  //keep Block PersistentData for each device! If device not yet in map create Persistent Data
  static map<int, BlockPersistentData> _DevicePersistent;

  BlockPersistentData * _BlockPersistent;


public:

  GenerateBeadTraceStream(streamResources * res, WorkerInfoQueueItem item);
  ~GenerateBeadTraceStream();


  // Implementation of stream execution methods
  bool InitJob();
  void ExecuteJob(); // execute async functions
  int handleResults(); // non async results handling
  void printStatus();

  static size_t getMaxHostMem();
  static size_t getMaxDeviceMem();



private:
  //helper
  bool ValidJob();
  void cleanUp();

  //allocators and destroyers
  void PreProcessAndAllocatePerRegionBuffers();
  void PostProcessAndDestroyPerRegionBuffers();

  void AllocatePersistentData();
  static void DestroyPersistentData(); //not called yet since it has tbd when to call to erase persistent data safely

  void AllocRawDevice();
  void CopyRawDevice();
  void DestroyRawDevice();

  // execution order
  void prepareInputs();  // not async stuff
  void copyToDevice();
  void executeKernel();
  void copyToHost();

  //kernel wrapper
  void ExecuteCreatePersistentMetaDataKernel();
  void ExecuteGenerateBeadTraceKernel();
};



#endif // GENERATEBEADTRACESTREAM_H
