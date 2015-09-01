/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * SampleHistory.h
 *
 *  Created on: Jul 10, 2015
 *      Author: Jakob Siegel
 */

#ifndef SAMPLEHISTORY_H_
#define SAMPLEHISTORY_H_


#include "DeviceParamDefines.h"
#include "HostParamDefines.h"
#include "SignalProcessingFitterQueue.h"
#include "GpuPipelineDefines.h"
#include "LayoutCubeRegionsTemplate.h"

#include <vector>

#define MAX_FLOW_HISTORY 20

class SampleCollectionConst
{

protected:

  int numFlows;  //maximum number of flows that can be collected in the history
  int writeBuffer; //index of next buffer to write to (after numFlows are written it also points to the oldest buffer)
  int collectedFlows; //max number of collected FLows == numFLows. if this limit is reached we turn into a fifo queue.

  short* SampleCompressedTracesBuffers[MAX_FLOW_HISTORY];
  int realFlow[MAX_FLOW_HISTORY]; //todo: remove, this is just for debugging


  __host__ __device__ inline
  int wrapAroundIdx(const int idx) const
  {
    return idx%numFlows;
  }

  __host__ __device__ inline
  int calcIdx( int nIndex) const
  {
    //if not yet wrapped around nIndex is actual index
    if(collectedFlows < numFlows) return nIndex;
    //if already wrapped around the writeBuffer is also the oldest buffer
    return wrapAroundIdx(writeBuffer + nIndex);
  }

public:

  // nIndex == 0 will return the oldest ( currentFlow - numFlows) sample buffer.
  __device__ inline
  const short * getTraces(int nIndex) const
  {
    int idx =calcIdx(nIndex);
    return SampleCompressedTracesBuffers[idx];
  }

  __host__ __device__ inline
  const short * getLatestTraces() const
  {
    int idx = writeBuffer-1;
    idx = (idx < 0)?(collectedFlows+idx):(idx);
    return SampleCompressedTracesBuffers[idx];
  }

  __device__ inline
  short * getWriteBuffer()
  {
    return SampleCompressedTracesBuffers[writeBuffer];
  }


  __device__ inline
  int getRealFlowNum(int nIndex) const
  {
    int idx =calcIdx(nIndex);
    return realFlow[idx];
  }


  __host__ __device__  inline
  int getNumSampleFlows() const
  {
    //returns number of flows currently collected in flow history
    return collectedFlows;
  }

  __host__ __device__  inline
  int getMaxNumSampleFlows() const
  {
    //returns max number of flows that can be collected in flow history
    return numFlows;
  }

  __host__ __device__  inline
    void print() const
    {
      const short * address=  getLatestTraces();
      printf("SampleCollection:\n collected Sample Flows: %d\n max Sample Flows: %d\n write Index: %d\n latest Buffer: %p\n",collectedFlows,numFlows,writeBuffer,address);
      for(int i=0; i<numFlows;i++)
        printf(" %2d(%3d):%s%p%s\n", i, realFlow[i], (i<collectedFlows)?("*"):(" "),SampleCompressedTracesBuffers[i],(i==writeBuffer)?("<-"):("  "));
    }


};



class SampleCollection : public SampleCollectionConst
{

  ImgRegParams ImgP;

  std::vector<LayoutCubeWithRegions<short>*> HostSampleCompressedTraces;
  std::vector<LayoutCubeWithRegions<short>*> DeviceSampleCompressedTraces;


protected:

  void writeSampleBeadToFlowBuffers(const FG_BUFFER_TYPE * fgbuffer, const int numFrames);
  size_t extractSamplesOneRegionAllFlows(BkgModelWorkInfo* region_bkinfo, int flowBlockSize);

public:


  SampleCollection(const ImgRegParams & ImgP, int numFlowsInHistory,int maxCompressedFrames);
  ~SampleCollection();

  // to be executed during last flow before switching to GPU flow by flow pipeline.
  void extractSamplesAllRegionsAndFlows(BkgModelWorkInfo* bkinfo, int flowBlockSize=20, int extractNumFlows=0);
  void InitDeviceBuffersAndSymbol(int maxCompressedFrames);

  //updates meta data after Sample collection by kernel!
  int UpdateSampleCollection(int realFlowIndx);
  void RezeroWriteBuffer();

  // only returns correct data after at least one Sampel set was collected on the device and UpdateSampleCollectin was executed there after
  LayoutCubeWithRegions<short> * getLatestSampleFromDevice();

  //returns size in bytes for all the flow buffers combined
  size_t getSize() const { return  (HostSampleCompressedTraces[0]->getSize() * numFlows); }

};


#endif /* SAMPLEHISTORY_H_ */
