/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * SampleHistory.cpp
 *
 *  Created on: Jul 10, 2015
 *      Author: Jakob Siegel
 */


#include "DeviceSymbolCopy.h"
#include "LayoutTranslator.h"
#include "SignalProcessingFitterQueue.h"
#include "JobWrapper.h"
#include "SampleHistory.h"
#include "HostParamDefines.h"

#include <vector>
#include <assert.h>

using namespace std;

#define SAMPLECOLLECT_DEBUG 1

SampleCollection::SampleCollection(const ImgRegParams & img_region_params, int numFlowsInHistory, int maxCompressedFrames ){
  writeBuffer = 0;
  collectedFlows = 0;
  numFlows = numFlowsInHistory;
  ImgP = img_region_params;
  if(numFlowsInHistory>1)
    cout << "CUDA: SampleCollection: creating Sample Bead History of " << numFlowsInHistory << " flows for Regional Parameter Fitting" <<endl;

  for(int i=0;i<MAX_FLOW_HISTORY; i++){
    SampleCompressedTracesBuffers[i] = NULL;
    realFlow[i] = -1;
    if (i<numFlows) {
      HostSampleCompressedTraces.push_back(new LayoutCubeWithRegions<short>(ImgP.getGridParam(NUM_SAMPLES_RF),maxCompressedFrames, HostMem));
      HostSampleCompressedTraces.back()->memSet(0);
      HostSampleCompressedTraces.back()->setRWStrideZ();
    }
  }
}


SampleCollection::~SampleCollection()
{

  for (vector<LayoutCubeWithRegions<short>*>::iterator it = HostSampleCompressedTraces.begin(); it != HostSampleCompressedTraces.end(); ++it){
    if (*it != NULL) delete *it;
    *it = NULL;
  }
  for (vector<LayoutCubeWithRegions<short>*>::iterator it = DeviceSampleCompressedTraces.begin(); it != DeviceSampleCompressedTraces.end(); ++it){
    if (*it != NULL) delete *it;
    *it = NULL;
  }
}



////////////////////////////////////////////////////////////////////////
// HOST SIDE SAMPLE COLLECTIN FIRST BLOCK OF FLOWS

void SampleCollection::writeSampleBeadToFlowBuffers(const FG_BUFFER_TYPE * fgbuffer, const int numFrames)
{
  //only sample beads get passed into this function
  //for each flow buffer write all frames:
  collectedFlows = 0;
  for (std::vector<LayoutCubeWithRegions<short>*>::iterator it = HostSampleCompressedTraces.begin(); it != HostSampleCompressedTraces.end(); ++it)
  {
    for(int frame=0; frame < numFrames; frame++){
      (*it)->write(*fgbuffer);
      fgbuffer++; //next frame
    }
    collectedFlows++;
    if(collectedFlows == numFlows) break;
  }
}

size_t SampleCollection::extractSamplesOneRegionAllFlows(BkgModelWorkInfo* region_bkinfo, int flowBlockSize)
{

  WorkSet myJob(region_bkinfo);

  size_t regId = ImgP.getRegId(myJob.getRegCol(),myJob.getRegRow());
  int startFlow = flowBlockSize - numFlows;
  int numFrames = myJob.getNumFrames();
  int numLiveBeads = myJob.getNumBeads();
  FG_BUFFER_TYPE * localFgBuffer = myJob.getFgBuffer() + numFrames * startFlow; // move to first frame of first bead of first history flow
  //ToDo: remove after debugging
  for(int i = 0; i < numFlows; i++)
    realFlow[i] = startFlow + i;

  size_t numSamples = 0;
  for(int n=0; n<numLiveBeads; n++){
    // only if sampled and HQ add bead to sample set:
    if(region_bkinfo->bkgObj->region_data->my_beads.sampled[n]){
      if(region_bkinfo->bkgObj->region_data->my_beads.high_quality[n]){
        //set write pointer for all flow buffers to the current region/bead
        for (vector<LayoutCubeWithRegions<short>*>::iterator it = HostSampleCompressedTraces.begin(); it != HostSampleCompressedTraces.end(); ++it)
          (*it)->setRWPtrRegion(regId,numSamples);

        writeSampleBeadToFlowBuffers(localFgBuffer,numFrames);
        numSamples++;
      }
    }

    localFgBuffer += numFrames * flowBlockSize; //next bead
  }

  return numSamples;
}

void SampleCollection::extractSamplesAllRegionsAndFlows(BkgModelWorkInfo* bkinfo, int flowBlockSize, int extractNumFlows){

  if(extractNumFlows <= 0) extractNumFlows=numFlows; //no extract Num Flows provided extract maximum
  assert(extractNumFlows <= flowBlockSize);//we cannot extract more than number of flows in block
  assert(extractNumFlows <= numFlows);//we cannot extract more flows as we have buffers created

  writeBuffer = wrapAroundIdx(extractNumFlows); //next buffer to write to (or oldest buffer) in history
  collectedFlows = extractNumFlows; //number of flows in history collected (only needed if we want to collect a history of > flowBlockSize (20))
#if SAMPLECOLLECT_DEBUG
  LayoutCubeWithRegions<size_t> NumSampleMap(ImgP.getGridParam(),1,HostMem);
#endif

  for(size_t r=0; r < ImgP.getNumRegions(); r++){
    BkgModelWorkInfo* region_bkinfo = (BkgModelWorkInfo*) &bkinfo[r];
#if SAMPLECOLLECT_DEBUG
    WorkSet myJob(region_bkinfo);
    size_t regId = ImgP.getRegId(myJob.getRegCol(),myJob.getRegRow());
    NumSampleMap[regId] = extractSamplesOneRegionAllFlows(region_bkinfo,flowBlockSize);
#else
    extractSamplesOneRegionAllFlows(region_bkinfo,flowBlockSize);
#endif

  }
#if SAMPLECOLLECT_DEBUG
  cout << "DEBUG: GPU Pipeline, collected samples for: " << collectedFlows << " flows for Regional Fitting" <<endl;
  cout << "DEBUG: GPU Pipeline, num Samples per region Host side:" <<endl;
  NumSampleMap.printRegionTable<size_t>();
#endif
}

// END HOST SIDE SAMPLE COLLECTIN FIRST BLOCK OF FLOWS
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// INITIAL DEVICE INIT:
void SampleCollection::InitDeviceBuffersAndSymbol( int maxCompressedFrames)
{
  //allocate and cop host buffers to device:

  for (vector<LayoutCubeWithRegions<short>*>::iterator it = HostSampleCompressedTraces.begin(); it != HostSampleCompressedTraces.end(); ++it)
  {
    DeviceSampleCompressedTraces.push_back(new LayoutCubeWithRegions<short>( *(*it) ,DeviceGlobal));
  }

  while( DeviceSampleCompressedTraces.size() < numFlows ){
    DeviceSampleCompressedTraces.push_back(new LayoutCubeWithRegions<short>(ImgP.getGridParam(NUM_SAMPLES_RF), maxCompressedFrames, DeviceGlobal));
    DeviceSampleCompressedTraces.back()->memSet(0);
  }

  int i = 0;
  for (vector<LayoutCubeWithRegions<short>*>::iterator it = DeviceSampleCompressedTraces.begin(); it != DeviceSampleCompressedTraces.end(); ++it)
    SampleCompressedTracesBuffers[i++] = (*it)->getPtr(); //store device pointer in aray for constant memory symbol

  //update symbol on device
  copySymbolsToDevice((*this));
  cout << "CUDA: SampleCollection: InitDeviceBuffersAndSymbol: created " << DeviceSampleCompressedTraces.size() << " Device Sample Buffers (" << getSize()/(1024.0*1024.0) << "MB), and initialized Device control symbol" <<endl;
  print();
}


//
////////////////////////////////////////////////////////////////////////


int SampleCollection::UpdateSampleCollection(int realFlowIndx)
{
    if(collectedFlows < numFlows) collectedFlows++;

    //todo: remove after debuging
    realFlow[writeBuffer] = realFlowIndx;

    writeBuffer = wrapAroundIdx(writeBuffer + 1);

    //update symbol on device
    copySymbolsToDevice((*this));
    print();
    return collectedFlows;
  }

void SampleCollection::RezeroWriteBuffer()
{
  DeviceSampleCompressedTraces[writeBuffer]->memSet(0);
}



LayoutCubeWithRegions<short> * SampleCollection::getLatestSampleFromDevice()
{
  int latestIndex = calcIdx(writeBuffer -1);
  (*HostSampleCompressedTraces[latestIndex]).copy((*DeviceSampleCompressedTraces[latestIndex]));
  return (HostSampleCompressedTraces[latestIndex]);
}


