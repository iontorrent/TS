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

#define SAMPLECOLLECT_DEBUG 2

HistoryCollection::HistoryCollection(const ImgRegParams & img_region_params, const int numFlowsInHistory,const int maxCompressedFrames,const int uncompressedFrames ){
  writeBuffer = 0;
  collectedFlows = 0;
  numFlows = numFlowsInHistory;
  latestSampleFlowOnHost = 0;
  latestEmptyFlowOnHost = 0;
  latestRegParamOnHost = 0;
  ImgP = img_region_params;
  if(numFlowsInHistory>1)
    cout << "CUDA: HistoryCollection: creating Sample Bead History of " << numFlowsInHistory << " flows for Regional Parameter Fitting" <<endl;



  for(int i=0;i<MAX_FLOW_HISTORY; i++){
    ppSampleCompressedTracesBuffers[i] = NULL;
    realFlow[i] = -1;
    nucIds[i] = -1;
    if (i<numFlows) {
      HostSampleCompressedTraces.push_back(new LayoutCubeWithRegions<short>(ImgP.getGridParam(NUM_SAMPLES_RF),maxCompressedFrames, HostMem));
      HostSampleCompressedTraces.back()->memSet(0);
      HostSampleCompressedTraces.back()->setRWStrideZ();

      HostEmptyTraceAvg.push_back(new LayoutCubeWithRegions<float>(ImgP.getGridParam(uncompressedFrames), 1, HostMem));
      HostEmptyTraceAvg.back()->memSet(0);
      HostEmptyTraceAvg.back()->setRWStrideX();
    }
  }

  HostPerFlowRegP = new  LayoutCubeWithRegions<PerFlowParamsRegion>(ImgP.getGridParam(),1,HostMem);

  DeviceSampleCompressedTraces.clear();
  DeviceEmptyTraceAvg.clear();
  DevicePerFlowRegP = NULL;

}


HistoryCollection::~HistoryCollection()
{


  for (vector<LayoutCubeWithRegions<short>*>::iterator it = HostSampleCompressedTraces.begin(); it != HostSampleCompressedTraces.end(); ++it){
    if (*it != NULL) delete *it;
    *it = NULL;
  }
  for (vector<LayoutCubeWithRegions<short>*>::iterator it = DeviceSampleCompressedTraces.begin(); it != DeviceSampleCompressedTraces.end(); ++it){
    if (*it != NULL) delete *it;
    *it = NULL;
  }

  for (vector<LayoutCubeWithRegions<float>*>::iterator it = HostEmptyTraceAvg.begin(); it != HostEmptyTraceAvg.end(); ++it){
    if (*it != NULL) delete *it;
    *it = NULL;
  }
  for (vector<LayoutCubeWithRegions<float>*>::iterator it = DeviceEmptyTraceAvg.begin(); it != DeviceEmptyTraceAvg.end(); ++it){
    if (*it != NULL) delete *it;
    *it = NULL;
  }

  if(HostPerFlowRegP != NULL) delete HostPerFlowRegP;
  if(DevicePerFlowRegP != NULL) delete DevicePerFlowRegP;

}


////////////////////////////////////////////////////////////////////////
// HOST SIDE SAMPLE COLLECTIN FIRST BLOCK OF FLOWS

void HistoryCollection::writeSampleBeadToFlowBuffers(const FG_BUFFER_TYPE * fgbuffer, const int numFrames)
{
  //only sample beads get passed into this function
  //for each flow buffer write all frames:
  int flows = 0;
  for (std::vector<LayoutCubeWithRegions<short>*>::iterator it = HostSampleCompressedTraces.begin(); it != HostSampleCompressedTraces.end(); ++it)
  {
    for(int frame=0; frame < numFrames; frame++){
      (*it)->write(*fgbuffer);
      fgbuffer++; //next frame
    }
    flows++;
    if(flows == collectedFlows) break;
  }
}

size_t HistoryCollection::extractSamplesOneRegionAllFlows(BkgModelWorkInfo* region_bkinfo, const int flowBlockSize, const size_t regId)
{

  WorkSet myJob(region_bkinfo);
  int startFlow = flowBlockSize - collectedFlows;
  int numFrames = myJob.getNumFrames();
  int numLiveBeads = myJob.getNumBeads();
  FG_BUFFER_TYPE * localFgBuffer = myJob.getFgBuffer() + numFrames * startFlow; // move to first frame of first bead of first history flow
  //ToDo: remove after debugging
  for(int i = 0; i < numFlows; i++){
    realFlow[i] = startFlow + i;
    nucIds[i] = myJob.getNucIdForFlow(realFlow[i]);
  }
  cout << "extractSamplesOneRegionAllFlows: " << regId << " numFrames: " << numFrames << " numLiveBeads: " << numLiveBeads << endl;
  size_t numSamples = 0;
  for(int n=0; n<numLiveBeads; n++){
    // only if sampled and HQ add bead to sample set:
    if(region_bkinfo->bkgObj->region_data->my_beads.sampled[n]){
      if(region_bkinfo->bkgObj->region_data->my_beads.high_quality[n]){
        //set write pointer for all flow buffers to the current region/bead
        assert(numSamples < NUM_SAMPLES_RF && "GPU Flow by FLow Pipeline, Region Sample limit exceeded!");
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

void HistoryCollection::extractEmptyTracesOneRegionAllFlows(BkgModelWorkInfo* region_bkinfo, const int flowBlockSize, const size_t regId)
{
  WorkSet myJob(region_bkinfo);
  int startFlow = flowBlockSize - collectedFlows;
  int numFrames = myJob.getNumFrames();
  float * perFlowEmptyTraces = myJob.getShiftedBackground() + numFrames * startFlow;

  int flow = 0;

  for (std::vector<LayoutCubeWithRegions<float>*>::iterator it = HostEmptyTraceAvg.begin(); it != HostEmptyTraceAvg.end(); ++it)
  {
    (*it)->setRWPtrRegion(regId);
#if SAMPLECOLLECT_DEBUG > 1
    cout << "DEBUG GPU EmptytraceAvg History," << regId <<"," << flow << ",";
#endif
    for(int frame=0; frame < numFrames; frame++){
      (*it)->write(*perFlowEmptyTraces);
#if SAMPLECOLLECT_DEBUG > 1
      cout << *perFlowEmptyTraces <<",";
#endif
      perFlowEmptyTraces++; //next frame
    }
#if SAMPLECOLLECT_DEBUG > 1
    cout << endl;
#endif
    flow++;
    if( flow == collectedFlows) break;
  }
}

void HistoryCollection::initRegionalParametersOneRegion(BkgModelWorkInfo* region_bkinfo, const size_t regId)
{

  WorkSet myJob(region_bkinfo);
  reg_params * rp = &(region_bkinfo->bkgObj->region_data->my_regions.rp);

  PerFlowParamsRegion & ref = HostPerFlowRegP->refAtReg(regId);

  ref.setCopyDrift(rp->CopyDrift);
  ref.setDarkness(rp->darkness[0]);
  ref.setRatioDrift(rp->RatioDrift);
  ref.setSigma(*(rp->AccessSigma()));
  ref.setCoarseStart(region_bkinfo->bkgObj->region_data->my_regions.cache_step.i_start_coarse_step[0]);
  ref.setFineStart(region_bkinfo->bkgObj->region_data->my_regions.cache_step.i_start_fine_step[0]);
  ref.setTMidNuc(rp->AccessTMidNuc()[0]);
  ref.setTMidNucShift(rp->nuc_shape.t_mid_nuc_shift_per_flow[0]);
  ref.setTshift(rp->tshift);

#if SAMPLECOLLECT_DEBUG > 1
  cout << "DEBUG GPU Regional Param initialization regId " << regId << endl;
  ref.print();
#endif


}

//////////////////////////////////////
//all regions extration


void HistoryCollection::extractHistoryAllRegionsAllFlows(BkgModelWorkInfo* bkinfo, int flowBlockSize, int extractNumFlows){

  if(extractNumFlows <= 0) extractNumFlows=numFlows; //no extract Num Flows provided extract maximum
  assert(extractNumFlows <= flowBlockSize);//we cannot extract more than number of flows in block
  assert(extractNumFlows <= numFlows);//we cannot extract more flows as we have buffers created


  writeBuffer = wrapAroundIdx(extractNumFlows); //next buffer to write to (or oldest buffer) in history
  collectedFlows = extractNumFlows; //number of flows in history collected (only needed if we want to collect a history of > flowBlockSize (20))

  getHostPerFlowRegParams().memSet(0);
#if SAMPLECOLLECT_DEBUG > 0
  LayoutCubeWithRegions<size_t> NumSampleMap(ImgP.getGridParam(),1,HostMem);
  ImgP.print();
#endif
  for(size_t r=0; r < ImgP.getNumRegions(); r++){
    BkgModelWorkInfo* region_bkinfo = (BkgModelWorkInfo*) &bkinfo[r];
    WorkSet myJob(region_bkinfo);
    if (!myJob.getNumBeads()) continue;

    size_t regId = ImgP.getRegId(myJob.getRegCol(),myJob.getRegRow());

#if SAMPLECOLLECT_DEBUG > 0
    NumSampleMap[regId] = extractSamplesOneRegionAllFlows(region_bkinfo,flowBlockSize,regId);
#else
    extractSamplesOneRegionAllFlows(region_bkinfo,flowBlockSize,regId);
#endif
    extractEmptyTracesOneRegionAllFlows(region_bkinfo,flowBlockSize,regId);

    initRegionalParametersOneRegion(region_bkinfo,regId);

  }
  int flow = bkinfo->flow;
  latestSampleFlowOnHost = flow;
  latestEmptyFlowOnHost = flow;
  latestRegParamOnHost = flow;
#if SAMPLECOLLECT_DEBUG > 0
  cout << "DEBUG: GPU Pipeline, collected samples for: " << collectedFlows << " flows for Regional Fitting" <<endl;
  cout << "DEBUG: GPU Pipeline, num Samples per region Host side:" <<endl;
  NumSampleMap.printRegionTable<size_t>();
#endif
}


// END HOST SIDE SAMPLE COLLECTIN FIRST BLOCK OF FLOWS
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// INITIAL DEVICE INIT:
void HistoryCollection::InitDeviceBuffersAndSymbol( const ConstantFrameParams & cFrmP)
{
  //allocate and cop host buffers to device:

  DevicePerFlowRegP = new  LayoutCubeWithRegions<PerFlowParamsRegion>(ImgP.getGridParam(),1,DeviceGlobal);
  DevicePerFlowRegP->copy(getHostPerFlowRegParams());

  int hId=0;
  for (vector<LayoutCubeWithRegions<short>*>::iterator it = HostSampleCompressedTraces.begin(); it != HostSampleCompressedTraces.end(); ++it)
  {

    DeviceSampleCompressedTraces.push_back(new LayoutCubeWithRegions<short>( *(*it) ,DeviceGlobal));

    (*it)->setRWStrideZ();

    for(size_t regId =0 ; regId < ImgP.getNumRegions(); regId++)
    {
      for(int sId = 0; sId < 20; sId++)
        cout << "Sample:"<< hId <<":"<< regId << ":" << sId <<","<< (*it)->getCSVatReg<short>(regId,sId,0,0,(*it)->getDimZ()) << endl;
    }
    hId++;
  }


  hId=0;
  for (vector<LayoutCubeWithRegions<float>*>::iterator it = HostEmptyTraceAvg.begin(); it != HostEmptyTraceAvg.end(); ++it)
  {
    DeviceEmptyTraceAvg.push_back(new LayoutCubeWithRegions<float>( *(*it) ,DeviceGlobal));

    (*it)->setRWStrideX();

    //for(size_t regId =0 ; regId < ImgP.getNumRegions(); regId++)
    //{
    //  cout << "emptyAvg:"<< hId << ":" << regId << ","<< (*it)->getCSVatReg<float>(regId,0,0,0,(*it)->getRegW(regId)) << endl;
    //}
    hId++;
  }

  assert(DeviceSampleCompressedTraces.size() == DeviceEmptyTraceAvg.size());

  while( DeviceSampleCompressedTraces.size() < numFlows ){
    DeviceSampleCompressedTraces.push_back(new LayoutCubeWithRegions<short>(ImgP.getGridParam(NUM_SAMPLES_RF), cFrmP.getMaxCompFrames(), DeviceGlobal));
    DeviceSampleCompressedTraces.back()->memSet(0);
    DeviceEmptyTraceAvg.push_back(new LayoutCubeWithRegions<float>(ImgP.getGridParam(cFrmP.getUncompFrames()), 1, DeviceGlobal));
    DeviceEmptyTraceAvg.back()->memSet(0);
  }

  int i=0;
  for (vector<LayoutCubeWithRegions<short>*>::iterator it = DeviceSampleCompressedTraces.begin(); it != DeviceSampleCompressedTraces.end(); ++it)
    ppSampleCompressedTracesBuffers[i++] = (*it)->getPtr(); //store device pointer in aray for constant memory symbol

  i=0;
  for (vector<LayoutCubeWithRegions<float>*>::iterator it = DeviceEmptyTraceAvg.begin(); it != DeviceEmptyTraceAvg.end(); ++it)
    ppEmptyTraceAvg[i++] = (*it)->getPtr(); //store device pointer in aray for constant memory symbol

  //update symbol on device
  copySymbolsToDevice((*this));
  cout << "CUDA: HistoryCollection: InitDeviceBuffersAndSymbol: created " << DeviceSampleCompressedTraces.size() << " Device History Buffers (" << getSize()/(1024.0*1024.0) << "MB), and initialized Device control symbol" <<endl;
  print();
}


bool HistoryCollection::deviceBuffersInitialized()
{
  return (DevicePerFlowRegP != NULL && DeviceSampleCompressedTraces.size() > 0 &&  DeviceEmptyTraceAvg.size()>0);
}

////////////////////////////////////////////////////////////////////////

//has to be called right after all history elements are updated.
int HistoryCollection::UpdateHistoryCollection(const PerFlowParamsGlobal & Fp)
{
  if(collectedFlows < numFlows) collectedFlows++;

  //todo: remove after debuging
  realFlow[writeBuffer] = Fp.getRealFnum();
  nucIds[writeBuffer] = Fp.getNucId();

  writeBuffer = wrapAroundIdx(writeBuffer + 1);

  //update symbol on device
  copySymbolsToDevice((*this));
  //print();
  return collectedFlows;
}

void HistoryCollection::RezeroWriteBuffer()
{
  DeviceSampleCompressedTraces[writeBuffer]->memSet(0);
  DeviceEmptyTraceAvg[writeBuffer]->memSet(0);
}


LayoutCubeWithRegions<short> * HistoryCollection::getLatestSampleTraces(int currentFlow)
{
  int idx = writeBuffer-1;
  idx = (idx < 0)?(collectedFlows+idx):(idx);
  if(currentFlow == 0 || currentFlow != latestSampleFlowOnHost){
    if(DeviceSampleCompressedTraces[idx] != NULL){
      (*HostSampleCompressedTraces[idx]).copy((*DeviceSampleCompressedTraces[idx]));
      latestSampleFlowOnHost = currentFlow;
    }
  }
  return (HostSampleCompressedTraces[idx]);
}
LayoutCubeWithRegions<float> * HistoryCollection::getLatestEmptyTraceAvgs(int currentFlow)
{
  int idx = writeBuffer-1;
  idx = (idx < 0)?(collectedFlows+idx):(idx);
  if(currentFlow == 0 || currentFlow != latestEmptyFlowOnHost){
    if(DeviceEmptyTraceAvg[idx] != NULL){
      (*HostEmptyTraceAvg[idx]).copy((*DeviceEmptyTraceAvg[idx]));
      latestEmptyFlowOnHost = 0;
    }
  }
  return (HostEmptyTraceAvg[idx]);
}

LayoutCubeWithRegions<PerFlowParamsRegion> * HistoryCollection::getCurrentRegParams(int currentFlow)
{
  int idx = writeBuffer-1;
  idx = (idx < 0)?(collectedFlows+idx):(idx);
  if(currentFlow == 0 || currentFlow != latestRegParamOnHost){
    if(DevicePerFlowRegP != NULL){
      HostPerFlowRegP->copy(*DevicePerFlowRegP);
      latestRegParamOnHost=currentFlow;
    }
  }
  return HostPerFlowRegP;
}



void HistoryCollection::CopySerializationDataFromDeviceToHost()
{

  if(deviceBuffersInitialized()){

    cout << "CUDA: HistoryCollection: Copying history, regional param and polyclonal buffers from device to Host for serialization" << endl;
    vector<LayoutCubeWithRegions<short>*>::iterator hTraceit = HostSampleCompressedTraces.begin();
    for (vector<LayoutCubeWithRegions<short>*>::iterator dTraceit = DeviceSampleCompressedTraces.begin();
        dTraceit != DeviceSampleCompressedTraces.end(); ++dTraceit, ++hTraceit){
      if (*dTraceit != NULL)
        (*hTraceit)->copy(*(*dTraceit));
    }
    vector<LayoutCubeWithRegions<float>*>::iterator hEmptyit = HostEmptyTraceAvg.begin();
    for (vector<LayoutCubeWithRegions<float>*>::iterator dEmptyit = DeviceEmptyTraceAvg.begin(); dEmptyit != DeviceEmptyTraceAvg.end(); ++dEmptyit, ++hEmptyit){
      if (*dEmptyit != NULL)
        (*hEmptyit)->copy(*(*dEmptyit));
    }
    if(DevicePerFlowRegP != NULL)
      getHostPerFlowRegParams().copy(getDevPerFlowRegParams());
  }

}


/*

 void HistoryCollection::CopyHistoryToDeviceDeviceAfterRestart()
{
  vector<LayoutCubeWithRegions<short>*>::iterator hTraceit = HostSampleCompressedTraces.begin();
  for (vector<LayoutCubeWithRegions<short>*>::iterator dTraceit = DeviceSampleCompressedTraces.begin();
      dTraceit != DeviceSampleCompressedTraces.end(); ++dTraceit, ++hTraceit){
     if (*dTraceit != NULL)
      (*dTraceit)->copy(*(*hTraceit));
   }
  vector<LayoutCubeWithRegions<float>*>::iterator hEmptyit = HostEmptyTraceAvg.begin();
  for (vector<LayoutCubeWithRegions<float>*>::iterator dEmptyit = DeviceEmptyTraceAvg.begin(); dEmptyit != DeviceEmptyTraceAvg.end(); ++dEmptyit, ++hEmptyit){
     if (*dEmptyit != NULL)
       (*dEmptyit)->copy(*(*hEmptyit));
   }
  getDevPerFlowRegParams().copy(getHostPerFlowRegParams());
}
 */


