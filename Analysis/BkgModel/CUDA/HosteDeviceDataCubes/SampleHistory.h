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


class HistoryCollectionConst
{

protected:

  int numFlows;  //maximum number of flows that can be collected in the history
  int writeBuffer; //index of next buffer to write to (after numFlows are written it also points to the oldest buffer)
  int collectedFlows; //max number of collected FLows == numFLows. if this limit is reached we turn into a fifo queue.

  int realFlow[MAX_FLOW_HISTORY]; //todo: remove, this is just for debugging
  int nucIds[MAX_FLOW_HISTORY];


  short* ppSampleCompressedTracesBuffers[MAX_FLOW_HISTORY];
  float* ppEmptyTraceAvg[MAX_FLOW_HISTORY];

  __host__ __device__ inline
  int wrapAroundIdx(const int idx) const
  {
    return idx%numFlows;
  }

  __host__ __device__ inline
  int calcRbIdx( int nIndex) const
  {
    //if not yet wrapped around nIndex is actual index
    if(collectedFlows < numFlows) return nIndex;
    //if already wrapped around the writeBuffer is also the oldest buffer
    //and we just have to wrap around if we run over the end of the buffer list
    return wrapAroundIdx(writeBuffer + nIndex);
  }

public:

  // nIndex == 0 will return the oldest ( currentFlow - numFlows) sample buffer.
  __device__ inline
  const short * getSampleTraces(int nIndex) const
  {
    int idx =calcRbIdx(nIndex);
    return ppSampleCompressedTracesBuffers[idx];
  }

  __device__ inline
  const float * getEmptyTraces(int nIndex) const
  {
    int idx =calcRbIdx(nIndex);
    return ppEmptyTraceAvg[idx];
  }

  __device__ inline
  int getNucId(int nIndex) const
  {
    int idx =calcRbIdx(nIndex);
    return nucIds[idx];
  }


  __host__ __device__ inline
  const short * getLatestSampleTraces() const
  {
    int idx = writeBuffer-1;
    idx = (idx < 0)?(collectedFlows+idx):(idx);
    return ppSampleCompressedTracesBuffers[idx];
  }
  __host__ __device__ inline
  const float * getLatestEmptyTraces() const
  {
    int idx = writeBuffer-1;
    idx = (idx < 0)?(collectedFlows+idx):(idx);
    return ppEmptyTraceAvg[idx];
  }
  __device__ inline
  int getLatestNucId() const
  {
    int idx = writeBuffer-1;
    idx = (idx < 0)?(collectedFlows+idx):(idx);
    return nucIds[idx];
  }


  __device__ inline
  short * getWriteBufferSampleTraces()
  {
    return ppSampleCompressedTracesBuffers[writeBuffer];
  }
  __device__ inline
  float * getWriteBufferEmptyTraceAvg()
  {
    return ppEmptyTraceAvg[writeBuffer];
  }

  __device__ inline
  int getRealFlowNum(int nIndex) const
  {
    int idx =calcRbIdx(nIndex);
    return realFlow[idx];
  }

  __host__ __device__  inline
  int getNumHistoryFlows() const
  {
    //returns number of flows currently collected in flow history
    return collectedFlows;
  }

  __host__ __device__  inline
  int getMaxNumHistoryFlows() const
  {
    //returns max number of flows that can be collected in flow history
    return numFlows;
  }

  __host__ __device__  inline
  void print() const
  {
    const short * SmplAddress=  getLatestSampleTraces();
    const float * EmptyAddress=  getLatestEmptyTraces();
    printf("CUDA: HistoryCollection:\n collected Sample Flows: %d\n max Sample Flows: %d\n write Index: %d\n latest Sample Trace Buffer: %p\n  latest EmptyTrace Buffer: %p\n",collectedFlows,numFlows,writeBuffer,SmplAddress,EmptyAddress);
    for(int i=0; i<numFlows;i++)
      printf(" %2d(%3d,%2d):%s%p,%p%s\n", i, realFlow[i], nucIds[i], (i<collectedFlows)?("*"):(" "),ppSampleCompressedTracesBuffers[i],ppEmptyTraceAvg[i] ,(i==writeBuffer)?("<-"):("  "));
  }


};


class HistoryCollection : public HistoryCollectionConst
{

  ImgRegParams ImgP;

  int latestSampleFlowOnHost;
  int latestEmptyFlowOnHost;
  int latestRegParamOnHost;

  std::vector<LayoutCubeWithRegions<short>*> HostSampleCompressedTraces;
  std::vector<LayoutCubeWithRegions<short>*> DeviceSampleCompressedTraces;

  std::vector<LayoutCubeWithRegions<float>*> HostEmptyTraceAvg;
  std::vector<LayoutCubeWithRegions<float>*> DeviceEmptyTraceAvg;

  LayoutCubeWithRegions<PerFlowParamsRegion> * HostPerFlowRegP;
  LayoutCubeWithRegions<PerFlowParamsRegion> * DevicePerFlowRegP;



protected:

  void writeSampleBeadToFlowBuffers(const FG_BUFFER_TYPE * fgbuffer, const int numFrames);

public:
  size_t extractSamplesOneRegionAllFlows(BkgModelWorkInfo* region_bkinfo,const int flowBlockSize, const size_t regId);
  void extractEmptyTracesOneRegionAllFlows(BkgModelWorkInfo* region_bkinfo, const int flowBlockSize, const size_t regId);
  void initRegionalParametersOneRegion(BkgModelWorkInfo* region_bkinfo, const size_t regId);



  HistoryCollection(const ImgRegParams & ImgP, const int numFlowsInHistory, const int maxCompressedFrames, const int uncompressedFrames );
  ~HistoryCollection();


  // to be executed during last flow before switching to GPU flow by flow pipeline.
  // extractNumFlows == 0 -> will extract flowblockSize flows
  void extractHistoryAllRegionsAllFlows(BkgModelWorkInfo* bkinfo, int flowBlockSize=20, int extractNumFlows=0);
  void InitDeviceBuffersAndSymbol(const ConstantFrameParams & cFrmP);
  bool deviceBuffersInitialized();

  //updates meta data after Sample collection by kernel!
  int UpdateHistoryCollection(const PerFlowParamsGlobal & Fp);
  void RezeroWriteBuffer();

  //returns pointer to host copy of latest Data. if current FLow is passed dat afrom device will only be copied if currentFlow == 0 OR currentFlow != the flow idx of the last update of the according host buffer
  LayoutCubeWithRegions<short> * getLatestSampleTraces(int currentFlow = 0);
  LayoutCubeWithRegions<float> * getLatestEmptyTraceAvgs(int currentFlow = 0);
  LayoutCubeWithRegions<PerFlowParamsRegion> * getCurrentRegParams(int currentFlow = 0);

  void CopySerializationDataFromDeviceToHost();
  //void CopyHistoryToDeviceDeviceAfterRestart(); // not needed done by InitDeviceBuffers and Symbols if host buffers are initlaized correctly

  LayoutCubeWithRegions<PerFlowParamsRegion> & getHostPerFlowRegParams() {return *HostPerFlowRegP;}
  LayoutCubeWithRegions<PerFlowParamsRegion> & getDevPerFlowRegParams(){ return *DevicePerFlowRegP;}

  //returns size in bytes for all the flow buffers combined
  size_t getSize() const { return  (HostSampleCompressedTraces[0]->getSize() + HostEmptyTraceAvg[0]->getSize()) * numFlows + HostPerFlowRegP->getSize(); }

  //};

private:

  HistoryCollection(){
    numFlows=0;
    writeBuffer=0;
    collectedFlows=0;
    latestSampleFlowOnHost=0;
    latestEmptyFlowOnHost=0;
    latestRegParamOnHost=0;
    HostPerFlowRegP = NULL;
    DevicePerFlowRegP = NULL;
    HostSampleCompressedTraces.clear();
    DeviceSampleCompressedTraces.clear();
    HostEmptyTraceAvg.clear();
    DeviceEmptyTraceAvg.clear();
  };

  friend class boost::serialization::access;
  template<typename Archive>
  void load(Archive& ar, const unsigned version) {

    cout << "CUDA: HistoryCollection: serialization, loading non dynamic members" << endl;
    //load non dynamic members
    ar  >> ImgP
    >> numFlows  //maximum number of flows that can be collected in the history
    >> writeBuffer //index of next buffer to write to (after numFlows are written it also points to the oldest buffer)
    >> collectedFlows //max number of collected FLows == numFLows. if this limit is reached we turn into a fifo queue.
    >> realFlow //just debugging shpould be removed
    >> nucIds;

    // load per Region Regional Parameters
    vector<PerFlowParamsRegion> * tmpRegParams = NULL; //new vector<PerFlowParamsRegionSerilization>(ImgP.getNumRegions());
    ar & tmpRegParams;
    if(tmpRegParams != NULL){
      cout << "CUDA: HistoryCollection: serialization, loading Regional Parameters" << endl;
      HostPerFlowRegP = new LayoutCubeWithRegions<PerFlowParamsRegion>(ImgP.getGridParam(),1, HostMem);
      HostPerFlowRegP->copyIn(tmpRegParams->data());
    }


    //load SampleTrace history and empty trace average hsitory
    size_t sampleFrames = 0;
    ar >> sampleFrames;
    size_t emptyFrames = 0;
    ar >> emptyFrames;
    cout << "CUDA: HistoryCollection: loading max sample Frames " << sampleFrames << " max avg empty frames "<< emptyFrames << endl;
    //temporary vectors for serializations
    if(sampleFrames > 0 && emptyFrames > 0){
      vector<short> * tmpSampleVector[MAX_FLOW_HISTORY] = {NULL}; //((ImgP.getGridParam(NUM_SAMPLES_RF)).getImgSize() * sampleFrames);
      vector<float> * tmpEmptyVector[MAX_FLOW_HISTORY] = {NULL}; //((ImgP.getGridParam(emptyFrames)).getImgSize());

      cout << "CUDA: HistoryCollection: serialization, loading Sample Traces and Empty Average History for " << collectedFlows << " flows" << endl;

      for( int i = 0; i < collectedFlows; i++ )
      {
        ar >> tmpSampleVector[i];

        HostSampleCompressedTraces.push_back(new LayoutCubeWithRegions<short>(ImgP.getGridParam(NUM_SAMPLES_RF),sampleFrames, HostMem));
        HostSampleCompressedTraces.back()->setRWStrideZ();
        HostSampleCompressedTraces.back()->copyIn(tmpSampleVector[i]->data());

        for(size_t regId =0 ; regId < ImgP.getNumRegions(); regId++)
        {
          for(int sId = 0; sId < 20; sId++)
            cout << "Sample:" << i << ":"<< regId << ":" << sId <<","<< HostSampleCompressedTraces.back()->getCSVatReg<short>(regId,sId,0,0,sampleFrames) << endl;
        }

        ar >> tmpEmptyVector[i];

        HostEmptyTraceAvg.push_back(new LayoutCubeWithRegions<float>(ImgP.getGridParam(emptyFrames), 1, HostMem));
        HostEmptyTraceAvg.back()->setRWStrideX();
        HostEmptyTraceAvg.back()->copyIn(tmpEmptyVector[i]->data());

        for(size_t regId =0 ; regId < ImgP.getNumRegions(); regId++)
        {
          cout << "emptyAvg:" << i << ":"<< regId << ","<< HostEmptyTraceAvg.back()->getCSVatReg<float>(regId,0,0,0,emptyFrames) << endl;
        }

        delete tmpSampleVector[i];
        delete tmpEmptyVector[i];

      }



    }

  }

  template<typename Archive>
  void save(Archive& ar, const unsigned version) const {

    //store non dynamic members
    cout << "CUDA: HistoryCollection: serialization, storing non dynamic members" << endl;

    ar  << ImgP
    << numFlows  //maximum number of flows that can be collected in the history
    << writeBuffer //index of next buffer to write to (after numFlows are written it also points to the oldest buffer)
    << collectedFlows //max number of collected FLows == numFLows. if this limit is reached we turn into a fifo queue.
    << realFlow //just debugging should be removed
    << nucIds;

    // store per REgion Regional Parameters
    vector<PerFlowParamsRegion> * tmpRegParams = NULL;
    if(HostPerFlowRegP != NULL){
      cout << "CUDA: HistoryCollection: serialization, storing Regional Parameters" << endl;
      tmpRegParams = new vector<PerFlowParamsRegion>(HostPerFlowRegP->getPtr(), HostPerFlowRegP->getPtr() + HostPerFlowRegP->getNumElements());// ImgP.getGridParam(emptyFrames));
    }
    ar & tmpRegParams;
    if(tmpRegParams != NULL) delete tmpRegParams;

    // store Sample Trace and Empty Trace average history
    size_t sampleFrames = 0;
    size_t emptyFrames = 0;
    if(!HostSampleCompressedTraces.empty()){
      sampleFrames = HostSampleCompressedTraces.front()->getDimZ();
      emptyFrames = HostEmptyTraceAvg.front()->getRegW(0);
      cout << "CUDA: HistoryCollection: max sample Frames " << sampleFrames << " max avg empty frames "<< emptyFrames << endl;
      //size plane size from ImgP
    }else{
      cout << "CUDA: HistoryCollection: no history data for serialization available" << endl;
    }
    ar << sampleFrames;
    ar << emptyFrames;

    if(sampleFrames > 0 && emptyFrames > 0 ){
      for( int i = 0; i < collectedFlows; i++ )
      {
        vector<short> * tmpSampleVector = new vector<short>(HostSampleCompressedTraces[i]->getNumElements());//ImgP.getGridParam(NUM_SAMPLES_RF) * sampleFrames);
        vector<float> * tmpEmptyVector = new vector<float>(HostEmptyTraceAvg[i]->getNumElements());// ImgP.getGridParam(emptyFrames));

        tmpSampleVector->assign(HostSampleCompressedTraces[i]->getPtr(), HostSampleCompressedTraces[i]->getPtr() + HostSampleCompressedTraces[i]->getNumElements() );
        ar << tmpSampleVector;

        HostSampleCompressedTraces[i]->setRWStrideZ();

        for(size_t regId =0 ; regId < ImgP.getNumRegions(); regId++)
          {
            for(int sId = 0; sId < 20; sId++)
              cout << "Sample:" << i << ":" << regId << ":" << sId <<","<< HostSampleCompressedTraces[i]->getCSVatReg<short>(regId,sId,0,0,sampleFrames) << endl;
          }

        tmpEmptyVector->assign(HostEmptyTraceAvg[i]->getPtr(), HostEmptyTraceAvg[i]->getPtr() + HostEmptyTraceAvg[i]->getNumElements() );
        ar << tmpEmptyVector;

        HostEmptyTraceAvg[i]->setRWStrideX();
        for(size_t regId =0 ; regId < ImgP.getNumRegions(); regId++)
        {
          cout << "emptyAvg:" << i << ":"<< regId << ","<< HostEmptyTraceAvg[i]->getCSVatReg<float>(regId,0,0,0,emptyFrames) << endl;
        }

        delete tmpSampleVector;
        delete tmpEmptyVector;

      }



      cout << "CUDA: HistoryCollection: no history data for serialization available" << endl;

    }

  }


  BOOST_SERIALIZATION_SPLIT_MEMBER()

};

#endif /* SAMPLEHISTORY_H_ */
