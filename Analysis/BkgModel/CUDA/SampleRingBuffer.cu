/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * BkgGpuPipeline.cu
 *
 *  Created on: September 5, 2014
 *      Author: jakob
 */


#include <iostream>
#include "LayoutTranslator.h"
#include "LayoutTester.h"
#include "MasterKernel.h"
#include "DeviceParamDefines.h"
#include "SignalProcessingFitterQueue.h"
#include "JobWrapper.h"
#include "GpuPipelineDefines.h"
#include "SampleRingBuffer.h"

//////////////////
//Host and Device-side implementation of the Sample Ring-Buffer


///////////////////////////////
// base class
SampleRingBuffer::SampleRingBuffer(int Samples):numBuffers(Samples),write(0)
{
  resetRead();
}

int SampleRingBuffer::getNumBuffers() {return numBuffers;}

void SampleRingBuffer::moveRead() //more read pointer to older sample, backwards
{
  if(--read < 0 )
    read = numBuffers - 1;
}
void SampleRingBuffer::moveWrite() // move write pointer one forward
{
  if(++write == numBuffers)
    write = 0;
}
void SampleRingBuffer::setWrite(int idx)
{
  write = idx % numBuffers;
  resetRead();
}


void SampleRingBuffer::resetRead()
{
  read = write-1;
  if(read<0) read = numBuffers-1;
}




////////////////////////////////////////////////
//Host Side Ring-Buffer implementation
SampleRingBufferHost::SampleRingBufferHost(int N, ImgRegParams ImgP, size_t nFrames):
      SampleRingBuffer(N),
      SamplesPerRegion(ImgP.getGridParam(), 1, HostMem),
      SampleCoord(ImgP.getGridParam(NUM_SAMPLES_RF), 1, HostMem),
      SampleCompressedTraces(ImgP.getGridParam(NUM_SAMPLES_RF),nFrames * N, HostMem), //(200 per region * maxFrames) * N
      NucIdMap(NULL),
      numFrames(nFrames)
{

  cout << "SampleRingBufferHost: creating sample ring buffer with " << ImgP.getNumRegions() << " regions, " << N << "x" << NUM_SAMPLES_RF <<" samples and " << nFrames << " max frames" << endl;
  NucIdMap = new int[N];

  SampleCompressedTraces.setRWStrideZ();
  //SampleParamCube.setRWStrideZ();

};

SampleRingBufferHost::~SampleRingBufferHost()
{
  if(NucIdMap != NULL){
    delete[] NucIdMap;
    NucIdMap = NULL;
  }
}


void SampleRingBufferHost::FillWholeRingBuffer( void * bkinfo)
{

  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);
  ImgRegParams irP;

  irP.init(myJob.getImgWidth(),myJob.getImgHeight(), myJob.getRegionWidth(),myJob.getRegionHeight());
  size_t regId =  irP.getRegId(myJob.getRegCol(),myJob.getRegRow());

  //flow to start from
  int sampleFlow =  myJob.getFlowBlockSize() - numBuffers - 1;
  SampleCoord.setRWPtrRegion(regId);
  SampleCoord.setRWStrideX();
  int numSamples = 0;
  for(int idx = 0; idx < myJob.getNumBeads() && numSamples < NUM_SAMPLES_RF  ; idx++)
  {
    if( info->bkgObj->region_data->my_beads.sampled[idx] && info->bkgObj->region_data->my_beads.high_quality[idx] )
    {

      //write location (not part of ring buffer
      BeadParams * bP = &(myJob.getBeadParams()[idx]);
      SampleCoordPair loc(bP->x,bP->y);
      SampleCoord.write(loc);
      cout << "regId," << regId <<",x," << loc.x << ",y," << loc.y << endl;

      // fill actual Ring Buffer with Sample Traces for flows!!!!
      setWrite(0);
      for(int b= 0; b < numBuffers; b++ ){
        SampleCompressedTraces.setRWPtrRegion(regId,numSamples,0,(write*numFrames));
        FG_BUFFER_TYPE * fgPtr = myJob.getFgBuffer()+myJob.getNumFrames()*(sampleFlow+b);

        for(int f=0; f < myJob.getNumFrames(); f++){
          SampleCompressedTraces.write( *fgPtr);
          fgPtr++;
        }
        moveWrite();
      }
      numSamples++;
    }
  }
  cout <<"SampleNucMap,";
  for(int b= 0; b < numBuffers; b++ ){
    NucIdMap[b] = myJob.getFlowIdxMap()[sampleFlow+b];
    cout << NucIdMap[b]  << ",";
  }
  cout << endl;

  SamplesPerRegion.setRWPtrRegion(regId);
  SamplesPerRegion.write(numSamples);

}

void SampleRingBufferHost::DumpToFile(string filename)
{
  ofstream myFile (filename.c_str(), ios::binary);


}




//Device side Ring-Buffer implementation

