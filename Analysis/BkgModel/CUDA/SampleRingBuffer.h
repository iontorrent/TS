/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * BkgGpuPipeline.h
 *
 *  Created on: September 5, 2014
 *      Author: jakob
 */

#ifndef SAMPLERINGBUFFER_H_
#define SAMPLERINGBUFFER_H_


#include "LayoutTranslator.h"
#include "DeviceParamDefines.h"
#include "SignalProcessingFitterQueue.h"
#include "GpuPipelineDefines.h"



//ring buffer logic to store N elements
//to simulate LIFO behavior, read pointer moves backwards write buffer forward
//read pointer gets always set to latest written element.
//read pointer can always be reset to the latest element with resetRead()
//programmer has to guarantee that N elements are stored in buffer when reading

class SampleRingBuffer
{


protected:

  int numBuffers;
  int write;
  int read;


public:
  SampleRingBuffer(int Samples);

  int getNumBuffers();
  void moveRead();
  void moveWrite();
  void setWrite(int idx);
  void resetRead();

};



class SampleRingBufferHost : public SampleRingBuffer
{


  LayoutCubeWithRegions<int> SamplesPerRegion;  // one value per region
  LayoutCubeWithRegions<SampleCoordPair> SampleCoord; //x/y coordinates per sample (200 per region)
  LayoutCubeWithRegions<short> SampleCompressedTraces; //maxCompFrames * 200 per region
  int * NucIdMap;

  int numFrames;



public:

  //ToDo: constructor that reads sample ring buffer from file after restart
  //SampleRingBufferHost(string filename);

  SampleRingBufferHost(int N, ImgRegParams ImgP, size_t nFrames);
  ~SampleRingBufferHost();


  void setRWPtrsRegion(size_t regId, size_t sampleIdx);

  void FillWholeRingBuffer( void * bkinfo);

  void DumpToFile(string filename);




};

/*
class SampleRingBufferDevice
{
  short* CompressedTraces;
  float* ParamCubes;
  int * numBuffers;

  SampleRingBufferDevice();
  ~SampleRingBufferDevice();
};
*/




#endif /* SAMPLERINGBUFFER_H_ */
