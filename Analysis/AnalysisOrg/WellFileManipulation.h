/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef WELLFILEMANIPULATION_H
#define WELLFILEMANIPULATION_H

#include <stdio.h>
#include <string.h>

#include "CommandLineOpts.h"
#include "RawWells.h"
#include "Utils.h"
#include "pThreadWrapper.h"


class Mask;
class ImageSpecClass;


void GetMetaDataForWells(char *dirExt, RawWells &rawWells, const char *chipType);
void SetWellsToLiveBeadsOnly(RawWells &rawWells, Mask *maskPtr);
void CreateWellsFileForWriting (RawWells &rawWells, Mask *maskPtr,
                                CommandLineOpts &inception_state,
                                int numFlows,
                                int numRows, int numCols,
                                const char *chipType);



///////////////////////////////////////
//turning this into a class that contains thread creation and handles all the needed data elements


class WriteFlowDataClass : public pThreadWrapper {

  string filePath;
  int numCols;
  size_t stepSize;
  bool saveAsUShort;
  unsigned int queueSize;
  SemQueue* packQueuePtr;
  SemQueue* writeQueuePtr;

protected:

  virtual void InternalThreadFunction();

public:

  WriteFlowDataClass( unsigned int saveQueueSize, CommandLineOpts &inception_state, ImageSpecClass &my_image_spec  ,const RawWells & rawWells);
  ~WriteFlowDataClass();


  SemQueue & GetPackQueue(){ return *packQueuePtr; }
  SemQueue & GetWriteQueue(){ return *writeQueuePtr; }

  unsigned int getQueueSize(){return queueSize;}

  bool start();
  void join();
};


#endif // WELLFILEMANIPULATION_H
