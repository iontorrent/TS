/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "PinnedInFlowReplay.h"
#include <vector>
#include <assert.h>

using namespace std;

// *********************************************************************
// reader specific functions

PinnedInFlowReader::PinnedInFlowReader(Mask *maskPtr, int numFlows, CommandLineOpts &clo)
  : PinnedInFlow(maskPtr, numFlows)
{
  // specify the file and section to read from
  reader_1 = new H5ReplayReader(clo, PINNEDINFLOW);
  reader_2 = new H5ReplayReader(clo, PINSPERFLOW);

  mW = maskPtr->W();
  mH = maskPtr->H();
}

PinnedInFlowReader::~PinnedInFlowReader()
{
  delete reader_1;
  delete reader_2;
}

void PinnedInFlowReader::Initialize(Mask *maskPtr)
{
  InitializePinnedInFlow(maskPtr);
  InitializePinsPerFlow();
}

void PinnedInFlowReader::InitializePinnedInFlow(Mask *maskPtr)
{
  int w = maskPtr->W();
  int h = maskPtr->H();

  // hyperslab layout in file
  // h * w = h x w
  vector<hsize_t> offset(2);
  offset[0] = 0;
  offset[1] = 0;

  // read out h x w
  vector<hsize_t> count(2);
  count[0] = h;
  count[1] = w;

  // memory layout is w * h
  // start position in memory slab is [0]
  vector<hsize_t> offset_out(1);
  offset_out[0] = 0;

  // read in h * w
  vector<hsize_t> count_out(1);
  count_out[0] = h * w;

  reader_1->Read(offset, count,  offset_out, count_out, &mPinnedInFlow[0]);
  fprintf(stdout, "PinnedInFlowReader: Width %d, Height %d: %d, %d, ...\n", w, h, mPinnedInFlow[0], mPinnedInFlow[1]);
}

void PinnedInFlowReader::InitializePinsPerFlow()
{
  vector<hsize_t> offset(1);
  offset[0] = 0;

  // read out numFlows
  vector<hsize_t> count(1);
  count[0] = mNumFlows;

  // memory layout is 1 x numFlows
  // start position in memory slab is [0]
  vector<hsize_t> offset_out(1);
  offset_out[0] = 0;

  // read in numFlows
  vector<hsize_t> count_out(1);
  count_out[0] = mNumFlows;

  reader_2->Read(offset, count,  offset_out, count_out, &mPinsPerFlow[0]);
}

int PinnedInFlowReader::Update (int flow, Image *img)
{
  // img is ignored
  return(mPinsPerFlow[flow]);
}

// *********************************************************************
// recorder specific functions

PinnedInFlowRecorder::PinnedInFlowRecorder(Mask *maskPtr, int numFlows, CommandLineOpts &clo)
  : PinnedInFlow(maskPtr, numFlows)
{
  // specify the file and section to record to
  recorder_1 = new H5ReplayRecorder(clo, PINNEDINFLOW);
  recorder_2 = new H5ReplayRecorder(clo, PINSPERFLOW);

  mW = maskPtr->W();
  mH = maskPtr->H();
}
 
PinnedInFlowRecorder::~PinnedInFlowRecorder()
{
  // save mPinnedInFlow to disk
  CreateDataset_1();
  WriteBuffer_1();

  delete recorder_1 ;

  // save mPinsPerFlow to disk
  CreateDataset_2();
  WriteBuffer_2();

  delete recorder_2;
}

// create the dataset for mPinnedInFlow
void PinnedInFlowRecorder::CreateDataset_1()
{
  vector<hsize_t> chunk_dims(2);
  chunk_dims[0] = mH;
  chunk_dims[1] = mW;

  // create the dataset, chunk_dims are total dimensions
  recorder_1->CreateDataset(chunk_dims);
}

// write the dataset for mPinnedInFlow
void PinnedInFlowRecorder::WriteBuffer_1()
{
  // hyperslab layout on disk
  // height x width
  // start position [0 0]
  vector<hsize_t> offset(2);
  offset[0] = 0;
  offset[1] = 0;

  // write out mH x mW
  vector<hsize_t> count(2);
  count[0] = mH;
  count[1] = mW;

  // memory layout of slab we write from is h * w
  // start position in memory slab is [0]
  // memory slab matches mPinnedInFlow
  vector<hsize_t> offset_in(1);
  offset_in[0] = 0;

  // write out mH * mW entries in mPinnedInFlow
  vector<hsize_t> count_in(1);
  count_in[0] = mH * mW;

  fprintf(stdout, "PinnedInWellRecorder: Width %d, Height %d: %d, %d, ...\n", mW, mH, mPinnedInFlow[0], mPinnedInFlow[1]);

  recorder_1->Write(offset, count, offset_in, count_in, &mPinnedInFlow[0]);
}

// create the dataset for mPinsPerFlow
void PinnedInFlowRecorder::CreateDataset_2()
{
  vector<hsize_t> chunk_dims(1);
  chunk_dims[0] = mNumFlows;

  // create the dataset, chunk_dims are total dimensions
  recorder_2->CreateDataset(chunk_dims);
}

// write the dataset for mPinsPerFlow
void PinnedInFlowRecorder::WriteBuffer_2()
{
  // hyperslab layout on disk 1 x numFlows
  // start position [0]
  vector<hsize_t> offset(1);
  offset[0] = 0;

  // write out numFlows
  vector<hsize_t> count(1);
  count[0] = mNumFlows;

  // memory layout of slab we write from is 1 x numFlows
  // start position in memory slab is [0]
  // memory slab matches mPinsInFlow
  vector<hsize_t> offset_in(1);
  offset_in[0] = 0;

  // write out mH * mW entries in mPinsInFlow
  vector<hsize_t> count_in(1);
  count_in[0] = mNumFlows;

  fprintf(stdout, "PinnedInFlowRecorder: numFlows %d: %d, %d, ...\n", mNumFlows, mPinsPerFlow[0], mPinsPerFlow[1]);

  recorder_2->Write(offset, count, offset_in, count_in, &mPinsPerFlow[0]);
}

