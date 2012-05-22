/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "EmptyTraceReplay.h"

#include <vector>
#include <assert.h>
#include <pthread.h>
using namespace std;

// *********************************************************************
// reader specific functions

EmptyTraceReader::EmptyTraceReader(CommandLineOpts &clo) : EmptyTrace(clo)
{
  // specify the file and section to read from
  reader = new H5ReplayReader(clo, EMPTYTRACE);

  bufferCount = 0;

  pthread_mutex_t tmp_mutex = PTHREAD_MUTEX_INITIALIZER;
  read_mutex = new pthread_mutex_t(tmp_mutex);
}

EmptyTraceReader::~EmptyTraceReader()
{
  delete reader;
  delete read_mutex;
}


void EmptyTraceReader::GenerateAverageEmptyTrace(Region *region, PinnedInFlow& pinnedInFlow, Mask *bfmask, Image *img, int flow)
{
  //fprintf(stdout, "ETR::GenerateAverageEmptyTrace with thread %u\n", (unsigned int)pthread_self());

  ReadEmptyTraceBuffer(flow);
}

void  EmptyTraceReader::Allocate(int numfb, int _imgFrames)
{
  EmptyTrace::Allocate(numfb, _imgFrames);
}

void EmptyTraceReader::ReadEmptyTraceBuffer(int flow)
{
  pthread_mutex_lock(read_mutex);

  if (bufferCount == 0)
  {
    // in the next block, populate the whole buffer from file
    memset (bg_buffers,0,sizeof (float [numfb*imgFrames]));
    memset (bg_dc_offset,0,sizeof (float[numfb]));

    int block_start =  (flow/numfb)*numfb;
    FillBuffer(block_start, regionIndex);
  }

  bufferCount = (bufferCount+1) % numfb;

  pthread_mutex_unlock(read_mutex);
}

void EmptyTraceReader::FillBuffer(int flow, int regionindex)
{
  // hyperslab layout in file
  // regions x flows x imgFrames
  // start position [regionindex, flow, 0]
  vector<hsize_t> offset(3);
  offset[0] = regionindex;
  offset[1] = flow;
  offset[2] = 0;

  // read out 1 x numfb x imgFrames
  vector<hsize_t> count(3);
  count[0] = 1;
  count[1] = numfb;
  count[2] = imgFrames;

  // memory layout for each slab we read in imgFrames x numfb
  // start position in memory slab is [0, 0]
  vector<hsize_t> offset_out(2);
  offset_out[0] = 0;
  offset_out[1] = 0;

  // read in numfb x imgFrames
  vector<hsize_t> count_out(2);
  count_out[0] = numfb;
  count_out[1] = imgFrames;

  reader->Read(offset, count,  offset_out, count_out, bg_buffers);
  fprintf(stdout, "H5 read buffer: flow %d, region %d: %f, %f, ...\n", flow, regionindex, bg_buffers[0], bg_buffers[1]);
}

// *********************************************************************
// recorder specific functions

EmptyTraceRecorder::EmptyTraceRecorder(CommandLineOpts &clo) : EmptyTrace(clo)
{
  // specify the file and section to record to
  recorder = new H5ReplayRecorder(clo ,EMPTYTRACE);

  bufferCount = 0;

  pthread_mutex_t tmp_mutex = PTHREAD_MUTEX_INITIALIZER;
  write_mutex = new pthread_mutex_t(tmp_mutex);
}

EmptyTraceRecorder::~EmptyTraceRecorder()
{
  delete recorder;
  delete write_mutex;
}

void  EmptyTraceRecorder::Allocate(int numfb, int _imgFrames)
{
  EmptyTrace::Allocate(numfb, _imgFrames);

  // initialize the chunking strategy based on the bg_buffer size
  // assumption is that bg_buffers always align to a chunk start
  // data layout is regions x flows (numfb) x trace (imgFrames)
  // chunk size is per region, i.e.: 1 x numfb x imgFrames
  // chunk size is 20 x 105 size(float) = 9800 bytes
  // chunk size matches the memory slabs
  vector<hsize_t> chunk_dims(3);
  chunk_dims[0] = 1;
  chunk_dims[1] = numfb;
  chunk_dims[2] = _imgFrames;

  // create the dataset
  recorder->CreateDataset(chunk_dims);

}

void EmptyTraceRecorder::GenerateAverageEmptyTrace(Region *region, PinnedInFlow& pinnedInFlow, Mask *bfmask, Image *img, int flow)
{
  EmptyTrace::GenerateAverageEmptyTrace(region, pinnedInFlow, bfmask, img, flow);
  WriteEmptyTraceBuffer(flow);
}

void  EmptyTraceRecorder::WriteEmptyTraceBuffer(int flow)
{

  pthread_mutex_lock(write_mutex);

  bufferCount = bufferCount+1;

  if ( bufferCount == numfb){

    // write the whole buffer to file
    int block_start =  (flow/numfb)*numfb;
    WriteBuffer(block_start, regionIndex);
    bufferCount = 0;
  }

  pthread_mutex_unlock(write_mutex);

}

void EmptyTraceRecorder::WriteBuffer(int block_start, int regionindex)
{
  // hyperslab layout on disk
  // imgFrames x flows x regions
  // start position [regionindex, start of block for this flow, 0]
  vector<hsize_t> offset(3);
  offset[0] = regionindex;
  offset[1] = block_start;
  offset[2] = 0;
  // write out 1 block x numfb x imgFrames
  vector<hsize_t> count(3);
  count[0] = 1;
  count[1] = numfb;
  count[2] = imgFrames;

  // memory layout of slab we write from is numfb x imgFrames
  // start position in memory slab is [0]
  // memory slab matches bg_buffers
  vector<hsize_t> offset_in(1);
  offset_in[0] = 0;

  // write out numfb x imgFrames
  vector<hsize_t> count_in(1);
  count_in[0] = numfb*imgFrames;

  vector<hsize_t> extension(3);
  extension[0] = regionindex+1;
  extension[1] = block_start+numfb;
  extension[2] = imgFrames;

  vector<float> newbuff(numfb*imgFrames);
  int ii = 0;
  for (int i=0; i<numfb; i++)
    for (int j=0; j<imgFrames; j++){
      newbuff[ii] = bg_buffers[ii] + bg_dc_offset[i];
      ii++;
    }

  recorder->ExtendDataSet(extension); // extend if necessary
  // fprintf(stdout, "H5 buffer, flow %d, region %d: %f, %f, ...\n", flow, regionIndex, newbuff[0], newbuff[1]);
  recorder->Write(offset, count, offset_in, count_in, &newbuff[0]);
}
