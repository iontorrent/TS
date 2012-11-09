/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef DELTACOMPFST_H
#define DELTACOMPFST_H

#include <vector>
#include <stdint.h>
#include "SynchDatSerialize.h"
#include "BitHandler.h"

using namespace std;

class DeltaCompFst : public TraceCompressor {

public:
  DeltaCompFst() : mCompressed(NULL), mMaxSize(0), mSize(0) {}
  ~DeltaCompFst() { if (mCompressed != NULL) { free(mCompressed); } }
  virtual int GetCompressionType() { return TraceCompressor::DeltaCompFst; }
  virtual void Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize, size_t *maxsize);
  virtual void Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size);
  void decompress(const int8_t *compressed, size_t size, TraceChunk &chunk);
  void compress(TraceChunk &chunk, size_t nRows, size_t nCols, size_t nFrames, int8_t **compressed, size_t *outsize, size_t *maxsize);
  
private:
  int8_t *mCompressed;
  size_t mMaxSize;
  size_t mSize;
};
#endif // DELTACOMPFST_H
