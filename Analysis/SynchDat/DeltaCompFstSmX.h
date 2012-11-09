/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef DELTACOMPFSTSMX_H
#define DELTACOMPFSTSMX_H

#include <vector>
#include <stdint.h>
#include "SynchDatSerialize.h"
#include "BitHandler.h"

using namespace std;

class DeltaCompFstSmX : public TraceCompressor {

public:
  DeltaCompFstSmX() : mWellsCompacted(20) {}
  ~DeltaCompFstSmX() {}
  virtual int GetCompressionType() { return TraceCompressor::DeltaCompFstSmX; }
  virtual void Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize, size_t *maxsize);
  virtual void Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size);
  void decompress(const int8_t *compressed, size_t size, TraceChunk &chunk);
  void compress(TraceChunk &chunk, size_t nRows, size_t nCols, size_t nFrames, int8_t **compressed, size_t *outsize, size_t *maxsize);
  
private:
  uint8_t mWellsCompacted;
};
#endif // DELTACOMPFSTSMX_H
