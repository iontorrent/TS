/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef DELTACOMP_H
#define DELTACOMP_H

#include <vector>
#include <stdint.h>
#include "SynchDatSerialize.h"
#include "BitHandler.h"

using namespace std;

class DeltaComp : public TraceCompressor {

public:
  virtual int GetCompressionType() { return TraceCompressor::DeltaComp; }
  virtual void Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize, size_t *maxsize);
  virtual void Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size);
  void compress(const std::vector<uint16_t>& data, size_t nRows, size_t nCols, size_t nFrames, std::vector<uint8_t> &compressed);
  void decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols, size_t &nFrames, std::vector<uint16_t> &data);
private:
  std::vector<uint16_t> mTranspose;
  std::vector<uint8_t> mCompressed;
  CompressorNH mCompressor;
};
#endif // DELTACOMP_H
