/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef VENCOLOSSLESS_H
#define VENCOLOSSLESS_H

#include <vector>
#include <stdint.h>
#include "SynchDatSerialize.h"
#include "BitHandler.h"
#include "HuffmanEncode.h"

using namespace std;
using namespace __gnu_cxx; 


class VencoLossless : public TraceCompressor {

public:
  virtual int GetCompressionType() { return TraceCompressor::LosslessVenco; }
  virtual void Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize, size_t *maxsize);
  virtual void Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size);
  void compress(const std::vector<uint16_t>& data, size_t nRows, size_t nCols, size_t nFrames, std::vector<uint8_t> &compressed);
  void decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols, size_t &nFrames, std::vector<uint16_t> &data);
private:
  std::vector<uint16_t> mTranspose;
  std::vector<uint8_t> mCompressed;
  Compressor mCompressor;
};
#endif // VENCOLOSSLESS_H
