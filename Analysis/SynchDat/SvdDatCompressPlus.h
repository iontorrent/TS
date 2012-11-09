/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SVDDATCOMPRESSPLUS_H
#define SVDDATCOMPRESSPLUS_H

#include <vector>
#include <stdint.h>
#include "SynchDatSerialize.h"

class SvdDatCompressPlus : public TraceCompressor {

public:
  virtual int GetCompressionType() { return TraceCompressor::LossySvdDatPlus; }
  virtual void Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize);
  virtual void Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size);
  void compress(const std::vector<uint16_t>& data, size_t nRows, size_t nCols, size_t nFrames, std::vector<uint8_t> &compressed);
  void decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols, size_t &nFrames, std::vector<uint16_t> &data);

  //private:
  protected:
  std::vector<uint16_t> mTranspose;
  std::vector<uint8_t> mCompressed;
};


class SvdDatCompressPlusPlus : public SvdDatCompressPlus {
//class SvdDatCompressPlusPlus : public TraceCompressor {
 public:
 SvdDatCompressPlusPlus():errCon(49),rankGood(8),pivot(0.0) {};
  void Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize);
  void Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size);
  int GetCompressionType() { return TraceCompressor::LossySvdDatPlusPlus; }
  void compress(const std::vector<uint16_t>& data, size_t nRows, size_t nCols, size_t nFrames, std::vector<uint8_t> &compressed);
  void decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols, size_t &nFrames, std::vector<uint16_t>& data);
  void SetErrCon(float ec) { errCon = ec; }
  void SetRankGood(int rg) { rankGood = rg; }
  void SetPivot(float piv) { pivot = piv; }
 private:
  float errCon;
  int rankGood;
  float pivot;
};

#endif // SVDDATCOMPRESSPLUS_H
