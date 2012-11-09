/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SVDDATCOMPRESS_H
#define SVDDATCOMPRESS_H

#include <vector>
#include <stdint.h>
#include <armadillo>
#include "SynchDatSerialize.h"

class SvdDatCompress : public TraceCompressor {

public:
  SvdDatCompress() { Init(10.0f, 6); }
  SvdDatCompress(float precision, int numVec) { Init(precision, numVec); }
  void Init(float precision, int numVec) { mPrecision = precision; mNumEvec = numVec; }
  virtual int GetCompressionType() { return TraceCompressor::LossySvdDat; }
  virtual void Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize, size_t *maxsize);
  virtual void Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size);
private:
  arma::Mat<float> Y, X, B, Cov, EVec;
  arma::Mat<short> BB;
  arma::Col<float> EVal;
  float mPrecision;
  int mNumEvec;
  std::vector<uint8_t> mCompressed;
};

#endif // SVDDATCOMPRESS_H
