/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

//#define SUBMISSION 
//#define SCORE 
//#define RESULT_STATS
//#define CLOCK
//#define INLINE
//#define DEBUG_OUTPUT

#define TIME_LIMIT (60)

#include <string>
#include <vector>
#include <stdexcept>
#include <map>
#include <list>
#include <set>
#include <queue>
#include <bitset>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <complex>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <sstream>
#include <sys/time.h>
#include <x86intrin.h>
#include <string.h>
#include "DeltaComp.h"
#include "BitHandler.h"

//#include <gmp.h>

using namespace std;


void DeltaComp::Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize, size_t *maxsize) {
  mTranspose.resize(chunk.mData.size());
  size_t count = 0;
  size_t rowEnd = chunk.mRowStart + chunk.mHeight;
  size_t colEnd = chunk.mColStart + chunk.mWidth;
  size_t frameEnd = chunk.mFrameStart + chunk.mDepth;
  for (size_t row = chunk.mRowStart; row < rowEnd; row++) {
    for (size_t col = chunk.mColStart; col < colEnd; col++) {
      for (size_t frame = chunk.mFrameStart; frame < frameEnd; frame++) {
        mTranspose[count++] = chunk.At(row, col, frame);
      }
    }
  }

  compress(mTranspose, chunk.mHeight, chunk.mWidth, chunk.mDepth, mCompressed);
  if (mCompressed.size() > *maxsize) { ReallocBuffer(mCompressed.size(), compressed, maxsize); }  
  //  (*compressed) =  (int8_t*) malloc(sizeof(int8_t) * mCompressed.size());
  *outsize = mCompressed.size();
  memcpy(*compressed, &mCompressed[0], *outsize);
}

void DeltaComp::Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size) {
  mCompressed.resize(size);
  memcpy(&mCompressed[0], compressed, size);
  size_t rows, cols, frames;
  decompress(mCompressed, rows, cols, frames, mTranspose);
  ION_ASSERT(rows == chunk.mHeight && cols == chunk.mWidth && frames == chunk.mDepth, "Dimensions don't match.");
  size_t rowEnd = chunk.mRowStart + chunk.mHeight;
  size_t colEnd = chunk.mColStart + chunk.mWidth;
  size_t frameEnd = chunk.mFrameStart + chunk.mDepth;
  size_t count = 0;
  chunk.mData.resize(rows * cols * frames);
  for (size_t row = chunk.mRowStart; row < rowEnd; row++) {
    for (size_t col = chunk.mColStart; col < colEnd; col++) {
      for (size_t frame = chunk.mFrameStart; frame < frameEnd; frame++) {
        chunk.At(row, col, frame) = mTranspose[count++];
      }
    }
  }
}

void DeltaComp::compress(const std::vector<uint16_t>& data, size_t nRows, size_t nCols, size_t nFrames, std::vector<uint8_t> &compressed) {
  mCompressor.compress(data, nRows, nCols, nFrames, compressed);
}

void DeltaComp::decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols, size_t &nFrames, std::vector<uint16_t> & data) {
  DecompressorNH p;
  p.decompress(compressed, nRows, nCols, nFrames, data);
}
