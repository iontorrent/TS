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
#include "SvdDatCompressPlus.h"
#include "DelicatoCompression.h"
#include "BitHandler.h"
#include "HuffmanEncode.h"
#include "SampleStats.h"

//#include <gmp.h>

using namespace std; using namespace __gnu_cxx; typedef vector<string> VS;
typedef long long i64;typedef unsigned long long u64;typedef unsigned char u8;
typedef unsigned u32; typedef unsigned short u16; typedef signed char byte;
//typedef vector<int> VI; typedef vector<VI> VVI; typedef vector<double> VD;
typedef vector<uint16_t> V16; typedef vector<int> VI; typedef vector<VI> VVI; typedef vector<double> VD;

void SvdDatCompressPlus::Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize) {
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
  (*compressed) = new int8_t [mCompressed.size()];
  *outsize = mCompressed.size();
  memcpy(*compressed, &mCompressed[0], *outsize);
}

void SvdDatCompressPlus::Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size) {
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

//using namespace SNSPLUS;

void SvdDatCompressPlus::compress(const V16& data, size_t nRows, size_t nCols, size_t nFrames, std::vector<uint8_t> &compressed) {
  std::vector<uint8_t> tmp;
  DelicatoCompression dc = DelicatoCompression();
  dc.compress(data, nRows, nCols, nFrames, tmp);
  mCompressed.clear(); 
  BitPacker bc;//(mCompressed);
  bc.put_u32((uint32_t)tmp.size());
  bc.put_compressed(&tmp[0], tmp.size());
  bc.flush();
  mCompressed = bc.get_data();
  cout << "Bytes per well: " << mCompressed.size() / ((float)nRows * nCols) << " Compression ratio: " << data.size() * 2.0 / mCompressed.size() << endl;
}

void SvdDatCompressPlus::decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols, size_t &nFrames, V16& data) {
  std::vector<uint8_t> tmp;
  BitUnpacker bc(compressed);
  uint32_t size  = bc.get_u32();
  tmp.resize(size);
  bc.get_compressed(&tmp[0], size);
  DelicatoCompression dc = DelicatoCompression();
  dc.decompress(tmp, data, nRows, nCols, nFrames);
  //  cout << "Compression ratio: " << data.size() * 2.0 / compressed.size() << endl;

}


void SvdDatCompressPlusPlus::compress(const vector<uint16_t>& data, size_t nRows, size_t nCols, size_t nFrames, std::vector<uint8_t> &compressed) {
  std::vector<uint8_t> tmp;
  DelicatoCompression dc = DelicatoCompression();
  dc.SetDoBoth(true);
  //dc.compress(data, nRows, nCols, nFrames, tmp, errCon);
  dc.compress(data, nRows, nCols, nFrames, tmp, errCon,rankGood,pivot);
  mCompressed.clear(); 
  //BitPacker bc(mCompressed);
  BitPacker bc;
  bc.put_u32((uint32_t)tmp.size());
  bc.put_compressed(&tmp[0], tmp.size());
  bc.flush();
  mCompressed = bc.get_data();
  cout << "Doing svd++ Bytes per well: " << mCompressed.size() / ((float)nRows * nCols) << " Compression ratio: " << data.size() * 2.0 / mCompressed.size() << endl;
}

void SvdDatCompressPlusPlus::decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols, size_t &nFrames, vector<uint16_t>& data) {
  std::vector<uint8_t> tmp;
  BitUnpacker bc(compressed);
  uint32_t size  = bc.get_u32();
  tmp.resize(size);
  bc.get_compressed(&tmp[0], size);
  DelicatoCompression dc = DelicatoCompression();
  dc.SetDoBoth(true);
  dc.decompress(tmp, data, nRows, nCols, nFrames);
}

void SvdDatCompressPlusPlus::Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize) {
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
  (*compressed) = new int8_t [mCompressed.size()];
  *outsize = mCompressed.size();
  memcpy(*compressed, &mCompressed[0], *outsize);
}

void SvdDatCompressPlusPlus::Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size) {
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
