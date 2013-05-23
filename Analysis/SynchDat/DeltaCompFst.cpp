/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

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
#include <arpa/inet.h>
#include "DeltaCompFst.h"
#include "BitHandler.h"
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
using namespace std;

void DeltaCompFst::decompress(const int8_t *compressed, size_t size, TraceChunk &chunk) {
    // @todo - neorder these bits                                                  
  int *header = (int *)(compressed);
  unsigned X = ntohl(header[0]);
  unsigned Y = ntohl(header[1]);
  unsigned L = ntohl(header[2]);
  size_t S = X*Y;

  size_t current = 12;
  size_t end = X*Y*(L+1)+12;
  chunk.mData.resize(S*L);
  int16_t* p = &chunk.mData[0];
  const int16_t *eptr = (const int16_t *)(&compressed[end]);
  // Pull off the first frame
  const int8_t *s = &compressed[0] + current;
  size_t i = 0;
  while(likely(i++ < S)) {
    *p = (uint8_t)(*s++);
    *p++ += (uint16_t) (*s) << 8;
    s++;
  }
  i = 1;
  while(likely(i++ < L)) {
    int16_t *pS = p - S;
    size_t j = 0;
    while(likely(j++ < S)) {
      if (*s == 127) {
  	*(p++) = ntohs(*eptr++) + *(pS++);
	s++;
      }
      else {
  	*(p++) = *(s++) + *(pS++);
      }
    }
  }
}

void DeltaCompFst::compress(TraceChunk &chunk, size_t nRows, size_t nCols, 
                            size_t nFrames, int8_t **compressed, size_t *outsize, size_t *maxSize) {
  size_t S = nRows * nCols;
  size_t L = nFrames;

  // Reallocate memory if needed
  size_t msize = sizeof(int8_t) * S * (L+1) * 3 + 12;
  if (msize > *maxSize) { ReallocBuffer(msize, compressed, maxSize); }

  // Write header about dimensions in network order bytes
  int *header = (int *)(*compressed);
  header[0] = htonl(nRows);
  header[1] = htonl(nCols);
  header[2] = htonl(L);
  int8_t *c = *compressed + 12;
  // End of byte sized deltas where we'll put things that don't compress into a byte
  int16_t *end = (int16_t *)((*compressed + S * (L+1) + 12));
  int16_t *endstart = end;

  // First frame as full 16 bits
  const int16_t* p = &chunk.mData[0];
  size_t i = 0;
  while (likely(i++ < S)) {
    *c++ = *p;
    *c++ = *p++ >> 8;
  }

  i = 1;
  while(likely(i++ < L)) { 
    size_t j = 0;
    const int16_t *pS = p - S;
    while (likely(j++ < S)) {
    //    for (size_t j = 0; j < S; j++) { // for each well
      int16_t d = *p++ - *pS++;
      if (d >= 127 || d <= -127) {
	*end++ = htons(d);
	d = 127;
      }
      *c++ = d;
    }
  }
  *outsize = (c - *compressed) + sizeof(int16_t) * (end - endstart);
}


void DeltaCompFst::Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize, size_t *maxsize) {
  compress(chunk, chunk.mHeight, chunk.mWidth, chunk.mDepth, compressed, outsize, maxsize);
}

void DeltaCompFst::Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size) {
  decompress(compressed, size, chunk);
}

