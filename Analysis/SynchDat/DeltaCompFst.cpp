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
#include <emmintrin.h>
#include <string.h>
#include <arpa/inet.h>
#include "DeltaCompFst.h"
#include "BitHandler.h"

using namespace std;

#ifdef __INTEL_COMPILER
#define v4si __m128i
#define ADD_PACKED_INT( x, y) _mm_add_epi32( (x), (y));
#define SUB_PACKED_INT( x, y) _mm_sub_epi32( (x), (y));
#else
typedef int v4si __attribute__ ( (vector_size (sizeof (int) *4)));
#define ADD_PACKED_INT( x, y) ( (x) + (y) );
#define SUB_PACKED_INT( x, y) ( (x) - (y) );
#endif // __INTEL_COMPILER

#define DC_STEP_SIZE 4
#define DC_STEP_SIZEU 4ul
#define WELLS_COMPACTED 16

union CompI {
  v4si v;
  int vv[DC_STEP_SIZE];
};

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
    uint16_t* p = &chunk.mData[0];
    const int16_t *eptr = (const int16_t *)(&compressed[end]);
    const int8_t *sptr = &compressed[0] + current;
    size_t sIdx = 0;
    size_t eIdx = 0;
    // Pull off the first frame
    for(size_t i = 0; i < S; i++) {
      uint16_t v = (uint8_t)(sptr[sIdx++]);
      uint16_t u = (uint8_t)(sptr[sIdx++]);
      *p++ = v + (u << 8 );
    }
    CompI cur;
    CompI last;
    for (size_t i = 1 ; i < L; i++) {
      for (size_t j = 0; j < S; j+=DC_STEP_SIZE) {
        size_t next = std::min(DC_STEP_SIZEU, S-j);
        for (size_t x = 0; x < next; x++) {
          cur.vv[x] = (int8_t)sptr[sIdx+x];
          last.vv[x] = *(p-S+x);
        }
        sIdx += next;
        for (size_t x = 0; x < next; x++) {
          if(cur.vv[x] == 127) {
            cur.vv[x] = ntohs(eptr[eIdx++]);
          }
        }
	cur.v = ADD_PACKED_INT(last.v, cur.v);
        for (size_t x = 0; x < next; x++) {
          *(p+x) = cur.vv[x];
        }
        p += next;
      }
    }
}

// Memory is organized by frames
void DeltaCompFst::compress(TraceChunk &chunk, size_t nRows, size_t nCols, 
                            size_t nFrames, int8_t **compressed, size_t *outsize, size_t *maxSize) {
  unsigned X = nRows, Y = nCols;
  size_t S = X*Y; // framestep
  size_t L = nFrames;

  // Reallocate memory if needed
  size_t msize = sizeof(int8_t) * S * (L+1) * 3 + 12;
  if (msize > *maxSize) { ReallocBuffer(msize, compressed, maxSize); }

  // Write header about dimensions in network order bytes
  int *header = (int *)(*compressed);
  header[0] = htonl(X);
  header[1] = htonl(Y);
  header[2] = htonl(L);
  size_t current = 12;

  // End of byte sized deltas where we'll put things that don't compress into a byte
  int16_t *end = (int16_t *)((*compressed + S * (L+1) + 12));
  size_t eIx = 0;

  // First frame as full 16 bits
  const uint16_t* p = &chunk.mData[0];
  for (size_t i = 0; i < S; i++) {
    int16_t v = *p++;
    (*compressed)[current++] = v;
    (*compressed)[current++] = v >> 8;
  }

  // Delta for all the other frames
  CompI cur;
  CompI last;
  CompI zero;
  for (size_t i = 0; i < DC_STEP_SIZEU; i++) {
    zero.vv[i] = 0.0f;
  }
  for (size_t i = 1 ; i < L; i++) {
    for (size_t j = 0; j < S; j+=DC_STEP_SIZE) {
      cur.v = zero.v;
      last.v = zero.v;
      size_t next = std::min(DC_STEP_SIZEU, S-j);
      const uint16_t *pS = p - S;
      for (size_t x = 0; x < next; x++) {
        cur.vv[x] = *(p + x);
        last.vv[x] = *(pS + x);
      }
      p += next;
      cur.v = SUB_PACKED_INT( cur.v, last.v);
      for (size_t x= 0; x < next; x++) {
        // If doesn't fit in a byte store it at the end as a short.
        if (abs(cur.vv[x]) >= 127) {
          int16_t v = cur.vv[x];
          end[eIx++] = htons(v); // network order for shorts
          cur.vv[x] = 127;
        }
        (*compressed)[current++] = cur.vv[x];
      }
    }
  }
  *outsize = current + sizeof(int16_t) * eIx;
}


void DeltaCompFst::Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize, size_t *maxsize) {
  compress(chunk, chunk.mHeight, chunk.mWidth, chunk.mDepth, compressed, outsize, maxsize);
}

void DeltaCompFst::Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size) {
  decompress(compressed, size, chunk);
}

