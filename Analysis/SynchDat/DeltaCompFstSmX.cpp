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
#include "DeltaCompFstSmX.h"
#include "SampleQuantiles.h"
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
// #define WELLS_COMPACTED 16
#define unlikely(x)     __builtin_expect((x),0)
union CompI {
  v4si v;
  int vv[DC_STEP_SIZE];
};


void DeltaCompFstSmX::Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize, size_t *maxsize) {
  compress(chunk, chunk.mHeight, chunk.mWidth, chunk.mDepth, compressed, outsize, maxsize);
}

void DeltaCompFstSmX::Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size) {
  decompress(compressed, size, chunk);
}

void DeltaCompFstSmX::compress(TraceChunk &chunk, size_t nRows, size_t nCols, 
                            size_t nFrames, int8_t **compressed, size_t *outsize, size_t *maxsize) {
  unsigned X = nRows, Y = nCols;
  size_t S = X*Y; // framestep
  size_t L = nFrames;
  if (S * L * 3 > *maxsize) { ReallocBuffer(S*3*L, compressed, maxsize); }
  int *header = (int *)(*compressed);
  uint8_t *ucompressed = (uint8_t *)(*compressed);
  header[0] = htonl(X);
  header[1] = htonl(Y);
  header[2] = htonl(L);
  int needMult = ceil(mWellsCompacted / DC_STEP_SIZE);
  size_t current = 12;  
  ucompressed[current++] = mWellsCompacted;
  int maxOffset = 62;
  int minOffset = -62;
  CompI cur[needMult];
  CompI last[needMult];
  CompI delta[needMult];
  int m4 = 7, m6 = 31, m8 = 127;
  CompI zero, offset4;
  for (size_t i = 0; i < DC_STEP_SIZE; i++) {
    zero.vv[i] = 0;
  }
  int16_t *p = &chunk.mData[0];
  for (size_t i = 0; i < S; i++) {
    int16_t v = *p++;
    ucompressed[current++] = v;
    ucompressed[current++] = v >> 8;
  }
  int vals[mWellsCompacted];
  int16_t* pTemp = NULL;
  // Subsequent frames delta compressed. 8 at a time with lowest delta encoding 00=2 bits, 01=4 bits, 10=8 bits 11 = uncompressed (full 14);
  for (size_t fIx = 1; fIx < chunk.mDepth; fIx++) {
    for (size_t wIx = 0; wIx < S; wIx+=mWellsCompacted) {
      // First loop through and figure out the minimum encoding..
      int offset = 0;
      int minV = 0;
      size_t maxCur = min((size_t)mWellsCompacted, S - wIx);
      size_t cIx = 0;
      for (size_t curIx = 0; curIx < maxCur; curIx += DC_STEP_SIZE, cIx++) {
        size_t next = std::min((size_t)DC_STEP_SIZE, maxCur - curIx);
        cur[cIx].v = zero.v;
        last[cIx].v = zero.v;
        for (size_t i = 0; i < next; i++) {
          pTemp = p + i;
          cur[cIx].vv[i] = *(pTemp);
          last[cIx].vv[i] = *(pTemp - S);
        }
        p += next;
        delta[cIx].v = SUB_PACKED_INT(cur[cIx].v, last[cIx].v);
        if (curIx == 0) {
          // average of first 4 values as offset
          offset = delta[cIx].vv[0] + delta[cIx].vv[1] + delta[cIx].vv[2] + delta[cIx].vv[3];
          offset = offset >> 2; // divide by 4
          // limit to fit in 6 bits;
          offset = offset > maxOffset ? maxOffset : offset;
          offset = offset < minOffset ? minOffset : offset;
          // limit to even number
          offset &= ~0x1;
          offset4.vv[0] = offset4.vv[1] = offset4.vv[2] = offset4.vv[3] = offset;
        }
        delta[cIx].v = SUB_PACKED_INT(delta[cIx].v, offset4.v);
        // Figure out largest difference for this block of wells
        for (size_t i = 0; i < next; i++) {
          size_t x = curIx + i;
          vals[x] = delta[cIx].vv[i];
          minV = max(minV, abs(vals[x]));
        }
      }
    
      // Encode the max difference
      uint8_t c = 0;
      if (minV <= m4)
        c = 0;
      else if (minV <= m6)
        c = 1;
      else if (minV <= m8)
        c = 2;
      else 
        c = 3;

      // Encode number of bits per well and the offset for this block
      uint8_t v = c << 6 | ((offset + maxOffset) >> 1);
      ucompressed[current++] = v;
      // here we are storing the entire packet of mWellsCompacted, but only the maxCur first entries are actually valid
      // Pack into 4, 6, 8 or 16 bits per well.
      switch (c) 
        {
        case (0) :
          // put each val in a nibble (4 bits)
          for (size_t i = 0; i < mWellsCompacted; i+=2) {
            ucompressed[current++] = (vals[i] + 7) << 4 | (vals[i+1] + 7); // all of information should be in bottom 4 bits
          }
          break;
        case (1) :
          // 6 bytes per value
          for (size_t i = 0; i < mWellsCompacted; i+=4) {
            vals[i] += 31;
            vals[i+1] += 31;
            vals[i+2] += 31;
            vals[i+3] += 31;
            ucompressed[current++] = (vals[i]) << 2 | (vals[i+1]) >> 4; // 6 bits of first one and 2 bits of second
            ucompressed[current++] = (vals[i+1]) << 4 | (((vals[i+2]) >> 2) & 0xF); // 4 bits of second one and 4 bits of third
            ucompressed[current++] = (vals[i+2]) << 6 | (vals[i+3]); // 2 bits of third one and 6 bits of second
          }
          break;
        case (2) :
          // 1 byte per value
          for (size_t i = 0; i < mWellsCompacted; i++) {
            ucompressed[current++] = vals[i] + 127; // all of information in 8 bytes
          }
          break;
        default:
          // 2 bytes per value
          for (size_t i = 0; i < mWellsCompacted; i++) {
            ucompressed[current++] = cur[i/DC_STEP_SIZE].vv[i%DC_STEP_SIZE];
            ucompressed[current++] = cur[i/DC_STEP_SIZE].vv[i%DC_STEP_SIZE] >> 8;
          }
        }
    }
  }
  *outsize = current;
}

// void DeltaCompFstSmX::compress(TraceChunk &chunk, size_t nRows, size_t nCols, 
//                             size_t nFrames, int8_t **compressed, size_t *outsize, size_t *maxsize) {
//   unsigned X = nRows, Y = nCols;
//   size_t S = X*Y; // framestep
//   size_t L = nFrames;
//   if (S * L * 3 > *maxsize) { ReallocBuffer(S*3*L, compressed, maxsize); }
//   int *header = (int *)(*compressed);
//   uint8_t *ucompressed = (uint8_t *)(*compressed);
//   header[0] = htonl(X);
//   header[1] = htonl(Y);
//   header[2] = htonl(L);
//   int needMult = ceil(mWellsCompacted / DC_STEP_SIZE);
//   size_t current = 12;  
//   ucompressed[current++] = mWellsCompacted;
//   int maxOffset = 62;
//   int minOffset = -62;
//   CompI cur[needMult];
//   CompI last[needMult];
//   CompI delta[needMult];
//   int m4 = 7, m6 = 31, m8 = 127;
//   CompI zero, offset4;
//   for (size_t i = 0; i < DC_STEP_SIZE; i++) {
//     zero.vv[i] = 0;
//   }
//   int16_t *p = &chunk.mData[0];
//   for (size_t i = 0; i < S; i++) {
//     int16_t v = *p++;
//     ucompressed[current++] = v;
//     ucompressed[current++] = v >> 8;
//   }
//   int vals[mWellsCompacted];
//   int16_t* pTemp = NULL;
//   // Subsequent frames delta compressed. 8 at a time with lowest delta encoding 00=2 bits, 01=4 bits, 10=8 bits 11 = uncompressed (full 14);
//   for (size_t fIx = 1; fIx < chunk.mDepth; fIx++) {
//     for (size_t wIx = 0; wIx < S; wIx+=mWellsCompacted) {
//       // First loop through and figure out the minimum encoding..
//       int offset = 0;
//       int minV = 0;
//       size_t maxCur = min((size_t)mWellsCompacted, S - wIx);
//       size_t cIx = 0;
//       for (size_t curIx = 0; curIx < maxCur; curIx += DC_STEP_SIZE, cIx++) {
//         size_t next = std::min((size_t)DC_STEP_SIZE, maxCur - curIx);
//         cur[cIx].v = zero.v;
//         last[cIx].v = zero.v;
//         for (size_t i = 0; i < next; i++) {
//           pTemp = p + i;
//           cur[cIx].vv[i] = *(pTemp);
//           last[cIx].vv[i] = *(pTemp - S);
//         }
//         p += next;
//         delta[cIx].v = SUB_PACKED_INT(cur[cIx].v, last[cIx].v);
//         if (curIx == 0) {
//           // average of first 4 values as offset
//           offset = delta[cIx].vv[0] + delta[cIx].vv[1] + delta[cIx].vv[2] + delta[cIx].vv[3];
//           offset = offset >> 2; // divide by 4
//           // limit to fit in 6 bits;
//           offset = offset > maxOffset ? maxOffset : offset;
//           offset = offset < minOffset ? minOffset : offset;
//           // limit to even number
//           offset &= ~0x1;
//           offset4.vv[0] = offset4.vv[1] = offset4.vv[2] = offset4.vv[3] = offset;
//         }
//         delta[cIx].v = SUB_PACKED_INT(delta[cIx].v, offset4.v);
//         // Figure out largest difference for this block of wells
//         for (size_t i = 0; i < next; i++) {
//           size_t x = curIx + i;
//           vals[x] = delta[cIx].vv[i];
//           minV = max(minV, abs(vals[x]));
//         }
//       }
    
//       // Encode the max difference
//       uint8_t c = 0;
//       if (minV <= m4)
//         c = 0;
//       else if (minV <= m6)
//         c = 1;
//       else if (minV <= m8)
//         c = 2;
//       else 
//         c = 3;

//       // Encode number of bits per well and the offset for this block
//       uint8_t v = c << 6 | ((offset + maxOffset) >> 1);
//       ucompressed[current++] = v;
//       // here we are storing the entire packet of mWellsCompacted, but only the maxCur first entries are actually valid
//       // Pack into 4, 6, 8 or 16 bits per well.
//       switch (c) 
//         {
//         case (0) :
//           // put each val in a nibble (4 bits)
//           for (size_t i = 0; i < mWellsCompacted; i+=2) {
//             ucompressed[current++] = (vals[i] + 7) << 4 | (vals[i+1] + 7); // all of information should be in bottom 4 bits
//           }
//           break;
//         case (1) :
//           // 6 bytes per value
//           for (size_t i = 0; i < mWellsCompacted; i+=4) {
//             vals[i] += 31;
//             vals[i+1] += 31;
//             vals[i+2] += 31;
//             vals[i+3] += 31;
//             ucompressed[current++] = (vals[i]) << 2 | (vals[i+1]) >> 4; // 6 bits of first one and 2 bits of second
//             ucompressed[current++] = (vals[i+1]) << 4 | (((vals[i+2]) >> 2) & 0xF); // 4 bits of second one and 4 bits of third
//             ucompressed[current++] = (vals[i+2]) << 6 | (vals[i+3]); // 2 bits of third one and 6 bits of second
//           }
//           break;
//         case (2) :
//           // 1 byte per value
//           for (size_t i = 0; i < mWellsCompacted; i++) {
//             ucompressed[current++] = vals[i] + 127; // all of information in 8 bytes
//           }
//           break;
//         default:
//           // 2 bytes per value
//           for (size_t i = 0; i < mWellsCompacted; i++) {
//             ucompressed[current++] = cur[i/DC_STEP_SIZE].vv[i%DC_STEP_SIZE];
//             ucompressed[current++] = cur[i/DC_STEP_SIZE].vv[i%DC_STEP_SIZE] >> 8;
//           }
//         }
//     }
//   }
//   *outsize = current;
// }




void DeltaCompFstSmX::decompress(const int8_t *compressed, size_t size, TraceChunk &chunk) {
  int *header = (int *)(compressed);
  unsigned X = ntohl(header[0]);
  unsigned Y = ntohl(header[1]);
  unsigned L = ntohl(header[2]);
  ION_ASSERT(chunk.mDepth == L && X == chunk.mHeight && Y == chunk.mWidth, "Blocks don't match.");
  size_t S = X*Y;
  uint8_t *ucompressed = (uint8_t *)compressed;
  size_t current = 12;
  int wellsCompacted = ucompressed[current++];
  int maxOffset = 62;
  // First frame;
  int16_t* p = &chunk.mData[0];
  for(size_t i = 0; i < S; i++) {
    uint16_t v = ucompressed[current++];
    uint16_t u = ucompressed[current++];
    p[i] = v + (u << 8 );
  }
  p += S;
  int vals[wellsCompacted];
  // Subsequent frames delta compressed. 8 at a time with lowest delta encoding 01=4 bits, 10=6 bits, 10=8 bits 11 = uncompressed (full 14);
  for (size_t fIx = 1; fIx < chunk.mDepth; fIx++) {
    for (size_t wIx = 0; wIx < S; wIx+=mWellsCompacted) {
      size_t maxCur = min((size_t)mWellsCompacted, S - wIx);
      uint8_t x = ucompressed[current++];
      uint8_t c = x >> 6;
      int offset = x & 0x3F;
      offset = (offset << 1) - maxOffset;
      int16_t* pp = p;
      switch (c) {
      case (0) :
        // put each val in a nibble (4 bits)
        offset -= 7;
        pp = p - S;
        for (int i = 0; i < wellsCompacted; i+=2) {
          vals[i] = (ucompressed[current] >> 4) + pp[i] + offset;
          vals[i+1] = (ucompressed[current] & 0x000F) + pp[i+1] + offset;
          current++;
        }
        break;
      case (1) :
        // 6 bits per value
        offset -= 31;
        pp = p - S;
        for (int i = 0; i < wellsCompacted; i+=4) {
          vals[i]   = (ucompressed[current] >> 2) + offset + pp[0];
          vals[i+1] = ((ucompressed[current] & 0x3) << 4 | ucompressed[current+1] >> 4) + offset + pp[1];
          vals[i+2] = (((ucompressed[current+1] << 2) & 0x3F) | ucompressed[current+2] >> 6) + offset + pp[2];
          vals[i+3] = (ucompressed[current+2] & 0x3F) + offset + pp[3];
          current+=3;
          pp += 4;
        }
        break;
      case (2) :
        offset -= 127;
        pp = p-S;
        // 8 bits, 1 bypte per value
        for (int i = 0; i < wellsCompacted; i++) {
          vals[i] = ucompressed[current++] + offset + pp[i];
        }
        break;
      default:
        // uncompressed full 16 bits, 2 words each.
        for (int i = 0; i < wellsCompacted; i++) {
          vals[i] =  (ucompressed[current+1] << 8) | ucompressed[current];
          current += 2;
        }
      }
      for (size_t i = 0; i < maxCur; i++) {
        *p++ = vals[i];
      }
    }
  }
}

