/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/**
 * unpackMatrix.cpp
 * Exported interface: unpackMatrix
 * This is the file that contains all linear algebra on the decompression side
 * 
 * @author Magnus Jedvert
 * @version 1.1 april 2012
 */

#include <armadillo>
#include <vector>
#include <cassert>
#include "matrixRounding.h"
#include "unpackMatrix.h"
#include "VencoLossless.h"
#include "BitHandler.h"
//#include "ByteHandler.h"

using namespace std;
using namespace arma;
//using namespace BPacker;

// typedef uint8_t u8;  // defined in arma for some reason
// typedef uint16_t u16;
typedef int16_t i16;
typedef int32_t i32;

/**
 * unpackMatrix - Reverse of packMatrix. Stores the matrix in memory pointed
 * to by dst. Pops the information from src.
 */
void unpackMatrix(ByteUnpacker &src, u16 *dst, size_t L) {
  // extract key numbers: 
  const vector<size_t> header = src.pop<size_t>(5);   
  const size_t nGood = header[0];
  const size_t nBad = header[1];
  const size_t RANK_GOOD = header[2];
  const size_t rankBad = header[3];
  const bool uses16Bit = header[4];
  const int N = nGood + nBad; 
  //    assert(rankBad == L);

  // dont copy the memory, use it directly instead:
#define POP_CONSTRUCTOR(T, H, W) src.popPtr<T>(H*W), H, W, false, true

  // extract basis:
  const fmat basis( POP_CONSTRUCTOR(float, L, rankBad) );
  const fmat basisGood = basis.cols(0, RANK_GOOD-1);
  // extract partion:
  const Col<u8> partion( src.popPtr<u8>(N), N, false, true );
  const uvec goodIdx = find(partion == 0);
  const uvec badIdx = find(partion == 1);
  assert( goodIdx.n_elem == nGood && badIdx.n_elem == nBad);

  // make a matrix using the memory pointed to by dst:
  Mat<uint16_t> restored(dst, L, N, false, true);
    
  if ( uses16Bit ) {
    // extract scores:
    const Mat<i16> scoreGood( POP_CONSTRUCTOR(i16, nGood, RANK_GOOD) );
    const Mat<i16> scoreBad( POP_CONSTRUCTOR(i16, nBad, rankBad) );
    // fill this matrix with the matrix multiplications:
    //restored.cols(goodIdx) = convToValidU16( basisGood * convToFloat(scoreGood).t() );
    //restored.cols(badIdx) = convToValidU16( basis * convToFloat(scoreBad).t() );
    restored.cols(goodIdx) = convToValidU16( basisGood * scoreGood.t() );
    restored.cols(badIdx) = convToValidU16( basis * scoreBad.t() );
  }
  else {
    // extract scores:
    const Mat<i32> scoreGood( POP_CONSTRUCTOR(i32, nGood, RANK_GOOD) );
    const Mat<i32> scoreBad( POP_CONSTRUCTOR(i32, nBad, rankBad) );

    // fill this matrix with the matrix multiplications:
    //restored.cols(goodIdx) = convToValidU16( basisGood * conv_to<fmat>::from(scoreGood).t() );
    //restored.cols(badIdx) = convToValidU16( basis * conv_to<fmat>::from(scoreBad).t() );

    restored.cols(goodIdx) = convToValidU16( basisGood * scoreGood.t() );
    restored.cols(badIdx) = convToValidU16( basis * scoreBad.t() );
  }
    Mat<uint16_t> gtmp = restored.cols(goodIdx);
    gtmp.save("svdp_decomGood.txt",raw_ascii);

    Mat<uint16_t> btmp = restored.cols(badIdx);
    btmp.save("svdp_decomBad.txt",raw_ascii);

}
#define POP_CONSTRUCTOR(T, H, W) src.popPtr<T>(H*W), H, W, false, true  

void unpackMatrixPlus(ByteUnpacker &src, u16 *dst, size_t L) {
  // extract key numbers: 
  const vector<size_t> header = src.pop<size_t>(5);   
  const size_t nGood = header[0];
  const size_t nBad = header[1];
  const size_t RANK_GOOD = header[2];
  //const size_t rankBad = header[3];
  const bool uses16Bit = header[4];
  const int N = nGood + nBad; 
  //assert(rankBad == L);

  // extract partion:
  const Col<u8> partion( src.popPtr<u8>(N), N, false, true );
  const uvec goodIdx = find(partion == 0);
  const uvec badIdx = find(partion == 1);
  assert( goodIdx.n_elem == nGood && badIdx.n_elem == nBad);
    
  // make a matrix using the memory pointed to by dst:
  Mat<u16> restored(dst, L, N, false, true);

  //decompress the good traces:
  if ( nGood > 0) {
    // extract basis:
    vector<size_t> gSize = src.pop<size_t>(2);
    vector<uint8_t> gHuffOut = src.pop<uint8_t>(gSize[0]);
    BitUnpacker gbitPacker(gHuffOut);
    vector<uint8_t> tmpGood;
    tmpGood.resize(gSize[1]);
    gbitPacker.get_compressed(&tmpGood[0],gSize[1]);
    ByteUnpacker gbp((char*)tmpGood.data());
    const Mat<float> basisGood( gbp.popPtr<float>( L * RANK_GOOD), L, RANK_GOOD, false, true);
    if ( uses16Bit ) {
      // extract scores:
      const Mat<int16_t> scoreGood( gbp.popPtr<int16_t>( nGood * RANK_GOOD), nGood, RANK_GOOD, false, true);
      // fill this matrix with the matrix multiplications:
      restored.cols(goodIdx) = convToValidU16( basisGood * convToFloat(scoreGood).t() );
      //restored.cols(goodIdx) = convToValidU16( basisGood * scoreGood.t() );
    }
    else {
      // extract scores:
      const Mat<int32_t> scoreGood( gbp.popPtr<int32_t>(nGood * RANK_GOOD), nGood, RANK_GOOD, false, true);
      // fill this matrix with the matrix multiplications:
      restored.cols(goodIdx) = convToValidU16( basisGood * conv_to<fmat>::from(scoreGood).t() );
      //restored.cols(goodIdx) = convToValidU16( basisGood * scoreGood.t() );
    }

  }
  //Decompress the bad data, output the uncompressed to restored.cols(badIdx);
  if ( nBad > 0)
    {
      vector<size_t> sSize = src.pop<size_t>(1);
      const vector<uint8_t> deltaIn = src.pop<uint8_t>(sSize[0]);
      Decompressor dc;
      size_t nCols;
      size_t nRows;
      size_t oL;
      vector<uint16_t> deltaOut;
      dc.decompress(deltaIn,nRows,nCols, oL, deltaOut);
      Mat<uint16_t> dataBadMat(&deltaOut[0], L, nBad, false, true);
      restored.cols(badIdx) = dataBadMat;
    }
}

/* unpackMatrixPlusV2 - */

void unpackMatrixPlusV2(ByteUnpacker &src, u16 *dst, size_t L) {
  // extract key numbers: 
  const vector<size_t> header = src.pop<size_t>(5);   
  const size_t nGood = header[0];
  const size_t nBad = header[1];
  const size_t RANK_GOOD = header[2];
  //const size_t rankBad = header[3];
  const bool uses16Bit = header[4];
  const int N = nGood + nBad; 
  //assert(rankBad == L);

  // extract partion:
  const Col<u8> partion( src.popPtr<u8>(N), N, false, true );
  const uvec goodIdx = find(partion == 0);
  const uvec badIdx = find(partion == 1);
  assert( goodIdx.n_elem == nGood && badIdx.n_elem == nBad);
    
  // make a matrix using the memory pointed to by dst:
  Mat<u16> restored(dst, L, N, false, true);

  //decompress the good traces:
  if ( nGood > 0) {
    // extract basis:
    const fmat basisGood(POP_CONSTRUCTOR(float, L, RANK_GOOD));
    if ( uses16Bit ) {
      // extract scores:
      const Mat<int16_t> scoreGood( POP_CONSTRUCTOR(i16, nGood,RANK_GOOD));
      restored.cols(goodIdx) = convToValidU16( basisGood * scoreGood.t() );
    }
    else {
      // extract scores:
      const Mat<int32_t> scoreGood( POP_CONSTRUCTOR(i32, nGood,RANK_GOOD));
      restored.cols(goodIdx) = convToValidU16( basisGood * scoreGood.t() );
      //restored.cols(goodIdx) = convToValidU16( basisGood * conv_to<fmat>::from(scoreGood).t() );
    }
  }
  //Decompress the bad data, output the uncompressed to restored.cols(badIdx);
  if ( nBad > 0)
    {
      vector<size_t> sSize = src.pop<size_t>(1);
      const vector<uint8_t> deltaIn = src.pop<uint8_t>(sSize[0]);
      Decompressor dc;
      size_t nCols;
      size_t nRows;
      size_t oL;
      vector<uint16_t> deltaOut;
      dc.byteDecompress(deltaIn,nRows,nCols, oL, deltaOut);
      Mat<uint16_t> dataBadMat(&deltaOut[0], L, nBad, false, true);
      restored.cols(badIdx) = dataBadMat;
    }
}

void unpackMatrixPlusV3(ByteUnpacker &src, u16 *dst, size_t L) {
  // extract key numbers: 
  const vector<size_t> header = src.pop<size_t>(5);   
  const size_t nGood = header[0];
  const size_t nBad = header[1];
  const size_t RANK_GOOD = header[2];
  //const size_t rankBad = header[3];
  const bool uses16Bit = header[4];
  const int N = nGood + nBad; 
  //assert(rankBad == L);
  const fmat basisGood( POP_CONSTRUCTOR(float, L, RANK_GOOD) );
  // extract partion:
  const Col<u8> partion( src.popPtr<u8>(N), N, false, true );
  const uvec goodIdx = find(partion == 0);
  const uvec badIdx = find(partion == 1);
  assert( goodIdx.n_elem == nGood && badIdx.n_elem == nBad);
    
  // make a matrix using the memory pointed to by dst:
  Mat<u16> restored(dst, L, N, false, true);

  //decompress the good traces:
  if ( nGood > 0) {
    if ( uses16Bit ) {
      // extract scores:
      const Mat<i16> scoreGood( POP_CONSTRUCTOR(i16, nGood, RANK_GOOD) );
      //restored.cols(goodIdx) = convToValidU16( basisGood * convToFloat(scoreGood).t() );
      restored.cols(goodIdx) = convToValidU16( basisGood * scoreGood.t() );
    }
    else {
      // extract scores:
      const Mat<i32> scoreGood( POP_CONSTRUCTOR(i32, nGood, RANK_GOOD) );
      // fill this matrix with the matrix multiplications:
      //restored.cols(goodIdx) = convToValidU16( basisGood * conv_to<fmat>::from(scoreGood).t() );
      restored.cols(goodIdx) = convToValidU16( basisGood * scoreGood.t() );
    }
  }
  //Decompress the bad data, output the uncompressed to restored.cols(badIdx);
  if ( nBad > 0)
    {
      vector<size_t> sSize = src.pop<size_t>(1);
      const vector<uint8_t> deltaIn = src.pop<uint8_t>(sSize[0]);
      Decompressor dc;
      size_t nCols;
      size_t nRows;
      size_t oL;
      vector<uint16_t> deltaOut;
      dc.decompress(deltaIn,nRows,nCols, oL, deltaOut);
      Mat<uint16_t> dataBadMat(&deltaOut[0], L, nBad, false, true);
      restored.cols(badIdx) = dataBadMat;
    }
}
