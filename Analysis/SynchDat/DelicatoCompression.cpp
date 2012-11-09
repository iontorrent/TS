/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/**
 * DelicatoCompressionCompression.cpp
 * Implementation of final interface
 * Exported interface: compress/decompress
 *  
 * @author Magnus Jedvert
 * @version 1.1 april 2012
*/

#include <vector>
#include <armadillo>
#include <stdint.h>
#include "packMatrix.h"
#include "unpackMatrix.h"
#include "compression.h"
#include "DelicatoCompression.h"
using namespace std;
using namespace arma;

//typedef uint16_t u16;
//typedef uint8_t u8;

#define rep(i, n) for (size_t i = 0; i < size_t(n); ++i)

// split incoming data in chunks, limits the memory footprint
// this is the maximum number of columns in each chunk:
static const int NCOLUMNS_IN_CHUNK = 2000000;

// change these to zip with something else:
// static const string ZIP_COMMAND = "pigz -c -3";
// static const string UNZIP_COMMAND = "pigz -d";
static const string ZIP_COMMAND = "gzip -c --fast";
static const string UNZIP_COMMAND = "gzip -d";

// rawdataCompress - helper function for compress:
static void rawdataCompress(const vector<u16> &input_data, size_t w, size_t h, size_t L, vector<u8> &compressed, float errCon, bool doBoth, int rankGood, float pivot) {
    BytePacker bPacker(compressed); 
    const int N = w * h;

    // push header:
    const size_t header[3] = {w, h, L};
    bPacker.push( (char*)header, sizeof(header) );      

    // push each chunk:
    const int nChunks = (int)ceil( double(N) / NCOLUMNS_IN_CHUNK );
    const uvec rowIdx = linspace<uvec>(0, N, nChunks+1);
    rep(i, nChunks) {
        const size_t a = rowIdx(i), b = rowIdx(i+1), n = b - a;
	if ( doBoth )
	  //packMatrixPlus(bPacker, input_data.data() + L * a, n, L, errCon, rankGood, pivot);
	  //packMatrixPlusV2(bPacker, input_data.data() + L * a, n, L, errCon, rankGood, pivot);
	  packMatrixPlusV3(bPacker, input_data.data() + L * a, n, L, errCon, rankGood, pivot);
	else
	  packMatrix(bPacker, input_data.data() + L * a, n, L);
    }

    bPacker.finalize(); 
}

/**
 * compress - takes as input a pointer to a packed array of uint16_t and 
 * interprets this as a Lx(w*h) matrix (column-major order). This is the data
 * from a well with width w, height h, and number of frames L. Returns a 
 * compressed version of the data in argument compressed. 
 * Returns 0 if success, 1 otherwise.
 */
int DelicatoCompression::compress(const vector<u16> &input_data, size_t w, size_t h, size_t L, vector<u8> &compressed, float errCon,int rankGood,float pivot) {
  //    vector<u8> tmp;
  rawdataCompress(input_data, w, h, L, compressed, errCon,doBoth, rankGood, pivot);
  //    filterThroughProcess(ZIP_COMMAND, tmp, compressed);
  return 0;
}

// rawdataDecompress - helper function for decompress:
static void rawdataDecompress(const vector<u8> &compressed, vector<u16> &output_data, size_t &w, size_t &h, size_t &L, bool doBoth) {
    ByteUnpacker bUnpack( (char*)compressed.data() );
    // unpack header:    
    const vector<size_t> globHeader = bUnpack.pop<size_t>(3);
    w = globHeader[0];
    h = globHeader[1];
    L = globHeader[2];  

    const size_t N = w * h;
    output_data.resize( N * L );

    // unpack each chunk:
    const int nChunks = (int)ceil( double(N) / NCOLUMNS_IN_CHUNK );
    const uvec rowIdx = linspace<uvec>(0, N, nChunks+1);    
    rep(i, nChunks) {
      if ( doBoth )
	//unpackMatrixPlus(bUnpack, output_data.data() + L * rowIdx(i), L);
	//unpackMatrixPlusV2(bUnpack, output_data.data() + L * rowIdx(i), L);
	unpackMatrixPlusV3(bUnpack, output_data.data() + L * rowIdx(i), L);
      else
	unpackMatrix(bUnpack, output_data.data() + L * rowIdx(i), L);
    }
}

/**
 * decompress - Reverse of compress. Takes the compressed data as input and
 * fills in output_data with uncompressed data, and also fills in the size of 
 * the region (w, h) and number of frames (L). 
 * Returns 0 if success, 1 otherwise.
 */
int DelicatoCompression::decompress(const vector<u8> &compressed, vector<u16> &output_data, size_t &w, size_t &h, size_t &L) {
  //    vector<u8> unzipped;
  //    filterThroughProcess(UNZIP_COMMAND, compressed, unzipped);
  rawdataDecompress(compressed, output_data, w, h, L, doBoth);
  return 0;
}

