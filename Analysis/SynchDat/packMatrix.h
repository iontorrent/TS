/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/**
 * packMatrix.h
 * Exported interface: packMatrix
 * Represents all linear algebra on the compression side
 * 
 * @author Magnus Jedvert
 * @version 1.1 april 2012
*/
#ifndef PACKMATRIX_H
#define PACKMATRIX_H
#include <cstdlib>
#include "compression.h"
#include <stdint.h>
//#include "ByteHandler.h"
/**
 * packMatrix - takes as input a pointer to a packed array of uint16_t and 
 * interprets this as a LxN matrix (column-major order). The data for the 
 * resulting matrix approximation is pushed onto input argument dst.
 */
void packMatrix(BytePacker &dst, const uint16_t *input_data, size_t N, size_t L);
void packMatrixPlus(BytePacker &dst, const uint16_t *input_data, size_t N, size_t L, double errCon, int rankGood, float pivot=0);
void packMatrixPlusV2(BytePacker &dst, const uint16_t *input_data, size_t N, size_t L, double errCon, int rankGood, float pivot=0);
void packMatrixPlusV3(BytePacker &dst, const uint16_t *input_data, size_t N, size_t L, double errCon, int rankGood, float pivot=0);
float getMAB(fvec &diff);
#endif // PACKMATRIX_H
