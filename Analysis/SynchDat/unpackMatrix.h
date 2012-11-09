/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/**
 * unpackMatrix.h
 * Exported interface: unpackMatrix
 * This is the file that contains all linear algebra on the decompression side
 * 
 * @author Magnus Jedvert
 * @version 1.1 april 2012
 */
#ifndef UNPACKMATRIX_H
#define UNPACKMATRIX_H

#include "compression.h"
#include <stdint.h>
//#include "ByteHandler.h"

/**
 * unpackMatrix - Reverse of packMatrix. Stores the matrix in memory pointed
 * to by dst. Pops the information from src.
 */

void unpackMatrix(ByteUnpacker &src, uint16_t *dst, size_t L);
void unpackMatrixPlus(ByteUnpacker &src, uint16_t *dst, size_t L);
void unpackMatrixPlusV2(ByteUnpacker &src, uint16_t *dst, size_t L);
void unpackMatrixPlusV3(ByteUnpacker &src, uint16_t *dst, size_t L);

#endif // UNPACKMATRIX_H
