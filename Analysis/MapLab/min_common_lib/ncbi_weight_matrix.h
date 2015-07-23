/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __ncbi_weight_matrix_h__
#define __ncbi_weight_matrix_h__

#include "rerror.h"

//#ifndef __ncbi_weight_matrix_cpp__
extern const char* ERR_BadMatrix;

#ifndef _MSC_VER
extern const int STD_PROT_MATRIX_SIZE;
#else
const int STD_PROT_MATRIX_SIZE = 24;
#endif

MAKE_ERROR_TYPE (BadMatrixFormat, ERR_BadMatrix);


// reads standard (square) NCBI weight matrix file;
// saves read alphabet and values into passed buffers.
// assumes alpha_buf is at least max_alpha_size bytes long and value_buf is at least max_alpha_size * max_alpha_size bytes long
// returns actual alphabet size
// on format error, throws BadMatrixFormat
unsigned readNcbiMatrix (const char* fname, unsigned max_alpha_size, char* alpha_buf, int* value_buf);

#endif // __ncbi_weight_matrix_h__
