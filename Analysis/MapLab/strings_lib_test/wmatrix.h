/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __wmatrix_h__
#define __wmatrix_h__

#include <wmatrix_templ.h>
#include <resource.h>
#include <ncbi_weight_matrix.h>

namespace genstr
{

typedef WeightMatrix<char, int, double> ProtFMatrix;
typedef WeightMatrix<char, int, int> ProtIMatrix;

template <typename ProtMatrixType>
void readProtMatrix (const char* fname, ProtMatrixType& matrix)
{
    char symbols [STD_PROT_MATRIX_SIZE];
    int values  [STD_PROT_MATRIX_SIZE * STD_PROT_MATRIX_SIZE];
    int read_alpha_size = readNcbiMatrix (fname, STD_PROT_MATRIX_SIZE, symbols, values);
    if (read_alpha_size != STD_PROT_MATRIX_SIZE) ers << "Matrix is too small in " << fname << ThrowEx (BadMatrixFormat);
    matrix.configure (symbols, STD_PROT_MATRIX_SIZE, values);
}


};

#endif //__wmatrix_h__
