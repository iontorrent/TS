//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.

#ifndef _LA_GENMD_H
#define _LA_GENMD_H

/** @file
 * @brief Generation functions for matrices 
 *
 * This file defines some generation functions for random
 * matrices.
 */

#include "arch.h"
#include "lapack.h"
#include "f2c.h"

#ifdef _LA_TRIDIAG_MAT_DOUBLE_H_
DLLIMPORT void LaGenerateMatDouble(LaTridiagMatDouble &A);
#endif

#ifdef _LA_SYMM_TRIDIAG_MAT_DOUBLE_H_
DLLIMPORT void LaGenerateMatDouble(LaSymmTridiagMatDouble &A);
#endif

#ifdef _LA_GEN_MAT_DOUBLE_H_
DLLIMPORT void LaGenerateMatDouble(LaGenMatDouble &A);

DLLIMPORT LaGenMatDouble& LaRandUniform(LaGenMatDouble &A, 
					double low, double high);
#endif // _LA_GEN_MAT_DOUBLE_H_

#ifdef _LA_GEN_MAT_COMPLEX_H_
DLLIMPORT LaGenMatComplex& LaRandUniform(LaGenMatComplex &A, 
					 double low, double high);
#endif // _LA_GEN_MAT_COMPLEX_H_


#ifdef _LA_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT void LaGenerateMatDouble(LaUpperTriangMatDouble &A);
#endif

#ifdef _LA_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT void LaGenerateMatDouble(LaLowerTriangMatDouble &A);
#endif


#ifdef _LA_SYMM_MAT_DOUBLE_H_
DLLIMPORT void LaGenerateMatDouble(LaSymmMatDouble &A);
#endif

#ifdef _LA_SPD_MAT_DOUBLE_H_
DLLIMPORT void LaGenerateMatDouble(LaSpdMatDouble &A);
#endif

#ifdef _LA_SPD_BAND_MAT_DOUBLE_H_
DLLIMPORT void LaGenerateMatDouble(LaSpdBandMatDouble &A);
#endif

#ifdef _LA_BAND_MAT_DOUBLE_H_
DLLIMPORT void LaGenerateMatDouble(LaBandMatDouble &A);
#endif


#endif // _LA_GENMD_H
