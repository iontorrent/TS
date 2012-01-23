// -*-C++-*- 

// Copyright (C) 2004 
// Christian Stimming <stimming@tuhh.de>

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2, or (at
// your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License along
// with this library; see the file COPYING.  If not, write to the Free
// Software Foundation, 59 Temple Place - Suite 330, Boston, MA 02111-1307,
// USA.

//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.

/** @file blas2pp.h
 * @brief Blas Level-2 Routines: Vector-Matrix Operations
 */


#ifndef _BLAS2_PP_H_
#define _BLAS2_PP_H_

#include "blas2.h"
#include "lafnames.h"
#include "arch.h" // needed for DLLIMPORT
#include LA_GEN_MAT_DOUBLE_H
#include LA_VECTOR_DOUBLE_H
#ifdef LA_COMPLEX_SUPPORT
# include LA_GEN_MAT_COMPLEX_H
# include LA_VECTOR_COMPLEX_H
#endif

/** @name Real-valued general matrices and vectors */
//@{
/** Perform the matrix-vector operation y := alpha*A'*x + beta*y */
DLLIMPORT
void Blas_Mat_Trans_Vec_Mult(const LaGenMatDouble &A, 
			     const LaVectorDouble &dx, 
			     LaVectorDouble &dy,
			     double alpha = 1.0, double beta = 0.0);

/** Perform the matrix-vector operation y := alpha*A*x + beta*y */
DLLIMPORT
void Blas_Mat_Vec_Mult(const LaGenMatDouble &A, 
		       const LaVectorDouble &dx, 
		       LaVectorDouble &dy, 
		       double alpha = 1.0, double beta = 0.0);

/** Perform the rank 1 operation A := alpha*dx*dy' + A */
DLLIMPORT
void Blas_R1_Update(LaGenMatDouble &A, const LaVectorDouble &dx, 
		    const LaVectorDouble &dy, double alpha = 1.0);
//@}


#ifdef LA_COMPLEX_SUPPORT
/** @name Complex-valued matrices and vectors */
//@{
/** Perform the matrix-vector operation y := alpha*A*x + beta*y */
DLLIMPORT
void Blas_Mat_Vec_Mult(const LaGenMatComplex &A, 
		       const LaVectorComplex &dx, 
		       LaVectorComplex &dy, 
		       LaComplex alpha = 1.0, LaComplex beta = 0.0);

/** Perform the matrix-vector operation y := alpha*A'*x + beta*y */
DLLIMPORT
void Blas_Mat_Trans_Vec_Mult(const LaGenMatComplex &A, 
			     const LaVectorComplex &dx, 
			     LaVectorComplex &dy,
			     LaComplex alpha = 1.0, LaComplex beta = 0.0);

/** Perform the rank 1 operation A := alpha*dx*dy' + A */
DLLIMPORT
void Blas_R1_Update(LaGenMatComplex &A, const LaVectorComplex &dx, 
		    const LaVectorComplex &dy, LaComplex alpha = 1.0);
//@}
#endif // LA_COMPLEX_SUPPORT

#ifdef _LA_SYMM_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Vec_Mult(const LaSymmMatDouble &A, const LaVectorDouble &dx, 
		       LaVectorDouble &dy, 
		       double alpha = 1.0, double beta = 1.0);
DLLIMPORT
void Blas_R1_Update(LaSymmMatDouble &A, const LaVectorDouble &dx,
		    double alpha = 1.0);
DLLIMPORT
void Blas_R2_Update(LaSymmMatDouble &A, const LaVectorDouble &dx, 
		    const LaVectorDouble &dy, double alpha = 1.0);
#endif

#ifdef _LA_SYMM_BAND_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Vec_Mult(const LaSymmBandMatDouble &A, const LaVectorDouble &dx, 
		       LaVectorDouble &dy, 
		       double alpha = 1.0, double beta = 1.0);
#endif

#ifdef _LA_SPD_MAT_DOUBLE_H_ 
DLLIMPORT
void Blas_Mat_Vec_Mult(const LaSpdMatDouble &AP, const LaVectorDouble &dx, 
		       LaVectorDouble &dy, 
		       double alpha = 1.0, double beta = 1.0);
DLLIMPORT
void Blas_R1_Update(LaSpdMatDouble &AP, const LaVectorDouble &dx,
		    double alpha = 1.0);
DLLIMPORT
void Blas_R2_Update(LaSpdMatDouble &AP, const LaVectorDouble &dx, 
		    const LaVectorDouble &dy, double alpha = 1.0);
#endif

#ifdef _LA_BAND_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Vec_Mult(const LaBandMatDouble &A, const LaVectorDouble &dx, 
		       LaVectorDouble &dy, 
		       double alpha = 1.0, double beta = 1.0);
#endif

#ifdef _LA_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Vec_Mult(const LaLowerTriangMatDouble &A, LaVectorDouble &dx);
#endif



#ifdef _LA_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Vec_Mult(const LaUpperTriangMatDouble &A, LaVectorDouble &dx);
#endif



#ifdef _LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H_ 
DLLIMPORT
void Blas_Mat_Vec_Mult(const LaUnitLowerTriangMatDouble &A, 
		       LaVectorDouble &dx);
#endif

#ifdef _LA_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Vec_Solve(LaLowerTriangMatDouble &A, LaVectorDouble &dx);
#endif



#ifdef _LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Vec_Mult(const LaUnitUpperTriangMatDouble &A, 
		       LaVectorDouble &dx);
#endif

#ifdef _LA_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Vec_Solve(LaUpperTriangMatDouble &A, LaVectorDouble &dx);
#endif


#ifdef _LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Vec_Solve(LaUnitLowerTriangMatDouble &A, 
			LaVectorDouble &dx);
#endif

#ifdef _LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Vec_Solve(LaUnitUpperTriangMatDouble &A, LaVectorDouble &dx);
#endif


#endif 
    //_BLAS2_PP_H_
