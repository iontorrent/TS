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

/** @file blas3pp.h
 * @brief Blas Level-3 Routines: Matrix-Matrix Operations and Matrix norms.
 */


#ifndef _BLAS3_PP_H_
#define _BLAS3_PP_H_

#include <cmath>
#include "lafnames.h"
#include "arch.h" // needed for DLLIMPORT

#ifdef LA_COMPLEX_SUPPORT
# include "lacomplex.h"
#endif

#ifndef DOXYGEN_IGNORE
// Forward declaration of classes
class LaGenMatDouble;
class LaVectorDouble;
class LaGenMatComplex;
class LaSymmMatDouble;

class LaBandMatDouble;
class LaSpdMatDouble;
class LaSymmTridiagMatDouble;
class LaTridiagMatDouble;
#endif // DOXYGEN_IGNORE

/** @name Real-valued general matrices */
//@{
/** Perform the matrix-matrix operation C := alpha*A*B + beta*C
 * where A and B are used in either non-transposed or transposed
 * form, depending on the function arguments.
 *
 * Internally this uses @c dgemm .
 *
 * @param transpose_A If true, use transposed A, i.e. A' instead
 * of A. If false, use A directly in non-transposed form.
 *
 * @param transpose_B If true, use transposed B, i.e. B' instead
 * of B. If false, use B directly in non-transposed form.
 *
 * (New in lapackpp-2.4.14.) */
DLLIMPORT
void Blas_Mat_Mat_Mult(const LaGenMatDouble &A, 
		       const LaGenMatDouble &B, LaGenMatDouble &C, 
		       bool transpose_A, bool transpose_B = false,
		       double alpha = 1.0, double beta = 0.0);

/** Perform the matrix-matrix operation C := alpha*A*B + beta*C */
DLLIMPORT
void Blas_Mat_Mat_Mult(const LaGenMatDouble &A, 
            const LaGenMatDouble &B, LaGenMatDouble &C, 
            double alpha = 1.0, double beta = 0.0);

/** Perform the matrix-matrix operation C := alpha*A'*B + beta*C */
DLLIMPORT
void Blas_Mat_Trans_Mat_Mult(const LaGenMatDouble &A, 
            const LaGenMatDouble &B, LaGenMatDouble &C, 
            double alpha = 1.0, double beta = 0.0);

/** Perform the matrix-matrix operation C := alpha*A*B' + beta*C */
DLLIMPORT
void Blas_Mat_Mat_Trans_Mult(const LaGenMatDouble &A, 
            const LaGenMatDouble &B, LaGenMatDouble &C, 
            double alpha = 1.0, double beta = 0.0);

/** Perform a matrix-matrix multiplication, returning only the
    diagonal of the result: C := diag(A * B). FIXME: needs verification. */
DLLIMPORT
void Blas_Mat_Mat_Mult(const LaGenMatDouble &A, 
		       const LaGenMatDouble &B, LaVectorDouble &C);

/** Perform a matrix-matrix multiplication, returning only the
    diagonal of the result: C := diag(A' * B). FIXME: needs verification.  */
DLLIMPORT
void Blas_Mat_Trans_Mat_Mult(const LaGenMatDouble &A, 
			     const LaGenMatDouble &B, LaVectorDouble &C);

/** Perform a matrix-matrix multiplication, returning only the
    diagonal of the result: C := diag(A * B'). FIXME: needs verification.  */
DLLIMPORT
void Blas_Mat_Mat_Trans_Mult(const LaGenMatDouble &A, 
			     const LaGenMatDouble &B, LaVectorDouble &C);

/** Matrix scaling: A := s * A
 *
 * (New in lapackpp-2.4.7.) */
DLLIMPORT
void Blas_Scale(double s, LaGenMatDouble &A);

//@}


#ifdef LA_COMPLEX_SUPPORT
/** @name Complex-valued matrices */
//@{
/** Perform the matrix-matrix operation C := alpha*A*B + beta*C
 * where A and B are used in either non-hermitian or hermitian
 * form (matrix transpose and complex conjugate), depending on the
 * function arguments.
 *
 * Internally this uses @c zgemm .
 *
 * @param hermit_A If true, use hermitian A, i.e. A* (sometimes
 * denoted conj(A')), the matrix transpose and complex conjugate,
 * instead of A. If false, use A directly in non-hermitian form.
 *
 * @param hermit_B If true, use hermitian B, i.e. B* (sometimes
 * denoted conj(B')), the matrix transpose and complex conjugate,
 * instead of B. If false, use B directly in non-hermitian form.
 *
 * (New in lapackpp-2.4.14.) */
DLLIMPORT
void Blas_Mat_Mat_Mult(const LaGenMatComplex &A, 
		       const LaGenMatComplex &B, LaGenMatComplex &C, 
		       bool hermit_A, bool hermit_B = false, 
		       LaComplex alpha = 1.0, LaComplex beta = 0.0);

/** Perform the matrix-matrix operation C := alpha*A*B + beta*C */
DLLIMPORT
void Blas_Mat_Mat_Mult(const LaGenMatComplex &A, 
            const LaGenMatComplex &B, LaGenMatComplex &C, 
            LaComplex alpha = 1.0, LaComplex beta = 0.0);

/** Perform the matrix-matrix operation C := alpha*A'*B + beta*C */
DLLIMPORT
void Blas_Mat_Trans_Mat_Mult(const LaGenMatComplex &A, 
            const LaGenMatComplex &B, LaGenMatComplex &C, 
            LaComplex alpha = 1.0, LaComplex beta = 0.0);

/** Perform the matrix-matrix operation C := alpha*A*B' + beta*C */
DLLIMPORT
void Blas_Mat_Mat_Trans_Mult(const LaGenMatComplex &A, 
            const LaGenMatComplex &B, LaGenMatComplex &C, 
            LaComplex alpha = 1.0, LaComplex beta = 0.0);

/** Matrix scaling: A := s * A
 *
 * (New in lapackpp-2.4.7.) */
DLLIMPORT
void Blas_Scale(COMPLEX s, LaGenMatComplex &A);
//@}
#endif // LA_COMPLEX_SUPPORT

#ifdef _LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H_

DLLIMPORT
void Blas_Mat_Mat_Solve(LaUnitLowerTriangMatDouble &A, 
            LaGenMatDouble &B, double alpha = 1.0);

#endif

#ifdef _LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Mat_Mult(LaUnitUpperTriangMatDouble &A,
            LaGenMatDouble &B, double alpha = 1.0);

DLLIMPORT
void Blas_Mat_Mat_Solve(LaUnitUpperTriangMatDouble &A, 
            LaGenMatDouble &B, double alpha = 1.0);

#endif

#ifdef _LA_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Mat_Mult(LaLowerTriangMatDouble &A,
            LaGenMatDouble &B, double alpha = 1.0);

DLLIMPORT
void Blas_Mat_Mat_Solve(LaLowerTriangMatDouble &A, 
            LaGenMatDouble &B, double alpha = 1.0);
#endif


#ifdef _LA_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Mat_Mult(LaUpperTriangMatDouble &A,
            LaGenMatDouble &B, double alpha = 1.0);

DLLIMPORT
void Blas_Mat_Mat_Solve(LaUpperTriangMatDouble &A, 
            LaGenMatDouble &B, double alpha = 1.0);
#endif


#ifdef _LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Trans_Mat_Solve(LaUnitLowerTriangMatDouble &A,
            LaGenMatDouble &B, double alpha = 1.0);
#endif

#ifdef _LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Trans_Mat_Solve(LaUnitUpperTriangMatDouble &A,
            LaGenMatDouble &B, double alpha = 1.0);
#endif

#ifdef _LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Mat_Mult(LaUnitLowerTriangMatDouble &A,
            LaGenMatDouble &B, double alpha = 1.0);

#endif

#ifdef _LA_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT
void Blas_Mat_Trans_Mat_Solve(LaLowerTriangMatDouble &A,
            LaGenMatDouble &B, double alpha = 1.0);
#endif


#ifdef _LA_UPPER_TRIANG_MAT_DOUBLE_H_ 
DLLIMPORT
void Blas_Mat_Trans_Mat_Solve(LaUpperTriangMatDouble &A,
            LaGenMatDouble &B, double alpha = 1.0);

#endif

/** @name Symmetric matrices */
//@{

/**
* Perform one of the matrix-matrix operations
*  - A := alpha*B*C + beta*A if b_left_side is true
*  - A := alpha*C*B + beta*A if b_left_side is false
*
*/
DLLIMPORT
void Blas_Mat_Mat_Mult(LaSymmMatDouble &A, LaGenMatDouble &B, 
		       LaGenMatDouble &C,
		       double alpha = 1.0, double beta = 1.0,
		       bool b_left_side = true);

/**
* Perform one of the matrix-matrix operations
*  - C := alpha*A*A' + beta*C if right_transposed is true
*  - C := alpha*A'*A + beta*C if right_transposed is false
*
* with A' as transposition of A.
*
*/
DLLIMPORT
void Blas_R1_Update(LaSymmMatDouble &C, LaGenMatDouble &A,
		    double alpha = 1.0, double beta = 1.0,
		    bool right_transposed = true);

/**
* Perform one of the matrix-matrix operations
*  - C := alpha*A*B' + alpha*B*A' + beta*C if right_transposed is true
*  - C := alpha*A'*B + alpha*B'*A + beta*C if right_transposed is false
*
* with A' and B' as transposition of A and B.
*
*/
DLLIMPORT
void Blas_R2_Update(LaSymmMatDouble &C, LaGenMatDouble &A,
		    LaGenMatDouble &B,
		    double alpha = 1.0, double beta = 1.0,
		    bool right_transposed = true);

//@}



//-------------------------------------
/// @name Matrix Norms
//-------------------------------------
//@{
/** \brief 1-Norm: Maximum column sum 
 *
 * Returns the 1-Norm of matrix A, which is the maximum
 * absolute column sum: \f$||A||_{1}=\max_j\sum_{i=1}^M|a_{ij}|\f$,
 *
 * \see Eric W. Weisstein. "Matrix Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/MatrixNorm.html */
DLLIMPORT
double Blas_Norm1(const LaGenMatDouble &A);

/** \brief Infinity-Norm: Maximum row sum
 *
 * Returns the Infinity-Norm of matrix A, which is the maximum
 * absolute row sum: \f$||A||_{\infty}=\max_i\sum_{j=1}^N|a_{ij}|\f$,
 *
 * \see Eric W. Weisstein. "Matrix Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/MatrixNorm.html */
DLLIMPORT
double Blas_Norm_Inf(const LaGenMatDouble &A);

/** \brief Frobenius-Norm
 *
 * Returns the Frobenius-Norm of matrix A (also called the Schur-
 * or Euclidean norm): \f$||A||_F=\sqrt{\sum_{i,j}|a_{ij}|^2}\f$,
 * i.e. the square root of the sum of the absolute squares of its
 * elements. 
 *
 * \see Eric W. Weisstein. "Matrix Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/MatrixNorm.html */
DLLIMPORT
double Blas_NormF(const LaGenMatDouble &A);

/** \brief 1-Norm: Maximum column sum 
 *
 * Returns the 1-Norm of matrix A, which is the maximum
 * absolute column sum: \f$||A||_{\infty}=\max_j\sum_{i=1}^M|a_{ij}|\f$,
 *
 * \see Eric W. Weisstein. "Matrix Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/MatrixNorm.html */
DLLIMPORT
double Blas_Norm1(const LaGenMatComplex &A);

/** \brief Infinity-Norm: Maximum row sum
 *
 * Returns the Infinity-Norm of matrix A, which is the maximum
 * absolute row sum: \f$||A||_{\infty}=\max_i\sum_{j=1}^N|a_{ij}|\f$,
 *
 * \see Eric W. Weisstein. "Matrix Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/MatrixNorm.html */
DLLIMPORT
double Blas_Norm_Inf(const LaGenMatComplex &A);

/** \brief Frobenius-Norm
 *
 * Returns the Frobenius-Norm of matrix A (also called the Schur-
 * or Euclidean norm): \f$||A||_F=\sqrt{\sum_{i,j}|a_{ij}|^2}\f$,
 * i.e. the square root of the sum of the absolute squares of its
 * elements. 
 *
 * \see Eric W. Weisstein. "Matrix Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/MatrixNorm.html */
DLLIMPORT
double Blas_NormF(const LaGenMatComplex &A);

#ifndef DOXYGEN_IGNORE
/** DEPRECATED, use Blas_Norm_Inf instead. */
DLLIMPORT
double Norm_Inf(const LaGenMatDouble &A);
/** DEPRECATED, use Blas_Norm_Inf instead. */
DLLIMPORT
double Norm_Inf(const LaGenMatComplex &A);
#endif // DOXYGEN_IGNORE


DLLIMPORT
double Norm_Inf(const LaBandMatDouble &A);
DLLIMPORT
double Norm_Inf(const LaSymmMatDouble &S);
DLLIMPORT
double Norm_Inf(const LaSpdMatDouble &S);
DLLIMPORT
double Norm_Inf(const LaSymmTridiagMatDouble &S);
DLLIMPORT
double Norm_Inf(const LaTridiagMatDouble &T);
//@}


#endif 
    // _BLAS3_PP_H_
            
