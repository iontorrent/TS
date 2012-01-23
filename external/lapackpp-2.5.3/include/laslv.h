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

/** @file
 * @brief Functions for solving linear equations
 */

#ifndef _LASLV_H
#define _LASLV_H

#include "lafnames.h"
#include LA_GEN_MAT_DOUBLE_H
#include LA_VECTOR_DOUBLE_H
#ifdef LA_COMPLEX_SUPPORT
# include LA_GEN_MAT_COMPLEX_H
# include LA_VECTOR_COMPLEX_H
#endif
#include LA_VECTOR_LONG_INT_H


/** @name Real-valued matrices */
//@{
 /** Compute the solution to a real system of linear equations A*X=B
 *
 * Depending on the dimensions of A, either a LU or a QR decomposition
 * is used.
 *
 * @note This function was broken with non-square matrices and QR
 * decomposition for a long time. It is fixed and verified only
 * since lapackpp-2.4.11.
 */
DLLIMPORT
void LaLinearSolve( const LaGenMatDouble& A, LaGenMatDouble& X, 
		    const LaGenMatDouble& B );

/** Compute the solution to a real system of linear equations A*X=B
 * in-place.
 *
 * Depending on the dimensions of A, either a LU or a QR decomposition
 * is used.
 *
 * In-place means: The contents of A are overwritten during the
 * calculation. B is not overwritten but always copied during this
 * operation.
 *
 * @note This function was broken with non-square matrices and QR
 * decomposition for a long time. It is fixed and verified only
 * since lapackpp-2.4.11.
 */
DLLIMPORT
void LaLinearSolveIP( LaGenMatDouble& A, LaGenMatDouble& X, 
		      const LaGenMatDouble& B );


/** Compute the solution to a real system of linear equations A*X=B by
 * using the LU decomposition. This only works for a squares matrix A.
 */
DLLIMPORT
void LaLULinearSolve( const LaGenMatDouble& A, LaGenMatDouble& X, 
		      const LaGenMatDouble& B );

/** Compute the solution to a real system of linear equations A*X=B by
 * using the LU decomposition in-place. This only works for a squares
 * matrix A.
 *
 * In-place means: The contents of A are overwritten during the
 * calculation. B is not overwritten but always copied during this
 * operation.
 */
DLLIMPORT
void LaLULinearSolveIP( LaGenMatDouble& A, LaGenMatDouble& X, 
			const LaGenMatDouble& B );

/** Compute the solution to a real system of linear equations A*X=B by
 * using QR decomposition, which works for any rectangular matrix A.
 *
 * @note This function was broken for a long time. It is fixed and
 * verified only since lapackpp-2.4.11.
 */
DLLIMPORT
void LaQRLinearSolve( const LaGenMatDouble& A, LaGenMatDouble& X, 
		      const LaGenMatDouble& B );

/** Compute the solution to a real system of linear equations A*X=B by
 * using the QR decomposition in-place. This works for any rectangular 
 * matrix A.
 *
 * In-place means: The contents of A are overwritten during the
 * calculation. B is not overwritten but always copied during this
 * operation.
 *
 * @note This function was broken for a long time. It is fixed and
 * verified only since lapackpp-2.4.11.
 */
DLLIMPORT
void LaQRLinearSolveIP( LaGenMatDouble& A, LaGenMatDouble& X, 
			const LaGenMatDouble& B );

/** @brief Compute the LU factorization
 *
 * Compute the LU factorization (in-place) of a general M-by-N
 * matrix A.
 *
 * More info: See man @c dgetrf. 
 *
 * In-place means: The contents of GM are overwritten during the
 * calculation.
 *
 * @param GM Matrix to be factorized in-place.
 *
 * @param PIV Vector to return the pivoting indices. This vector
 * *has* to be at least as long as min(M,N).
 */
DLLIMPORT
void LUFactorizeIP(LaGenMatDouble &GM, LaVectorLongInt &PIV);
//@}



#ifdef LA_COMPLEX_SUPPORT
/** @name Complex-valued matrices */
//@{
 /** Compute the solution to a complex-valued system of linear
 * equations A*X=B
 *
 * Depending on the dimensions of A, either a LU or a QR decomposition
 * is used.
 *
 * @note This function was broken with non-square matrices and QR
 * decomposition for a long time. It is fixed and verified only
 * since lapackpp-2.4.11.
 */
DLLIMPORT
void LaLinearSolve( const LaGenMatComplex& A, LaGenMatComplex& X, 
		    const LaGenMatComplex& B );

/** Compute the solution to a complex-valued system of linear
 * equations A*X=B in-place.
 *
 * Depending on the dimensions of A, either a LU or a QR decomposition
 * is used.
 *
 * In-place means: The contents of A are overwritten during the
 * calculation. B is not overwritten but always copied during this
 * operation.
 *
 * @note This function was broken with non-square matrices and QR
 * decomposition for a long time. It is fixed and verified only
 * since lapackpp-2.4.11.
 */
DLLIMPORT
void LaLinearSolveIP( LaGenMatComplex& A, LaGenMatComplex& X, 
		      const LaGenMatComplex& B );


/** Compute the solution to a complex-valued system of linear
 * equations A*X=B by using the LU decomposition. This only works for
 * a squares matrix A.
 */
DLLIMPORT
void LaLULinearSolve( const LaGenMatComplex& A, LaGenMatComplex& X, 
		      const LaGenMatComplex& B );

/** Compute the solution to a complex-valued system of linear
 * equations A*X=B by using the LU decomposition in-place. This only
 * works for a squares matrix A.
 *
 * In-place means: The contents of A are overwritten during the
 * calculation. B is not overwritten but always copied during this
 * operation.
 */
DLLIMPORT
void LaLULinearSolveIP( LaGenMatComplex& A, LaGenMatComplex& X, 
			const LaGenMatComplex& B );

/** Compute the solution to a complex-valued system of linear
 * equations A*X=B by using QR decomposition, which works for any
 * rectangular matrix A.
 *
 * @note This function was broken for a long time. It is fixed and
 * verified only since lapackpp-2.4.11.
 */
DLLIMPORT
void LaQRLinearSolve( const LaGenMatComplex& A, LaGenMatComplex& X, 
		      const LaGenMatComplex& B );

/** Compute the solution to a complex-valued system of linear
 * equations A*X=B by using the QR decomposition in-place. This works
 * for any rectangular matrix A.
 *
 * In-place means: The contents of A are overwritten during the
 * calculation. B is not overwritten but always copied during this
 * operation.
 *
 * @note This function was broken for a long time. It is fixed and
 * verified only since lapackpp-2.4.11.
 */
DLLIMPORT
void LaQRLinearSolveIP( LaGenMatComplex& A, LaGenMatComplex& X, 
			const LaGenMatComplex& B );

/** @brief Compute the LU factorization
 *
 * Compute the LU factorization (in-place) of a general M-by-N
 * matrix A.
 *
 * In-place means: The contents of GM are overwritten during the
 * calculation.
 *
 * More info: See man @c zgetrf. 
 *
 * @param GM Matrix to be factorized in-place.
 * @param PIV Vector to return the pivoting indices. 
 * This vector *has* to be at least as long as min(M,N).
 */
DLLIMPORT
void LUFactorizeIP(LaGenMatComplex &GM, LaVectorLongInt &PIV);
//@}

/** @brief Compute the inverse of a matrix from LU factorization
 *
 * Compute the inverse of a matrix in-place based on output
 * from LUFactorizeIP
 *
 * In-place means: The contents of A are overwritten during the
 * calculation.
 *
 * @param A Matrix factorized output matrix from LUFactorizeIP
 * @param PIV Vector pivoting indices output from LUFactorizeIP.
 */
DLLIMPORT
void LaLUInverseIP(LaGenMatComplex &A, LaVectorLongInt &PIV);

/** @brief Compute the inverse of a matrix from LU factorization
 *
 * Compute the inverse of a matrix in-place based on output
 * from LUFactorizeIP
 *
 * In-place means: The contents of A are overwritten during the
 * calculation.
 *
 * @param A Matrix factorized output matrix from LUFactorizeIP
 * @param PIV Vector pivoting indices output from LUFactorizeIP.
 * @param work Vector temporary work area (can be reused for efficiency).
 * work.size() must be at least A.size(0), if it is less, it will get
 * resized.
 */
DLLIMPORT
void LaLUInverseIP(LaGenMatComplex &A, LaVectorLongInt &PIV, LaVectorComplex &work);

#endif // LA_COMPLEX_SUPPORT


#ifdef _LA_SPD_MAT_DOUBLE_H_

DLLIMPORT
void LaLinearSolve( const LaSpdMatDouble& A, LaGenMatDouble& X, 
    LaGenMatDouble& B );
DLLIMPORT
void LaLinearSolveIP( LaSpdMatDouble& A, LaGenMatDouble& X, LaGenMatDouble& B );

DLLIMPORT
void LaCholLinearSolve( const LaSpdMatDouble& A, LaGenMatDouble& X, 
        LaGenMatDouble& B );
DLLIMPORT
void LaCholLinearSolveIP( LaSpdMatDouble& A, LaGenMatDouble& X, 
        LaGenMatDouble& B );
#endif // _LA_SPD_MAT_DOUBLE_H_

#ifdef _LA_SYMM_MAT_DOUBLE_H_

/** FIXME: Document me! FIXME: Needs verification. */
DLLIMPORT
void LaLinearSolve( const LaSymmMatDouble& A, LaGenMatDouble& X, 
		    const LaGenMatDouble& B );
/** FIXME: Document me! FIXME: Needs verification. */
DLLIMPORT
void LaLinearSolveIP( LaSymmMatDouble& A, LaGenMatDouble& X, 
		      const LaGenMatDouble& B );

/** FIXME: Document me! FIXME: Needs verification. */
DLLIMPORT
void LaCholLinearSolve( const LaSymmMatDouble& A, LaGenMatDouble& X, 
			const LaGenMatDouble& B );
/** FIXME: Document me! FIXME: Needs verification. */
DLLIMPORT
void LaCholLinearSolveIP( LaSymmMatDouble& A, LaGenMatDouble& X, 
			  const LaGenMatDouble& B );

// Eigenvalue problems 

DLLIMPORT
void LaEigSolve(const LaSymmMatDouble &S, LaVectorDouble &eigvals);
DLLIMPORT
void LaEigSolve(const LaSymmMatDouble &S, LaVectorDouble &eigvals, 
    LaGenMatDouble &eigvec);
DLLIMPORT
void LaEigSolveIP(LaSymmMatDouble &S, LaVectorDouble &eigvals);
#endif // _LA_SYMM_MAT_DOUBLE_H_

#ifdef LA_COMPLEX_SUPPORT
/** This function calculates all eigenvalues and eigenvectors of a
 * <i>general</i> matrix A. Uses \c dgeev . A wrapper for the other
 * function that uses two LaVectorDouble's for the eigenvalues.
 *
 * Uses @c dgeev
 *
 * @param A On entry, the general matrix A of dimension N x N. 
 *
 * @param eigvals On exit, this vector contains the eigenvalues.
 * Complex conjugate pairs of
 * eigenvalues appear consecutively with the eigenvalue having the
 * positive imaginary part first. The given argument must be a
 * vector of length N whose content will be overwritten.
 *
 * @param VR On exit, the right eigenvectors v(j) are stored one
 * after another in the columns of \c VR, in the same order as
 * their eigenvalues.  If the j- th eigenvalue is real, then v(j)
 * = VR(:,j), the j-th column of VR.  If the j-th and (j+1)-st
 * eigenvalues form a complex con- jugate pair, then v(j) =
 * VR(:,j) + i*VR(:,j+1) and v(j+1) = VR(:,j) - i*VR(:,j+1). The
 * given argument can be of size NxN, in which case the content
 * will be overwritten, or of any other size, in which case it
 * will be resized to dimension NxN.
*/
DLLIMPORT
void LaEigSolve(const LaGenMatDouble &A, LaVectorComplex &eigvals, LaGenMatDouble &VR);
#endif
/** This function calculates all eigenvalues and eigenvectors of a
 * <i>general</i> matrix A.
 *
 * Uses @c dgeev
 *
 * @param A On entry, the general matrix A of dimension N x N. 
 *
 * @param eigvals_real On exit, this vector contains the real
 * parts of the eigenvalues. Complex conjugate
 * pairs of eigenvalues appear consecutively with the eigenvalue
 * having the positive imaginary part first. The given argument
 * must be a vector of length N whose content will be overwritten.
 *
 * @param eigvals_imag On exit, this vector contains the imaginary
 * parts of the eigenvalues. The given argument
 * must be a vector of length N whose content will be overwritten.
 *
 * @param VR On exit, the right eigenvectors v(j) are stored one
 * after another in the columns of \c VR, in the same order as
 * their eigenvalues.  If the j- th eigenvalue is real, then v(j)
 * = VR(:,j), the j-th column of VR.  If the j-th and (j+1)-st
 * eigenvalues form a complex con- jugate pair, then v(j) =
 * VR(:,j) + i*VR(:,j+1) and v(j+1) = VR(:,j) - i*VR(:,j+1). The
 * given argument can be of size NxN, in which case the content
 * will be overwritten, or of any other size, in which case it
 * will be resized to dimension NxN.
 *
 * FIXME: Needs verification! */
DLLIMPORT
void LaEigSolve(const LaGenMatDouble &A, LaVectorDouble &eigvals_real,
		LaVectorDouble &eigvals_imag, LaGenMatDouble &VR);

/** FIXME: This is a misleading function! This function calculates all
 * eigenvalues and eigenvectors of a <i>symmetric</i> matrix A, <i>not
 * a general matrix A</i>!
 *
 * In-place means: The contents of A_symmetric are overwritten
 * during the calculation.
 *
 * @param A_symmetric On entry, the symmetric (not a general!!)
 * matrix A. The leading N-by-N lower triangular part of A is used
 * as the lower triangular part of the matrix to be decomposed. On
 * exit, A contains the orthonormal eigenvectors of the matrix.
 *
 * @param eigvals Vector of length at least N. On exit, this
 * vector contains the N eigenvalues.
 *
 * FIXME: Needs verification! FIXME: This uses dsyev which only works
 * on symmetric matrices. Instead, this should be changed to use dgeev
 * or even better dgeevx. For the complex case, we would have to write
 * another function that uses zgeev or zgeevx.
 *
 * New in lapackpp-2.4.9.
 */
DLLIMPORT
void LaEigSolveSymmetricVecIP(LaGenMatDouble &A_symmetric,
			      LaVectorDouble &eigvals);

/** DEPRECATED, has been renamed into LaEigSolveSymmetricVecIP().
 *
 * This is a misleading function! This function calculates all
 * eigenvalues and eigenvectors of a <i>symmetric</i> matrix A,
 * <i>not a general matrix A</i>! 
 *
 * This function just passes on the arguments to
 * LaEigSolveSymmetricVecIP().
 *
 * \deprecated
 */
DLLIMPORT
void LaEigSolveVecIP(LaGenMatDouble &A_symmetric, LaVectorDouble &eigvals);

#ifdef LA_COMPLEX_SUPPORT
/** Compute for an N-by-N complex nonsymmetric matrix A the
 * eigenvalues, and the right eigenvectors. Uses \c zgeev .
 *
 * (FIXME: Should add the option to select calculation of left
 * eigenvectors instead of the right eigenvectors, or both, or
 * none.)
 *
 * @param A On entry, the general matrix A of dimension N x N. 
 *
 * @param W Contains the computed eigenvalues. The given argument
 * must be a vector of length N whose content will be overwritten.
 *
 * @param VR On exit, the right eigenvectors v(j) are stored one
 * after another in the columns of \c VR, in the same order as
 * their eigenvalues.  The given argument can be of size NxN or
 * greater, in which case the content will be overwritten, or of
 * any other size, in which case it will be resized to dimension
 * NxN.
 *
 * FIXME: Needs verification! */
DLLIMPORT
void LaEigSolve(const LaGenMatComplex &A, LaVectorComplex &W,
		LaGenMatComplex &VR);
#endif // LA_COMPLEX_SUPPORT

/** @brief Compute the inverse of a matrix from LU factorization
 *
 * Compute the inverse of a matrix in-place based on output
 * from LUFactorizeIP
 *
 * In-place means: The contents of A are overwritten during the
 * calculation.
 *
 * @param A Matrix factorized output matrix from LUFactorizeIP
 * @param PIV Vector pivoting indices output from LUFactorizeIP.
 */
DLLIMPORT
void LaLUInverseIP(LaGenMatDouble &A, LaVectorLongInt &PIV);

/** @brief Compute the inverse of a matrix from LU factorization
 *
 * Compute the inverse of a matrix in-place based on output
 * from LUFactorizeIP
 *
 * In-place means: The contents of A are overwritten during the
 * calculation.
 *
 * @param A Matrix factorized output matrix from LUFactorizeIP
 * @param PIV Vector pivoting indices output from LUFactorizeIP.
 * @param work Vector temporary work area (can be reused for efficiency).
 * work.size() must be at least A.size(0), if it is less, it will get
 * resized.
 */
DLLIMPORT
void LaLUInverseIP(LaGenMatDouble &A, LaVectorLongInt &PIV, LaVectorDouble &work);

#endif // _LASLV_H
