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

// You should have received a copy of the GNU Lesser General
// Public License along with this library; see the file COPYING.
// If not, write to the Free Software Foundation, 59 Temple Place
// - Suite 330, Boston, MA 02111-1307, USA.

//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.

/** @file
 * @brief Functions for Singular Value Decomposition
 */

#ifndef _LASVD_H
#define _LASVD_H

#include "lafnames.h"
#include LA_GEN_MAT_DOUBLE_H
#ifdef LA_COMPLEX_SUPPORT
# include LA_GEN_MAT_COMPLEX_H
#endif
#include LA_VECTOR_LONG_INT_H


#ifdef LA_COMPLEX_SUPPORT
/** @name Complex-valued matrices */
//@{

/** Compute the Singular Value Decomposition. 
 *
 * Compute the singular value decomposition (SVD) of a complex M-by-N
 * matrix A, also computing the left and right singular vectors, by
 * using a divide-and-conquer method.
 *
 * In lapack this is zgesdd. zgesdd computes the singular value
 * decomposition (SVD) of a complex M-by-N matrix A, optionally
 * computing the left and/or right singular vectors, by using
 * divide-and-conquer method. The SVD is written 
 *
 * \f[A = U \cdot Sigma \cdot V^T\f]
 *
 * where Sigma is an M-by-N matrix which is zero except for its \c
 * min(m,n) diagonal elements, U is an M-by-M unitary matrix, and V is
 * an N-by-N unitary matrix.  The diagonal elements of SIGMA are the
 * singular values of A; they are real and non- negative, and are
 * returned in descending order.  The first \c min(m,n) columns of U
 * and V are the left and right singular vectors of A.
 *
 * Note that the routine returns VT = V**H (conjugate-transpose), not V.
 *
 * @param A The M-by-N input matrix to be decomposed. It will be
 * destroyed during the computation.
 *
 * @param Sigma A real-valued vector of length \c min(M,N) that will
 * return the singular values. WATCH OUT: The length has to be \e
 * exactly \c min(M,N) or else an exception will be thrown.
 *
 * @param U The M-by-M matrix of the left singular vectors.
 * @param VT The N-by-N matrix of the right singular vectors.
 */
DLLIMPORT
void LaSVD_IP(LaGenMatComplex& A, 
	      LaVectorDouble &Sigma, 
	      LaGenMatComplex& U, 
	      LaGenMatComplex& VT );

/** Compute the Singular Values.
 *
 * Compute the singular values of a singular value decomposition (SVD)
 * of a complex M-by-N matrix A.
 *
 * In lapack this is zgesdd. zgesdd computes the singular value
 * decomposition (SVD) of a complex M-by-N matrix A, optionally
 * computing the left and/or right singular vectors, by using
 * divide-and-conquer method. The SVD is written 
 *
 * \f[A = U \cdot Sigma \cdot V^T\f]
 *
 * where Sigma is an M-by-N matrix which is zero except for its \c
 * min(m,n) diagonal elements, U is an M-by-M unitary matrix, and V is
 * an N-by-N unitary matrix.  The diagonal elements of SIGMA are the
 * singular values of A; they are real and non- negative, and are
 * returned in descending order.
 *
 * @param A The M-by-N input matrix to be decomposed. It will be
 * destroyed during the computation.
 *
 * @param Sigma A real-valued vector of length \c min(M,N) that will
 * return the singular values. WATCH OUT: The length has to be \e
 * exactly \c min(M,N) or else an exception will be thrown.
 */
DLLIMPORT
void LaSVD_IP(LaGenMatComplex& A, 
	      LaVectorDouble &Sigma);
//@}
#endif // LA_COMPLEX_SUPPORT


/** @name Real-valued matrices */
//@{

/** Compute the Singular Value Decomposition. 
 *
 * Compute the singular value decomposition (SVD) of a complex M-by-N
 * matrix A, also computing the left and right singular vectors, by
 * using a divide-and-conquer method.
 *
 * In lapack this is dgesdd. dgesdd computes the singular value
 * decomposition (SVD) of a real M-by-N matrix A, optionally computing
 * the left and/or right singular vectors, by using divide-and-conquer
 * method. The SVD is written
 *
 * \f[A = U \cdot Sigma \cdot V^T\f]
 *
 * where Sigma is an M-by-N matrix which is zero except for its \c
 * min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and V
 * is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA are
 * the singular values of A; they are real and non- negative, and are
 * returned in descending order.  The first \c min(m,n) columns of U
 * and V are the left and right singular vectors of A.
 *
 * Note that the routine returns VT = V**T (transposed), not V.
 *
 * Now watch out: This routine has several modes of operation,
 * depending on the size of the input matrices \c U and \c VT. This is:
 *
 * - If \c U is M-by-M and \c VT is N-by-N (the normal mode), then \e
 * all left and right singular vectors are calculated and are returned
 * in \c U and \c VT.
 *
 * - If \c U is M-by-min(M,N) and \c VT is min(M,N)-by-N, then the
 * first min(M,N) left and right singular vectors are calculated,
 * respectively, and are returned in \c U and \c VT. FIXME: needs verification.
 *
 * - If M >= N, \c U is of size 0, and \c VT is N-by-N, then the first
 * N left singular vectors are calculated and returned in the first
 * columns of \c A, and all right singular vectors are calculated and
 * returned in \c VT. In this mode, \c U is unused. FIXME: needs verification.
 *
 * - If M < N, \c U is M-by-M, and \c VT is of size 0, then all left
 * singular vectors are calculated and returned in \c U, and the first
 * M right singular vectors are calculated and returned in the first M
 * rows of \c A. In this mode, \c VT is unused. FIXME: needs verification.
 *
 * In any other combination of matrix sizes, an exception is thrown.
 *
 * @param A The M-by-N input matrix to be decomposed. It will be
 * destroyed during the computation.
 *
 * @param Sigma A real-valued vector of length \c min(M,N) that will
 * return the singular values. WATCH OUT: The length has to be \e
 * exactly \c min(M,N) or else an exception will be thrown.
 *
 * @param U In the normal mode of calculation, the M-by-M matrix of
 * the left singular vectors. In other modes this might be unused.
 *
 * @param VT In the normal mode of calculation, the N-by-N matrix of
 * the right singular vectors. In other modes this might be unused.
 */
DLLIMPORT
void LaSVD_IP(LaGenMatDouble& A, LaVectorDouble &Sigma,
	      LaGenMatDouble& U, LaGenMatDouble& VT );

/** Compute the Singular Values.
 *
 * Compute the singular values of a singular value decomposition (SVD)
 * of a complex M-by-N matrix A.
 *
 * In lapack this is dgesdd. dgesdd computes the singular value
 * decomposition (SVD) of a real M-by-N matrix A, optionally computing
 * the left and/or right singular vectors, by using divide-and-conquer
 * method. The SVD is written
 *
 * \f[A = U \cdot Sigma \cdot V^T\f]
 *
 * where Sigma is an M-by-N matrix which is zero except for its \c
 * min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and V
 * is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA are
 * the singular values of A; they are real and non- negative, and are
 * returned in descending order.
 *
 * @param A The M-by-N input matrix to be decomposed. It will be
 * destroyed during the computation.
 *
 * @param Sigma A real-valued vector of length \c min(M,N) that will
 * return the singular values. WATCH OUT: The length has to be \e
 * exactly \c min(M,N) or else an exception will be thrown.
 */
DLLIMPORT
void LaSVD_IP(LaGenMatDouble& A, LaVectorDouble &Sigma);

//@}

#endif // _LASVD_H
