//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.

// Dominik Wagenfuehr <dominik.wagenfuehr@arcor.de>
// Copyright (C) 2006

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


#ifndef _LA_SYMM_BAND_FACT_DOUBLE_H_
#define _LA_SYMM_BAND_FACT_DOUBLE_H_

/** @file
 * @brief Factorization and solving of real symmetric positive definite band matrices
*/

#include "arch.h"
#include "lafnames.h"
#include LA_SYMM_BAND_MAT_DOUBLE_H

/**
 * Factorize a real-valued symmetric positive definite band matrix
 * with Cholesky factorization, in-place.
 *
 * @param A On entry, a real-valued symmetric positive definite
 * band matrix A of dimension N x N.  On exit, the lower
 * triangular Cholesky factorization B with \f$ B^T \cdot B = A \f$
 */
void LaSymmBandMatFactorizeIP(LaSymmBandMatDouble &A);

/**
 * Factorize a real-valued symmetric positive definite band matrix
 * with Cholesky.
 *
 * @param A On entry, a real-valued symmetric positive definite
 * band matrix A of dimension N x N.
 *
 * @param AF On exit, the lower triangular Cholesky factorization
 * of A with \f$ AF^T \cdot AF = A \f$
 */
void LaSymmBandMatFactorize(const LaSymmBandMatDouble &A,
			    LaSymmBandMatDouble& AF);

/**
 * Solve A*X=B in-place where A is a real-valued symmetric
 * positive definite band matrix.
 *
 * The solution will be calulated in-place that means that A is
 * overwritten during the process with the Cholesky-factorization
 * and B will hold the solution afterwards.
 *
 * @param A On entry, the real-valued symmetric positive definite
 * band matrix A of dimension N x N.  On exit, the cholesky
 * factorization.
 *
 * @param B On entry, the general matrix B of dimension N x M.
 * On exit, the solution matrix X for A*X = B.
 */
void LaLinearSolveIP(LaSymmBandMatDouble &A, LaGenMatDouble &B);

/**
 * Solve A*X=B where A is a real-valued symmetric positive
 * definite band matrix.
 *
 * @param A On entry, a real-valued symmetric positive definite
 * band matrix A of dimension N x N.
 *
 * @param B On entry, the general matrix B of dimension N x M.
 *
 * @param X On exit, the solution matrix X with A*X = B.
 */
void LaLinearSolve(const LaSymmBandMatDouble A, LaGenMatDouble &X,
		   const LaGenMatDouble &B);

#endif 
// _LA_SYMM_BAND_FACT_DOUBLE_H_
