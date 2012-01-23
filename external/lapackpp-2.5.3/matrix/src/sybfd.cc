// Dominik Wagenfuehr <dominik.wagenfuehr@arcor.de>
// Copyright (C) 2006

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2, or
// (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License along
// with this library; see the file COPYING.  If not, write to the Free
// Software Foundation, 59 Temple Place - Suite 330, Boston, MA 02111-1307,
// USA.


// constructor/destructor functions

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "lafnames.h"
#include LA_EXCEPTION_H
#include LA_SYMM_BAND_FACT_DOUBLE_H
#include "lapack.h"

/// Factorize a real-valued symmetric positive definite band matrix with Cholesky in-place.
void LaSymmBandMatFactorizeIP(LaSymmBandMatDouble &A)
{
  char uplo='L';
  integer n = A.size(1), kd = A.subdiags(), lda = A.gdim(0), info = 0;

  F77NAME(dpbtrf)( &uplo, &n, &kd, &A(0,0), &lda, &info);

  LA_ASSERTZERO(info);
}

/// Factorize a real-valued symmetric positive definite band matrix with Cholesky.
void LaSymmBandMatFactorize(const LaSymmBandMatDouble &A,
			    LaSymmBandMatDouble& AF)
{
  AF.copy(A);
  LaSymmBandMatFactorizeIP(AF);
}

/// Solve A*X=B in-place where A is a real-valued symmetric positive definite band matrix.
void LaLinearSolveIP(LaSymmBandMatDouble &A, LaGenMatDouble &B)
{
  assert (A.size(1) == B.size(0));

  LaSymmBandMatFactorizeIP(A);

  char uplo = 'L';
  integer N = A.size(1);
  integer KD = A.subdiags();
  integer M = B.size(1);
  integer lda = A.gdim(0);
  integer ldb = B.gdim(0);
  integer info = 0;

  F77NAME(dpbtrs)(&uplo, &N, &KD, &M, &A(0,0), &lda, &B(0,0), &ldb, &info);
  
  assert (info == 0);
}

/// Solve A*X=B where A is a real-valued symmetric positive definite band matrix.
void LaLinearSolve(const LaSymmBandMatDouble A, LaGenMatDouble &X,
                            const LaGenMatDouble &B)
{
  LaSymmBandMatDouble AF(A);
  X.copy(B);
  LaLinearSolveIP(AF, X);
}

