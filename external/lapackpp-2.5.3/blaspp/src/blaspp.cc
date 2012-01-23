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

// requires
//

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include "lafnames.h"
#include LA_EXCEPTION_H
#include "blas1pp.h"
#include "blas2pp.h"
#include "blas3pp.h"
#include <cmath>
#include LA_VECTOR_DOUBLE_H

#include LA_BAND_MAT_DOUBLE_H
#include LA_LOWER_TRIANG_MAT_DOUBLE_H
#include LA_SPD_MAT_DOUBLE_H
#include LA_SYMM_BAND_MAT_DOUBLE_H 
#include LA_SYMM_MAT_DOUBLE_H
#include LA_SYMM_TRIDIAG_MAT_DOUBLE_H
#include LA_TRIDIAG_MAT_DOUBLE_H
#include LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H
#include LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H
#include LA_UPPER_TRIANG_MAT_DOUBLE_H

#include "blaspp.h"
#include "blas3.h"


// Only enable this when LA_NO_DEPRECATED is not defined
#ifndef LA_NO_DEPRECATED

//-------------------------------------
// Vector/Vector operators
//-------------------------------------

LaVectorDouble operator*(const LaVectorDouble &x, double a)
{
    int N = x.size();
    LaVectorDouble t(N);

    for (int i=0; i<N; i++)
    {
        t(i) = a * x(i);
    }

    return t;
}
    

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
double operator*(const LaVectorDouble &dx, 
                                const LaVectorDouble &dy)
{
    assert(dx.size()==dy.size());
    integer incx = dx.inc(), incy = dy.inc(), n = dx.size();

    return F77NAME(ddot)(&n, &dx(0), &incx, &dy(0), &incy);
}
                      
/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
LaVectorDouble operator+(const LaVectorDouble &dx, 
                                const LaVectorDouble &dy)
{
    assert(dx.size()==dy.size());
    integer incx = dx.inc(), incy = dx.inc(), n = dx.size();
    double da = 1.0;

    LaVectorDouble tmp((int) n);
    tmp = dy;

    F77NAME(daxpy)(&n, &da, &dx(0), &incx, &tmp(0), &incy);
    return tmp;
}

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
LaVectorDouble operator-(const LaVectorDouble &dx, 
                                const LaVectorDouble &dy)
{
    assert(dx.size()==dy.size());
    integer incx = dx.inc(), incy = dy.inc(), n = dx.size();
    double da = -1.0;

    LaVectorDouble tmp(n);
    tmp = dx;

    F77NAME(daxpy)(&n, &da, &dy(0), &incx, &tmp(0), &incy);
    return tmp;
}
//@}

//-------------------------------------
/// @name Matrix/Vector operators (deprecated) 
//-------------------------------------
//@{

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
LaVectorDouble operator*(const LaGenMatDouble &A, 
                                const LaVectorDouble &dx)
{
    char trans = 'N';
    double alpha = 1.0, beta = 0.0;
    integer M = A.size(0), N = A.size(1), lda = A.gdim(0);

    LaVectorDouble dy(M);
    integer incx = dx.inc();
    integer incy = dy.inc();

    // dy = 0.0; -- unneeded since beta is zero

    F77NAME(dgemv)(&trans, &M, &N, &alpha, &A(0,0), &lda, &dx(0), &incx, 
        &beta, &dy(0), &incy); 
    return dy; 
        
}

#ifdef _LA_BAND_MAT_DOUBLE_H_
LaVectorDouble operator*(const LaBandMatDouble &A, 
                                const LaVectorDouble &dx)
{
    char trans = 'N';
    double alpha = 1.0, beta = 0.0;
    integer M = A.size(0), N = A.size(1), lda = A.gdim(0),
        kl = A.subdiags(), ku = A.superdiags(); 

    LaVectorDouble dy(N);
    integer incx = dx.inc(), incy = dy.inc();

    F77NAME(dgbmv)(&trans, &M, &N, &kl, &ku, &alpha, &A(0,0), &lda,
                   &dx(0), &incx, &beta, &dy(0), &incy);
    return dy;
}
#endif

#ifdef _LA_SYMM_MAT_DOUBLE_H_
LaVectorDouble operator*(const LaSymmMatDouble &A, 
                                const LaVectorDouble &dx)
{
    char uplo = 'L';
    double alpha = 1.0, beta = 0.0;
    integer N = A.size(1), lda = A.gdim(0);

    LaVectorDouble dy(N);
    integer incx = dx.inc(), incy = dy.inc();

    F77NAME(dsymv)(&uplo, &N, &alpha, &A(0,0), &lda, &dx(0), &incx,
                   &beta, &dy(0), &incy);
    return dy;
}
#endif

#ifdef _LA_SYMM_BAND_MAT_DOUBLE_H_ 
LaVectorDouble operator*(const LaSymmBandMatDouble &A, 
        const LaVectorDouble &dx) 
{
    char uplo = 'L';
    double alpha = 1.0, beta = 0.0;
    integer N = A.size(1), lda = A.gdim(0), k = A.subdiags();

    LaVectorDouble dy(N);
    integer incx = dx.inc(), incy = dy.inc();

    F77NAME(dsbmv)(&uplo, &N, &k, &alpha, &A(0,0), &lda, &dx(0), &incx,
                   &beta, &dy(0), &incy);
    return dy;
}

#endif


#ifdef _LA_SPD_MAT_DOUBLE_H_
LaVectorDouble operator*(const LaSpdMatDouble &AP, 
                                const LaVectorDouble &dx)
{
    char uplo = 'L';
    double alpha = 1.0, beta = 0.0;
    integer N = AP.size(1), incx = dx.inc(); 
    integer lda = AP.gdim(0);

    LaVectorDouble dy(N);
    integer incy = dy.inc();

    F77NAME(dsymv)(&uplo, &N, &alpha, &AP(0,0), &lda, &dx(0), &incx, &beta,
                    &dy(0), &incy);
    return dy;
}
#endif

#ifdef _LA_LOWER_TRIANG_MAT_DOUBLE_H_
LaVectorDouble operator*(const LaLowerTriangMatDouble &A, 
                                const LaVectorDouble &dx)
{
    char uplo = 'L', trans = 'N', diag = 'N';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();

    LaVectorDouble dy(dx);

    F77NAME(dtrmv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dy(0), &incx);

    return dy;
}
#endif

#ifdef _LA_UPPER_TRIANG_MAT_DOUBLE_H_
LaVectorDouble operator*(const LaUpperTriangMatDouble &A, 
                                const LaVectorDouble &dx)
{
    char uplo = 'U', trans = 'N', diag = 'N';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();

    LaVectorDouble dy(dx);

    F77NAME(dtrmv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dy(0), &incx);

    return dy;
}
#endif

#ifdef _LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H_
LaVectorDouble operator*(const LaUnitLowerTriangMatDouble &A,
                                const LaVectorDouble &dx)
{
    char uplo = 'L', trans = 'N', diag = 'U';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();

    LaVectorDouble dy(dx);

    F77NAME(dtrmv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dy(0), &incx);

    return dy;
}

#endif

#ifdef _LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H_
LaVectorDouble operator*(const LaUnitUpperTriangMatDouble &A,
                                const LaVectorDouble &dx)
{
    char uplo = 'U', trans = 'N', diag = 'U';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();

    LaVectorDouble dy(dx);

    F77NAME(dtrmv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dy(0), &incx);

    return dy;
}
#endif
//@}

//-------------------------------------
/// @name Matrix/Matrix operators (deprecated) 
//-------------------------------------
//@{
/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
LaGenMatDouble operator*(const LaGenMatDouble &A, 
                                const LaGenMatDouble &B)
{
    char t = 'N';
    integer m = A.size(0), k = A.size(1), n = B.size(1);
    integer lda = A.gdim(0), ldb = B.gdim(0);
    double alpha = 1.0, beta = 0.0;

    LaGenMatDouble C(m,n);
    integer ldc = A.gdim(0);

    //C = 0.0; -- beta is zero, doesn't need to be set

  F77NAME(dgemm)(&t, &t, &m, &n, &k, &alpha, &A(0,0), &lda, &B(0,0), &ldb,
                &beta, &C(0,0), &ldc);

    return C.shallow_assign();
}

#ifdef _LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H_
LaGenMatDouble operator*(const LaUnitLowerTriangMatDouble &A,
                                const LaGenMatDouble &B)
{
        char side = 'L', uplo = 'L', transa = 'N', diag = 'U';
        double alpha = 1.0;
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

        LaGenMatDouble C(B);

  F77NAME(dtrmm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &C(0,0), &ldb);

        return C;
}
#endif

#ifdef _LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H_
LaGenMatDouble operator*(const LaUnitUpperTriangMatDouble &A,
                                const LaGenMatDouble &B)
{
        char side = 'L', uplo = 'U', transa = 'N', diag = 'U';
        double alpha = 1.0;
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

        LaGenMatDouble C(B);

  F77NAME(dtrmm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &C(0,0), &ldb);

        return C;
}
#endif

#ifdef _LA_LOWER_TRIANG_MAT_DOUBLE_H_
LaGenMatDouble operator*(const LaLowerTriangMatDouble &A,
                                const LaGenMatDouble &B)
{
        char side = 'L', uplo = 'L', transa = 'N', diag = 'N';
        double alpha = 1.0;
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

        LaGenMatDouble C(B);

  F77NAME(dtrmm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &C(0,0), &ldb);

        return C;
}
#endif

#ifdef _LA_UPPER_TRIANG_MAT_DOUBLE_H_
LaGenMatDouble operator*(const LaUpperTriangMatDouble &A,
                                const LaGenMatDouble &B)
{
        char side = 'L', uplo = 'U', transa = 'N', diag = 'N';
        double alpha = 1.0;
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

        LaGenMatDouble C(B);

  F77NAME(dtrmm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &C(0,0), &ldb);

        return C;
}
#endif

#ifdef _LA_SYMM_MAT_DOUBLE_H_
LaGenMatDouble operator*(const LaSymmMatDouble &A, 
                                const LaGenMatDouble &B)
{
        char side = 'L', uplo = 'L';
        double alpha = 1.0, beta = 0.0;
        LaGenMatDouble C(B.size(1),A.size(1));
        integer m = C.size(0), n = C.size(1), lda = A.gdim(0),
                ldb = B.gdim(0), ldc = C.gdim(0);

  F77NAME(dsymm)(&side, &uplo, &m, &n, &alpha, &A(0,0), &lda,
                &B(0,0), &ldb, &beta, &C(0,0), &ldc);

        return C;
}
#endif

#ifdef _LA_SYMM_TRIDIAG_MAT_DOUBLE_H_
LaVectorDouble operator*(const LaSymmTridiagMatDouble& A, 
                                const LaVectorDouble& X)
{
    integer M = A.size();
    integer N = X.size();
    LaVectorDouble R(M);

    R(0) = ((A.diag(0)(0) * X(0)) + (A.diag(-1)(0) * X(1)));

    for (integer i = 1; i < M-2; i++)
    {
        R(i) = ((A.diag(-1)(i-1) * X(i-1)) +
                (A.diag(0)(i) * X(i)) +
                (A.diag(-1)(i) * X(i+1)));
    }

    R(M-1) = ((A.diag(0)(M-1) * X(N-1)) + (A.diag(-1)(M-2) * X(N-2)));

    return R;
}
#endif

#ifdef  _LA_TRIDIAG_MAT_DOUBLE_H_
LaVectorDouble operator*(const LaTridiagMatDouble& A, 
                                const LaVectorDouble& X)
{
    integer M = A.size();
    integer N = X.size();
    LaVectorDouble R(M);

    R(0) = ((A.diag(0)(0) * X(0)) + (A.diag(-1)(0) * X(1)));

    for (integer i = 1; i < M-2; i++)
    {
        R(i) = ((A.diag(-1)(i-1) * X(i-1)) +
                (A.diag(0)(i) * X(i)) +
                (A.diag(1)(i) * X(i+1)));
    }

    R(M-1) = ((A.diag(0)(M-1) * X(N-1)) + (A.diag(1)(M-2) * X(N-2)));

    return R;
}
#endif

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
LaGenMatDouble operator-(const LaGenMatDouble &A, 
                                const LaGenMatDouble &B)
{
#ifndef HPPA
    const char fname[] = "operator+(A,B)";
#else
    char *fname = NULL;
#endif

    integer M = A.size(0);
    integer N = A.size(1);

    if (M != B.size(0) || N != B.size(1))
    {
        throw(LaException(fname, "matrices non-conformant."));
    }

    LaGenMatDouble C(M,N);

    // slow mode
    // we'll hook the BLAS in later

    for (integer i=0;  i<M; i++)
        for(integer j=0; j<N; j++)
            C(i,j) = A(i,j) - B(i,j);

    return C;
}

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
LaGenMatDouble operator+(const LaGenMatDouble &A, 
                                const LaGenMatDouble &B)
{
#ifndef HPPA
    const char fname[] = "operator+(A,B)";
#else
    char *fname = NULL;
#endif

    integer M = A.size(0);
    integer N = A.size(1);

    if (M != B.size(0) || N != B.size(1))
    {
        throw(LaException(fname, "matrices non-conformant."));
    }

    LaGenMatDouble C(M,N);

    // slow mode
    // we'll hook the BLAS in later

    for (integer i=0;  i<M; i++)
        for(integer j=0; j<N; j++)
            C(i,j) = A(i,j) + B(i,j);

    return C;
}

# ifdef LA_COMPLEX_SUPPORT
LaGenMatComplex operator+(const LaGenMatComplex &A, 
                                const LaGenMatComplex &B)
{
#ifndef HPPA
    const char fname[] = "operator+(A,B)";
#else
    char *fname = NULL;
#endif

    integer M = A.size(0);
    integer N = A.size(1);

    if (M != B.size(0) || N != B.size(1))
    {
        throw(LaException(fname, "matrices non-conformant."));
    }

    LaGenMatComplex C(M,N);

    // slow mode
    // we'll hook the BLAS in later

    for (integer i=0;  i<M; i++)
        for(integer j=0; j<N; j++)
            C(i,j) = LaComplex(A(i,j)) + LaComplex(B(i,j));

    return C;
}

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
LaGenMatComplex operator-(const LaGenMatComplex &A, 
                                const LaGenMatComplex &B)
{
#ifndef HPPA
    const char fname[] = "operator+(A,B)";
#else
    char *fname = NULL;
#endif

    integer M = A.size(0);
    integer N = A.size(1);

    if (M != B.size(0) || N != B.size(1))
    {
        throw(LaException(fname, "matrices non-conformant."));
    }

    LaGenMatComplex C(M,N);

    // slow mode
    // we'll hook the BLAS in later

    for (integer i=0;  i<M; i++)
        for(integer j=0; j<N; j++)
            C(i,j) = LaComplex(A(i,j)) - LaComplex(B(i,j));

    return C;
}
//@}
# endif // LA_COMPLEX_SUPPORT

#endif // LA_NO_DEPRECATED

