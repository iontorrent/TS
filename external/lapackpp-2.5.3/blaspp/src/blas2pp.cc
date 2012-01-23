//
//              LAPACK++ 1.1 Linear Algebra Package 1.1
//               University of Tennessee, Knoxvilee, TN.
//            Oak Ridge National Laboratory, Oak Ridge, TN.
//        Authors: J. J. Dongarra, E. Greaser, R. Pozo, D. Walker
//                 (C) 1992-1996 All Rights Reserved
//
//                             NOTICE
//
// Permission to use, copy, modify, and distribute this software and
// its documentation for any purpose and without fee is hereby granted
// provided that the above copyright notice appear in all copies and
// that both the copyright notice and this permission notice appear in
// supporting documentation.
//
// Neither the Institutions (University of Tennessee, and Oak Ridge National
// Laboratory) nor the Authors make any representations about the suitability 
// of this software for any purpose.  This software is provided ``as is'' 
// without express or implied warranty.
//
// LAPACK++ was funded in part by the U.S. Department of Energy, the
// National Science Foundation and the State of Tennessee.

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "lafnames.h"
#include LA_EXCEPTION_H
#include LA_VECTOR_DOUBLE_H
#include LA_SYMM_MAT_DOUBLE_H
#include LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H
#include LA_UPPER_TRIANG_MAT_DOUBLE_H
#include LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H
#include LA_LOWER_TRIANG_MAT_DOUBLE_H
#include LA_SPD_MAT_DOUBLE_H
#include LA_SYMM_BAND_MAT_DOUBLE_H
#include LA_TRIDIAG_MAT_DOUBLE_H

#ifdef LA_COMPLEX_SUPPORT
#  include "f2c.h"
#  include "lapackc.h"
#  include LA_VECTOR_COMPLEX_H
#  include LA_GEN_MAT_COMPLEX_H
#endif

#include "blas2pp.h"

void Blas_Mat_Trans_Vec_Mult(const LaGenMatDouble &A, const LaVectorDouble &dx,
            LaVectorDouble &dy, double alpha, double beta)
{
    char trans = 'T';
    integer M = A.size(0), N = A.size(1), lda = A.gdim(0),
        incx = dx.inc(), incy = dy.inc();
    assert(A.size(1) == dy.size());
    assert(A.size(0) == dx.size());


    F77NAME(dgemv)(&trans, &M, &N, &alpha, &A(0,0), &lda, &dx(0), &incx,
                    &beta, &dy(0), &incy);

}

void Blas_Mat_Vec_Mult(const LaGenMatDouble &A, const LaVectorDouble &dx,
            LaVectorDouble &dy, double alpha , double beta )
{
    char trans = 'N';
    integer M = A.size(0), N = A.size(1), lda = A.gdim(0),
        incx = dx.inc(), incy = dy.inc();
    assert(A.size(0) == dy.size());
    assert(A.size(1) == dx.size());


    F77NAME(dgemv)(&trans, &M, &N, &alpha, &A(0,0), &lda, &dx(0), &incx,
                    &beta, &dy(0), &incy);

}

void Blas_Mat_Vec_Mult(const LaSymmBandMatDouble &A, const LaVectorDouble &dx, 
            LaVectorDouble &dy, double alpha , double beta )
{
    char uplo = 'L';
    integer N = A.size(1), lda = A.gdim(0),
        k = A.subdiags(), incx = dx.inc(), incy = dy.inc();
    assert(A.size(0) == dy.size());
    assert(A.size(1) == dx.size());


    F77NAME(dsbmv)(&uplo, &N, &k, &alpha, &A(0,0), &lda, &dx(0), &incx,
                   &beta, &dy(0), &incy);

}


void Blas_Mat_Vec_Mult(const LaSpdMatDouble &AP, const LaVectorDouble &dx, 
            LaVectorDouble &dy, double alpha , double beta )
{
    char uplo = 'L';
    integer N = AP.size(1), incx = dx.inc(), incy = dy.inc();
    assert(AP.size(0) == dy.size());
    assert(AP.size(1) == dx.size());

    F77NAME(dspmv)(&uplo, &N, &alpha, const_cast<double*>(&AP(0,0)), &dx(0), &incx, &beta, 
                   &dy(0), &incy);

}



void Blas_Mat_Vec_Mult(const LaLowerTriangMatDouble &A, LaVectorDouble &dx)
{
    char uplo = 'L', trans = 'N', diag = 'N';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();
    assert(A.size(0) == dx.size());
    assert(A.size(1) == dx.size());


    F77NAME(dtrmv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dx(0), &incx);

}



void Blas_Mat_Vec_Mult(const LaUpperTriangMatDouble &A, LaVectorDouble &dx)
{
    char uplo = 'U', trans = 'N', diag = 'N';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();
    assert(A.size(0) == dx.size());
    assert(A.size(1) == dx.size());


    F77NAME(dtrmv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dx(0), &incx);

}



void Blas_Mat_Vec_Mult(const LaUnitLowerTriangMatDouble &A, 
		       LaVectorDouble &dx)
{
    char uplo = 'L', trans = 'N', diag = 'U';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();
    assert(A.size(0) == dx.size());
    assert(A.size(1) == dx.size());


    F77NAME(dtrmv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dx(0), &incx);

}



void Blas_Mat_Vec_Mult(const LaUnitUpperTriangMatDouble &A, 
		       LaVectorDouble &dx)
{
    char uplo = 'U', trans = 'N', diag = 'U';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();
    assert(A.size(0) == dx.size());
    assert(A.size(1) == dx.size());


    F77NAME(dtrmv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dx(0), &incx);

}


void Blas_Mat_Vec_Solve(LaLowerTriangMatDouble &A, LaVectorDouble &dx)
{
    char uplo = 'L', trans = 'N', diag = 'N';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();


    F77NAME(dtrsv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dx(0), &incx);

}


void Blas_Mat_Vec_Solve(LaUpperTriangMatDouble &A, LaVectorDouble &dx)
{
    char uplo = 'U', trans = 'N', diag = 'N';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();


    F77NAME(dtrsv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dx(0), &incx);

}


void Blas_Mat_Vec_Solve(LaUnitLowerTriangMatDouble &A, 
                                LaVectorDouble &dx)
{
    char uplo = 'L', trans = 'N', diag = 'U';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();


    F77NAME(dtrsv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dx(0), &incx);

}


void Blas_Mat_Vec_Solve(LaUnitUpperTriangMatDouble &A, 
                                LaVectorDouble &dx)
{
    char uplo = 'U', trans = 'N', diag = 'U';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();


    F77NAME(dtrsv)(&uplo, &trans, &diag, &N, &A(0,0), &lda, &dx(0), &incx);

}


void Blas_R1_Update(LaGenMatDouble &A, const LaVectorDouble &dx, 
		    const LaVectorDouble &dy, double alpha )
{
    integer M = A.size(0), N = A.size(1), lda = A.gdim(0),
        incx = dx.inc(), incy = dy.inc();
    assert(A.size(0) == dx.size());
    assert(A.size(1) == dy.size());


    F77NAME(dger)(&M, &N, &alpha, &dx(0), 
		  &incx, &dy(0), &incy, 
                  &A(0,0), &lda);

}


void Blas_R1_Update(LaSymmMatDouble &A, const LaVectorDouble &dx,
                double alpha )
{
    char uplo = 'L';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc();
    assert(A.size(0) == dx.size());
    assert(A.size(1) == dx.size());


    F77NAME(dsyr)(&uplo, &N, &alpha, &dx(0), &incx, &A(0,0), &lda);

}


void Blas_R1_Update(LaSpdMatDouble &AP, const LaVectorDouble &dx,
                double alpha )
{
    char uplo = 'L';
    integer N = AP.size(1),
        incx = dx.inc();
    assert(AP.size(0) == dx.size());
    assert(AP.size(1) == dx.size());


    F77NAME(dspr)(&uplo, &N, &alpha, &dx(0), &incx, &AP(0,0));

}



void Blas_R2_Update(LaSymmMatDouble &A, const LaVectorDouble &dx, 
		    const LaVectorDouble &dy, double alpha )
{
    char uplo = 'L';
    integer N = A.size(1), lda = A.gdim(0),
        incx = dx.inc(), incy = dy.inc();


    F77NAME(dsyr2)(&uplo, &N, &alpha, &dx(0), &incx, &dy(0), &incy, 
                  &A(0,0), &lda);

}


void Blas_R2_Update(LaSpdMatDouble &AP, const LaVectorDouble &dx, 
		    const LaVectorDouble &dy, double alpha )
{
    char uplo = 'L';
    integer N = AP.size(1), incx = dx.inc(), incy = dy.inc();
    assert(AP.size(0) == dx.size());
    assert(AP.size(0) == dy.size());


    F77NAME(dspr2)(&uplo, &N, &alpha, &dx(0), &incx, &dy(0), &incy,
                  &AP(0,0));

}

#ifdef LA_COMPLEX_SUPPORT
void Blas_Mat_Vec_Mult(const LaGenMatComplex &A, 
		       const LaVectorComplex &dx, 
		       LaVectorComplex &dy, 
		       LaComplex _alpha, LaComplex _beta)
{
    char trans = 'N';
    doublecomplex alpha(_alpha);
    doublecomplex beta(_beta);
    integer M = A.size(0), N = A.size(1), lda = A.gdim(0),
        incx = dx.inc(), incy = dy.inc();
    assert(A.size(0) == dy.size());
    assert(A.size(1) == dx.size());

    F77NAME(zgemv)(&trans, &M, &N, &alpha, &A(0,0), &lda, &dx(0), &incx,
                    &beta, &dy(0), &incy);

}
void Blas_Mat_Trans_Vec_Mult(const LaGenMatComplex &A, 
			     const LaVectorComplex &dx, 
			     LaVectorComplex &dy,
			     LaComplex _alpha, LaComplex _beta)
{
    char trans = 'C';
    integer M = A.size(0), N = A.size(1), lda = A.gdim(0),
        incx = dx.inc(), incy = dy.inc();
    doublecomplex alpha(_alpha);
    doublecomplex beta(_beta);
    assert(A.size(1) == dy.size());
    assert(A.size(0) == dx.size());


    F77NAME(zgemv)(&trans, &M, &N, &alpha, &A(0,0), &lda, &dx(0), &incx,
                    &beta, &dy(0), &incy);

}
void Blas_R1_Update(LaGenMatComplex &A, const LaVectorComplex &dx, 
		    const LaVectorComplex &dy, LaComplex _alpha)
{
    integer M = A.size(0), N = A.size(1), lda = A.gdim(0),
        incx = dx.inc(), incy = dy.inc();
    doublecomplex alpha(_alpha);
    assert(A.size(0) == dx.size());
    assert(A.size(1) == dy.size());


    F77NAME(zgerc)(&M, &N, &alpha, &dx(0), 
		  &incx, &dy(0), &incy, 
                  &A(0,0), &lda);

}
#endif /*  LA_COMPLEX_SUPPORT */
