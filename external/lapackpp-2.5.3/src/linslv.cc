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

#include <iostream>
#include "lapack.h"
#include "lapackc.h"
#include "lafnames.h"
#include LA_EXCEPTION_H
#include LA_GEN_MAT_DOUBLE_H
#include LA_VECTOR_DOUBLE_H
#include LA_VECTOR_LONG_INT_H 
#ifdef LA_COMPLEX_SUPPORT
#  include LA_GEN_MAT_COMPLEX_H
#  include LA_VECTOR_COMPLEX_H
#endif
#include LA_SPD_MAT_DOUBLE_H
#include LA_SYMM_MAT_DOUBLE_H
#include "blas3pp.h"

#include LA_SOLVE_DOUBLE_H
#include LA_UTIL_H


void LaLinearSolve( const LaGenMatDouble& A, LaGenMatDouble& X,
		    const LaGenMatDouble& B)
{
    int M = A.size(0), N = A.size(1);

    if ( M == N ) 
        LaLULinearSolve(A,X,B);
    else 
        LaQRLinearSolve(A,X,B);
}   
    
void LaLinearSolve(const LaSpdMatDouble &A, LaGenMatDouble& X, 
		   LaGenMatDouble& B )
{
    LaCholLinearSolve(A, X, B );
}

void LaLinearSolve(const LaSymmMatDouble &A, LaGenMatDouble& X, 
		   const LaGenMatDouble& B )
{
    LaCholLinearSolve(A, X, B );
}

void LaLinearSolveIP(LaSpdMatDouble &A, LaGenMatDouble& X, LaGenMatDouble& B )
{
    LaCholLinearSolveIP(A, X, B );
}

void LaLinearSolveIP(LaSymmMatDouble &A, LaGenMatDouble& X, 
		     const LaGenMatDouble& B )
{
    LaCholLinearSolveIP(A, X, B );
}

void LaLinearSolveIP( LaGenMatDouble& A, LaGenMatDouble& X,
		      const LaGenMatDouble& B)
{
    int M = A.size(0), N = A.size(1);

    if ( M == N ) 
        LaLULinearSolveIP(A,X,B);
    else 
        LaQRLinearSolveIP(A,X,B);
}   

void LaLULinearSolve(const LaGenMatDouble& A, LaGenMatDouble& X, 
		     const LaGenMatDouble& B )
{
    LaGenMatDouble A1(A);   // exception if out of memory
    LaLULinearSolveIP(A1, X, B);
}


// General LU Solver
// 
//                        N x N               N x nrhs          N x nrhs
//
void LaLULinearSolveIP( LaGenMatDouble& A, LaGenMatDouble& X, 
			const LaGenMatDouble& B )
{
#ifndef HPPA
     const char fname[] = "LaLULinearSolveIP(LaGenMatDouble &A, &X, &B)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif


    // let's not worry about non-unit column strides for the moment
    if ( A.inc(0) != 1 || A.inc(1) != 1)
        throw(LaException(fname, "A is  non-contiguous."));

    if (!(X.size(0) == B.size(0) && X.size(1) == B.size(1)))
        throw(LaException(fname, "X and B are non-conformant."));

    X.inject(B);            // will throw exception if not conformant


    // in the future this can call the linear least square routines
    // to handle non-square matrices

    if (A.size(0) != A.size(1))
        throw(LaException(fname, "Square matrix expected.\n"));

    if (A.size(1) != X.size(0))
        throw(LaException(fname, "A and X are non-comformant."));

    integer info = 0;
    int M = A.size(0);
    integer Ml = M;
    //integer N = A.size(1);
    integer K = X.size(1);
    integer lda = A.inc(0) * A.gdim(0);
    integer ldx = X.inc(0) * X.gdim(0);

    LaVectorLongInt ipiv( M);        

    F77NAME(dgesv) (&Ml, &K, &A(0,0), &lda, &ipiv(0), &X(0,0), &ldx, &info);

    if (info < 0)
	throw(LaException(fname, "Internal error in LAPACK: DGESV() with illegal argument value"));
    else if (info > 0)
	throw(LaException(fname, "Internal error in LAPACK: DGESV() Factor U was exactly singular"));
}



    

void LaQRLinearSolve(const LaGenMatDouble& A, LaGenMatDouble& X, 
		     const LaGenMatDouble& B )
{
    LaGenMatDouble A1(A);
    LaQRLinearSolveIP(A1, X, B);
}

    
// General QR solver
//
//                          M x N              N x nrhs           M  x nrhs
//
void LaQRLinearSolveIP(LaGenMatDouble& A, LaGenMatDouble& X, 
		       const LaGenMatDouble& B )
{
#ifndef HPPA
    const char fname[] = "LaQRLinearSolveIP(LaGenMatDouble &A, &X, &B)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif

    // let's not worry about non-unit column strides for the moment
    if ( A.inc(0) != 1 || A.inc(1) != 1)
        throw(LaException(fname, "A is  non-contiguous."));
    if ( A.size(0) == 0 || A.size(1) == 0 )
        throw(LaException(fname, "Matrix A is empty; one dimension is zero."));

    if (!(  A.size(0) == B.size(0) &&
            A.size(1) == X.size(0) &&
            X.size(1) == B.size(1) ))
        throw(LaException(fname, "input matrices are non-conformant."));

    integer info = 0;
    int M = A.size(0);
    int N = A.size(1);
    integer Ml = M;
    integer Nl = N;
    int nrhs = X.size(1);
    integer nrhsl = nrhs;
    integer lda = A.inc(0) * A.gdim(0);

    int nb = LaEnvBlockSize("DGELS", A);
    integer lwork = M * N + nb * std::max(M * N, nrhs);
    //std::cout << fname << ": nb= " << nb << "  lwork=" << lwork << std::endl;

    LaVectorDouble WORK(lwork);         
    char trans = 'N';
    
    if (M != N)
    {
	// Typically is A non-square, so we need to create tmp X because 
	// X is N x nrhs, while B is M x nrhs.  We need to make copies of
	// these so that the routine won't corrupt data around X and B.

        LaGenMatDouble Xtmp(std::max(M, N), nrhs);
        integer ldx = Xtmp.inc(0) * Xtmp.gdim(0);

	// Copy B into the temporary X matrix that is passed to dgels()
        Xtmp(LaIndex(0,M-1), LaIndex()).inject( B );

        F77NAME(dgels) (&trans, &Ml, &Nl, &nrhsl, &A(0,0), &lda, &Xtmp(0,0), 
                &ldx, &WORK(0), &lwork, &info);

	// And copy the result from the larger matrix back into
	// the actual result matrix.
        X.inject(Xtmp(LaIndex(0,N-1), LaIndex()));
    }
    else
    {
        integer ldx = X.inc(0) * X.gdim(0);

	// Copy B into the X matrix that is passed to dgels()
        X.inject( B );

        F77NAME(dgels) (&trans, &Ml, &Nl, &nrhsl, &A(0,0), &lda, &X(0,0), 
                &ldx, &WORK(0), &lwork, &info);
    }


    // this shouldn't really happen.
    //
    if (info < 0)
        throw(LaException(fname, "Internal error in LAPACK: SGELS()"));

}
// ////////////////////////////////////////////////////////////

#ifdef LA_COMPLEX_SUPPORT
void LaLinearSolve( const LaGenMatComplex& A, LaGenMatComplex& X,
		    const LaGenMatComplex& B)
{
    int M = A.size(0), N = A.size(1);

    if ( M == N ) 
        LaLULinearSolve(A,X,B);
    else 
        LaQRLinearSolve(A,X,B);
}   

void LaLinearSolveIP( LaGenMatComplex& A, LaGenMatComplex& X,
		      const LaGenMatComplex& B)
{
    int M = A.size(0), N = A.size(1);

    if ( M == N ) 
        LaLULinearSolveIP(A,X,B);
    else 
        LaQRLinearSolveIP(A,X,B);
}   

void LaLULinearSolve(const  LaGenMatComplex& A, LaGenMatComplex& X, 
		     const LaGenMatComplex& B )
{
    LaGenMatComplex A1(A);   // exception if out of memory
    LaLULinearSolveIP(A1, X, B);
}


// General LU Solver
// 
//                        N x N               N x nrhs          N x nrhs
//
void LaLULinearSolveIP( LaGenMatComplex& A, LaGenMatComplex& X, 
			const LaGenMatComplex& B )
{
#ifndef HPPA
     const char fname[] = "LaLULinearSolveIP(LaGenMatComplex &A, &X, &B)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif


    // let's not worry about non-unit column strides for the moment
    if ( A.inc(0) != 1 || A.inc(1) != 1)
        throw(LaException(fname, "A is  non-contiguous."));

    if (!(X.size(0) == B.size(0) && X.size(1) == B.size(1)))
        throw(LaException(fname, "X and B are non-conformant."));

    X.inject(B);            // will throw exception if not conformant


    // in the future this can call the linear least square routines
    // to handle non-square matrices

    if (A.size(0) != A.size(1))
        throw(LaException(fname, "Square matrix expected.\n"));

    if (A.size(1) != X.size(0))
        throw(LaException(fname, "A and X are non-comformant."));

    integer info = 0;
    int M = A.size(0);
    integer Ml = M;
    //integer N = A.size(1);
    integer K = X.size(1);
    integer lda = A.inc(0) * A.gdim(0);
    integer ldx = X.inc(0) * X.gdim(0);

    LaVectorLongInt ipiv( M);        

    F77NAME(zgesv) (&Ml, &K, &A(0,0), &lda, &ipiv(0), &X(0,0), &ldx, &info);

    if (info < 0)
	throw(LaException(fname, "Internal error in LAPACK: DGESV() with illegal argument value"));
    else if (info > 0)
	throw(LaException(fname, "Internal error in LAPACK: DGESV() Factor U was exactly singular"));
}



    

void LaQRLinearSolve(const LaGenMatComplex& A, LaGenMatComplex& X, 
		     const LaGenMatComplex& B )
{
    LaGenMatComplex A1(A);
    LaQRLinearSolveIP(A1, X, B);
}

    
// General QR solver
//
//                          M x N              N x nrhs           M  x nrhs
//
void LaQRLinearSolveIP(LaGenMatComplex& A, LaGenMatComplex& X, const LaGenMatComplex& B )
{
#ifndef HPPA
    const char fname[] = "LaQRLinearSolveIP(LaGenMatComplex &A, &X, &B)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif

    // let's not worry about non-unit column strides for the moment
    if ( A.inc(0) != 1 || A.inc(1) != 1)
        throw(LaException(fname, "A is  non-contiguous."));
    if ( A.size(0) == 0 || A.size(1) == 0 )
        throw(LaException(fname, "Matrix A is empty; one dimension is zero."));

    if (!(  A.size(0) == B.size(0) &&
            A.size(1) == X.size(0) &&
            X.size(1) == B.size(1) ))
        throw(LaException(fname, "input matrices are non-conformant."));

    integer info = 0;
    int M = A.size(0);
    int N = A.size(1);
    integer Ml = M;
    integer Nl = N;
    int nrhs = X.size(1);
    integer nrhsl = nrhs;
    integer lda = A.inc(0) * A.gdim(0);

        
    //int nb = 32;
    int nb = LaEnvBlockSize("ZGELS", A);
    integer lwork = M * N + nb * std::max(M*N, nrhs);
    //std::cout << "Block size: " << nb << std::endl;

    LaVectorComplex WORK(lwork);         
    char trans = 'N';
    
    if (M != N)
    {
	// Typically is A non-square, so we need to create tmp X because 
	// X is N x nrhs, while B is M x nrhs.  We need to make copies of
	// these so that the routine won't corrupt data around X and B.

        LaGenMatComplex Xtmp(std::max(M, N), nrhs);
        integer ldx = Xtmp.inc(0) * Xtmp.gdim(0);

	// Copy B into the temporary X matrix that is passed to zgels()
        Xtmp(LaIndex(0,M-1),LaIndex()).inject( B );

        F77NAME(zgels) (&trans, &Ml, &Nl, &nrhsl, &A(0,0), &lda, &Xtmp(0,0), 
                &ldx, &WORK(0), &lwork, &info);

	// And copy the result from the larger matrix back into
	// the actual result matrix.
	X.inject(Xtmp(LaIndex(0,N-1), LaIndex()));
    }
    else
    {
        integer ldx = X.inc(0) * X.gdim(0);

	// Copy B into the X matrix that is passed to dgels()
        X.inject( B );

        F77NAME(zgels) (&trans, &Ml, &Nl, &nrhsl, &A(0,0), &lda, &X(0,0), 
                &ldx, &WORK(0), &lwork, &info);
    }


    // this shouldn't really happen.
    //
    if (info < 0)
        throw(LaException(fname, "Internal error in LAPACK: ZGELS()"));

}

#endif // LA_COMPLEX_SUPPORT



// ////////////////////////////////////////////////////////////

void  LaCholLinearSolve( const LaSpdMatDouble& A, LaGenMatDouble& X,
        LaGenMatDouble& B )
{
        LaSpdMatDouble A1(A);
        LaCholLinearSolveIP(A1, X, B);
}

void  LaCholLinearSolve( const LaSymmMatDouble& A, LaGenMatDouble& X,
			 const LaGenMatDouble& B )
{
        LaSymmMatDouble A1(A);
        LaCholLinearSolveIP(A1, X, B);
}

// A is NxN, X is MxN and B is MxN
//
void LaCholLinearSolveIP( LaSpdMatDouble& A, LaGenMatDouble& X, 
        LaGenMatDouble& B )
{
#ifndef HPPA
    const char fname[] = "LaCholLinearSolveIP(LaSpdMatDouble &A, &X, &B)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif

    // let's not worry about non-unit column strides for the moment
    if ( A.inc(0) != 1 || A.inc(1) != 1)
        throw(LaException(fname, "A is  non-contiguous."));

    if (!(X.size(0) == B.size(0) && X.size(1) == B.size(1)))
        throw(LaException(fname, "X and B are non-conformant."));

    X.inject(B);            // will throw exception if not conformant


    // in the future this can call the linear least square routines
    // to handle non-square matrices

    if (A.size(0) != A.size(1))
        throw(LaException(fname, "Square matrix expected.\n"));

    if (A.size(1) != X.size(0))
        throw(LaException(fname, "A and X are non-comformant."));

    integer info = 0;
    integer M = A.size(0);
    //integer N = A.size(1);
    integer K = X.size(1);
    integer lda = A.inc(0) * A.gdim(0);
    integer ldx = X.inc(0) * X.gdim(0);
    char uplo = 'L';

    F77NAME(dposv) (&uplo, &M, &K, &A(0,0), &lda,  &X(0,0), &ldx, &info);

    // this shouldn't really happen.
    //
    if (info < 0)
        throw(LaException(fname, "Internal error in LAPACK: SGESV()"));

    if (info > 0)
        throw (LaException(fname, "A is not symmetric-positive-definite."));


}

void LaCholLinearSolveIP( LaSymmMatDouble& A, LaGenMatDouble& X, 
			  const LaGenMatDouble& B )
{
#ifndef HPPA
    const char fname[] = "LaCholLinearSolveIP(LaSymmMatDouble &A, &X, &B)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif

    // let's not worry about non-unit column strides for the moment
    if ( A.inc(0) != 1 || A.inc(1) != 1)
        throw(LaException(fname, "A is  non-contiguous."));

    if (!(X.size(0) == B.size(0) && X.size(1) == B.size(1)))
        throw(LaException(fname, "X and B are non-conformant."));

    X.inject(B);            // will throw exception if not conformant


    // in the future this can call the linear least square routines
    // to handle non-square matrices

    if (A.size(0) != A.size(1))
        throw(LaException(fname, "Square matrix expected.\n"));

    if (A.size(1) != X.size(0))
        throw(LaException(fname, "A and X are non-comformant."));

    integer info = 0;
    integer M = A.size(0);
    //integer N = A.size(1);
    integer K = X.size(1);
    integer lda = A.inc(0) * A.gdim(0);
    integer ldx = X.inc(0) * X.gdim(0);
    char uplo = 'L';

    LaVectorLongInt ipiv(M);
    integer lwork = -1;
    LaVectorDouble work(1);
    // Workspace query
    F77NAME(dsysv) (&uplo, &M, &K, &A(0,0), &lda,  &ipiv(0), &X(0,0), &ldx, 
		    &work(0), &lwork, &info);
    lwork = integer(work(0));
    work.resize(lwork, 1);

    F77NAME(dsysv) (&uplo, &M, &K, &A(0,0), &lda,  &ipiv(0), &X(0,0), &ldx, 
		    &work(0), &lwork, &info);

    // this shouldn't really happen.
    //
    if (info < 0)
        throw(LaException(fname, "Internal error in LAPACK: DSYSV()"));

    if (info > 0)
        throw (LaException(fname, "Matrix is singular."));


}


void LUFactorizeIP(LaGenMatDouble &GM, LaVectorLongInt &PIV)
{
    integer m = GM.size(0), n = GM.size(1), lda = GM.gdim(0);
    integer info=0;
    assert(PIV.size() >= (m<n ? m : n));

    // Copied from LaGenMatFactorize in fmd.h of LAPACK++, but the
    // version there didn't admit to being in-place modifying.
    F77NAME(dgetrf)(&m, &n, &GM(0,0), &lda, &(PIV(0)), &info);

    if (info < 0)
	throw LaException("LUFactorizeIP", "Error in argument");
}

#ifdef LA_COMPLEX_SUPPORT
void LUFactorizeIP(LaGenMatComplex &GM, LaVectorLongInt &PIV)
{
    integer m = GM.size(0), n = GM.size(1), lda = GM.gdim(0);
    integer info=0;
    assert(PIV.size() >= (m<n ? m : n));

    // Copied from LaGenMatFactorize in fmd.h of LAPACK++, but the
    // version there didn't admit to being in-place modifying.
    F77NAME(zgetrf)(&m, &n, &GM(0,0), &lda, &(PIV(0)), &info);

    if (info < 0)
	throw LaException("LUFactorizeIP", "Error in argument");
}

void LaLUInverseIP(LaGenMatComplex &A, LaVectorLongInt &PIV)
{
    LaVectorComplex work; // will be resized in other function
    LaLUInverseIP(A, PIV, work);
}
void LaLUInverseIP(LaGenMatComplex &A, LaVectorLongInt &PIV, LaVectorComplex &work)
{
    integer N = A.size(1), lda = A.gdim(0), info = 0;
    if(A.size(0) != A.size(1))
      throw LaException("LaLUInverseIP", "Input must be square");
    integer W = work.size();

    // Check for minimum work size
    if ( W < A.size(0) ) 
    {
      int nb = LaEnvBlockSize("ZGETRI", A);
      W = N*nb;
      work.resize(W, 1);
    }

    F77NAME(zgetri)(&N, &(A(0,0)), &lda, &(PIV(0)), &work(0), &W, &info);
    if (info < 0)
	throw LaException("LaLUInverseIP", "Error in zgetri argument");
    if (info > 0)
	throw LaException("LaLuInverseIP", "Matrix is singlular, cannot compute inverse");
}
#endif // LA_COMPLEX_SUPPORT

void LaLUInverseIP(LaGenMatDouble &A, LaVectorLongInt &PIV)
{
    LaVectorDouble work; // will be resized in other function
    LaLUInverseIP(A, PIV, work);
#if 0
    // above code is shorter - remove this code soon.
    integer N = A.size(1), lda = A.gdim(0), info = 0;
    if(A.size(0) != A.size(1))
      throw LaException("LaLUInverseIP", "Input must be square");
    int nb = LaEnvBlockSize("DGETRI", A);
    integer W = N*nb;
    LaVectorDouble work(W);         

    F77NAME(dgetri)(&N, &(A(0,0)), &lda, &(PIV(0)), &work(0), &W, &info);
    if (info < 0)
	throw LaException("inv", "Error in dgetri argument");
#endif
}

void LaLUInverseIP(LaGenMatDouble &A, LaVectorLongInt &PIV, LaVectorDouble &work)
{
    integer N = A.size(1), lda = A.gdim(0), info = 0;
    if(A.size(0) != A.size(1))
      throw LaException("LaLUInverseIP", "Input must be square");
    integer W = work.size();

    // Check for minimum work size
    if ( W < A.size(0) ) 
    {
      int nb = LaEnvBlockSize("DGETRI", A);
      W = N*nb;
      work.resize(W, 1);
    }

    F77NAME(dgetri)(&N, &(A(0,0)), &lda, &(PIV(0)), &work(0), &W, &info);
    if (info < 0)
	throw LaException("LaLUInverseIP", "Error in dgetri argument");
    if (info > 0)
	throw LaException("LaLuInverseIP", "Matrix is singlular, cannot compute inverse");
}
