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
#include LA_GEN_MAT_DOUBLE_H
#include LA_VECTOR_DOUBLE_H
#include LA_VECTOR_LONG_INT_H 
#include LA_EXCEPTION_H
#ifdef LA_COMPLEX_SUPPORT
#  include LA_GEN_MAT_COMPLEX_H
#  include LA_VECTOR_COMPLEX_H
#endif

#include LA_SOLVE_DOUBLE_H
#include LA_UTIL_H
#include "lasvd.h"


#ifdef LA_COMPLEX_SUPPORT
    

void LaSVD_IP(LaGenMatComplex& A, LaVectorDouble &Sigma, LaGenMatComplex& U, LaGenMatComplex& VT )
{
#ifndef HPPA
    const char fname[] = "LaSVD_IP(LaGenMatComplex &A, &Sigma, &U, &VT)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif

    // let's not worry about non-unit column strides for the moment
    if ( A.inc(0) != 1 || A.inc(1) != 1)
        throw(LaException(fname, "A is  non-contiguous."));

    char jobz = 'A';
    integer info = 0;
    int M = A.size(0);
    int N = A.size(1);
    integer Ml = M;
    integer Nl = N;
    integer lda = A.inc(0) * A.gdim(0);
//     int nrhs = X.size(1);
//     integer nrhsl = nrhs;

    if (Sigma.size() != std::min(M,N))
	throw LaException(fname, "Sigma is not of correct size");
    if (U.size(0) != U.size(1) || U.size(0) != M)
	throw LaException(fname, "U is not of correct size");
    if (VT.size(0) != VT.size(1) || VT.size(0) != N)
	throw LaException(fname, "VT is not of correct size");
        
    //#define MAX3(A,B,C) ((A)>(B) ? ((A)>(C) ? (A) : (C)) : ((B)>(C) ? (B) : (C)))
    //  #define MIN(A,B) (A < B ? A : B )
    
    //std::cout << "Block size: " << LaEnvBlockSize("DGELSV", A) << std::endl;
    //int nb = 32;
    //int nb = LaEnvBlockSize("ZGESDD", A);
    integer W = std::min(M,N)*std::min(M,N) + 
	2*std::min(M,N) + std::max(M,N);
    LaVectorComplex work(W);
    work = 0.0;

    int lrwork = 5*std::min(M,N)*(std::min(M,N) + 1);
    LaVectorDouble rwork(lrwork);

    int liwork = 8*std::min(M,N);
    LaVectorLongInt iwork(liwork);

    integer ldu = U.inc(0) * U.gdim(0);
    integer ldvt = VT.inc(0) * VT.gdim(0);

#if 0
    F77NAME(zgesvd)(&jobz, &jobz, &Ml, &Nl, &A(0,0), &lda, 
		    &Sigma(0), &U(0,0), &ldu, &VT(0,0), &ldvt, 
		    &work(0), &W, &rwork(0),
		    &info);
#else
    F77NAME(zgesdd)(&jobz, &Ml, &Nl, &A(0,0), &lda, 
		    &Sigma(0), &U(0,0), &ldu, &VT(0,0), &ldvt, 
		    &work(0), &W, 
		    &rwork(0), &iwork(0),
		    &info);
#endif

    LA_ASSERTZERO(info);
}


void LaSVD_IP(LaGenMatComplex& A, LaVectorDouble &Sigma)
{
#ifndef HPPA
    const char fname[] = "LaSVD_IP(LaGenMatComplex &A, &Sigma)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif

    // let's not worry about non-unit column strides for the moment
    if ( A.inc(0) != 1 || A.inc(1) != 1)
        throw(LaException(fname, "A is  non-contiguous."));

    char jobz = 'N';
    integer info = 0;
    int M = A.size(0);
    int N = A.size(1);
    integer Ml = M;
    integer Nl = N;
    integer lda = A.inc(0) * A.gdim(0);
//     int nrhs = X.size(1);
//     integer nrhsl = nrhs;

    LaGenMatComplex U(1,1);
    LaGenMatComplex VT(1,1);

    if (Sigma.size() != std::min(M,N))
	throw LaException(fname, "Sigma is not of correct size");

    integer lwork = 2*std::min(M,N) + std::max(M,N);
    LaVectorComplex work(lwork);
    //work = 0.0;

    int lrwork = 7*std::min(M,N);
    LaVectorDouble rwork(lrwork);

    int liwork = 8*std::min(M,N);
    LaVectorLongInt iwork(liwork);

    integer ldu = 1;
    integer ldvt = 1;

#if 0
    F77NAME(zgesvd)(&jobz, &jobz, &Ml, &Nl, &A(0,0), &lda, 
		    &Sigma(0), &U(0,0), &ldu, &VT(0,0), &ldvt, 
		    &work(0), &lwork, &rwork(0),
		    &info);
#else
    F77NAME(zgesdd)(&jobz, &Ml, &Nl, &A(0,0), &lda, 
		    &Sigma(0), &U(0,0), &ldu, &VT(0,0), &ldvt, 
		    &work(0), &lwork, 
		    &rwork(0), &iwork(0),
		    &info);
#endif

    LA_ASSERTZERO(info);
}



#endif // LA_COMPLEX_SUPPORT


// ////////////////////////////////////////////////////////////
// Now the real-valued matrices

void LaSVD_IP(LaGenMatDouble& A, LaVectorDouble &Sigma, LaGenMatDouble& U, LaGenMatDouble& VT )
{
#ifndef HPPA
    const char fname[] = "LaSVD_IP(LaGenMatDouble &A, &X, &B)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif

    // let's not worry about non-unit column strides for the moment
    if ( A.inc(0) != 1 || A.inc(1) != 1)
        throw(LaException(fname, "A is  non-contiguous."));

    char jobz = '?';
    integer info = 0;
    int M = A.size(0);
    int N = A.size(1);
    int MNmin = std::min(M,N);
    integer Ml = M;
    integer Nl = N;
    integer lda = A.inc(0) * A.gdim(0);

    if (Sigma.size() != MNmin)
	throw LaException(fname, "Sigma is not of correct size");

    if ((U.size(0) == M && U.size(1) == M) && (VT.size(0) == N && VT.size(1) == N))
      jobz = 'A';
    else if ((U.size(0) == M && U.size(1) == MNmin )
	     && (VT.size(0) == MNmin && VT.size(1) == N))
      jobz = 'S';
    else if (M >= N && U.size(0) == 0 && (VT.size(0) == N && VT.size(1) == N))
      jobz = 'O';
    else if (M < N && (U.size(0) == M && U.size(1) == M) && VT.size(0) == 0)
      jobz = 'O';
    else
      throw LaException(fname, "U or VT is not of correct size");
    
    //if (U.size(0) != U.size(1) || U.size(0) != M)
    //throw LaException(fname, "U is not of correct size");
    //if (VT.size(0) != VT.size(1) || VT.size(0) != N)
    //throw LaException(fname, "VT is not of correct size");
        
    integer ldu = U.inc(0) * U.gdim(0);
    integer ldvt = VT.inc(0) * VT.gdim(0);
    // If Vt is not referenced, set the LD to 1
    if (ldvt == 0 && jobz == 'O' && VT.size(0) == 0)
      ldvt=1;
    // If U is not referenced, set the LD to 1
    if (ldu == 0 && jobz == 'O' && U.size(0) == 0)
      ldu=1;

    int liwork = 8*MNmin;
    LaVectorLongInt iwork(liwork);

    integer lwork = -1;
    LaVectorDouble work(1);
    // Calculate the optimum temporary workspace
    F77NAME(dgesdd)(&jobz, &Ml, &Nl, &A(0,0), &lda, 
		    &Sigma(0), &U(0,0), &ldu, &VT(0,0), &ldvt, 
		    &work(0), &lwork, &iwork(0),
		    &info);
    lwork = int(work(0));
    work.resize(lwork, 1);

    // Now the real calculation
    F77NAME(dgesdd)(&jobz, &Ml, &Nl, &A(0,0), &lda, 
		    &Sigma(0), &U(0,0), &ldu, &VT(0,0), &ldvt, 
		    &work(0), &lwork, &iwork(0),
		    &info);

    if (info != 0) {
        throw(LaException(fname, "Internal error in LAPACK: dgesdd()"));
    }

}


void LaSVD_IP(LaGenMatDouble& A, LaVectorDouble &Sigma)
{
#ifndef HPPA
    const char fname[] = "LaSVD_IP(LaGenMatDouble &A, &X, &B)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif

    // let's not worry about non-unit column strides for the moment
    if ( A.inc(0) != 1 || A.inc(1) != 1)
        throw(LaException(fname, "A is  non-contiguous."));

    char jobz = 'N';
    integer info = 0;
    int M = A.size(0);
    int N = A.size(1);
    integer Ml = M;
    integer Nl = N;
    integer lda = A.inc(0) * A.gdim(0);

    LaGenMatDouble U(1,1);
    LaGenMatDouble VT(1,1);

    if (Sigma.size() != std::min(M,N))
	throw LaException(fname, "Sigma is not of correct size");

    integer ldu = 1;
    integer ldvt = 1;

    //integer lwork = 4*std::min(M,N)*std::min(M,N) + std::max(M,N) + 9*min(M,N);
/*    integer lwork = 3*std::min(M,N)*std::min(M,N) + 
       std::max(std::max(M,N), 
		4*std::min(M,N)*std::min(M,N)+4*std::min(M,N)); */
    integer lwork = 3*std::min(M,N) + std::max( std::max(M,N),
					        6*std::min(M,N));
    LaVectorDouble work(lwork);
    //work = 0.0;

    int liwork = 8*std::min(M,N);
    LaVectorLongInt iwork(liwork);

    F77NAME(dgesdd)(&jobz, &Ml, &Nl, &A(0,0), &lda, 
		    &Sigma(0), &U(0,0), &ldu, &VT(0,0), &ldvt, 
		    &work(0), &lwork, &iwork(0),
		    &info);

    if (info != 0) {
        throw(LaException(fname, "Internal error in LAPACK: dgesdd()"));
    }

}

