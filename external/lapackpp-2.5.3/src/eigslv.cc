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
#include "lafnames.h"
#ifdef LA_COMPLEX_SUPPORT
#  include LA_GEN_MAT_COMPLEX_H
#  include LA_VECTOR_COMPLEX_H
#endif
#include LA_GEN_MAT_DOUBLE_H
#include LA_VECTOR_DOUBLE_H
#include LA_VECTOR_INT_H 
#include LA_SPD_MAT_DOUBLE_H
#include LA_SYMM_MAT_DOUBLE_H
#include LA_EXCEPTION_H
#include LA_SOLVE_DOUBLE_H
#include LA_UTIL_H

void LaEigSolve(const LaSymmMatDouble &S, LaVectorDouble &eigvals)
{   
//#ifndef HPPA
//     const char fname[] = "LaEigSolve(LaSymmMatDouble &A, &eigvals)";
//#else
//    char *fname = NULL;  // HP C++ does not support string initalization!
//#endif

    LaSymmMatDouble tmp(S);

    LaEigSolveIP(tmp, eigvals);


}

void LaEigSolve(const LaSymmMatDouble &S, LaVectorDouble &eigvals, 
    LaGenMatDouble &eigvec)
{   
//#ifndef HPPA
//   const char fname[] = "LaEigSolve(LaSymmMatDouble &A, &eigvals, &eigvecs)";
//#else
//    char *fname = NULL;  // HP C++ does not support string initalization!
//#endif

  // It is unclear whether this function implementation really is
  // a good thing. Was introduced in revision 1.3 on 2004-09-08,
  // see
  // http://sourceforge.net/mailarchive/message.php?msg_id=9480567
  integer N = S.size(0);
  integer i,j;
 
  for(j=0;j<N;j++){
     for(i=j;i<N;i++){
	eigvec(i,j)=S(i,j);
     }
  }
  LaEigSolveSymmetricVecIP(eigvec, eigvals);

}

#ifdef LA_COMPLEX_SUPPORT
void LaEigSolve(const LaGenMatDouble &A, LaVectorComplex &eigvals, LaGenMatDouble &eigvec)
{
  if(eigvals.size() != A.size(0))
    throw LaException("LaEigSolve(LaGenMatDouble &, LaVectorComplex &, LaGenMatDouble &", "eigenvalue vectors must be same size as one dimension of input matrix");
  
  LaVectorDouble eigvals_real(eigvals.size());
  LaVectorDouble eigvals_imag(eigvals.size());

  LaEigSolve(A,eigvals_real,eigvals_imag,eigvec);
  //std::cout << "EigenVectors inside function\n" << eigvec << "\n";
  for(int i=0; i < eigvals.size(); i++)
  {
    eigvals(i).r=eigvals_real(i);
    eigvals(i).i=eigvals_imag(i);
  }
}
#endif

void LaEigSolve(const LaGenMatDouble &A, LaVectorDouble &eigvals_real, LaVectorDouble &eigvals_imag, LaGenMatDouble &eigvec)
{
    char jobvl = 'N';
    char jobvr = 'V';
    integer n = A.size(0);
    integer lda = A.gdim(0);
    integer ldvl = 1;
    integer ldvr = n;
    integer lwork = 4*n;
    integer info = 0;
    
    if (A.size(0) != A.size(1))
      throw LaException("LaEigSolve(LaGenMatDouble &, LaVectorDouble &, LaVectorDouble &, LaGenMatDouble &", "Matrix must be square!");
    if((eigvals_real.size() != n) || (eigvals_imag.size() != n))
      throw LaException("LaEigSolve(LaGenMatDouble &, LaVectorDouble &, LaVectorDouble &, LaGenMatDouble &", "eigenvalue vectors must be same size as one dimension of input matrix");

    LaGenMatDouble tmp;
    tmp.copy(A);
    LaVectorDouble work(lwork);
    if (eigvec.size(0) != n || eigvec.size(1) != n)
      eigvec.resize(n, n);
    
    F77NAME(dgeev)(&jobvl, &jobvr, &n, &tmp(0,0), &lda, &eigvals_real(0), &eigvals_imag(0), 
                   NULL, &ldvl, &eigvec(0,0), &ldvr, &work(0), &lwork, &info);

    if (info != 0)
        throw(LaException("LaEigSolve(LaGenMatDouble &, LaVectorDouble &, LaVectorDouble &, LaGenMatDouble &", "Internal error in LAPACK: SSYEV()"));
}


#ifdef LA_COMPLEX_SUPPORT
void LaEigSolve(const LaGenMatComplex &A, LaVectorComplex &W,
		LaGenMatComplex &VR)
{
    char jobvl = 'N';
    char jobvr = 'V';
    integer N = A.size(0);
    integer lda = A.gdim(0);
    if (A.size(0) != A.size(1))
      throw LaException("LaEigSolve()", "Matrix must be square!");
    if (W.size() != N)
      throw LaException("LaEigSolve(LaGenMatDouble &, LaVectorDouble &, LaVectorDouble &, LaGenMatDouble &", "eigenvalue vectors must be same size as one dimension of input matrix");

    integer ldvl = 1;
    if (VR.size(0) < N || VR.size(1) < N)
      VR.resize(N, N);
    integer ldvr = VR.gdim(0);

    integer lwork = 4*N;
    LaVectorComplex work(lwork);
    LaVectorDouble rwork(2*N);

    integer info = 0;

    LaGenMatComplex tmp;
    tmp.copy(A);
    
    F77NAME(zgeev)(&jobvl, &jobvr, &N, &tmp(0,0), &lda, &W(0), 
                   NULL, &ldvl, &VR(0,0), &ldvr,
		   &work(0), &lwork, &rwork(0), &info);

    if (info != 0)
        throw(LaException("LaEigSolve()", "Internal error in LAPACK: ZGEEV()"));
}
#endif // LA_COMPLEX_SUPPORT

void LaEigSolveIP(LaSymmMatDouble &S, LaVectorDouble &eigvals)
{   
#ifndef HPPA
     const char fname[] = "LaEigSolveIP(LaGenMatDouble &A, &v)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif
    integer N = S.size(0);
    char jobz = 'N';
    char uplo = 'L';
    integer info = 0;
    integer lda = S.gdim(0);

    if (eigvals.size() < N)
    {
        throw(LaException(fname, "Not enough room to store eigenvalues"));
    }
        

    integer w = (LaEnvBlockSize("SSYTRD", S) +2) * N;
    LaVectorDouble Work(w);


    F77NAME(dsyev)(&jobz, &uplo, &N, S.addr(), &lda, &eigvals(0), &Work(0),
        &w, &info);

    if (info != 0)
        throw(LaException(fname, "Internal error in LAPACK: SSYEV()"));

}

void LaEigSolveVecIP(LaGenMatDouble &A_symmetric, LaVectorDouble &eigvals)
{
   LaEigSolveSymmetricVecIP(A_symmetric, eigvals);
}

void LaEigSolveSymmetricVecIP(LaGenMatDouble &A, LaVectorDouble &eigvals)
{   
#ifndef HPPA
     const char fname[] = "LaEigSolveSymmetricVecIP(LaGenMatDouble &A, &eigvals)";
#else
    char *fname = NULL;  // HP C++ does not support string initalization!
#endif
    integer N = A.size(0);
    char jobz = 'V';
    char uplo = 'L';
    integer info = 0;
    integer lda = A.gdim(0);

    if (eigvals.size() < N)
    {
        throw(LaException(fname, "Not enough room to store eigenvalues"));
    }
        

    integer w = (LaEnvBlockSize("SSYTRD", A) +2) * N;
    LaVectorDouble Work(w);


    F77NAME(dsyev)(&jobz, &uplo, &N, A.addr(), &lda, &eigvals(0), &Work(0),
        &w, &info);

    if (info != 0)
        throw(LaException(fname, "Internal error in LAPACK: SSYEV()"));

}


