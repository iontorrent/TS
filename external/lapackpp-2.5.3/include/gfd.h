//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.


#ifndef _LA_GEN_FACT_DOUBLE_H
#define _LA_GEN_FACT_DOUBLE_H

/** @file

    Deprecated. Class for the LU factorization of a matrix. Note: This
    class is probably broken by design, because the matrices L and U
    do not exist separately in the internal lapack but they are part
    of the modified input matrix A.

    Do not use this unless you are really sure you understand what
    this class does.
*/

#include "lafnames.h"
#include LA_VECTOR_LONG_INT_H
#include LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H
#include LA_UPPER_TRIANG_MAT_DOUBLE_H

#include "lapack.h"
/** Class for the LU factorization of a matrix. Note: This class is
    probably broken by design, because the matrices L and U do not
    exist separately in the internal lapack but they are part of the
    modified input matrix A. 

    Do not use this unless you are really sure you understand what
    this class does. */
class LaGenFactDouble
{
    LaUnitLowerTriangMatDouble  L_;
    LaUpperTriangMatDouble      U_;
    LaVectorLongInt             pivot_;
    int                      info_;
    int                 transpose_;

public:

    // constructor

    inline LaGenFactDouble();
    inline LaGenFactDouble(int,int);
    inline LaGenFactDouble(LaGenFactDouble &);
    inline ~LaGenFactDouble();

    // extraction functions for components

    inline LaUnitLowerTriangMatDouble& L();
    inline LaUpperTriangMatDouble& U();
    inline LaVectorLongInt& pivot();
    inline int& info();
    inline int& transpose();

    // operators

    inline LaGenFactDouble& ref(LaGenFactDouble &);
    inline LaGenFactDouble& ref(LaGenMatDouble &);

};



    // constructor/destructor functions

inline LaGenFactDouble::LaGenFactDouble():L_(),U_(),pivot_()
{

    info_ = 0;
    transpose_ = 0;
}


inline LaGenFactDouble::LaGenFactDouble(int n, int m):L_(n,m),U_(n,m),pivot_(n*m)
{

    info_ = 0;
    transpose_ = 0;
}


inline LaGenFactDouble::LaGenFactDouble(LaGenFactDouble &F)
{

  L_.ref(F.L_);
  U_.ref(F.U_);
  pivot_.ref(F.pivot_);
  info_ = F.info_;
  transpose_ = F.transpose_;
}

inline LaGenFactDouble::~LaGenFactDouble()
{
}

    // member functions
inline LaUnitLowerTriangMatDouble& LaGenFactDouble::L()
{

    return L_;
}

inline LaUpperTriangMatDouble& LaGenFactDouble::U()
{

    return U_;
}

inline LaVectorLongInt& LaGenFactDouble::pivot()
{

    return pivot_;
}

inline int& LaGenFactDouble::info()
{

    return info_;
}

inline int& LaGenFactDouble::transpose()
{

    return transpose_;
}

    
    // operators


inline LaGenFactDouble& LaGenFactDouble::ref(LaGenFactDouble& F)
{

    L_.ref(F.L_);
    U_.ref(F.U_);
    pivot_.ref(F.pivot_);
    info_ = F.info_;
    transpose_ = F.transpose_;
    
    return *this;
}

inline LaGenFactDouble& LaGenFactDouble::ref(LaGenMatDouble &G)
{

  L_.ref(G);
  U_.ref(G);
  info_ = 0;
  transpose_ = 0;

  return *this;
}

#if 0
inline void LaLinearSolve(LaGenFactDouble &AF, LaGenMatDouble& X,
    LaGenMatDouble& B )
{
    char trans = 'N';
    integer n = AF.L().size(1), lda = AF.L().gdim(0), nrhs = X.size(1),
            ldb = B.size(0), info = 0;

    X.inject(B);
    F77NAME(dgetrs)(&trans, &n, &nrhs, &(AF.U()(0,0)), &lda, &(AF.pivot()(0)),
         &X(0,0), &ldb, &info);
}


/** Calculate the LU factorization of a matrix A. 
 *
 * Note: The class LaGenFactDouble is probably broken by design,
 * because the matrices L and U do not exist separately in the
 * internal lapack but they are part of the modified input matrix
 * A. The factorization classes need a complete redesign.
 *
 * However, the intended behaviour can be achieved when the
 * LaGenFactDouble object is constructed with the original matrix A
 * as argument. This work if and only if 1. the original matrix A is
 * allowed to be destroyed by the factorization, and 2. you use the
 * same original matrix for calling this function. Use the following
 * code: \verbatim
// original matrix A:
LaGenMatDouble A(m,n);
// fill A somehow. Then construct the factorization:
LaGenFactDouble AF(A);
LaGenMatFactorize(A, AF);
// AF refers to the factorization. A may no longer be used, which is
// fine. Now use the factorization:
LaLinearSolve(AF, X, B); // ... and so on. 
\endverbatim
 */
inline void LaGenMatFactorize(LaGenMatDouble &GM, LaGenFactDouble &GF)
{
    integer m = GM.size(0), n = GM.size(1), lda = GM.gdim(0);
    integer info=0;

    F77NAME(dgetrf)(&m, &n, &GM(0,0), &lda, &(GF.pivot()(0)), &info);
}

inline double LaGenMatCond(LaGenMatDouble &GM, bool use_one_norm = true)
{
    //integer M = GM.size(0);
    integer N = GM.size(1), lda = GM.gdim(0);
    integer info=0;
    char norm;
    double anorm;
    if (use_one_norm)
      {
	norm='1';
	anorm = Blas_Norm1(GM);
      }
    else
      {
	norm='I';
	anorm = Blas_Norm_Inf(GM);
      }
    double rcond;
    VectorDouble work(4*N);
    VectorLongInt iwork(N);

    F77NAME(dgecon)(&norm, &N, &GM(0,0), &lda, &anorm, &rcond, 
		    &work(0), &iwork(0),&info);

    return rcond;
}

inline void LaGenMatFactorizeUnblocked(LaGenMatDouble &A, LaGenFactDouble &F)
{
    integer m = A.size(0), n=A.size(1), lda = A.gdim(0);
    integer info=0;

    F77NAME(dgetf2)(&m, &n, &A(0,0), &lda, &(F.pivot()(0)), &info);
}
#endif

void LaLUFactorDouble(LaGenMatDouble &A, LaGenFactDouble &F, integer nb);
void LaLUFactorDouble(LaGenMatDouble &A, LaGenFactDouble &F);

#endif
