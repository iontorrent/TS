// -*-c++-*-
//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.

#ifndef _LA_TRIDIAG_FACT_DOUBLE_H_
#define _LA_TRIDIAG_FACT_DOUBLE_H_

/** @file
    @brief LU factorization of a tridiagonal matrix

    Class for the LU factorization of a tridiagonal matrix. 

    Note: It is unclear whether the design of this class needs a major
    overhaul.  Do not use this unless you are really sure you
    understand what this class does.
*/

#include LA_VECTOR_LONG_INT_H
#include LA_TRIDIAG_MAT_DOUBLE_H

#include "lapack.h"

/** 
    Class for the LU factorization of a tridiagonal matrix. 

    \see \ref LaTridiagMatDouble, LaTridiagMatFactorize()

    Note: It is unclear whether the design of this class needs
    some rewriting. Currently this class is only usable for
    solving an equation system with a tridiagonal matrix.

    As a code example for solving Ax=b with tridiagonal A, you
    would program the following lines:
\verbatim
LaTridiagMatDouble A(N); // define A
A.diag(0).inject(...); // fill the matrix with values
LaGenMatDouble B(N,1); // define right-hand-side B
B = ...; // fill B with values from somewhere
// To solve Ax=b:
LaTridiagFactDouble Afact;
LaGenMatDouble X(N,1);
LaTridiagMatFactorize(A, Afact); // calculate LU factorization
LaLinearSolve(Afact, X, B); // solve; result is in X
\endverbatim

*/
class LaTridiagFactDouble
{
    LaTridiagMatDouble T_;
    LaVectorLongInt pivot_;
    int size_;

public:

    // constructor

    LaTridiagFactDouble();
    LaTridiagFactDouble(int);
    LaTridiagFactDouble(LaTridiagFactDouble &);
    ~LaTridiagFactDouble();

    LaTridiagMatDouble& T() { return T_; }
    LaVectorLongInt& pivot() { return pivot_; }
    int size() { return size_; }

    LaVectorDouble& diag(int);
    const LaVectorDouble& diag(int k) const;

    // operators

    LaTridiagFactDouble& ref(LaTridiagMatDouble &);
    LaTridiagFactDouble& ref(LaTridiagFactDouble &);
    LaTridiagFactDouble& copy(const LaTridiagMatDouble &);
    LaTridiagFactDouble& copy(const LaTridiagFactDouble &);

};



    // constructor/destructor functions

inline LaTridiagFactDouble::LaTridiagFactDouble():T_(),pivot_(),size_(0)
{}


inline LaTridiagFactDouble::LaTridiagFactDouble(int N):T_(N),pivot_(N),size_(N)
{}


inline LaTridiagFactDouble::LaTridiagFactDouble(LaTridiagFactDouble &F)
{
  T_.copy(F.T_);
  pivot_.copy(F.pivot_);
  size_ = F.size_;
}

inline LaTridiagFactDouble::~LaTridiagFactDouble()
{}

    // member functions

inline LaVectorDouble& LaTridiagFactDouble::diag(int k)
{
    return T_.diag(k);
}

inline const LaVectorDouble& LaTridiagFactDouble::diag(int k) const
{
    return T_.diag(k);
}
    
    // operators


inline LaTridiagFactDouble& LaTridiagFactDouble::ref(LaTridiagFactDouble& F)
{
    T_.ref(F.T_);
    pivot_.ref(F.pivot_);
    size_ = F.size_;
    
    return *this;
}

inline LaTridiagFactDouble& LaTridiagFactDouble::ref(LaTridiagMatDouble& A)
{
    T_.ref(A);
    size_ = A.size();
    pivot_.resize(size_, 1);

    return *this;
}

inline LaTridiagFactDouble& LaTridiagFactDouble::copy(const LaTridiagFactDouble& F)
{
    T_.copy(F.T_);
    pivot_.copy(F.pivot_);
    size_ = F.size_;
    
    return *this;
}

inline LaTridiagFactDouble& LaTridiagFactDouble::copy(const LaTridiagMatDouble& A)
{
    T_.copy(A);
    size_ = A.size();
    pivot_.resize(size_, 1);

    return *this;
}

/** Calculate the LU factorization of a tridiagonal
 * matrix. Factorizes by @c dgttrf.
 *
 * \param A The matrix to be factorized. Will be unchanged.
 * \param AF The class where the factorization of A will be stored.
 *
 *  \see \ref LaTridiagFactDouble, LaLinearSolve(LaTridiagFactDouble &, LaGenMatDouble &, const LaGenMatDouble &)
 */
void DLLIMPORT LaTridiagMatFactorize(const LaTridiagMatDouble &A,
				  LaTridiagFactDouble &AF);

/** Solve Ax=b with tridiagonal A and the calculated LU
 * factorization of A as returned by
 * LaTridiagMatFactorize(). Solves by \c dgttrs.
 *
 * \param AF The LU factorization of the A matrix
 * \param X The matrix that will contain the result afterwards. Size must be correct.
 * \param B The right-hand-side of the equation system Ax=b.
 */
void DLLIMPORT LaLinearSolve(LaTridiagFactDouble &AF, LaGenMatDouble &X,
			  const LaGenMatDouble &B);

#endif 
// _LA_TRIDIAG_FACT_DOUBLE_H_
