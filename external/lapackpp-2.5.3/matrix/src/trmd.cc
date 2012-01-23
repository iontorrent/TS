
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

#include "arch.h"
#include "lafnames.h"
#include LA_EXCEPTION_H
#include LA_TRIDIAG_MAT_DOUBLE_H
#include LA_TRIDIAG_FACT_DOUBLE_H


DLLIMPORT double LaTridiagMatDouble::outofbounds_ = 0; // set outofbounds_. 

DLLIMPORT int LaTridiagMatDouble::debug_ = 0; // set debug to 0 initially.

DLLIMPORT int* LaTridiagMatDouble::info_= new int;  // turn off info print flag.

    // constructors

LaTridiagMatDouble::LaTridiagMatDouble()
   : du2_(), du_(), d_(), dl_(), size_(0)
{}

LaTridiagMatDouble::LaTridiagMatDouble(int N)
   : du2_(N-2), du_(N-1), d_(N), dl_(N-1), size_(N)
{}
    
LaTridiagMatDouble::LaTridiagMatDouble(const LaTridiagMatDouble& td)
   : du2_(td.du2_), du_(td.du_), d_(td.d_), dl_(td.dl_), size_(td.size_)
{
   assert(d_.size() - 1 == du_.size());
   assert(d_.size() - 1 == dl_.size());
   assert(d_.size() - 2 == du2_.size());
}

LaTridiagMatDouble::LaTridiagMatDouble(const LaVectorDouble& diag,
				       const LaVectorDouble& diaglower,
				       const LaVectorDouble& diagupper)
    : du2_(diag.size()-2)
    , du_(diagupper)
    , d_(diag)
    , dl_(diaglower)
    , size_(diag.size())
{
   assert(d_.size() - 1 == du_.size());
   assert(d_.size() - 1 == dl_.size());
}

LaTridiagMatDouble::~LaTridiagMatDouble()
{
}


LaVectorDouble& LaTridiagMatDouble::diag(int k)
{
    switch (k)
    {
        case 0:   // main
            return d_;
        case -1:  // lower
	    return dl_;
        case 1:   // upper
	    return du_;
        case 2:   // second upper
	    return du2_;
        default:
            std::cerr <<"Unrecognized integer representation of diagonal\n";
	    throw LaException("LaTridiagMatDouble::diag", "Unrecognized integer representation of diagonal");

    }
}


const LaVectorDouble& LaTridiagMatDouble::diag(int k) const
{
    switch (k)
    {
        case 0:   // main
            return d_;
        case -1:  // lower
	    return dl_;
        case 1:   // upper
	    return du_;
        case 2:   // second upper
	    return du2_;
        default:
            std::cerr <<"Unrecognized integer representation of diagonal\n";
	    throw LaException("LaTridiagMatDouble::diag", "Unrecognized integer representation of diagonal");

    }
}


LaTridiagMatDouble& LaTridiagMatDouble::copy(const LaTridiagMatDouble&T) 
{
   du2_.copy(T.du2_);
   du_.copy(T.du_);
   d_.copy(T.d_);
   dl_.copy(T.dl_);    
   size_ = T.size_;
   return *this;
}

LaTridiagMatDouble& LaTridiagMatDouble::inject(const LaTridiagMatDouble& T)
{
   assert(size_ == T.size_);
   du2_.inject(T.du2_);
   du_.inject(T.du_);
   d_.inject(T.d_);
   dl_.inject(T.dl_);    
   return *this;
}

std::ostream& operator<<(std::ostream& s, const LaTridiagMatDouble& td)
{
  if (*(td.info_))     // print out only matrix info, not actual values
  {
      *(td.info_) = 0; // reset the flag
      s << "superdiag: (" << td.du_.size() << ") " ;
      s <<" #ref: "<< td.du_.ref_count()<< std::endl;
      s << "maindiag: (" << td.d_.size() << ") " ;
      s <<" #ref: "<< td.d_.ref_count()<< std::endl;
      s << "subdiag: (" << td.dl_.size() << ") " ;
      s <<" #ref: "<< td.dl_.ref_count()<< std::endl;
  }
  else
  {
    s << td.diag(1);
    s << td.diag(0);
    s << td.diag(-1);
    s << std::endl;
  } 
  return s;
}

void LaTridiagMatFactorize(const LaTridiagMatDouble &A,
				  LaTridiagFactDouble &AF)
{
    integer N = A.size(), info = 0;
    AF.copy(A);
    double *DL = &AF.diag(-1)(0), *D = &AF.diag(0)(0),
         *DU = &AF.diag(1)(0), *DU2 = &AF.diag(2)(0);

    //std::cerr << " \t*\n";

    F77NAME(dgttrf)(&N, DL, D, DU, DU2, &(AF.pivot()(0)), &info);

    //std::cerr << " \t\t**\n";
}


/** Solve Ax=b with tridiagonal A and the calculated LU
 * factorization of A as returned by
 * LaTridiagMatFactorize(). Solves by \c dgttrs.
 *
 * \param AF The LU factorization of the A matrix
 * \param X The matrix that will contain the result afterwards. Size must be correct.
 * \param B The right-hand-side of the equation system Ax=b.
 */
void LaLinearSolve(LaTridiagFactDouble &AF, LaGenMatDouble &X,
			  const LaGenMatDouble &B)
{
    char trans = 'N';
    integer N = AF.size(), nrhs = X.size(1), ldb = B.size(0), info = 0;
    double *DL = &AF.diag(-1)(0), *D = &AF.diag(0)(0),
         *DU =  &AF.diag(1)(0), *DU2 = &AF.diag(2)(0);

    X.inject(B);
    F77NAME(dgttrs)(&trans, &N, &nrhs, DL, D, DU, DU2, &(AF.pivot()(0)),
                    &X(0,0), &ldb, &info);
}
