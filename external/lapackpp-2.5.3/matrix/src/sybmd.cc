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
#include LA_SYMM_BAND_MAT_DOUBLE_H


DLLIMPORT double LaSymmBandMatDouble::outofbounds_ = 0; // initialize outofbounds_.

DLLIMPORT int LaSymmBandMatDouble::debug_ = 0; // initialize debug to off.

DLLIMPORT int* LaSymmBandMatDouble::info_ = new int; // turn off info print flag.

  // constructors 

LaSymmBandMatDouble::LaSymmBandMatDouble()
    :data_()
{
  N_ = 0;
  kl_ = 0;
}

LaSymmBandMatDouble::LaSymmBandMatDouble(int n, int p)
    :data_(2*p+1, n) // This used to be broken in lapackpp-2.4.13 and any earlier version!
{
  N_ = n;
  kl_ = p;
}

LaSymmBandMatDouble::LaSymmBandMatDouble(const LaSymmBandMatDouble &A)
{
  data_.copy(A.data_);
  N_ = A.N_;
  kl_ = A.kl_;
}

  // destructor 

LaSymmBandMatDouble::~LaSymmBandMatDouble()
{
}

LaSymmBandMatDouble& LaSymmBandMatDouble::operator=(const LaSymmBandMatDouble &B)
{
    data_ = B.data_;
    N_ = B.N_;
    kl_ = B.kl_;

    return *this;
}

LaSymmBandMatDouble& LaSymmBandMatDouble::operator=(double scalar)
{
  int i,j;
  int n2;

  for (i=0; i<N_; i++)
  {
    n2= std::min(i+kl_+1 , N_);  // select minimum
    for (j=i; j<n2; j++)
    {
      (*this)(i,j) = scalar;
    }
  }

  return *this;
}

LaSymmBandMatDouble& LaSymmBandMatDouble::resize(const LaSymmBandMatDouble &ob)
{
  data_.resize(ob.data_);
  N_ = ob.N_;
  kl_=ob.kl_;
  return *this;
}

LaSymmBandMatDouble& LaSymmBandMatDouble::resize(int n, int p)
{
  data_.resize(2*p+1, n);
  N_ = n;
  kl_ = p;
  return *this;
}

LaSymmBandMatDouble& LaSymmBandMatDouble::copy(const LaSymmBandMatDouble &B)
{
  data_ = B.data_;
  N_ = B.N_;
  kl_ = B.kl_;

  return *this;
}

std::ostream& operator<<(std::ostream &s, const LaSymmBandMatDouble &ob)
{
  if (*(ob.info_))     // print out only matrix info, not actual values
  {
      *(ob.info_) = 0; // reset the flag
      s << "(" << ob.size(0) << "x" << ob.size(1) << ") " ;
      s << "Indices: " << ob.index(0) << " " << ob.index(1);
      s << " #ref: " << ob.ref_count() ;
      s << " sa:" << ob.shallow();
  }
  else
  {
    int i,j;
    int N_ = ob.N_;
    int kl_ = ob.kl_;

    for (i=0; i<N_; i++)
    {
      for (j=0; j<N_; j++)
        {
          if(((i>=j)&&(i-j<=kl_)))
            s << ob(i,j) << ' ';
          else if (((j>=i)&&(j-i<=kl_)))
            s << ob(j,i) << ' ';
        }
      s << "\n";
    }
  }
  return s;
}

