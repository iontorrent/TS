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
#include LA_BAND_MAT_DOUBLE_H

DLLIMPORT int LaBandMatDouble::debug_= 0;
DLLIMPORT int* LaBandMatDouble::info_ = new int;
DLLIMPORT double  LaBandMatDouble::outofbounds_ = 0.0;

  // constructors 

LaBandMatDouble::LaBandMatDouble() 
    : data_()
{

  N_ = kl_ = ku_ = 0;
}

LaBandMatDouble::LaBandMatDouble(int n,int l,int u)
    : data_(2*l+u+1,n)
{

  N_ = n;
  kl_ = l;
  ku_ = u;
}

LaBandMatDouble::LaBandMatDouble(const LaBandMatDouble &A)
{

  data_.copy(A.data_);
  N_ = A.N_;
  kl_ = A.kl_;
  ku_ = A.ku_;
}

  // destructor 

LaBandMatDouble::~LaBandMatDouble()
{
}

LaBandMatDouble& LaBandMatDouble::operator=(double a)
{
    int M = size(0);
    int N = size(1);
    int i,j;

    for (j=0; j<M; j++)
        for (i=0; i<N; i++)
            data_(i,j) = a;

    return *this;
}

LaBandMatDouble& LaBandMatDouble::resize(const LaBandMatDouble &ob)
{

  data_.resize(ob.data_);
  N_ = ob.N_;
  kl_ = ob.kl_;
  ku_ = ob.ku_;

  return *this;
}


LaBandMatDouble& LaBandMatDouble::operator=(const LaBandMatDouble &B)
{
    data_ = B.data_;
    N_ = B.N_;
    kl_ = B.kl_;
    ku_ = B.ku_;

    return *this;
}


LaBandMatDouble LaBandMatDouble::copy(const LaBandMatDouble &ob)
{

  int i,j;

  resize(ob);

  for (j=0; j<ob.N_; j++)
    for (i=0; i<ob.N_; i++)
        data_(i,j) = ob(i,j);

  return *this;
}

std::ostream& operator<<(std::ostream &s, const LaBandMatDouble &ob)
{
  if (*(ob.info_))     // print out only matrix info, not actual values
  {
      *(ob.info_) = 0; // reset the flag
      s << "(" << ob.size(0) << "x" << ob.size(1) << ") " ;
      s << "Indices: " << ob.index(0) << " " << ob.index(1);
      s << " #ref: " << ob.ref_count() ;
      s << " sa:" << ob.shallow();
  }
#if 0
  else
  {
    int i,j;
    int N_ = ob.N_;
    int ku_ = ob.ku_;
    int kl_ = ob.kl_;

    for (j=0; j<N_; j++)
    {
      for (i=0; i<N_; i++)
          if(((i>=j)&&(i-j<=kl_))||((j>i)&&(j-i<=ku_)))
            s << ob(i,j) << ' ';
          else
            s << "  ";
      s << "\n";
    }
  }
#endif //0
    
    else
    {
        int i,j;
        int N_ = ob.N_;
        
        for (j=0; j<N_; j++)
        {
            for (i=0; i<N_; i++)
                s << ob(i,j) << ' ';
            s << "\n";
        }
    }

  return s;
}

