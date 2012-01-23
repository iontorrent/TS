// -*-C++-*- 

// Copyright (C) 2004 
// Christian Stimming <stimming@tuhh.de>

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2, or
// (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License along
// with this library; see the file COPYING.  If not, write to the Free
// Software Foundation, 59 Temple Place - Suite 330, Boston, MA 02111-1307,
// USA.


// constructor/destructor functions

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "lafnames.h"
#include LA_GEN_QR_FACT_COMPLEX_H
#include LA_EXCEPTION_H
#include "lapackc.h"

LaGenQRFactComplex::LaGenQRFactComplex()
    : _matA()
    , _tau()
    , _work()
{
}

LaGenQRFactComplex::LaGenQRFactComplex(LaGenMatComplex& A)
    : _matA()
    , _tau()
    , _work()
{
   decomposeQR_IP(A);
}

LaGenQRFactComplex::LaGenQRFactComplex(LaGenQRFactComplex& q)
    : _matA()
    , _tau()
    , _work()
{
   _matA.ref(q._matA);
   _tau.ref(q._tau);
}

LaGenQRFactComplex::~LaGenQRFactComplex()
{
}

void LaGenQRFactComplex::decomposeQR_IP(LaGenMatComplex& A)
{
   integer m = A.size(0);
   integer n = A.size(1);
   integer lda = A.gdim(0);
   integer info = 0;

   _matA.ref(A);
   _tau.resize(std::min(m,n), 1);

   integer lwork;
   if (_work.size() >= n)
      lwork = _work.size();
   else 
   {
      // Calculate the optimal temporary workspace
      lwork = -1;
      _work.resize(1, 1);
      F77NAME(zgeqrf)(&m, &n, &_matA(0,0), &lda, &_tau(0),
		      &_work(0), &lwork, &info);
      lwork = int(la::abs(LaComplex(_work(0))));
      _work.resize(lwork, 1);
   }

   F77NAME(zgeqrf)(&m, &n, &_matA(0,0), &lda, &_tau(0),
		   &_work(0), &lwork, &info);
   
   // this shouldn't really happen.
   if (info < 0)
      throw(LaException("", "Internal error in LAPACK: SGELS()"));
}


LaGenMatComplex& LaGenQRFactComplex::generateQ_IP()
{
   generateQ_internal(_matA);
   return _matA.shallow_assign();
}

void LaGenQRFactComplex::generateQ(LaGenMatComplex& A) const
{
   A.copy(_matA);
   generateQ_internal(A);
}

void LaGenQRFactComplex::generateQ_internal(LaGenMatComplex& A) const
{
   integer m = A.size(0);
   integer n = A.size(1);
   integer k = std::min(m, n); 
   integer lda = A.gdim(0);
   integer info = 0;

   integer lwork;
   if (_work.size() >= n)
      lwork = _work.size();
   else 
   {
      // Calculate the optimal temporary workspace
      lwork = -1;
      _work.resize(1, 1);
      F77NAME(zungqr)(&m, &n, &k, &A(0,0), &lda, &_tau(0),
		      &_work(0), &lwork, &info);
      lwork = int(la::abs(LaComplex(_work(0))));
      _work.resize(lwork, 1);
   }

   F77NAME(zungqr)(&m, &n, &k, &A(0,0), &lda, &_tau(0),
		   &_work(0), &lwork, &info);
   
   // this shouldn't really happen.
   if (info < 0)
      throw(LaException("", "Internal error in LAPACK: SGELS()"));
}

void LaGenQRFactComplex::Mat_Mult(LaGenMatComplex& C, bool hermitian, 
				  bool from_left) const 
{
   char side = from_left ? 'L' : 'R';
   char trans = hermitian ? 'C' : 'N';
   integer m = C.size(0);
   integer n = C.size(1);
   integer k = std::min(m, n);
   integer ldc = C.gdim(0);
   integer lda = _matA.gdim(0);
   integer info = 0;

   integer lwork;
   if (_work.size() >= std::max(m,n))
      lwork = _work.size();
   else 
   {
      // Calculate the optimal temporary workspace
      lwork = -1;
      _work.resize(1, 1);
      F77NAME(zunmqr)(&side, &trans, &m, &n, &k, &_matA(0,0), &lda, &_tau(0),
		      &C(0,0), &ldc, &_work(0), &lwork, &info);
      lwork = int(la::abs(LaComplex(_work(0))));
      _work.resize(lwork, 1);
   }

   F77NAME(zunmqr)(&side, &trans, &m, &n, &k, &_matA(0,0), &lda, &_tau(0),
		   &C(0,0), &ldc, &_work(0), &lwork, &info);
   
   // this shouldn't really happen.
   if (info < 0)
      throw(LaException("", "Internal error in LAPACK: SGELS()"));
}

