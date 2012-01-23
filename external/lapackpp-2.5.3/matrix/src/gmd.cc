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
#include LA_PREFS_H
#include LA_GEN_MAT_DOUBLE_H
#include LA_GEN_MAT_FLOAT_H
#include LA_EXCEPTION_H
#include "mtmpl.h"
#include "blas3pp.h"

// The rest of the "template function wrapper methods" are
// implemented in the file gmtmpl.cc .

DLLIMPORT int LaGenMatDouble::debug_ = 0;     // turn off global deubg flag initially.
                                // use A.debug(1) to turn on/off,
                                // and A.debug() to check current status.

DLLIMPORT int* LaGenMatDouble::info_= new int;        // turn off info print flag.

LaGenMatDouble::~LaGenMatDouble()
{ }

LaGenMatDouble::LaGenMatDouble()
    : v(0)
    , shallow_(0)
{
   dim[0] = 0;
   dim[1] = 0;
   sz[0] = 0;
   sz[1] = 0;
   //*info_ = 0;
   //init(0, 0);
   if (debug())
   {
      std::cout << "*** LaGenMatDouble::LaGenMatDouble()\n";
   }
}

LaGenMatDouble::LaGenMatDouble(int m, int n) 
   : v(m*n)
{
   init(m, n);
}

// modified constructor to support row ordering (jg)
LaGenMatDouble::LaGenMatDouble(value_type *d, int m, int n, bool row_ordering)
   : v(d, m, n, row_ordering)
{
   init(m, n);
   if (debug())
   {
      std::cout << ">>> LaGenMatDouble::LaGenMatDouble(double *d, int m, int n)\n";
   }
}

LaGenMatDouble::LaGenMatDouble(const LaGenMatDouble& X)
   : v(0)
{
   debug_ = X.debug_;
   shallow_ = 0;  // do not perpeturate shallow copies, otherwise
   //  B = A(I,J) does not work properly...

   if (debug())
   {
      std::cout << ">>> LaGenMatDouble::LaGenMatDouble(const LaGenMatDouble&)\n";
   }

   if (X.shallow_)
   {
      v.ref(X.v);
      dim[0] = X.dim[0];
      dim[1] = X.dim[1];
      sz[0] = X.sz[0];
      sz[1] = X.sz[1];
      ii[0] = X.ii[0];
      ii[1] = X.ii[1];
   }
   else
   {
      if (X.debug())
	 std::cout << std::endl << "Data is being copied!\n" << std::endl;

      init(X.size(0), X.size(1));

      copy(X);
   }  

   if (debug())
   {
      std::cout << "*this: " << info() << std::endl;
      std::cout << "<<< LaGenMatDouble::LaGenMatDouble(const LaGenMatDouble& X)\n";
   }
}

LaGenMatDouble::LaGenMatDouble(const LaGenMatFloat& X)
   : v(X.size(0)*X.size(1))
{
   debug_ = X.debug();

   init(X.size(0), X.size(1));

   int M = X.size(0), N = X.size(1);
   for (int j=0; j<N; j++)
      for (int i=0; i<M; i++)
	 // Conversion from float to double
	 (*this)(i,j) = X(i,j);

   if (debug())
   {
      std::cout << "*this: " << info() << std::endl;
      std::cout << "<<< LaGenMatDouble::LaGenMatDouble(const LaGenMatFloat& X)\n";
   }
}

void LaGenMatDouble::init(int m, int n)
{
   if (m && n)
   {
      ii[0](0,m-1);
      ii[1](0,n-1);
   }
   dim[0] = m;
   dim[1] = n;
   sz[0] = m;
   sz[1] = n;
   *info_ = 0;
   shallow_= 0;
}

// ////////////////////////////////////////

LaGenMatDouble& LaGenMatDouble::ref(const LaGenMatDouble& s)
{
   // handle trivial M.ref(M) case
   if (this == &s) return *this;
   else
   {
      ii[0] = s.ii[0];
      ii[1] = s.ii[1];
      dim[0] = s.dim[0];
      dim[1] = s.dim[1];
      sz[0] = s.sz[0];
      sz[1] = s.sz[1];
      shallow_ = 0;

      v.ref(s.v);

      return *this;
   }
}

LaGenMatDouble& LaGenMatDouble::scale(double s)
{
   Blas_Scale(s, *this);
   return *this;
}
LaGenMatDouble& LaGenMatDouble::operator*=(double s)
{ return scale(s); }

LaGenMatDouble LaGenMatDouble::operator()(const LaIndex& II, const LaIndex& JJ) const
{
   if (debug())
   {
      std::cout << ">>> LaGenMatDouble::operator(const LaIndex& const LaIndex&) const\n";
   }
   LaIndex I, J;
   mtmpl::submatcheck(*this, II, JJ, I, J);

   LaGenMatDouble tmp;

   int Idiff = (I.end() - I.start())/I.inc();
   int Jdiff = (J.end() - J.start())/J.inc();

   tmp.dim[0] = dim[0];
   tmp.dim[1] = dim[1];
   tmp.sz[0] = Idiff + 1;
   tmp.sz[1] = Jdiff + 1;

   tmp.ii[0].start() =  ii[0].start() + I.start()*ii[0].inc();
   tmp.ii[0].inc() = ii[0].inc() * I.inc();
   tmp.ii[0].end() = Idiff * tmp.ii[0].inc() + tmp.ii[0].start();

   tmp.ii[1].start() =  ii[1].start() + J.start()*ii[1].inc();
   tmp.ii[1].inc() = ii[1].inc() * J.inc();
   tmp.ii[1].end() = Jdiff * tmp.ii[1].inc() + tmp.ii[1].start();

   tmp.v.ref(v);
   tmp.shallow_assign();

   if (debug())
   {
      std::cout << "    return value: " << tmp.info() << std::endl;
      std::cout << "<<< LaGenMatDouble::operator(const LaIndex& const LaIndex&) const\n";
   }

   return tmp;

   // A second return statement that returns a variable name different
   // from the first one in order to switch off the "name return value
   // optimization" that would otherwise kill the copy() semantics of
   // the copy constructor.
   //  return LaGenMatDouble();
}

LaGenMatDouble LaGenMatDouble::operator()(const LaIndex& II, const LaIndex& JJ) 
{
   if (debug())
   {
      std::cout << ">>> LaGenMatDouble::operator(const LaIndex& const LaIndex&)\n";
   }
   LaIndex I, J;
   mtmpl::submatcheck(*this, II, JJ, I, J);

   LaGenMatDouble tmp;

   int Idiff = (I.end() - I.start())/I.inc();
   int Jdiff = (J.end() - J.start())/J.inc();

   tmp.dim[0] = dim[0];
   tmp.dim[1] = dim[1];
   tmp.sz[0] = Idiff + 1;
   tmp.sz[1] = Jdiff + 1;

   tmp.ii[0].start() =  ii[0].start() + I.start()*ii[0].inc();
   tmp.ii[0].inc() = ii[0].inc() * I.inc();
   tmp.ii[0].end() = Idiff * tmp.ii[0].inc() + tmp.ii[0].start();

   tmp.ii[1].start() =  ii[1].start() + J.start()*ii[1].inc();
   tmp.ii[1].inc() = ii[1].inc() * J.inc();
   tmp.ii[1].end() = Jdiff * tmp.ii[1].inc() + tmp.ii[1].start();

   tmp.v.ref(v);
   tmp.shallow_assign();

   if (debug())
   {
      std::cout << "    return value: " << tmp.info() << std::endl;
      std::cout << "<<< LaGenMatDouble::operator(const LaIndex& const LaIndex&)\n";
   }

   return tmp;

   // A second return statement that returns a variable name different
   // from the first one in order to switch off the "name return value
   // optimization" that would otherwise kill the copy() semantics of
   // the copy constructor.
   //return LaGenMatDouble();
}


// The rest of the "template function wrapper methods" are
// implemented in the file gmtmpl.cc .
