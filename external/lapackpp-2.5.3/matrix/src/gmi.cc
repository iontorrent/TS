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
#include LA_GEN_MAT_INT_H
#include LA_EXCEPTION_H
#include "mtmpl.h"

// The rest of the "template function wrapper methods" are
// implemented in the file gmtmpl.cc .

DLLIMPORT int LaGenMatInt::debug_ = 0;     // turn off global deubg flag initially.
                                // use A.debug(1) to turn on/off,
                                // and A.debug() to check current status.

DLLIMPORT int* LaGenMatInt::info_= new int;        // turn off info print flag.

LaGenMatInt::~LaGenMatInt()
{ }

LaGenMatInt::LaGenMatInt()
   : v(0)
{
   init(0, 0);
}

LaGenMatInt::LaGenMatInt(int m, int n) 
   : v(m*n)
{
   init(m, n);
}

// modified constructor to support row ordering (jg)
LaGenMatInt::LaGenMatInt(value_type *d, int m, int n, bool row_ordering)
   : v(d, m, n, row_ordering)
{
   init(m, n);
   if (debug())
   {
      std::cout << ">>> LaGenMatInt::LaGenMatInt(double *d, int m, int n)\n";
   }
}

LaGenMatInt::LaGenMatInt(const LaGenMatInt& X)
   : v(0)
{
   debug_ = X.debug_;
   shallow_ = 0;  // do not perpeturate shallow copies, otherwise
   //  B = A(I,J) does not work properly...

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
      std::cout << "<<< LaGenMatInt::LaGenMatInt(const LaGenMatInt& X)\n";
   }
}

void LaGenMatInt::init(int m, int n)
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

LaGenMatInt& LaGenMatInt::ref(const LaGenMatInt& s)
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

LaGenMatInt LaGenMatInt::operator()(const LaIndex& II, const LaIndex& JJ) const
{
   if (debug())
   {
      std::cout << ">>> LaGenMatInt::operator(const LaIndex& const LaIndex&)\n";
   }
   LaIndex I, J;
   mtmpl::submatcheck(*this, II, JJ, I, J);

   LaGenMatInt tmp;

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
      std::cout << "<<< LaGenMatInt::operator(const LaIndex& const LaIndex&)\n";
   }

   return tmp;
}

LaGenMatInt LaGenMatInt::operator()(const LaIndex& II, const LaIndex& JJ) 
{
   if (debug())
   {
      std::cout << ">>> LaGenMatInt::operator(const LaIndex& const LaIndex&)\n";
   }
   LaIndex I, J;
   mtmpl::submatcheck(*this, II, JJ, I, J);

   LaGenMatInt tmp;

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
      std::cout << "<<< LaGenMatInt::operator(const LaIndex& const LaIndex&)\n";
   }

   return tmp;

}


// The rest of the "template function wrapper methods" are
// implemented in the file gmtmpl.cc .
