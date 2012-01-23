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
#include LA_GEN_MAT_COMPLEX_H
#include LA_EXCEPTION_H
#include "mtmpl.h"
#include LA_TEMPLATES_H
#include "blas3pp.h"

DLLIMPORT int LaGenMatComplex::debug_ = 0;     // turn off global deubg flag initially.
                                // use A.debug(1) to turn on/off,
                                // and A.debug() to check current status.

DLLIMPORT int* LaGenMatComplex::info_= new int;        // turn off info print flag.

LaGenMatComplex::~LaGenMatComplex()
{}

LaGenMatComplex::LaGenMatComplex()
   : v(0)
{
   init(0, 0);
}

LaGenMatComplex::LaGenMatComplex(int m, int n) 
   : v(m*n)
{
   init(m, n);
}

// modified constructor to support row ordering (jg)
LaGenMatComplex::LaGenMatComplex(value_type *d, int m, int n, bool row_ordering)
   : v(d, m, n, row_ordering)
{
   init(m, n);
   if (debug())
   {
      std::cout << ">>> LaGenMatComplex::LaGenMatComplex(double *d, int m, int n)\n";
   }
}

LaGenMatComplex::LaGenMatComplex(const LaGenMatComplex& X)
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
      size0 = X.size0;
      size1 = X.size1;
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
      std::cout << "<<< LaGenMatComplex::LaGenMatComplex(const LaGenMatComplex& X)\n";
   }
}


LaGenMatComplex::LaGenMatComplex(const LaGenMatDouble& s_real, 
				 const LaGenMatDouble& s_imag)
   : v(0)
{
   init(s_real.size(0), s_real.size(1));

   copy(s_real, s_imag);
}

void LaGenMatComplex::init(int m, int n)
{
   if (m && n)
   {
      ii[0](0,m-1);
      ii[1](0,n-1);
   }
   dim[0] = m;
   dim[1] = n;
   size0 = m;
   size1 = n;
   *info_ = 0;
   shallow_= 0;
}

// ////////////////////////////////////////
typedef LaGenMatComplex matrix_type;

LaGenMatComplex& LaGenMatComplex::operator=(const LaComplex& s)
{
    return operator=(s.toCOMPLEX());
}

LaGenMatComplex& LaGenMatComplex::operator+=(COMPLEX s)
{
  for(int j=0; j < size(1); j++)
    for(int i=0; i < size(0); i++)
    {
      (*this)(i,j).r+=s.r;
      (*this)(i,j).i+=s.i;
    }

  return *this;
}

LaGenMatComplex& LaGenMatComplex::scale(const LaComplex& s)
{
   Blas_Scale(s.toCOMPLEX(), *this);
   return *this;
}
LaGenMatComplex& LaGenMatComplex::scale(COMPLEX s)
{
   return scale(LaComplex(s));
}
LaGenMatComplex& LaGenMatComplex::operator*=(COMPLEX s)
{ return scale(s); }

LaGenMatComplex& LaGenMatComplex::ref(const LaGenMatComplex& s)
{

   // handle trivial M.ref(M) case
   if (this == &s) return *this;
   else
   {
      ii[0] = s.ii[0];
      ii[1] = s.ii[1];
      dim[0] = s.dim[0];
      dim[1] = s.dim[1];
      size0 = s.size0;
      size1 = s.size1;
      shallow_ = 0;

      v.ref(s.v);

      return *this;
   }
}


LaGenMatComplex& LaGenMatComplex::copy(const LaGenMatDouble& s_real, 
				       const LaGenMatDouble& s_imag)
{
   // current scheme in copy() is to detach the left-hand-side
   // from whatever it was pointing to.
   resize(s_real.size(0), s_real.size(1));

   // optimize later; for now use the correct but slow implementation
   int i, j,  M=size(0), N=size(1);

   LaGenMatComplex &dest = *this;
   if (s_imag.size(0) > 0 && s_imag.size(1) > 0)
      for (j=0; j<N; ++j)
	 for (i=0; i<M; ++i)
	 {
	    dest(i,j).r = s_real(i,j);
	    dest(i,j).i = s_imag(i,j);
	 }
   else
      for (j=0; j<N; ++j)
	 for (i=0; i<M; ++i)
	 {
	    dest(i,j).r = s_real(i,j);
	    dest(i,j).i = 0.0;
	 }

   return *this;
}


LaGenMatComplex LaGenMatComplex::operator()(const LaIndex& II, const LaIndex& JJ) const
{
   if (debug())
   {
      std::cout << ">>> LaGenMatComplex::operator(const LaIndex& const LaIndex&)\n";
   }
   LaIndex I, J;
   mtmpl::submatcheck(*this, II, JJ, I, J);

   LaGenMatComplex tmp;

   int Idiff = (I.end() - I.start())/I.inc();
   int Jdiff = (J.end() - J.start())/J.inc();

   tmp.dim[0] = dim[0];
   tmp.dim[1] = dim[1];
   tmp.size0 = Idiff + 1;
   tmp.size1 = Jdiff + 1;

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
      std::cout << "<<< LaGenMatComplex::operator(const LaIndex& const LaIndex&)\n";
   }

   return tmp;

}

LaGenMatComplex LaGenMatComplex::operator()(const LaIndex& II, const LaIndex& JJ) 
{
   if (debug())
   {
      std::cout << ">>> LaGenMatComplex::operator(const LaIndex& const LaIndex&)\n";
   }
   LaIndex I, J;
   mtmpl::submatcheck(*this, II, JJ, I, J);

   LaGenMatComplex tmp;

   int Idiff = (I.end() - I.start())/I.inc();
   int Jdiff = (J.end() - J.start())/J.inc();

   tmp.dim[0] = dim[0];
   tmp.dim[1] = dim[1];
   tmp.size0 = Idiff + 1;
   tmp.size1 = Jdiff + 1;

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
      std::cout << "<<< LaGenMatComplex::operator(const LaIndex& const LaIndex&)\n";
   }

   return tmp;
}


std::ostream& operator<<(std::ostream& s, const LaGenMatComplex& G)
{
    if (*(G.info_))     // print out only matrix info, not actual values
    {
        *(G.info_) = 0; // reset the flag
	G.Info(s);
    }
    else 
    {

        int i,j;
        LaPreferences::pFormat p = LaPreferences::getPrintFormat();
        bool newlines = LaPreferences::getPrintNewLines();

        if((p == LaPreferences::MATLAB) || (p == LaPreferences::MAPLE))
          s << "[";

        for (i=0; i<G.size0; i++)
        {
            if(p == LaPreferences::MAPLE)
              s << "[";
            for (j=0; j<G.size1; j++)
            {
              if(p == LaPreferences::NORMAL)
                s << G(i,j);
              if(p == LaPreferences::MATLAB)
                s << G(i,j).r << "+" << G(i,j).i << "i";
              if(p == LaPreferences::MAPLE)
                s << G(i,j).r << "+" << G(i,j).i << "*I";
              if(((p == LaPreferences::NORMAL) || (p == LaPreferences::MATLAB)) && (j != G.size(1)-1))
                s << "  ";
              if(((p == LaPreferences::MAPLE)) && (j != G.size(1)-1))
                s << ", ";              
            }
            if(p == LaPreferences::MAPLE)
            {
              s << "]";
              if(i != G.size(0)-1)
                s << ", ";
            }
            if((p == LaPreferences::MATLAB) && (i != G.size(0)-1))
              s << ";  ";
            if( ((newlines)||(p==LaPreferences::NORMAL)) && (i != G.size(0)-1)) // always print newline if in NORMAL mode
              s << "\n";
        }
        if((p == LaPreferences::MATLAB) || (p == LaPreferences::MAPLE))
          s << "]";

        s << "\n";
    }
    return s;
}

LaGenMatDouble LaGenMatComplex::real() const
{ return real_to_LaGenMatDouble().shallow_assign(); }

LaGenMatDouble LaGenMatComplex::imag() const
{ return imag_to_LaGenMatDouble().shallow_assign(); }


matrix_type matrix_type :: zeros (int N, int M)
{ 
   matrix_type mat(N, M == 0 ? N : M);
   mat = LaComplex(0, 0);
   return mat.shallow_assign();
}
matrix_type matrix_type :: ones (int N, int M)
{ 
   matrix_type mat(N, M == 0 ? N : M);
   mat = LaComplex(1, 0);
   return mat.shallow_assign();
}
matrix_type matrix_type :: eye (int N, int M)
{ 
   matrix_type mat(zeros(N, M));
   LaComplex one(1, 0);
   int nmin = (M == 0 ? N : (M < N ? M : N));
   for (int k = 0; k < nmin; ++k)
      mat(k, k) = one;
   return mat.shallow_assign();
}
matrix_type matrix_type :: from_diag (const matrix_type &vect)
{
  if (vect.rows() != 1 && vect.cols() != 1)
    throw LaException("diag<matT>(const matT& vect, matT& mat)",
		      "The argument 'vect' is not a vector: "
		      "neither dimension is equal to one");
  int nmax(vect.rows() > vect.cols() ? vect.rows() : vect.cols());
  matrix_type mat(nmax, nmax);
  if (vect.rows() == 1)
    for (int k = 0; k < nmax; ++k)
      mat(k, k) = vect(0, k);
  else
    for (int k = 0; k < nmax; ++k)
      mat(k, k) = vect(k, 0);
  return mat.shallow_assign();
}
bool matrix_type :: is_zero() const
{
  int i, j,  M=rows(), N=cols();

  COMPLEX zero = LaComplex(0);
  for (j=0;j<N;j++)
    for (i=0;i<M; i++)
      if (operator() (i, j) != zero)
	return false;
  return true;
}
matrix_type::value_type matrix_type :: trace () const
{
  int M=rows(), N=cols();

  LaComplex result(0);
  int nmin = (M == 0 ? N : (M < N ? M : N));
  for (int k = 0; k < nmin; ++k)
     result += LaComplex(operator() (k, k));
  return result;
}
matrix_type matrix_type :: linspace (matrix_type::value_type start, matrix_type::value_type end, int nr_points)
{
   LaGenMatDouble re(LaGenMatDouble::linspace(start.r, end.r, nr_points));
   LaGenMatDouble im(LaGenMatDouble::linspace(start.i, end.i, nr_points));
   return LaGenMatComplex(re, im).shallow_assign();
}
matrix_type matrix_type :: rand (int N, int M,
				 double low, double high) 
{
   LaGenMatDouble re(LaGenMatDouble::rand(N, M, low, high));
   LaGenMatDouble im(LaGenMatDouble::rand(N, M, low, high));
   return LaGenMatComplex(re, im).shallow_assign();
}

