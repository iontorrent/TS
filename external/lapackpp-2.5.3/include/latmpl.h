/*-*-c++-*-****************************************************************
 *                     latmpl.h C++ Templates for lapackpp classes
                       -------------------
 begin                : 2005-12-29
 copyright            : (C) 2005 by Christian Stimming
 email                : stimming@tuhh.de
***************************************************************************/

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2, or (at
// your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License along
// with this library; see the file COPYING.  If not, write to the Free
// Software Foundation, 59 Temple Place - Suite 330, Boston, MA 02111-1307,
// USA.

#ifndef _LATMPL_H
#define _LATMPL_H

/** @file
 * @brief Template functions for matrices
 *
 * The definition of template functions for easier usage of lapackpp's
 * matrix classes.
 */

#include <cstdlib>
#include <cmath>
#include <lafnames.h>
#include LA_EXCEPTION_H

// Watch out: Only some compilers correctly implement the C++
// standard specifications, where type names from inside template
// classes need to be preceded by the "typename" keyword. Other
// compilers do not accept the keyword "typename". For those we
// define it as empty here.
#ifndef __GNUC__
   // Non-gcc compiler
#  ifdef _MSC_VER
     // This is Microsoft Visual C++.
     //  Microsoft Visual C++ 7.1  _MSC_VER = 1310
     //  Microsoft Visual C++ 7.0  _MSC_VER = 1300
     //  Microsoft Visual C++ 6.0  _MSC_VER = 1200
     //  Microsoft Visual C++ 5.0  _MSC_VER = 1100
#    if _MSC_VER <= 1300
       // This is Microsoft Visual C++ 7.0 or older, so define the
       // keyword as empty
#      define typename
#    else
       // Microsoft Visual C++ 7.1 or newer. typename should be ok.
#    endif
#  else
     // Compiler unknown
#  endif
#endif

namespace la {

/** @name Matrix Predicates */
//@{
/** Returns true if this is an all-zero matrix. (New in lapackpp-2.4.4) */
template <class matT>
bool is_zero(const matT& mat)
{
  int i, j,  M=mat.rows(), N=mat.cols();

  // If your compiler gives an error in this line, please see the
  // note above on "typename".
  typename matT::value_type zero(0);

  for (j=0;j<N;j++)
    for (i=0;i<M; i++)
      if (mat(i, j) != zero)
	return false;
  return true;
}

/** Returns true if both matrices are exactly equal. (New in
 * lapackpp-2.4.4) */
template <class matT>
bool equal(const matT& mat1, const matT& mat2)
{
  int i, j,  M=mat1.rows(), N=mat1.cols();
  if (mat1.rows() != mat2.rows() || mat1.cols() != mat2.cols())
    throw LaException("equal<matT>(const matT&, const matT)",
		      "Both matrices have unequal sizes");
  for (j=0;j<N;j++)
    for (i=0;i<M; i++)
      if (mat1(i, j) != mat2(i, j))
	return false;
  return true;
}
//@}


/** @name Create elementary matrices */
//@{
/** Sets the given matrix \c M to an all-zero matrix of dimension \c
 * NxN, if \c M is not given, or \c NxM if \c M is given.
 * (New in lapackpp-2.4.4) */
template <class matT> 
void zeros(matT& mat, int N, int M=0)
{
  mat.resize(N, M == 0 ? N : M);
  mat = typename matT::value_type(0);
}

/** Sets the given matrix \c M to an all-one matrix of dimension \c
 * NxN, if \c M is not given, or \c NxM if \c M is given. (New in
 * lapackpp-2.4.4) */
template <class matT> 
void ones(matT& mat, int N, int M=0)
{
  mat.resize(N, M == 0 ? N : M);
  mat = typename matT::value_type(1);
}

/** Sets the given matrix \c M to an identity matrix (a diagonal of
 * ones and all other elements zeros) of square dimension \c NxN, if
 * \c M is not given, or of rectangular dimension \c NxM if \c M is
 * given. (New in lapackpp-2.4.4) */
template <class matT> 
void eye(matT& mat, int N, int M=0)
{
  int nmin = (M == 0 ? N : (M < N ? M : N));
  mat.resize(N, M == 0 ? N : M);
  mat = typename matT::value_type(0);
  typename matT::value_type one(1);
  for (int k = 0; k < nmin; ++k)
    mat(k, k) = one;
}

/** Fills the given matrix \c A with pseudo-random values. The
 * values are uniformly distributed in the interval \c (0,1) or,
 * if specified, \c (low,high).
 *
 * Note: Since this uses the system's \c rand() call, the
 * randomness of the values might be questionable -- don't use
 * this if you need really strong random numbers. */
template <class matT>
void rand(matT &A, typename matT::value_type low = 0,
	  typename matT::value_type high = 1)
{
   int i, j,  M = A.rows(), N = A.cols();
   typename matT::value_type scale = high - low;
   for (j=0; j<N; ++j)
      for (i=0; i<M; ++i)
	 A(i,j) = low + 
	    scale * double(std::rand()) / double(RAND_MAX);
}

/** Fills the given matrix \c A with pseudo-random values, where
 * the value type of the matrix is an integer type. The values are
 * uniformly distributed in the interval \c [0,1] or, if
 * specified, \c (low,high), in both cases including the interval
 * edges.
 *
 * Note: Since this uses the system's \c rand() call, the
 * randomness of the values might be questionable -- don't use
 * this if you need really strong random numbers. */
template <class matT>
void int_rand(matT &A, typename matT::value_type low = 0,
	      typename matT::value_type high = 1)
{
   int i, j,  M = A.rows(), N = A.cols();
   double bins = high - low + 1;
   for (j=0; j<N; ++j)
      for (i=0; i<M; ++i)
	 A(i,j) = low + 
	    typename matT::value_type(
	       std::floor(bins * double(std::rand()) / 
			  double(RAND_MAX)));
}

/** Sets the given matrix \c mat to a diagonal matrix with the
 * given vector \c vect on the diagonal and zeros elsewhere. The
 * matrix \c mat is allowed to be non-square, only the length of
 * the diagonal has to fit to the vector's length. The vector \c
 * vect is allowed to be either a row vector (dimension 1xN) or a
 * column vector (dimension Nx1). (New in lapackpp-2.4.4) */
template <class matT>
void from_diag(matT& mat, const matT& vect)
{
  int nmin = (mat.rows() < mat.cols() ? mat.rows() : mat.cols());
  mat = typename matT::value_type(0);
  if (vect.rows() != 1 && vect.cols() != 1)
    throw LaException("diag<matT>(const matT& vect, matT& mat)",
		      "The argument 'vect' is not a vector: "
		      "neither dimension is equal to one");
  if (vect.rows() * vect.cols() != nmin)
    throw LaException("diag<matT>(const matT& vect, matT& mat)",
		      "The size of the vector is unequal to the matrix diagonal");
  if (vect.rows() == 1)
    for (int k = 0; k < nmin; ++k)
      mat(k, k) = vect(0, k);
  else
    for (int k = 0; k < nmin; ++k)
      mat(k, k) = vect(k, 0);
}

//@}


/** @name Constructors for elementary and special matrices */
//@{
/** Returns a newly allocated all-zero matrix of dimension \c NxN, if
 * \c M is not given, or \c NxM if \c M is given.
 * (New in lapackpp-2.4.4) */
template <class matT>
matT zeros(int N, int M=0)
{
  matT mat;
  zeros(mat, N, M);
  return mat.shallow_assign();
}

/** Returns a newly allocated all-one matrix of dimension \c NxN, if
 * \c M is not given, or \c NxM if \c M is given.
 * (New in lapackpp-2.4.4) */
template <class matT>
matT ones(int N, int M=0)
{
  matT mat;
  ones(mat, N, M);
  return mat.shallow_assign();
}

/** Returns a newly allocated identity matrix of dimension \c NxN, if
 * \c M is not given, or a rectangular matrix \c NxM if \c M is given.
 * (New in lapackpp-2.4.4) */
template <class matT>
matT eye(int N, int M=0)
{
  matT mat;
  eye(mat, N, M);
  return mat.shallow_assign();
}

/** Returns a newly allocated matrix of dimension \c NxM with
 * pseudo-random values. The values are uniformly distributed in
 * the interval \c (0,1) or, if specified, \c (low,high).  (New in
 * lapackpp-2.4.5)
 *
 * Note: Since this uses the system's \c rand() call, the
 * randomness of the values might be questionable -- don't use
 * this if you need really strong random numbers. */
template <class matT>
matT rand(int N, int M,
	  typename matT::value_type low = 0,
	  typename matT::value_type high = 1)
{
   matT mat(N, M);
   rand(mat, low, high);
   return mat.shallow_assign();
}

/** Returns a newly allocated matrix of dimension \c NxM with
 * pseudo-random values, where the matrix element type is an
 * integer type. The values are uniformly distributed in the
 * interval \c [0,1] or, if specified, \c (low,high), in both
 * cases including the interval edges. (New in lapackpp-2.4.5)
 *
 * Note: Since this uses the system's \c rand() call, the
 * randomness of the values might be questionable -- don't use
 * this if you need really strong random numbers. */
template <class matT>
matT int_rand(int N, int M,
	  typename matT::value_type low = 0,
	  typename matT::value_type high = 1)
{
   matT mat(N, M);
   int_rand(mat, low, high);
   return mat.shallow_assign();
}

/** Returns a newly allocated diagonal matrix of dimension \c NxN
 * that has the vector \c vect of length \c N on the diagonal.
 * The vector \c vect is allowed to be either a row vector
 * (dimension 1xN) or a column vector (dimension Nx1). (New in
 * lapackpp-2.4.5) */
template <class matT>
matT from_diag(const matT& vect)
{
  if (vect.rows() != 1 && vect.cols() != 1)
    throw LaException("diag<matT>(const matT& vect, matT& mat)",
		      "The argument 'vect' is not a vector: "
		      "neither dimension is equal to one");
  int nmax(vect.rows() > vect.cols() ? vect.rows() : vect.cols());
  matT mat(nmax, nmax);
  from_diag(mat, vect);
  return mat.shallow_assign();
}

/** Returns a newly allocated matrix of type \c destT, containing
 * the element-by-element converted values of the matrix \c src
 * which was of type \c srcT. 
 *
 * The template argument \c destT must be specified; the template
 * argument \c srcT is deduced automatically. To convert a
 * LaGenMatDouble into a LaGenMatInt: 
 \verbatim
 LaGenMatDouble foo(5, 6);
 LaGenMatInt bar(la::convert_to<LaGenMatInt>(foo));
 \endverbatim
 * 
 * Note: This conversion should even work from and to various
 * matrices of the IT++ library, http://itpp.sourceforge.net (New
 * in lapackpp-2.4.5.)
*/
template<class destT, class srcT>
destT convert_mat(const srcT& src)
{
   destT res(src.rows(), src.cols());
   // optimize later; for now use the correct but slow implementation
   int i, j,  M=src.rows(), N=src.cols();
   for (j=0; j<N; ++j)
      for (i=0; i<M; ++i)
	 res(i, j) = typename destT::value_type ( src(i, j) );
   return res.shallow_assign();
}

/** Returns a newly allocated linarly spaced column vector with \c
 * nr_points elements, between and including \c start and \c
 * end. (New in lapackpp-2.4.5.) */
template <class matT>
matT linspace(typename matT::value_type start, 
	      typename matT::value_type end,
	      int nr_points)
{
   matT mat(nr_points, 1);
   typename matT::value_type stepsize = 
      (end - start) / typename matT::value_type (nr_points - 1);
   for (int k = 0; k < nr_points; ++k)
   {
      mat(k, 0) = start;
      start += stepsize;
   }
   return mat.shallow_assign();
}

/** Returns a newly allocated large matrix that consists of \c
 * M-by-N copies of the given matrix \c A. (New in
 * lapackpp-2.4.5.) */
template <class matT>
matT repmat(const matT& A, int M, int N)
{
   int i, j, origM = A.rows(), origN = A.cols();
   matT mat(origM * M, origN * N);
   for (j = 0; j < N; ++j)
      for (i = 0; i < M; ++i)
	 mat(LaIndex(i * origM, (i+1) * origM - 1),
	     LaIndex(j * origN, (j+1) * origN - 1))
	    .inject(A);

   return mat.shallow_assign();
}
//@}

/** @name Calculate some matrix measures */
//@{
/** Returns the trace, i.e. the sum of all diagonal elements of
 * the matrix. The matrix \c mat does not have to be square. (New
 * in lapackpp-2.4.4) */
template <class matT>
typename matT::value_type trace(const matT& mat)
{
  int nmin = (mat.rows() < mat.cols() ? mat.rows() : mat.cols());
  typename matT::value_type result(0);
  for (int k = 0; k < nmin; ++k)
    result += mat(k, k);
  return result;
}

/** Returns a newly allocated column vector of dimension \c Nx1 that
 * contains the diagonal of the given matrix \c mat. (New in
 * lapackpp-2.4.5) */
template <class matT>
matT diag(const matT& mat)
{
  int nmin = (mat.rows() < mat.cols() ? mat.rows() : mat.cols());
  matT vect(nmin, 1);
  for (int k = 0; k < nmin; ++k)
      vect(k, 0) = mat(k, k);
  return vect.shallow_assign();
}

//@}

} // namespace la

#endif // _LATMPL_H
