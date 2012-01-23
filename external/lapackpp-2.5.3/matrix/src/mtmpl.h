// -*-C++-*- 

// Copyright (C) 2005
// Christian Stimming <stimming@tuhh.de>

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

#ifndef MTMPL_H
#define MTMPL_H

#include "lafnames.h"
#include LA_PREFS_H
#include LA_EXCEPTION_H

/** This file and this namespace includes the template functions
 * that are common to all matrix classes.
 *
 * This way we do not start to switch from normal classes to
 * template classes to the outside, but in the inside
 * implementation all classes already use the identical function
 * code. Of course this has the advantage that one bugfix will be
 * available in all classes at once.*/
namespace mtmpl {

/** Set elements of \c mat to the scalar value \c scalar. No
 * new matrix is created, so that if there are other matrices
 * that reference this memory space, they will also be
 * affected. */
template<class MT> MT&
assign(MT& mat, typename MT::vec_type& vec, typename MT::value_type scalar)
{
   if (mat.debug())
      std::cout << ">>> matrix_type& matrix_type::operator=(value_type s)"
		<< std::endl
		<< "       s = " << scalar << std::endl;

   if (mat.has_unitstride() && ! mat.is_submatrixview())
      // Unit strides, not a submatrix view
      vec = scalar;
   else
   {
      // This is probably a submatrix view. Still try some
      // optimizations.
      int i, j,  M=mat.size(0), N=mat.size(1);

      // Mat::operator(i,j) is defined as
      // vec(mat.gdim(0)*(mat.start(1) + j*mat.inc(1)) +  mat.start(0) + i*mat.inc(0));
      int start01 = mat.gdim(0) * mat.start(1) + mat.start(0);
      int inc1 = mat.gdim(0) * mat.inc(1);

      if (M==1)
      {
	 // Special case: only one row; save the inner loop here.

	 // Naive implementation:
	 for (j=0; j<N; ++j)
	 //   //mat(0,j) = scalar;
	    vec(start01 + j*inc1) = scalar;

	 /*
	   Optimized version commented out because it might still
	   contain bugs.

	 // More optimized implementation from vtmpl.h assign():
	 typename MT::vec_type::value_type *iter = vec.addr() + start01;
	 typename MT::vec_type::value_type *end;
	 int nn = N % 5;
	 if (nn != 0)
	 {
	    end = iter + nn*inc1;
	    for ( ; iter != end; iter += inc1)
	       *iter = scalar;
	    if (N < 5)
	       return mat;
	 }
	 end = vec.addr() + start01 + N * inc1;
	 int _2inc1 = inc1 + inc1;
	 int _3inc1 = _2inc1 + inc1;
	 int _4inc1 = _3inc1 + inc1;
	 int _5inc1 = _4inc1 + inc1;
	 for ( ; iter != end; iter += _5inc1)
	 {
	    *iter = scalar;
	    iter[inc1] = scalar;
	    iter[_2inc1] = scalar;
	    iter[_3inc1] = scalar;
	    iter[_4inc1] = scalar;
	 }
	 */
      }
      else
      {
	 int inc0 = mat.inc(0);
	 int start01j;
	 for (j=0; j<N; ++j)
	 {
	    start01j = start01 + j*inc1;
	    for (i=0; i<M; ++i)
	       //mat(i,j) = scalar;
	       vec(start01j + i*inc0) = scalar;
	 }
      }
   }
   return mat;
}

/** Add the scalar value \c scalar to all elements of the matrix
 * \c mat. No new matrix is created, so that if there are other
 * matrices that reference this memory space, they will also be
 * affected. 
 *
 * @note This method is rather slow. In many cases, it can
 * be much faster to use Blas_Mat_Mult() with a Ones-Matrix
 * instead.
 */
template<class MT> MT&
add_scalar(MT& mat, typename MT::vec_type& vec, typename MT::value_type scalar)
{
   // Try some optimizations.
   int i, j,  M=mat.size(0), N=mat.size(1);

   // Mat::operator(i,j) is defined as
   // vec(mat.gdim(0)*(mat.start(1) + j*mat.inc(1)) +  mat.start(0) + i*mat.inc(0));
   int start01 = mat.gdim(0) * mat.start(1) + mat.start(0);
   int inc1 = mat.gdim(0) * mat.inc(1);

   if (M==1)
   {
      // Special case: only one row; save the inner loop here.

      // Naive implementation:
      for (j=0; j<N; ++j)
	 //   //mat(0,j) += scalar;
	 vec(start01 + j*inc1) += scalar;

   }
   else
   {
      int inc0 = mat.inc(0);
      int start01j;
      for (j=0; j<N; ++j)
      {
	 start01j = start01 + j*inc1;
	 for (i=0; i<M; ++i)
	    //mat(i,j) += scalar;
	    vec(start01j + i*inc0) += scalar;
      }
   }
   return mat;
}

/** Resize to a \e new matrix of size m x n. The element
 * values of the new matrix are \e uninitialized, even if
 * resizing to a smaller matrix. */
template<class MT> inline MT&
resize(MT& mat, int new_m, int new_n)
{
   assert(new_m >= 0);
   assert(new_n >= 0);

   if (mat.debug())
      std::cout << ">>> resize("<< new_m << "," 
		<< new_n << ")" << std::endl;

   // Check for submatrix view
   if (mat.is_submatrixview())
      // This is a submatrix view. Resize doesn't make sense.
//#ifdef DEBUG
	throw 
//#endif
	   LaException("LaGenMatDouble::resize(int,int)", "This is a submatrix view. Resize, copy() or operator=() does not make sense. Use inject() instead of copy() or operator=().");

   // first, reference 0x0 matrix, potentially freeing memory
   // this allows one to resize a matrix > 1/2 of the available
   // memory

   MT tmp1(0,0);
   mat.ref(tmp1);

   // now, reference an MxN matrix
   MT tmp(new_m,new_n);
   mat.ref(tmp);

   return mat;
}

/** Release left-hand side (reclaiming memory space if
 * possible) and copy elements of elements of \c s. Unline \c
 * inject(), it does not require conformity, and previous
 * references of left-hand side are unaffected. */
template<class MT> inline MT&
copy(MT& mat, const MT& src)
{
   if (mat.debug())
      std::cout << ">>> matrix_type& matrix_type::copy(const matrix_type& X)"
		<< std::endl
		<< " src: " << src.info()
		<< std::endl;

   // current scheme in copy() is to detach the left-hand-side
   // from whatever it was pointing to.
   mat.resize(src);

   mat.inject(src);

   return mat;
}

/** Copy elements of s into the memory space referenced by the
 * left-hand side, without first releasing it. The effect is
 * that if other matrices share memory with left-hand side,
 * they too will be affected. Note that the size of s must be
 * the same as that of the left-hand side matrix. 
 *
 * @note If you rather wanted to create a new copy of \c s,
 * you should use \c copy() instead. */
template<class MT> inline MT&
inject(MT& dest, typename MT::vec_type& vec,
       const MT& s, const typename MT::vec_type& srcvec)
{
   assert(s.size(0) == dest.size(0));
   assert(s.size(1) == dest.size(1));

   if (dest.has_unitstride()
       && dest.gdim(0) == s.gdim(0) && dest.gdim(1) == s.gdim(1)
       && dest.index(0) == s.index(0) && dest.index(1) == s.index(1))
      // Unit strides and exact identical memory layout
      vec.inject(srcvec);
   else
   {
      int i, j,  M=dest.size(0), N=dest.size(1);
      for (j=0;j<N;j++)
	 for (i=0;i<M; i++)
	    dest(i,j) = s(i,j);
   }

   return dest;
}

/** Used in the operator()(LaIndex, LaIndex) as a simplification */
template<class MT> inline void
submatcheck(const MT& mat, const LaIndex& II, const LaIndex& JJ,
	    LaIndex& I, LaIndex& J)
{
   if (II.null())
      I.set(0,mat.size(0)-1);
   else
      I = II;
   if (JJ.null())
      J.set(0,mat.size(1)-1);
   else
      J = JJ;

   //(JJ.null()) ?  J.set(0,size(1)) : J = JJ;

   const LaIndex& cI = I;
   const LaIndex& cJ = J;
   assert(cI.inc() != 0);
   assert(cJ.inc() != 0);

   if (cI.inc() > 0)
   {
      assert(cI.start() >= 0);
      assert(cI.start() <= cI.end());
      assert(cI.end() < mat.size(0));
   }
   else // cI.inc() < 0
   {
      assert(cI.start() < mat.size(0));
      assert(cI.start() >= cI.end());
      assert(cI.end() >= 0);
   }

   if (cJ.inc() > 0)
   {
      assert(cJ.start() >= 0);
      assert(cJ.start() <= cJ.end());
      assert(cJ.end() < mat.size(1));
   }
   else  // cJ.inc() < 0
   {
      assert(cJ.start() < mat.size(1));
      assert(cJ.start() >= cJ.end());
      assert(cJ.end() >= 0);
   }
}

/** Print the matrix to the given output stream. If the matrix
 * info flag is set, then this prints only the matrix info,
 * see LaGenMatDouble::info(). Otherwise all matrix elements
 * are printed. 
 *
 * @see LaPreferences::setPrintFormat() 
 */
template<class MT> inline std::ostream&
print(std::ostream& s, const MT& G, int *info_)
{
   if (*(info_))     // print out only matrix info, not actual values
   {
      *(info_) = 0; // reset the flag
      G.Info(s);
   }
   else 
   {
      int i,j;
      LaPreferences::pFormat p = LaPreferences::getPrintFormat();
      bool newlines = LaPreferences::getPrintNewLines();

      if((p == LaPreferences::MATLAB) || (p == LaPreferences::MAPLE))
	 s << "[";
      for (i=0; i<G.size(0); i++)
      {
	 if(p == LaPreferences::MAPLE)
	    s << "[";
	 for (j=0; j<G.size(1); j++)
	 {
	    s << G(i,j);
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

/** Create an index list */
template<class vT>
void indexList(vT& res, const LaIndex& II)
{
  res.resize(II.length(), 1);
   //std::cout << "indexList: start " << II.start() << " inc " << II.inc() << " end " << II.end() << std::endl;
   int i = 0;
   for (int k = II.start(); k <= II.end(); k += II.inc())
   {
      res(i) = k;
      ++i;
   }
   //std::cout << "indexList: result " << res << std::endl;
}

} // namespace

#endif // MTMPL_H
