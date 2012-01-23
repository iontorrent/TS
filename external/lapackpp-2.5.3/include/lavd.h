// -*-C++-*- 

// Copyright (C) 2004 
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

//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.

/** @file 
 * @brief Real-valued vector
 */


#ifndef _LA_VECTOR_DOUBLE_H_
#define _LA_VECTOR_DOUBLE_H_

#include "lafnames.h"
#include LA_GEN_MAT_DOUBLE_H


/** \brief Real-valued Vector class
 *
 *  A vector is simply an nx1 or 1xn, matrix, only that it can be
 *  constructed and accessed by a single dimension 
 *
 * Multiplication of this vector should be done by the functions in
 * blas1pp.h and blas2pp.h, e.g. Blas_Dot_Prod() or
 * Blas_Add_Mult(). (There are also some operators in blaspp.h, but we
 * advice against them because they will always allocate a new matrix
 * for the result even though you usually already have a matrix at
 * hand for writing the result into.)  Transpositions of vectors
 * usually do not have to be calculated explicitly, but you can
 * directly use the different multiplication functions that will use
 * this vector as a transposed one, e.g. Blas_R1_Update().
 *
 */
class DLLIMPORT LaVectorDouble: public LaGenMatDouble
{
   public:

      /** @name Declaration */
      //@{
      /** Constructs a column vector of length 0 (null). */
      LaVectorDouble();

      /** Constructs a column vector of length n */
      LaVectorDouble(int n);

      /** Constructs a vector of size \f$m\times n\f$. One of the two
       * dimensions must be one! */
      LaVectorDouble(int m, int n);  

      /** Constructs a column vector of length n by copying the values
       * from a one-dimensional C array of length n. */
      LaVectorDouble(double* v, int n);

      /** Constructs an \f$m\times n\f$ vector by copying the values
       * from a one-dimensional C array of length mn. One of the two
       * dimensions must be one! */
      //LaVectorDouble(double* v, int m, int n);

      /** Create a new vector from an existing matrix by copying. The
       * given matrix s must be a vector, i.e. one of its dimensions
       * must be one! */
      LaVectorDouble(const LaGenMatDouble& s);

      /** Resize to a column vector of length n.
       *
       * This method always resizes to column vector, similar to
       * the one-argument constructor. If you want a row vector,
       * use the two-argument resize(int,int) method.
       *
       * New in lapackpp-2.4.14.
       */
      void resize(int n);

      /** Resize vector to size \f$m\times n\f$. One of the two
       * dimensions must be one!
       *
       * New in lapackpp-2.4.14.
       */
      void resize(int m, int n);

      //@}

      /** @name Information */
      //@{
      /** Returns the length n of this vector. */
      inline int size() const;

      /** Returns the distance between memory locations (in terms of
       * number of elements) between consecutive elements along
       * dimension d. For example, if \c inc(d) returns 1, then
       * elements along the dth dimension are contiguous in
       * memory. */
      inline int inc() const;

      /** If the memory space used by this matrix is viewed as a
       * linear array, \c start(d) returns the starting offset of
       * the first element in dimension \c d. (See \ref LaIndex
       * class.) */
      inline int start() const;

      /** If the memory space used by this matrix is viewed as a
       * linear array, \c end(d) returns the starting offset of the
       * last element in dimension \c d. (See \ref LaIndex
       * class.) */
      inline int end() const;

      /** Returns the index specifying this submatrix view in
       * dimension \c d. (See \ref LaIndex class.) This will only
       * differ from a unit-stride index is the current matrix is
       * actually a submatrix view of some larger matrix. */
      inline LaIndex index() const;
      //@}

      /** @name Access functions */
      //@{
      /** Returns the \f$i\f$th element of this vector, with the
       * index i starting at zero (zero-based offset). This means
       * you have
       *
       * \f[ v = \left(\begin{array}{c} a_1 \\ a_2 \\ \vdots \\ a_N
       * \end{array}\right)
       * \f]
       * 
       * but for accessing the element \f$a_1\f$ you have to
       * write @c v(0).
       *
       * Optional runtime bounds checking (0<=i<=n) is set
       * by the compile time macro LA_BOUNDS_CHECK. */
      inline double& operator()(int i);

      /** Returns the \f$i\f$th element of this vector, with the
       * index i starting at zero (zero-based offset). This means
       * you have
       *
       * \f[ v = \left(\begin{array}{c} a_1 \\ a_2 \\ \vdots \\ a_N
       * \end{array}\right)
       * \f]
       * 
       * but for accessing the element \f$a_1\f$ you have to
       * write @c v(0).
       *
       * Optional runtime bounds checking (0<=i<=n) is set
       * by the compile time macro LA_BOUNDS_CHECK. */
      inline const double& operator()(int i) const ;

      /** Return a submatrix view specified by the index I. (See
       * \ref LaIndex class.) These indices specify start,
       * increment, and ending offsets, similar to triplet notation
       * of Matlab or Fortran 90. For example, if B is a 10 x 10
       * matrix, I is \c (0:2:2) and J is \c (3:1:4), then \c B(I,J)
       * denotes the 2 x 2 matrix
       *
       * \f[  \left(\begin{array}{cc} b_{0,3} & b_{2,3} \\
       * b_{0,4} & b_{4,4}
       * \end{array}\right) \f]
       */
      inline LaVectorDouble operator()(const LaIndex& i);

      /** Return a submatrix view specified by the index I. (See
       * \ref LaIndex class.) These indices specify start,
       * increment, and ending offsets, similar to triplet notation
       * of Matlab or Fortran 90. For example, if B is a 10 x 10
       * matrix, I is \c (0:2:2) and J is \c (3:1:4), then \c B(I,J)
       * denotes the 2 x 2 matrix
       *
       * \f[  \left(\begin{array}{cc} b_{0,3} & b_{2,3} \\
       * b_{0,4} & b_{4,4}
       * \end{array}\right) \f]
       */
      inline LaVectorDouble operator()(const LaIndex& i) const;
      //@}
    
      /** @name Assignments */
      //@{
      /** Set elements of left-hand size to the scalar value s. No
       * new vector is created, so that if there are other vectors
       * that reference this memory space, they will also be
       * affected. */
      LaVectorDouble& operator=(double s);

      /** Release left-hand side (reclaiming memory space if
       * possible) and copy elements of elements of \c s. Unline \c
       * inject(), it does not require conformity, and previous
       * references of left-hand side are unaffected. 
       *
       * This is an alias for copy().
       */
      LaVectorDouble& operator=(const LaGenMatDouble&s);

      /** Copy elements of s into the memory space referenced by the
       * left-hand side, without first releasing it. The effect is
       * that if other vectors share memory with left-hand side,
       * they too will be affected. Note that the size of s must be
       * the same as that of the left-hand side vector. 
       *
       * @note If you rather wanted to create a new copy of \c s,
       * you should use \c copy() instead. */
      LaVectorDouble& inject(const LaGenMatDouble &s);

      /** Release left-hand side (reclaiming memory space if
       * possible) and copy elements of elements of \c s. Unline \c
       * inject(), it does not require conformity, and previous
       * references of left-hand side are unaffected. */
      LaVectorDouble& copy(const LaGenMatDouble &s);

      /** Let this vector reference the given vector s, so that the
       * given vector memory s is now referenced by multiple objects
       * (by the given object s and now also by this object). Handle
       * this with care!
       *
       * This function releases any previously referenced memory of
       * this object. */
      LaVectorDouble& ref(const LaGenMatDouble &);
      //@}
};

// NOTE: we default to column vectors, since matrices are column
//  oriented.

inline LaVectorDouble::LaVectorDouble() : LaGenMatDouble(0,1) {}
inline LaVectorDouble::LaVectorDouble(int i) : LaGenMatDouble(i,1) {}

// NOTE: one shouldn't be using this method to initalize, but
// it is here so that the constructor can be overloaded with 
// a runtime test.
//
inline LaVectorDouble::LaVectorDouble(int m, int n) : LaGenMatDouble(m,n)
{
   assert(n==1 || m==1);
}

inline LaVectorDouble::LaVectorDouble(double *d, int m) : 
   LaGenMatDouble(d,m,1) {}

#if 0
inline LaVectorDouble::LaVectorDouble(double *d, int m, int n) : 
   LaGenMatDouble(d,m,n) {}
#endif

inline LaVectorDouble::LaVectorDouble(const LaGenMatDouble& G) : 
   LaGenMatDouble(G)
{
   assert(G.size(0)==1 || G.size(1)==1);
}

inline void LaVectorDouble::resize(int i)
{
   // Always resizes to column vector, similar to the one-argument
   // constructor. If you want a row vector, use the two-argument
   // resize().
   LaGenMatDouble::resize(i, 1);  // column vector
}

inline void LaVectorDouble::resize(int m, int n)
{
   assert(n==1 || m==1);
   LaGenMatDouble::resize(m, n);
}

        
//note that vectors can be either stored columnwise, or row-wise

// this will handle the 0x0 case as well.

inline int LaVectorDouble::size() const 
{
   return LaGenMatDouble::size(0)*LaGenMatDouble::size(1); 
}

inline double& LaVectorDouble::operator()(int i)
{
   if (LaGenMatDouble::size(0)==1 )
      return LaGenMatDouble::operator()(0,i);
   else
      return LaGenMatDouble::operator()(i,0);
}

inline const double& LaVectorDouble::operator()(int i) const
{
   if (LaGenMatDouble::size(0)==1 )
      return LaGenMatDouble::operator()(0,i);
   else
      return LaGenMatDouble::operator()(i,0);
}

inline LaVectorDouble LaVectorDouble::operator()(const LaIndex& I)
{
   if (LaGenMatDouble::size(0)==1)
      return LaGenMatDouble::operator()(LaIndex(0,0),I).shallow_assign(); 
   else
      return LaGenMatDouble::operator()(I,LaIndex(0,0)).shallow_assign(); 
}

inline LaVectorDouble LaVectorDouble::operator()(const LaIndex& I) const
{
   if (LaGenMatDouble::size(0)==1)
      return LaGenMatDouble::operator()(LaIndex(0,0),I).shallow_assign(); 
   else
      return LaGenMatDouble::operator()(I,LaIndex(0,0)).shallow_assign(); 
}


inline LaVectorDouble& LaVectorDouble::copy(const LaGenMatDouble &A)
{
   assert(A.size(0) == 1 || A.size(1) == 1);   //make sure rhs is a
   // a vector.
   LaGenMatDouble::copy(A);
   return *this;
}

inline LaVectorDouble& LaVectorDouble::operator=(const LaGenMatDouble &A)
{
   return copy(A);
}

inline LaVectorDouble& LaVectorDouble::ref(const LaGenMatDouble &A)
{
   assert(A.size(0) == 1 || A.size(1) == 1);
   LaGenMatDouble::ref(A);
   return *this;
}

inline LaVectorDouble& LaVectorDouble::operator=(double d)
{
   LaGenMatDouble::operator=(d);
   return *this;
}

inline LaVectorDouble& LaVectorDouble::inject(const LaGenMatDouble &A)
{
   assert(A.size(0) == 1 || A.size(1) == 1);
   LaGenMatDouble::inject(A);
   return *this;
}
    
inline int LaVectorDouble::inc() const
{
   if (LaGenMatDouble::size(1)==1 )
      return LaGenMatDouble::inc(0);
   else
      return LaGenMatDouble::inc(1)*LaGenMatDouble::gdim(0);
   // NOTE: This was changed on 2005-03-04 because without the dim[0]
   // this gives wrong results on non-unit-stride submatrix views.
}

inline LaIndex LaVectorDouble::index() const
{
   if (LaGenMatDouble::size(1)==1 )
      return LaGenMatDouble::index(0);
   else
      return LaGenMatDouble::index(1);
}

inline int LaVectorDouble::start() const
{
   if (LaGenMatDouble::size(1)==1 )
      return LaGenMatDouble::start(0);
   else
      return LaGenMatDouble::start(1);
}

inline int LaVectorDouble::end() const
{
   if (LaGenMatDouble::size(1)==1 )
      return LaGenMatDouble::end(0);
   else
      return LaGenMatDouble::end(1);
}

#endif 
// _LA_VECTOR_DOUBLE_H_
