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
 * @brief Vector of integers
 */

#ifndef _LA_VECTOR_INT_H_
#define _LA_VECTOR_INT_H_

#include "lafnames.h"

#include LA_GEN_MAT_INT_H

/** \brief Vector class for integers
 *
 * A vector is simply an nx1 or 1xn, matrix, only that it can be
 * constructed and accessed by a single dimension.
 *
 */
class LaVectorInt: public LaGenMatInt
{
 public:

  /** @name Declaration */
  //@{
  /** Constructs a column vector of length 0 (null). */
  inline LaVectorInt();

  /** Constructs a column vector of length n */
  inline LaVectorInt(int n);

  /** Constructs a vector of size \f$m\times n\f$. One of the two
   * dimensions must be one! */
  inline LaVectorInt(int m, int n);  

  /** Constructs a column vector of length n by copying the values
   * from a one-dimensional C array of length n. */
  inline LaVectorInt(int* v, int n);

  /** Constructs an \f$m\times n\f$ vector by copying the values
   * from a one-dimensional C array of length mn. One of the two
   * dimensions must be one! */
  inline LaVectorInt(int* v, int m, int n);

  /** Create a new vector from an existing matrix by copying. The
   * given matrix s must be a vector, i.e. one of its dimensions
   * must be one! */
  inline LaVectorInt(const LaGenMatInt&);

  /** Create this integer vector from the index counting of this
   * LaIndex() object. */
  LaVectorInt (const LaIndex& ind);
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
  inline int& operator()(int i);

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
  inline const int& operator()(int i) const ;

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
  inline LaVectorInt operator()(const LaIndex&);
  //@}

  /** @name Assignments */
  //@{
  /** Set elements of left-hand size to the scalar value s. No
   * new vector is created, so that if there are other vectors
   * that reference this memory space, they will also be
   * affected. */
  inline LaVectorInt& operator=(int);

  /** Release left-hand side (reclaiming memory space if
   * possible) and copy elements of elements of \c s. Unline \c
   * inject(), it does not require conformity, and previous
   * references of left-hand side are unaffected. 
   *
   * This is an alias for copy().
   */
  inline LaVectorInt& operator=(const LaGenMatInt&);


  /** Copy elements of s into the memory space referenced by the
   * left-hand side, without first releasing it. The effect is
   * that if other vectors share memory with left-hand side,
   * they too will be affected. Note that the size of s must be
   * the same as that of the left-hand side vector. 
   *
   * @note If you rather wanted to create a new copy of \c s,
   * you should use \c copy() instead. */
  inline LaVectorInt& inject(const LaGenMatInt &);

  /** Release left-hand side (reclaiming memory space if
   * possible) and copy elements of elements of \c s. Unline \c
   * inject(), it does not require conformity, and previous
   * references of left-hand side are unaffected. */
  inline LaVectorInt& copy(const LaGenMatInt &);

  /** Let this vector reference the given vector s, so that the
   * given vector memory s is now referenced by multiple objects
   * (by the given object s and now also by this object). Handle
   * this with care!
   *
   * This function releases any previously referenced memory of
   * this object. */
  inline LaVectorInt& ref(const LaGenMatInt &);
  //@}
    
};

// NOTE: we default to column vectors, since matrices are column
//  oriented.

inline LaVectorInt::LaVectorInt() : LaGenMatInt(0,1) {}
inline LaVectorInt::LaVectorInt(int i) : LaGenMatInt(i,1) {}

// NOTE: one shouldn't be using this method to initalize, but
// it is here so that the constructor can be overloaded with 
// a runtime test.
//
inline LaVectorInt::LaVectorInt(int m, int n) : LaGenMatInt(m,n)
{
  assert(n==1 || m==1);
}

inline LaVectorInt::LaVectorInt(int *d, int n) : 
  LaGenMatInt(d,n,1) {}

inline LaVectorInt::LaVectorInt(int *d, int n, int m) : 
  LaGenMatInt(d,n,m) {}

inline LaVectorInt::LaVectorInt(const LaGenMatInt &G) 
{
  assert(G.size(0)==1 || G.size(1)==1);

  (*this).ref(G);
}


//note that vectors can be either stored columnwise, or row-wise

// this will handle the 0x0 case as well.

inline int LaVectorInt::size() const 
{ return LaGenMatInt::size(0)*LaGenMatInt::size(1); }

inline int& LaVectorInt::operator()(int i)
{ if (LaGenMatInt::size(0)==1 )
  return LaGenMatInt::operator()(0,i);
 else
   return LaGenMatInt::operator()(i,0);
}

inline const int& LaVectorInt::operator()(int i) const
{ if (LaGenMatInt::size(0)==1 )
  return LaGenMatInt::operator()(0,i);
 else
   return LaGenMatInt::operator()(i,0);
}

inline LaVectorInt LaVectorInt::operator()(const LaIndex& I)
{ if (LaGenMatInt::size(0)==1)
  return LaGenMatInt::operator()(LaIndex(0,0),I).shallow_assign(); 
 else
   return LaGenMatInt::operator()(I,LaIndex(0,0)).shallow_assign(); 
}


inline LaVectorInt& LaVectorInt::copy(const LaGenMatInt &A)
{
  assert(A.size(0) == 1 || A.size(1) == 1);   //make sure rhs is a
  // a vector.
  LaGenMatInt::copy(A);
  return *this;
}

inline LaVectorInt& LaVectorInt::operator=(const  LaGenMatInt &A)
{
  return inject(A);
}

inline LaVectorInt& LaVectorInt::ref(const LaGenMatInt &A)
{
  assert(A.size(0) == 1 || A.size(1) == 1);
  LaGenMatInt::ref(A);
  return *this;
}

inline LaVectorInt& LaVectorInt::operator=(int d)
{
  LaGenMatInt::operator=(d);
  return *this;
}

inline LaVectorInt& LaVectorInt::inject(const LaGenMatInt &A)
{
  assert(A.size(0) == 1 || A.size(1) == 1);
  LaGenMatInt::inject(A);
  return *this;
}
    
inline int LaVectorInt::inc() const
{
   if (LaGenMatInt::size(1)==1 )
      return LaGenMatInt::inc(0);
   else
      return LaGenMatInt::inc(1)*LaGenMatInt::gdim(0);
   // NOTE: This was changed on 2005-03-04 because without the dim[0]
   // this gives wrong results on non-unit-stride submatrix views.
}

inline LaIndex LaVectorInt::index() const
{
  if (LaGenMatInt::size(1)==1 )
    return LaGenMatInt::index(0);
  else
    return LaGenMatInt::index(1);
}

inline int LaVectorInt::start() const
{
  if (LaGenMatInt::size(1)==1 )
    return LaGenMatInt::start(0);
  else
    return LaGenMatInt::start(1);
}

inline int LaVectorInt::end() const
{
  if (LaGenMatInt::size(1)==1 )
    return LaGenMatInt::end(0);
  else
    return LaGenMatInt::end(1);
}

#endif 
// _LA_VECTOR_INT_H_
