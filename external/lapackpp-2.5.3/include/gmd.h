// -*-C++-*- 

// Copyright (C) 2004 
// Christian Stimming <stimming@tuhh.de>
//
// Row-order modifications by Jacob (Jack) Gryn <jgryn at cs dot yorku dot ca>

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

/** @file 
 * @brief General Dense Rectangular Matrix Class
 */

//      LAPACK++ (V. 1.0a Beta)
//      (C) 1992-1996 All Rights Reserved.
//
//      Lapack++ Rectangular Matrix Class
//
//      Dense (nonsingular) matrix, assumes no special structure or properties.
//
//      ) allows 2-d indexing
//      ) non-unit strides
//      ) inject assignment
//      ) std::cout << A.info()  prints out internal states of A
//      ) indexing via A(i,j) where i,j are either integers or
//              LaIndex         

#ifndef _LA_GEN_MAT_DOUBLE_H_
#define _LA_GEN_MAT_DOUBLE_H_

#include "arch.h"
#include "lafnames.h"
#include VECTOR_DOUBLE_H
#include LA_INDEX_H
#include LA_GEN_MAT_FLOAT_H

class LaGenMatComplex;
class LaGenMatDouble;
class LaGenMatFloat;
class LaGenMatInt;
class LaGenMatLongInt;

/** \brief General Dense Rectangular Matrix Class
 *
 * This is the basic LAPACK++ matrix. It is a dense (nonsingular)
 * matrix, assumes no special structure or properties.
 *
 *  - allows 2-d indexing
 *  - non-unit strides
 *  - inject assignment
 *  - std::cout << A.info()  prints out internal states of A
 *  - indexing via A(i,j) where i,j are either integers or LaIndex         
 *
 * Multiplication of this matrix should be done by the functions in
 * blas1pp.h, blas2pp.h and blas3pp.h,
 * e.g. Blas_Mat_Mat_Mult(). (There are also some operators in
 * blaspp.h, but we advice against them because they will always
 * allocate a new matrix for the result even though you usually
 * already have a matrix at hand for writing the result into.)
 * Transpositions of matrices usually do not have to be calculated
 * explicitly, but you can directly use the different multiplication
 * functions that will use a matrix as a transposed one,
 * e.g. Blas_Mat_Trans_Mat_Mult().
 */
class DLLIMPORT LaGenMatDouble
{
   public:
      /** The type of the value elements. */
      typedef double value_type;
      /** Convenience typedef of this class to itself to make
       * common function definitions easier. (New in
       * lapackpp-2.4.5) */
      typedef LaGenMatDouble matrix_type;
      /** Internal wrapper type; don't use that in an
       * application. */
      typedef VectorDouble vec_type;
   private:
      vec_type     v;
      LaIndex           ii[2];
      int             dim[2];  // size of original matrix, not submatrix
      int             sz[2];   // size of this submatrix
      void init(int m, int n);
      static int  debug_; // trace all entry and exits into methods and 
      // operators of this class.  This variable is
      // explicitly initalized in gmd.cc

      static int      *info_;   // print matrix info only, not values
      //   originally 0, set to 1, and then
      //   reset to 0 after use.
      // use as in
      //
      //    std::cout << B.info() << std::endl;
      //
      // this *info_ member is unique in that it really isn't
      // part of the matrix info, just a flag as to how
      // to print it.   We've included in this beta release
      // as part of our testing, but we do not expect it 
      // to be user accessable.
      // It has to be declared as global static
      // so that we may monitor expresssions like
      // X::(const &X) and still utilize without violating
      // the "const" condition.
      // Because this *info_ is used at most one at a time,
      // there is no harm in keeping only one copy of it,
      // also, we do not need to malloc free space every time
      // we call a matrix constructor.


      int shallow_; // set flag to '0' in order to return matrices
                    // by value from functions without unecessary
                    // copying.


      // users shouldn't be able to modify assignment semantics..
      //
      //LaGenMatDouble& shallow_assign();

   public:

      /** @name Declaration */
      //@{
      /*::::::::::::::::::::::::::*/
      /* Constructors/Destructors */
      /*::::::::::::::::::::::::::*/

      /** Constructs a null 0x0 matrix. */
      LaGenMatDouble();

      /** Constructs a column-major matrix of size \f$m\times
       * n\f$. Matrix elements are NOT initialized! */
      LaGenMatDouble(int m, int n);

      /** Constructs an \f$m\times n\f$ matrix by using the values
       * from the one-dimensional C array \c v of length \c m*n. 
       *
       * \note If \c row_ordering is \c false, then the data will \e
       * not be copied but instead the C array will be shared
       * (shallow copy). In that case, you must not delete the C
       * array as long as you use this newly created matrix. Also,
       * if you need a copy (deep copy), construct one matrix \c A
       * by this constructor, and then copy this content into a
       * second matrix by \c B.copy(A). On the other hand, if \c
       * row_ordering is \c true, then the data will be copied
       * immediately (deep copy).
       *
       * \param v The one-dimensional C array of size \c m*n whose
       * data should be used. If \c row_ordering is \c false, then
       * the data will \e not be copied but shared (shallow
       * copy). If \c row_ordering is \c true, then the data will be
       * copied (deep copy).
       *
       * \param m The number of rows in the new matrix.
       *
       * \param n The number of columns in the new matrix.
       *
       * \param row_ordering If \c false, then the C array is used
       * in column-order, i.e. the first \c m elements of \c v are
       * used as the first column of the matrix, the next \c m
       * elements are the second column and so on. (This is the
       * default and this is also the internal storage format in
       * order to be compatible with the underlying Fortran
       * subroutines.) If this is \c true, then the C array is used
       * in row-order, i.e. the first \c n elements of \c v are used
       * as the first row of the matrix, the next \c n elements are
       * the second row and so on. (Internally, this is achieved by
       * allocating a new copy of the array and copying the array
       * into the internal ordering.)
       */
      LaGenMatDouble(double*v, int m, int n, bool row_ordering=false);

      /** Create a new matrix from an existing one by copying.
       *
       * Watch out! Due to the C++ "named return value optimization"
       * you cannot use this as an alias for copy() when declaring a
       * variable if the right-side is a return value of
       * operator(). More precisely, you cannot write the following:
       * \verbatim
       LaGenMatDouble x( y(LaIndex(),LaIndex()) ); // erroneous reference copy!
       \endverbatim
       *
       * Instead, if the initialization should create a new copy of
       * the right-side matrix, you have to write it this way:
       * \verbatim
       LaGenMatDouble x( y(LaIndex(),LaIndex()).copy() ); // correct deep-copy
       \endverbatim
       *
       * Or this way:
       * \verbatim
       LaGenMatDouble x;
       x = y(LaIndex(),LaIndex()); // correct deep-copy
       \endverbatim
       */
      LaGenMatDouble(const LaGenMatDouble&);

      /** Create a new matrix from an existing one by copying and
       * converting each element from a float to a double. */
      explicit LaGenMatDouble(const LaGenMatFloat&);

      /** Resize to a \e new matrix of size m x n. The element
       * values of the new matrix are \e uninitialized, even if
       * resizing to a smaller matrix. */
      LaGenMatDouble& resize(int m, int n);

      /** Resize to a \e new matrix of the same size as the given
       * matrix s. The element values of the new matrix are \e
       * uninitialized, even if resizing to a smaller matrix. */
      LaGenMatDouble& resize(const LaGenMatDouble& s);

      /** Destroy matrix and reclaim vector memory space if this is
       * the only structure using it. */
      virtual ~LaGenMatDouble();
      //@}


      /** @name Information Predicates */
      //@{
      /** Returns true if this is an all-zero matrix. (New in
       * lapackpp-2.4.5) */
      bool is_zero() const;

      /** Returns true if this matrix is only a submatrix view of
       * another (larger) matrix. (New in lapackpp-2.4.4) */
      bool is_submatrixview() const
      { return size(0) != gdim(0) || size(1) != gdim(1); };

      /** Returns true if this matrix has unit stride. 
       *
       * This is a necessary condition for not being a submatrix
       * view, but it's not sufficient. (New in lapackpp-2.4.4) */
      bool has_unitstride() const
      { return inc(0) == 1 && inc(1) == 1; };

      /** Returns true if the given matrix \c mat is exactly equal
       * to this object. (New in lapackpp-2.4.5) */
      bool equal_to(const LaGenMatDouble& mat) const;
      //@}


      /** @name Information */
      //@{
      /*::::::::::::::::::::::::::::::::*/
      /*  Indices and access operations */
      /*::::::::::::::::::::::::::::::::*/

      /** Returns the length n of the dth dimension, i.e. for a M x
       * N matrix, \c size(0) returns M and \c size(1) returns N. */
      inline int size(int d) const;   // submatrix size
      /** Returns the number of columns, i.e. for a M x N matrix
       * this returns N. New in lapackpp-2.4.4. */
      inline int cols() const { return size(1); }
      /** Returns the number of rows, i.e. for a M x N matrix this
       * returns M. New in lapackpp-2.4.4. */
      inline int rows() const { return size(0); }

      /** Returns the distance between memory locations (in terms of
       * number of elements) between consecutive elements along
       * dimension d. For example, if \c inc(d) returns 1, then
       * elements along the dth dimension are contiguous in
       * memory. */
      inline int inc(int d) const;    // explicit increment

      /** Returns the global dimensions of the (possibly larger)
       * matrix owning this space. This will only differ from \c
       * size(d) if the current matrix is actually a submatrix view
       * of some larger matrix. */
      inline int gdim(int d) const;   // global dimensions

      /** If the memory space used by this matrix is viewed as a
       * linear array, \c start(d) returns the starting offset of
       * the first element in dimension \c d. (See \ref LaIndex
       * class.) */
      inline int start(int d) const;  // return ii[d].start()

      /** If the memory space used by this matrix is viewed as a
       * linear array, \c end(d) returns the starting offset of the
       * last element in dimension \c d. (See \ref LaIndex
       * class.) */
      inline int end(int d) const;    // return ii[d].end()

      /** Returns the index specifying this submatrix view in
       * dimension \c d. (See \ref LaIndex class.) This will only
       * differ from a unit-stride index if the current matrix is
       * actually a submatrix view of some larger matrix. */
      inline LaIndex index(int d) const;// index

      /** Returns the number of data objects which utilize the same
       * (or portions of the same) memory space used by this
       * matrix. */
      inline int ref_count() const;

      /** Returns the memory address of the first element of the
       * matrix. \c G.addr() is equivalent to \c &G(0,0) . */
      inline double* addr() const;       // begining addr of data space
      //@}

      /** @name Access functions */
      //@{
      /** Returns the \f$(i,j)\f$th element of this matrix, with the
       * indices i and j starting at zero (zero-based offset). This
       * means you have
       *
       * \f[ A_{n\times m} = \left(\begin{array}{ccc} a_{11} & &
       * a_{1m} \\ & \ddots & \\ a_{n1} & & a_{nm}
       * \end{array}\right)
       * \f]
       * 
       * but for accessing the element \f$a_{11}\f$ you have to
       * write @c A(0,0).
       *
       * Optional runtime bounds checking (0<=i<m, 0<=j<n) is set
       * by the compile time macro LA_BOUNDS_CHECK. */
      inline double& operator()(int i, int j);

      /** Returns the \f$(i,j)\f$th element of this matrix, with the
       * indices i and j starting at zero (zero-based offset). This
       * means you have
       *
       * \f[ A_{n\times m} = \left(\begin{array}{ccc} a_{11} & &
       * a_{1m} \\ & \ddots & \\ a_{n1} & & a_{nm}
       * \end{array}\right) \f]
       * 
       * but for accessing the element \f$a_{11}\f$ you have to
       * write @c A(0,0).
       *
       * Optional runtime bounds checking (0<=i<m, 0<=j<n) is set
       * by the compile time macro LA_BOUNDS_CHECK. */
      inline const double& operator()(int i, int j) const;

      /** Return a submatrix view specified by the indices I and
       * J. (See \ref LaIndex class.) These indices specify start,
       * increment, and ending offsets, similar to triplet notation
       * of Matlab or Fortran 90. For example, if B is a 10 x 10
       * matrix, I is \c (0:2:2) and J is \c (3:1:4), then \c B(I,J)
       * denotes the 2 x 2 matrix
       *
       * \f[  \left(\begin{array}{cc} b_{0,3} & b_{2,3} \\
       * b_{0,4} & b_{4,4}
       * \end{array}\right) \f]
       */
      LaGenMatDouble operator()(const LaIndex& I, const LaIndex& J) ;

      /** Return a submatrix view specified by the indices I and
       * J. (See \ref LaIndex class.) These indices specify start,
       * increment, and ending offsets, similar to triplet notation
       * of Matlab or Fortran 90. For example, if B is a 10 x 10
       * matrix, I is \c (0:2:2) and J is \c (3:1:4), then \c B(I,J)
       * denotes the 2 x 2 matrix
       *
       * \f[  \left(\begin{array}{cc} b_{0,3} & b_{2,3} \\
       * b_{0,4} & b_{4,4}
       * \end{array}\right) \f]
       */
      LaGenMatDouble operator()(const LaIndex& I, const LaIndex& J) const;

      /** Returns a submatrix view for the specified row \c k of
       * this matrix. 
       *
       * The returned object references still the same memory as
       * this object, so if you modify elements, they will appear
       * modified in both objects.  (New in lapackpp-2.4.6) */
      LaGenMatDouble row(int k);
      /** Returns a submatrix view for the specified row \c k of
       * this matrix. 
       *
       * The returned object references still the same memory as
       * this object, so if you modify elements, they will appear
       * modified in both objects. (New in lapackpp-2.4.6) */
      LaGenMatDouble row(int k) const;
      /** Returns a submatrix view for the specified column \c k
       * of this matrix.
       *
       * The returned object references still the same memory as
       * this object, so if you modify elements, they will appear
       * modified in both objects. (New in lapackpp-2.4.6) */
      LaGenMatDouble col(int k);
      /** Returns a submatrix view for the specified column \c k
       * of this matrix.
       *
       * The returned object references still the same memory as
       * this object, so if you modify elements, they will appear
       * modified in both objects. (New in lapackpp-2.4.6) */
      LaGenMatDouble col(int k) const;
      //@}

      /** @name Assignments */
      //@{
      /** Set elements of left-hand side to the scalar value s. No
       * new matrix is created, so that if there are other matrices
       * that reference this memory space, they will also be
       * affected. */
      LaGenMatDouble& operator=(double s);

      /** Release left-hand side (reclaiming memory space if
       * possible) and copy elements of elements of \c s. Unline \c
       * inject(), it does not require conformity, and previous
       * references of left-hand side are unaffected. 
       *
       * This is an alias for copy().
       *
       * Watch out! Due to the C++ "named return value optimization"
       * you cannot use this as an alias for copy() when declaring a
       * variable if the right-side is a return value of
       * operator(). More precisely, you cannot write the following:
       * \verbatim
       LaGenMatDouble x = y(LaIndex(),LaIndex()); // erroneous reference copy!
       \endverbatim
       *
       * Instead, if the initialization should create a new copy of
       * the right-side matrix, you have to write it this way:
       * \verbatim
       LaGenMatDouble x = y(LaIndex(),LaIndex()).copy(); // correct deep-copy
       \endverbatim
       *
       * Or this way:
       * \verbatim
       LaGenMatDouble x;
       x = y(LaIndex(),LaIndex()); // correct deep-copy
       \endverbatim
       *
       * Note: The manual for lapack++-1.1 claimed that this
       * operator would be an alias for ref(), not for copy(),
       * i.e. this operator creates a reference instead of a deep
       * copy. However, since that confused many people, the
       * behaviour was changed so that B=A will now create B as a
       * deep copy instead of a reference. If you want a
       * reference, please write B.ref(A) explicitly.
       */
      LaGenMatDouble& operator=(const LaGenMatDouble& s);

      /** Add the scalar value s to elements of left-hand side. No
       * new matrix is created, so that if there are other matrices
       * that reference this memory space, they will also be
       * affected. 
       *
       * @note This method is rather slow. In many cases, it can
       * be much faster to use Blas_Mat_Mult() with a Ones-Matrix
       * instead. */
      LaGenMatDouble& operator+=(double s);

      /** Add the scalar value s to elements of left-hand side. No
       * new matrix is created, so that if there are other matrices
       * that reference this memory space, they will also be
       * affected. (New in lapackpp-2.4.7.) */
      LaGenMatDouble& add(double s);

      /** Scale the left-hand side matrix by the given scalar
       * value. No new matrix is created, so that if there are
       * other matrices that reference this memory space, they
       * will also be affected. (New in lapackpp-2.4.7.) */
      LaGenMatDouble& operator*=(double s);

      /** Scale the left-hand side matrix by the given scalar
       * value. No new matrix is created, so that if there are
       * other matrices that reference this memory space, they
       * will also be affected. (New in lapackpp-2.4.7.) */
      LaGenMatDouble& scale(double s);

      /** Copy elements of s into the memory space referenced by the
       * left-hand side, without first releasing it. The effect is
       * that if other matrices share memory with left-hand side,
       * they too will be affected. Note that the size of s must be
       * the same as that of the left-hand side matrix. 
       *
       * @note If you rather wanted to create a new copy of \c s,
       * you should use \c copy() instead. */
      LaGenMatDouble& inject(const LaGenMatDouble& s);

      /** Release left-hand side (reclaiming memory space if
       * possible) and copy elements of elements of \c s. Unline \c
       * inject(), it does not require conformity, and previous
       * references of left-hand side are unaffected. */
      LaGenMatDouble& copy(const LaGenMatDouble& s);

      /** Returns a newly allocated matrix that is an
       * element-by-element copy of this matrix.
       *
       * New in lapackpp-2.5.2 */
      LaGenMatDouble copy() const;

      /** This is an optimization for returning temporary matrices
       * from functions, without copying. The shallow_assign()
       * function essentially sets an internal flag which instructs
       * the \c X::X(&X) copy constructor to avoid the copying. */
      inline LaGenMatDouble& shallow_assign();

      /** Let this matrix reference the given matrix s, so that the
       * given matrix memory s is now referenced by multiple objects
       * (by the given object s and now also by this object). Handle
       * this with care!
       *
       * This function releases any previously referenced memory of
       * this object. */
      LaGenMatDouble& ref(const LaGenMatDouble& s);
      //@}

      /** @name Expensive access functions */
      //@{
      /** Returns a newly allocated large matrix that consists of
       * \c M-by-N copies of the given matrix. (New in
       * lapackpp-2.4.5.) */
      LaGenMatDouble repmat (int M, int N) const;
      /** Returns the trace, i.e. the sum of all diagonal elements
       * of the matrix. (New in lapackpp-2.4.5) */
      value_type trace () const;
      /** Returns a newly allocated column vector of dimension \c
       * Nx1 that contains the diagonal of the given matrix. (New
       * in lapackpp-2.4.5) */
      LaGenMatDouble diag () const;
      //@}

      /** @name Debugging information */
      //@{
      /** Returns global shallow flag */
      inline int shallow() const      // read global shallow flag
      { return shallow_;}
      /** Returns global debug flag */
      inline int debug() const;       // read global debug flag
      /** Set global debug flag */
      inline int debug(int d);        // set global debug flag
      /**
       // use as in
       //
       //    std::cout << B.info() << std::endl;
       //
       // this *info_ member is unique in that it really isn't
       // part of the matrix info, just a flag as to how
       // to print it.   We've included in this beta release
       // as part of our testing, but we do not expect it 
       // to be user accessable.
       */
      inline const LaGenMatDouble& info() const { 
	 *(const_cast<LaGenMatDouble*>(this)->info_) = 1; 
	 return *this;
      };
	
      /** Print the matrix info (not the actual elements) to the
       * given ostream. */
      inline std::ostream& Info(std::ostream& s) const
      {
	 s << "Size: (" << size(0) << "x" << size(1) << ") " ;
	 s << "Indeces: " << ii[0] << " " << ii[1];
	 s << "#ref: " << ref_count() 
	   << "addr: " << addr() << " shallow:" << shallow_ << std::endl;
	 return s;
      };
      //@}
      /** Print the matrix to the given output stream. If the matrix
       * info flag is set, then this prints only the matrix info,
       * see LaGenMatDouble::info(). Otherwise all matrix elements
       * are printed. 
       *
       * @see LaPreferences::setPrintFormat() 
       */
      friend DLLIMPORT std::ostream& operator<<(std::ostream&, const LaGenMatDouble&);

      /** @name Matrix type conversions */
      //@{
      /** Convert this matrix to a complex matrix with imaginary part zero. */
      LaGenMatComplex to_LaGenMatComplex() const;
      /** Convert this matrix to a float (floating-point single precision) matrix. */
      LaGenMatFloat to_LaGenMatFloat() const;
      /** Convert this matrix to an int matrix. */
      LaGenMatInt to_LaGenMatInt() const;
      /** Convert this matrix to a long int matrix. */
      LaGenMatLongInt to_LaGenMatLongInt() const;
      //@}


      /** @name Constructors for elementary matrices */
      //@{
      /** Returns a newly allocated all-zero matrix of dimension
       * \c NxN, if \c M is not given, or \c NxM if \c M is given.
       * (New in lapackpp-2.4.5) */
      static LaGenMatDouble zeros (int N, int M=0);
      /** Returns a newly allocated all-one matrix of dimension \c
       * NxN, if \c M is not given, or \c NxM if \c M is given.
       * (New in lapackpp-2.4.5) */
      static LaGenMatDouble ones (int N, int M=0);
      /** Returns a newly allocated identity matrix of dimension
       * \c NxN, if \c M is not given, or a rectangular matrix \c
       * NxM if \c M is given.  (New in lapackpp-2.4.5) */
      static LaGenMatDouble eye (int N, int M=0);
      /** Returns a newly allocated matrix of dimension \c NxM
       * with pseudo-random values. The values are uniformly
       * distributed in the interval \c (0,1) or, if specified, \c
       * (low,high).  (New in lapackpp-2.4.5)
       *
       * Note: Since this uses the system's \c rand() call, the
       * randomness of the values might be questionable -- don't
       * use this if you need really strong random numbers. */
      static LaGenMatDouble rand (int N, int M,
			       value_type low=0, value_type high=1);
      /** Returns a newly allocated diagonal matrix of dimension
       * \c NxN that has the vector \c vect of length \c N on the
       * diagonal.  (New in lapackpp-2.4.5) */
      static LaGenMatDouble from_diag (const LaGenMatDouble &vect);
      /** Returns a newly allocated linarly spaced column vector
       * with \c nr_points elements, between and including \c
       * start and \c end. (New in lapackpp-2.4.5.) */
      static LaGenMatDouble linspace (value_type start, value_type end,
				   int nr_points);
      //@}

};  //* End of LaGenMatDouble Class *//

namespace la {
   /** The basic matrix data type containing double values. */
   typedef LaGenMatDouble mat;
} // namespace

/** Print the matrix to the given output stream. If the matrix
 * info flag is set, then this prints only the matrix info,
 * see LaGenMatDouble::info(). Otherwise all matrix elements
 * are printed. 
 *
 * @see LaPreferences::setPrintFormat() 
 */
DLLIMPORT
std::ostream& operator<<(std::ostream&, const LaGenMatDouble&);
        

//* Member Functions *//
 
inline int LaGenMatDouble::size(int d) const
{
   return sz[d];
}

inline int LaGenMatDouble::inc(int d) const
{
   return ii[d].inc();
}

inline int LaGenMatDouble::gdim(int d) const
{
   return dim[d];
}

inline int LaGenMatDouble::start(int d) const
{
   return ii[d].start();
}

inline int LaGenMatDouble::end(int d) const
{
   return ii[d].end();
}

inline int LaGenMatDouble::ref_count() const
{
   return v.ref_count();
}


inline LaIndex LaGenMatDouble::index(int d)  const
{
   return ii[d];
}

inline double* LaGenMatDouble::addr() const
{
   return  v.addr();
}

inline int LaGenMatDouble::debug() const
{
   return debug_;
}

inline int LaGenMatDouble::debug(int d)
{
   return debug_ = d;
}

inline double& LaGenMatDouble::operator()(int i, int j)
{
#ifdef LA_BOUNDS_CHECK
   assert(i>=0);
   assert(i<size(0));
   assert(j>=0);
   assert(j<size(1));
#endif
   return v( dim[0]*(ii[1].start() + j*ii[1].inc()) + 
	     ii[0].start() + i*ii[0].inc());
}

inline const double& LaGenMatDouble::operator()(int i, int j) const
{

#ifdef LA_BOUNDS_CHECK
   assert(i>=0);
   assert(i<size(0));
   assert(j>=0);
   assert(j<size(1));
#endif

   return v( dim[0]*(ii[1].start() + j*ii[1].inc()) + 
	     ii[0].start() + i*ii[0].inc());
}

inline  LaGenMatDouble&  LaGenMatDouble::shallow_assign()
{
   shallow_ = 1;
   return *this;
}


#endif 
// _LA_GEN_MAT_H_
