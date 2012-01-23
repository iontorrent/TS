// -*-C++-*- 

// Copyright (C) 2004 
// Christian Stimming <stimming@tuhh.de>

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
 * @brief General Dense Rectangular Matrix Class with float elements
 */

//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.
//
//      Lapack++ Rectangular Matrix Class
//
//      Dense (nonsingular) matrix, assumes no special structure or properties.
//
//      ) allows 2-d indexing
//      ) non-unit strides
//      ) deep (copy) assignment
//      ) std::cout << A.info()  prints out internal states of A
//      ) indexing via A(i,j) where i,j are either integers or
//              LaIndex         

#ifndef _LA_GEN_MAT_FLOAT_H_
#define _LA_GEN_MAT_FLOAT_H_

#include "arch.h"
#include "lafnames.h"
#include VECTOR_FLOAT_H
#include LA_INDEX_H

class LaGenMatComplex;
class LaGenMatDouble;
class LaGenMatFloat;
class LaGenMatInt;
class LaGenMatLongInt;


class DLLIMPORT LaGenMatFloat
{
   public:
      /** The type of the value elements. */
      typedef float value_type;
      /** Convenience typedef of this class to itself to make
       * common function definitions easier. (New in
       * lapackpp-2.4.5) */
      typedef LaGenMatFloat matrix_type;
      /** Internal wrapper type; don't use that in an
       * application. */
      typedef VectorFloat vec_type;
   private:
    vec_type     v;
    LaIndex         ii[2];
    int             dim[2];  // size of original matrix, not submatrix
    int             sz[2];   // size of this submatrix
    void init(int m, int n);
    static int  debug_; // trace all entry and exits into methods and 
                        // operators of this class.  This variable is
                        // explicitly initalized in lagenmatfloat.cc

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
    //LaGenMatFloat& shallow_assign();

public:



        /*::::::::::::::::::::::::::*/

        /* Constructors/Destructors */

        /*::::::::::::::::::::::::::*/


        LaGenMatFloat();
        LaGenMatFloat(int, int);

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
        LaGenMatFloat(float* v, int m, int n, bool row_ordering=false);

      /** Create a new matrix from an existing one by copying.
       *
       * Watch out! Due to the C++ "named return value optimization"
       * you cannot use this as an alias for copy() when declaring a
       * variable if the right-side is a return value of
       * operator(). More precisely, you cannot write the following:
       * \verbatim
       LaGenMatFloat x( y(LaIndex(),LaIndex()) ); // erroneous reference copy!
       \endverbatim
       *
       * Instead, if the initialization should create a new copy of
       * the right-side matrix, you have to write it this way:
       * \verbatim
       LaGenMatFloat x( y(LaIndex(),LaIndex()).copy() ); // correct deep-copy
       \endverbatim
       *
       * Or this way:
       * \verbatim
       LaGenMatFloat x;
       x = y(LaIndex(),LaIndex()); // correct deep-copy
       \endverbatim
       */
        LaGenMatFloat(const LaGenMatFloat&);
    virtual ~LaGenMatFloat();


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
      bool equal_to(const matrix_type& mat) const;
      //@}


        /*::::::::::::::::::::::::::::::::*/

        /*  Indices and access operations */

        /*::::::::::::::::::::::::::::::::*/

    inline int size(int d) const;   // submatrix size
      /** Returns the number of columns, i.e. for a M x N matrix
       * this returns N. New in lapackpp-2.4.4. */
      inline int cols() const { return size(1); }
      /** Returns the number of rows, i.e. for a M x N matrix this
       * returns M. New in lapackpp-2.4.4. */
      inline int rows() const { return size(0); }
    inline int inc(int d) const;    // explicit increment
    inline int gdim(int d) const;   // global dimensions
    inline int start(int d) const;  // return ii[d].start()
    inline int end(int d) const;    // return ii[d].end()
    inline LaIndex index(int d) const;// index
    inline int ref_count() const;
    inline LaGenMatFloat& shallow_assign();
    inline float* addr() const;       // begining addr of data space
    
    inline float& operator()(int i, int j);
    inline const float& operator()(int i, int j) const;
    LaGenMatFloat operator()(const LaIndex& I, const LaIndex& J) ;
    LaGenMatFloat operator()(const LaIndex& I, const LaIndex& J) const;
      /** Returns a submatrix view for the specified row \c k of
       * this matrix.
       *
       * The returned object references still the same memory as
       * this object, so if you modify elements, they will appear
       * modified in both objects.  (New in lapackpp-2.4.6) */
      LaGenMatFloat row(int k);
      /** Returns a submatrix view for the specified row \c k of
       * this matrix.
       *
       * The returned object references still the same memory as
       * this object, so if you modify elements, they will appear
       * modified in both objects.  (New in lapackpp-2.4.6) */
      LaGenMatFloat row(int k) const;
      /** Returns a submatrix view for the specified column \c k
       * of this matrix.
       *
       * The returned object references still the same memory as
       * this object, so if you modify elements, they will appear
       * modified in both objects.  (New in lapackpp-2.4.6) */
      LaGenMatFloat col(int k);
      /** Returns a submatrix view for the specified column \c k
       * of this matrix.
       *
       * The returned object references still the same memory as
       * this object, so if you modify elements, they will appear
       * modified in both objects.  (New in lapackpp-2.4.6) */
      LaGenMatFloat col(int k) const;

            LaGenMatFloat& operator=(float s);
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
       LaGenMatFloat x = y(LaIndex(),LaIndex()); // erroneous reference copy!
       \endverbatim
       *
       * Instead, if the initialization should create a new copy of
       * the right-side matrix, you have to write it this way:
       * \verbatim
       LaGenMatFloat x = y(LaIndex(),LaIndex()).copy(); // correct deep-copy
       \endverbatim
       *
       * Or this way:
       * \verbatim
       LaGenMatFloat x;
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
    LaGenMatFloat& operator=(const LaGenMatFloat& s); //copy

    LaGenMatFloat& operator+=(float s);
    LaGenMatFloat& add(float s);

    LaGenMatFloat& resize(int m, int n);
    LaGenMatFloat& resize(const LaGenMatFloat& s);
    LaGenMatFloat& ref(const LaGenMatFloat& s);
    LaGenMatFloat& inject(const LaGenMatFloat& s);
    LaGenMatFloat& copy(const LaGenMatFloat& s);

      /** Returns a newly allocated matrix that is an
       * element-by-element copy of this matrix.
       *
       * New in lapackpp-2.5.2 */
      LaGenMatFloat copy() const;

      /** @name Expensive access functions */
      //@{
      /** Returns a newly allocated large matrix that consists of
       * \c M-by-N copies of the given matrix. (New in
       * lapackpp-2.4.5.) */
      matrix_type repmat (int M, int N) const;
      /** Returns the trace, i.e. the sum of all diagonal elements
       * of the matrix. (New in lapackpp-2.4.5) */
      value_type trace () const;
      /** Returns a newly allocated column vector of dimension \c
       * Nx1 that contains the diagonal of the given matrix. (New
       * in lapackpp-2.4.5) */
      matrix_type diag () const;
      //@}


    inline int shallow() const      // read global shallow flag
        { return shallow_;}
    inline int debug() const;       // read global debug flag
    inline int debug(int d);        // set global debug flag
    inline const LaGenMatFloat& info() const { 
            int *t = info_; 
            *t = 1; 
            return *this;};

    //* I/O *//
    friend DLLIMPORT std::ostream& operator<<(std::ostream&, const LaGenMatFloat&);
    std::ostream& Info(std::ostream& s) const
    {
        s << "Size: (" << size(0) << "x" << size(1) << ") " ;
        s << "Indeces: " << ii[0] << " " << ii[1];
        s << "#ref: " << ref_count() << "addr: " << addr() << std::endl;
        return s;
    };

      /** @name Matrix type conversions */
      //@{
      /** Convert this matrix to a complex matrix with imaginary part zero. */
      LaGenMatComplex to_LaGenMatComplex() const;
      /** Convert this matrix to a double (floating-point double precision) matrix. */
      LaGenMatDouble to_LaGenMatDouble() const;
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
      static matrix_type zeros (int N, int M=0);
      /** Returns a newly allocated all-one matrix of dimension \c
       * NxN, if \c M is not given, or \c NxM if \c M is given.
       * (New in lapackpp-2.4.5) */
      static matrix_type ones (int N, int M=0);
      /** Returns a newly allocated identity matrix of dimension
       * \c NxN, if \c M is not given, or a rectangular matrix \c
       * NxM if \c M is given.  (New in lapackpp-2.4.5) */
      static matrix_type eye (int N, int M=0);
      /** Returns a newly allocated matrix of dimension \c NxM
       * with pseudo-random values. The values are uniformly
       * distributed in the interval \c (0,1) or, if specified, \c
       * (low,high).  (New in lapackpp-2.4.5)
       *
       * Note: Since this uses the system's \c rand() call, the
       * randomness of the values might be questionable -- don't
       * use this if you need really strong random numbers. */
      static matrix_type rand (int N, int M,
			       value_type low=0, value_type high=1);
      /** Returns a newly allocated diagonal matrix of dimension
       * \c NxN that has the vector \c vect of length \c N on the
       * diagonal.  (New in lapackpp-2.4.5) */
      static matrix_type from_diag (const matrix_type &vect);
      /** Returns a newly allocated linarly spaced column vector
       * with \c nr_points elements, between and including \c
       * start and \c end. (New in lapackpp-2.4.5.) */
      static matrix_type linspace (value_type start, value_type end,
				   int nr_points);
      //@}

};  //* End of LaGenMatFloat Class *//


namespace la {
   /** The matrix data type containing (single-precision) \c float
       values. */
   typedef LaGenMatFloat fmat;
} // namespace

/** Print the matrix to the given output stream. If the matrix
 * info flag is set, then this prints only the matrix info,
 * see LaGenMatDouble::info(). Otherwise all matrix elements
 * are printed. 
 *
 * \see LaPreferences::setPrintFormat() 
 */
DLLIMPORT
std::ostream& operator<<(std::ostream&, const LaGenMatFloat&);

        

    //* Member Functions *//


 
inline int LaGenMatFloat::size(int d) const
{
    return sz[d];
}

inline int LaGenMatFloat::inc(int d) const
{
    return ii[d].inc();
}

inline int LaGenMatFloat::gdim(int d) const
{
    return dim[d];
}

inline int LaGenMatFloat::start(int d) const
{
    return ii[d].start();
}

inline int LaGenMatFloat::end(int d) const
{
    return ii[d].end();
}

inline int LaGenMatFloat::ref_count() const
{
    return v.ref_count();
}


inline LaIndex LaGenMatFloat::index(int d)  const
{
    return ii[d];
}

inline float* LaGenMatFloat::addr() const
{
    return  v.addr();
}

inline int LaGenMatFloat::debug() const
{
    return debug_;
}

inline int LaGenMatFloat::debug(int d)
{
    return debug_ = d;
}

inline float& LaGenMatFloat::operator()(int i, int j)
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

inline const float& LaGenMatFloat::operator()(int i, int j) const
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




inline  LaGenMatFloat&  LaGenMatFloat::shallow_assign()
{
    shallow_ = 1;
    return *this;
}





#endif 
// _LA_GEN_MAT_H_
