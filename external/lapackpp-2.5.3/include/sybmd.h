//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.

// Dominik Wagenfuehr <dominik.wagenfuehr@arcor.de>
// Copyright (C) 2006

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
 * @brief Symmetric Positive Definite Band Matrix Class
 */

#ifndef _LA_SYMM_BAND_MAT_DOUBLE_H_
#define _LA_SYMM_BAND_MAT_DOUBLE_H_

#include "arch.h"
#include "lafnames.h"
#include LA_GEN_MAT_DOUBLE_H

/** \brief Symmetric Positive Definite Band Matrix Class
 *
 * This matrix holds a symmetric positive definite n x n banded
 * matrix with bandwidth p, that it, with k subdiagonals and k
 * superdiagonals the bandwidth will be \f$ p=2k+1 \f$.
 *
 * Internally a general matrix with dimension \f$2p+1 \times n\f$
 * will be created for storage.
 *
 * For factorization of this matrix see functions in \ref sybfd.h .
 *
 * \see http://en.wikipedia.org/wiki/Band_matrix explains a
 * general band matrix (not necessarily symmetric).
*/

class DLLIMPORT LaSymmBandMatDouble
{
  LaGenMatDouble data_;  // internal storage.

  int N_;       // N_ is (NxN)
  int kl_;      // kl_ = # subdiags
  static double outofbounds_; // out of range value returned.
  static int debug_;         // print debug info.
  static int *info_;         // print matrix info only, not values
                             //   originally 0, set to 1, and then
                             //   reset to 0 after use.


public:

  /** @name Declaration */
  //@{
  /*::::::::::::::::::::::::::*/
  /* Constructors/Destructors */
  /*::::::::::::::::::::::::::*/

  /**
  * Constructs a null 0x0 matrix.
  */
  LaSymmBandMatDouble();

  /**
  * Constructs a n x n symmetric matrix with bandwidth p, that is,
  * with k subdiagonals and k superdiagonals the bandwidth will be
  * \f$ p=2k+1 \f$.
  */
  LaSymmBandMatDouble(int n, int p);

  /**
  * Create (deep) copy of another matrix.
  */
  LaSymmBandMatDouble(const LaSymmBandMatDouble& A);

  /**
  * Destroy matrix.
  */
  ~LaSymmBandMatDouble();

  /**
  * Resize to a \e new matrix of dimension n x n with bandwidth p.
  */
  LaSymmBandMatDouble& resize(int n, int p);

  /**
  * Resize to a \e new matrix with same dimension and bandwidth as A.
  */
  LaSymmBandMatDouble& resize(const LaSymmBandMatDouble& ob);
  //@}

  /** @name Assignments and Access */
  //@{

  /**
  * Set elements of left-hand side to the scalar value s.
  */
  LaSymmBandMatDouble& operator=(double scalar);

  /**
  * Copy elements of other matrix. This is an alias for copy().
  */
  LaSymmBandMatDouble& operator=(const LaSymmBandMatDouble& ob);

  /**
  * Return element (i,j) of the matrix. Start index is (0,0)
  * (zero-based offset).
  *
  * Optional runtime bounds checking (0 <= i,j < n) is set
  * by the compile time macro LA_BOUNDS_CHECK.
  */
  double& operator()(int i, int j);

  /**
  * Return element (i,j) of the matrix. Start index is (0,0)
  * (zero-based offset).
  *
  * Optional runtime bounds checking (0 <= i,j < n) is set
  * by the compile time macro LA_BOUNDS_CHECK.
  */
  const double& operator()(int i, int j) const;

  /**
  * Let this matrix reference the given matrix \c ob, so that the
  * given matrix memory s is now referenced by multiple objects
  * (by the given object ob and now also by this object).
  */
  inline LaSymmBandMatDouble& ref(LaSymmBandMatDouble &ob);

  /**
  * Release left-hand side and copy elements of elements
  * of \c ob.
  */
  LaSymmBandMatDouble& copy(const LaSymmBandMatDouble &ob);
  //@}

  /** @name Information */
  //@{

  /**
  * Returns the length N of the dth dimension. Because
  * the matrix is symmetric it is
  * \e size(0) == \e size(1) == \e N.
  *
  * <b>Important:</b> The size does not return the
  * size of the internal stored matrix as it
  * was in Lapack versions <= 2.4.13.
  */
  inline int size(int d) const;         // submatrix size

  /**
  * Returns the distance between memory locations (in terms of
  * number of elements) between consecutive elements along
  * dimension d. For example, if \c inc(d) returns 1, then
  * elements along the dth dimension are contiguous in
  * memory.
  */
  inline int inc(int d) const;          // explicit increment

  /**
  * Returns the global dimensions of the (possibly larger)
  * matrix owning this space. This will only differ from \c
  * size(d) if the current matrix is actually a submatrix view
  * of some larger matrix.
  */
  inline int gdim(int d) const;         // global dimensions

  /**
  * Returns the memory address of the first element of the
  * matrix. \c G.addr() is equivalent to \c &G(0,0) .
  */
  inline double* addr() const 
  {     
        return data_.addr();
  }
  
  /**
  * Returns the number of data objects which utilize the same
  * (or portions of the same) memory space used by this
  * matrix.
  */
  inline int ref_count() const  
  { 
        return data_.ref_count();
  }
  
  /**
  * Returns the index specifying this submatrix view in
  * dimension \c d. (See \ref LaIndex class.) This will only
  * differ from a unit-stride index if the current matrix is
  * actually a submatrix view of some larger matrix.
  */
  inline LaIndex index(int d) const
  { 
        return data_.index(d);
  }
  
  /** Returns bandwidth of matrix.
   *
   * Watch out: Contrary to the name of this method, it does \e
   * not return the number of subdiagonals. Instead the bandwidth
   * p is returned, that is, with k subdiagonals and k
   * superdiagonals the bandwidth will be \f$ p=2k+1 \f$.
   */
  inline int subdiags()
  { 
        return (kl_);
  }

  /** Returns bandwidth of matrix. 
   *
   * Watch out: Contrary to the name of this method, it does \e
   * not return the number of superdiagonals. Instead the
   * bandwidth p is returned, that is, with k subdiagonals and k
   * superdiagonals the bandwidth will be \f$ p=2k+1 \f$. 
   */
  inline int subdiags() const
  {
        return (kl_);
  }
  //@}

  /** @name Debugging information */
  //@{

  /** Returns global shallow flag */
  inline int shallow() const
  {
        return data_.shallow();
  }
  
  /** Returns global debug flag */
  inline int debug() const
  {
        return debug_;
  }
  
  /** Set global debug flag */
  inline int debug(int d)
  {
        return debug_ = d;
  }

  inline const LaSymmBandMatDouble& info() const
  {
        int *t = info_;
        *t = 1;
        return *this;
  };

  /**
  * Print the matrix info (not the actual elements)
  * to the standard output. */
  inline void print_data() const
  {
        std::cout << data_;
  }
  //@}

  /** Print the matrix to the given output stream. If the matrix
  * info flag is set, then this prints only the matrix info,
  * see LaGenMatDouble::info(). Otherwise all matrix elements
  * are printed.
  *
  * @see LaPreferences::setPrintFormat()
  */
  friend std::ostream& operator<<(std::ostream &s, const LaSymmBandMatDouble &ob);

};

  // member functions and operators

inline LaSymmBandMatDouble& LaSymmBandMatDouble::ref(LaSymmBandMatDouble &ob)
{
  data_.ref(ob.data_);
  N_ = ob.N_;
  kl_ = ob.kl_;

  return *this;
}

inline int LaSymmBandMatDouble::size(int d __attribute__((unused)) ) const
{
   return(data_.size(1));
}

inline int LaSymmBandMatDouble::inc(int d) const
{
   return(data_.inc(d));
}

inline int LaSymmBandMatDouble::gdim(int d) const
{
   return(data_.gdim(d));
}

inline double& LaSymmBandMatDouble::operator()(int i, int j)
{
#ifdef LA_BOUNDS_CHECK
   assert(i >= 0);
   assert(i < N_);
   assert(j >= 0);
   assert(j < N_);
#endif

   if (i >= j)
   {
      if (i-j <= kl_)
	 return data_(kl_ + i - j, j);
      else
      {
#ifdef LA_BOUNDS_CHECK
	 assert(0);
#else
	 return outofbounds_;
#endif
      }
   }
   else // if (j>i)
   {
      if (j-i <= kl_)
	 return data_(kl_ + j - i,i);
      else
      {
#ifdef LA_BOUNDS_CHECK
	 assert(0);
#else
	 return outofbounds_;
#endif
      }
   }
}

inline const double& LaSymmBandMatDouble::operator()(int i, int j) const
{
#ifdef LA_BOUNDS_CHECK
   assert(i >= 0);
   assert(i < N_);
   assert(j >= 0);
   assert(j < N_);
#endif

   if (i >= j)
   {
      if (i-j <= kl_)
	 return data_(kl_ + i - j, j);
      else
      {
#ifdef LA_BOUNDS_CHECK
	 assert(0);
#else
	 return outofbounds_;
#endif
      }
   }
   else // if (j>i)
   {
      if (j-i <= kl_)
	 return data_(kl_ + j - i,i);
      else
      {
#ifdef LA_BOUNDS_CHECK
	 assert(0);
#else
	 return outofbounds_;
#endif
      }
   }
}

#endif 
// _LA_SYMM_BAND_MAT_DOUBLE_H_

