// -*-C++-*- 
//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.


#include "arch.h"
#ifndef _LA_TRIDIAG_MAT_DOUBLE_
#define _LA_TRIDIAG_MAT_DOUBLE_

#include "lafnames.h"
#include LA_VECTOR_DOUBLE_H

/** \brief Tridiagonal square matrix class
 *
 * Unlike general banded matrices, this tridiagonal matrix is
 * stored by diagonals rather than columns. A tridiagonal matrix
 * of order N is stored in three one-dimensional array, one of
 * length N containing the diagonal elements and two of length N-1
 * containing the subdiagonal and superdiagonal elements with
 * element index 0 through N-2.
 *
 * One such matrix with the element indices looks as follows:
 *
 * \f[ A_{n\times n} = 
 * \left(\begin{array}{cccc}
 * a_{00} & a_{01} & \cdots & 0 \\
 * a_{10} & \ddots & & \\
 *  & & \ddots & a_{(n-2)(n-1)} \\
 * 0 & \cdots & a_{(n-1)(n-2)} & a_{(n-1)(n-1)}
 * \end{array}\right)
 * \f]
 * 
 * Multiplication of this matrix should be done by the functions
 * in blas1pp.h, blas2pp.h and blas3pp.h,
 * e.g. Blas_Mat_Mat_Mult(), except that currently there isn't any
 * function available for this class. Please ask on the
 * lapackpp-devel mailing list for support if you need any
 * assistance with this.
 *
 * \see \ref LaTridiagFactDouble, LaTridiagMatFactorize()
 */
class DLLIMPORT LaTridiagMatDouble
{   
      LaVectorDouble du2_;    /* second upper diag, N-2 */
      LaVectorDouble du_;     /* upper diag, N-1 */
      LaVectorDouble d_;      /* main diag, N */
      LaVectorDouble dl_;     /* lower diag, N-1 */
      int size_;

      static double outofbounds_; /* return this address, when addresing out
				     of bounds */
      static int debug_;        // print debug info.
      static int *info_;        // print matrix info only, not values
      //   originally 0, set to 1, and then
      //   reset to 0 after use.

   public:

      /** @name Declaration */
      //@{
      /** Constructs a null 0x0 matrix. */
      LaTridiagMatDouble();
      /** Constructs a tridiagonal matrix of size \f$N\times
       * N\f$. Matrix elements are NOT initialized! */
      LaTridiagMatDouble(int N);
      /** Create a new matrix from an existing one by copying
       * (deep-copy). */
      LaTridiagMatDouble(const LaTridiagMatDouble &);
      /** Create a new matrix from the given three diagonals. The
       * dimensions must match: diag must be of length N,
       * diaglower and diagupper of dimension N-1, otherwise a
       * failed assertion will terminate the program. */
      LaTridiagMatDouble(const LaVectorDouble& diag,
			 const LaVectorDouble& diaglower,
			 const LaVectorDouble& diagupper);

      /** Destroy matrix and reclaim vector memory space if this
       * is the only structure using it. */
      ~LaTridiagMatDouble();
      //@}

      /** @name Information */
      //@{
      /** Returns the size \f$N\times N\f$ of this tridiagonal
       * square matrix. */
      int size() const { return size_;}
      //@}

      /** @name Access functions */
      //@{
      /** Returns the \f$(i,j)\f$th element of this matrix, with the
       * indices i and j starting at zero (zero-based offset). This
       * means you have
       *
       * \f[ A_{n\times n} = 
       * \left(\begin{array}{cccc}
       * a_{11} & a_{12} & \cdots & 0 \\
       * a_{21} & \ddots & & \\
       *  & & \ddots & a_{(n-1)n} \\
       * 0 & \cdots & a_{n(n-1)} & a_{nn}
       * \end{array}\right)
       * \f]
       * 
       * but for accessing the element \f$a_{11}\f$ you have to
       * write @c A(0,0).
       *
       * Optional runtime bounds checking (0<=i<m, 0<=j<n) is set
       * by the compile time macro LA_BOUNDS_CHECK.
       *
       * @note This operator was broken all the way until
       * lapackpp-2.4.11 (regarding the lower diagonal) and is
       * fixed since lapackpp-2.4.12.
       */
      inline double &operator()(int i, int j);

      /** Returns the \f$(i,j)\f$th element of this matrix, with the
       * indices i and j starting at zero (zero-based offset). This
       * means you have
       *
       * \f[ A_{n\times n} = 
       * \left(\begin{array}{cccc}
       * a_{11} & a_{12} & \cdots & 0 \\
       * a_{21} & \ddots & & \\
       *  & & \ddots & a_{(n-1)n} \\
       * 0 & \cdots & a_{n(n-1)} & a_{nn}
       * \end{array}\right)
       * \f]
       * 
       * but for accessing the element \f$a_{11}\f$ you have to
       * write @c A(0,0).
       *
       * Optional runtime bounds checking (0<=i<m, 0<=j<n) is set
       * by the compile time macro LA_BOUNDS_CHECK. 
       *
       * @note This operator was broken all the way until
       * lapackpp-2.4.11 (regarding the lower diagonal) and is
       * fixed since lapackpp-2.4.12.
       */
      inline double operator()(int i, int j) const;

      /** Returns the diagonal diag_selection: 0 main, -1 lower, 1
       * upper, 2 second upper.
       *
       * (Actually, this class additionally stores the second
       * upper diagonal of length N-2, selected by
       * diag_selection==2, but this is only being used in
       * LaTridiagMatFactorize().)
       *
       * @note When assigning values or vectors to the returned
       * diagonals, make sure you only use LaVectorDouble::inject
       * instead of LaVectorDouble::copy or
       * LaVectorDouble::operator=(). Please write this as
       * follows: 
\verbatim
  LaVectorDouble newdiag(N);
  newdiag(0) = ...;
  LaTriagMatDouble triagmat(N);
  triagmat.diag(0).inject(newdiag); // correct
  // but don't write this:
  triagmat.diag(0) = newdiag; // wrong!
\endverbatim
       *
       * Watch out: You can directly manipulate the internal
       * storage with this method! In particular, it would be
       * possible to change the length of the returned vector --
       * don't do this or otherwise the tridiagonal matrix will
       * not be correct anymore.  */
      LaVectorDouble& diag(int diag_selection);

      /** Returns the diagonal diag_selection: 0 main, -1 lower, 1
       * upper, 2 second upper.
       *
       * (Actually, this class additionally stores the second
       * upper diagonal of length N-2, selected by
       * diag_selection==2, but this is only being used in
       * LaTridiagMatFactorize().)
       *
       * @note When assigning values or vectors to the returned
       * diagonals, make sure you only use LaVectorDouble::inject
       * instead of LaVectorDouble::copy or
       * LaVectorDouble::operator=(). Please write this as
       * follows: 
\verbatim
  LaVectorDouble newdiag(N);
  newdiag(0) = ...;
  LaTriagMatDouble triagmat(N);
  triagmat.diag(0).inject(newdiag); // correct
  // but don't write this:
  triagmat.diag(0) = newdiag; // wrong!
\endverbatim
      */
      const LaVectorDouble& diag(int diag_selection) const;
      //@}


      /** @name Assignments */
      //@{
      /** Release left-hand side (reclaiming memory space if
       * possible) and copy elements of elements of \c s. Unline \c
       * inject(), it does not require conformity, and previous
       * references of left-hand side are unaffected. */
      LaTridiagMatDouble& copy(const LaTridiagMatDouble& s); 

      /** Copy elements of s into the memory space referenced by the
       * left-hand side, without first releasing it. The effect is
       * that if other matrices share memory with left-hand side,
       * they too will be affected. Note that the size of s must be
       * the same as that of the left-hand side matrix. 
       *
       * @note If you rather wanted to create a new copy of \c s,
       * you should use \c copy() instead. */
      LaTridiagMatDouble& inject(const LaTridiagMatDouble& s);

      /** Let this matrix reference the given matrix s, so that the
       * given matrix memory s is now referenced by multiple objects
       * (by the given object s and now also by this object). Handle
       * this with care!
       *
       * This function releases any previously referenced memory of
       * this object. */
      inline LaTridiagMatDouble& ref(LaTridiagMatDouble&); 
      //@}

      /** @name Debugging information */
      //@{
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
      const LaTridiagMatDouble& info() const {
	 int *t = info_; *t = 1; return *this;}
      /** Returns global debug flag */
      int debug() const { return debug_;}
      //@}

      /** Print the matrix to the given output stream. If the matrix
       * info flag is set, then this prints only the matrix info,
       * see LaGenMatDouble::info(). Otherwise all matrix elements
       * are printed. 
       *
       * The printing format of this NxN tridiagonal matrix is as
       * follows: First the N-1 elements of the superdiagonal are
       * printed, then the N elements of the diagonal, then the
       * N-1 elements of the subdiagonal.
       */
      friend DLLIMPORT std::ostream& operator<<(std::ostream&,const LaTridiagMatDouble&);


};

DLLIMPORT std::ostream& operator<<(std::ostream& s, const LaTridiagMatDouble& td);


    // operators and member functions

inline double& LaTridiagMatDouble::operator()(int i,int j)
{
   switch (i-j)
   {
      case 0:   // main
#ifdef LA_BOUNDS_CHECK
	 if (i>d_.size()-1)
	    return outofbounds_;
	 else
#endif
	    return d_(i);
      case 1:  // lower
#ifdef LA_BOUNDS_CHECK
	 if (i>dl_.size()-1)
	    return outofbounds_;
	 else
#endif
	    // Before lapackpp-2.4.12 this was dl_(i) but that was WRONG!
	    return dl_(j);
      case -1:   // upper
#ifdef LA_BOUNDS_CHECK
	 if (i>du_.size()-1)
	    return outofbounds_;
	 else
#endif
	    return du_(i);
      default:
	 return outofbounds_;
   }
}


inline double LaTridiagMatDouble::operator()(int i,int j) const
{
   switch (i-j)
   {
      case 0:   // main
#ifdef LA_BOUNDS_CHECK
	 if (i>d_.size()-1)
	    return outofbounds_;
	 else
#endif
	    return d_(i);
      case 1:  // lower
#ifdef LA_BOUNDS_CHECK
	 if (i>dl_.size()-1)
	    return outofbounds_;
	 else
#endif
	    // Before lapackpp-2.4.12 this was dl_(i) but that was WRONG!
	    return dl_(j);
      case -1:   // upper
#ifdef LA_BOUNDS_CHECK
	 if (i>du_.size()-1)
	    return outofbounds_;
	 else
#endif
	    return du_(i);
      default:
	 return outofbounds_;
   }
}

inline LaTridiagMatDouble& LaTridiagMatDouble::ref(LaTridiagMatDouble&T) 
{
    du2_.ref(T.du2_);
    du_.ref(T.du_);
    d_.ref(T.d_);
    dl_.ref(T.dl_); 
    size_ = T.size_;

    return *this;
}







#endif 
// _LA_TRIDIAG_MAT_DOUBLE_
