// -*-C++-*- 

// Copyright (C) 2004 
// Christian Stimming <stimming@tuhh.de>

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2, or
// (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License along
// with this library; see the file COPYING.  If not, write to the Free
// Software Foundation, 59 Temple Place - Suite 330, Boston, MA 02111-1307,
// USA.

/** @file 
 * @brief QR factorization
 */

//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.


#ifndef _LA_GEN_QRFACT_COMPLEX_H
#define _LA_GEN_QRFACT_COMPLEX_H

#include "lafnames.h"
#include LA_VECTOR_COMPLEX_H
#include LA_GEN_MAT_COMPLEX_H

#include "lapack.h"

/** \brief Represent a QR decomposition. 
 *
 * This class calculates the <i>QR factorization</i> of a general
 * <i>m</i>-by-<i>n</i> Matrix \f$A\f$ given by \f[ A = Q
 * \left(\begin{array}{c}R\\ 0\end{array}\right) \f] for \f$m\geq n\f$,
 * where \f$R\f$ is an <i>n</i>-by-<i>n</i> upper triangular matrix
 * and \f$Q\f$ is an <i>m</i>-by-<i>m</i> unitary matrix. If \f$A\f$
 * is of full rank n, then \f$R\f$ is non-singular.
 *
 * See http://www.netlib.org/lapack/lug/node40.html for more
 * details. 
 *
*/
class DLLIMPORT LaGenQRFactComplex
{
      LaGenMatComplex _matA;
      LaVectorComplex _tau;

      mutable LaVectorComplex _work;

      // This assumes that the QR-decomposition is already copied into
      // A.
      void generateQ_internal(LaGenMatComplex &A) const;

   public:

      /** Null constructor. Use decomposeQR_IP() to actually use this
       * object. */
      LaGenQRFactComplex();

      /** Constructor that directly calculates the QR decomposition
       * from the given matrix A, in-place. See decomposeQR_IP() for
       * more about the implications of this. */
      LaGenQRFactComplex(LaGenMatComplex &A);

      /** Copy constructor. */
      LaGenQRFactComplex(LaGenQRFactComplex &QR);

      /** Default destructor. */
      ~LaGenQRFactComplex();

      /** Calculate the QR decomposition of A. 
       *
       * This is in-place, i.e. it destroys the input matrix A and
       * keeps a reference to its memory around. In other words, you
       * cannot do anything with your input matrix A anymore. You can
       * safely delete any references to A because this object will
       * keep its own references still around.
       *
       * Internally this uses the lapack routine \c zgeqrf . */
      void decomposeQR_IP(LaGenMatComplex& A);

      /** Generate the matrix Q explicitly. This is in-place, i.e. it
       * destroys the internal QR decomposition but only calculates
       * the matrix Q.
       *
       * Internally this uses the lapack routine \c zungqr .
       */
      LaGenMatComplex& generateQ_IP();

      /** Generate the matrix Q explicitly. The given matrix A will be
       * overwritten by the matrix Q.
       *
       * Internally this uses the lapack routine \c zungqr .
       */
      void generateQ(LaGenMatComplex &A) const;

      /** Multiply the matrix C by the matrix Q of the QR
       * decomposition that is represented through this object.  This
       * method calculates one out of the following four different
       * calculations:
       *
       *  - \f$C=C\cdot Q\f$
       *  - \f$C=C\cdot Q^H\f$
       *  - \f$C=Q\cdot C\f$ or
       *  - \f$C=Q^H\cdot C\f$
       *
       * This does not modify the internal QR decomposition, so this
       * multiplication can be applied repeatedly.
       *
       * Internally this uses the lapack routine \c zunmqr .
       *
       * \param C The matrix to be multiplied.
       *
       * \param hermitian If true, then \f$Q^H\f$ (hermitian, or
       * conjugate transposed) is used. If false, then \f$Q\f$ is used
       * directly.
       *
       * \param from_left If true, then Q or \f$Q^H\f$ is applied from
       * the left so that \c C=Q*C . If false, then Q is applied from
       * the right so that \c C=C*Q .
       */
      void Mat_Mult(LaGenMatComplex& C, bool hermitian, 
		    bool from_left) const;
};


#endif
