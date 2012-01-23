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

/** @file
 * @brief Some additional vector-matrix operators
 *
 * DEPRECATED. This file defines some additional operators for vectors
 * and matrices. But in general the Blas functions from blas1pp.h,
 * blas2pp.h and blas3pp.h are much faster and should be preferred.
 */

#ifndef _BLAS_PP_H_
#define _BLAS_PP_H_

// requires
//

#include "lafnames.h"
#include LA_EXCEPTION_H
#include "blas1pp.h"
#include "blas2pp.h"
#include "blas3pp.h"
#include <cmath>
#include LA_VECTOR_DOUBLE_H


// Only enable this when LA_NO_DEPRECATED is not defined
#ifndef LA_NO_DEPRECATED

//-------------------------------------
// Vector/Vector operators
//-------------------------------------

/** @name Vector operators (deprecated) */
//@{
/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
DLLIMPORT LaVectorDouble operator*(const LaVectorDouble &x, double a);
    
/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
inline LaVectorDouble operator*(double a, const LaVectorDouble &x)
{
    return operator*(x,a);
}

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
DLLIMPORT double operator*(const LaVectorDouble &dx, 
			   const LaVectorDouble &dy);
                      
/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
DLLIMPORT LaVectorDouble operator+(const LaVectorDouble &dx, 
				   const LaVectorDouble &dy);

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
DLLIMPORT LaVectorDouble operator-(const LaVectorDouble &dx, 
				   const LaVectorDouble &dy);
//@}

//-------------------------------------
/// @name Matrix/Vector operators (deprecated) 
//-------------------------------------
//@{

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
DLLIMPORT LaVectorDouble operator*(const LaGenMatDouble &A, 
				   const LaVectorDouble &dx);

#ifdef _LA_BAND_MAT_DOUBLE_H_
DLLIMPORT LaVectorDouble operator*(const LaBandMatDouble &A, 
				   const LaVectorDouble &dx);
#endif

#ifdef _LA_SYMM_MAT_DOUBLE_H_
DLLIMPORT LaVectorDouble operator*(const LaSymmMatDouble &A, 
				   const LaVectorDouble &dx);
#endif

#ifdef _LA_SYMM_BAND_MAT_DOUBLE_H_ 
DLLIMPORT LaVectorDouble operator*(const LaSymmBandMatDouble &A, 
				   const LaVectorDouble &dx);
#endif


#ifdef _LA_SPD_MAT_DOUBLE_H_
DLLIMPORT LaVectorDouble operator*(const LaSpdMatDouble &AP, 
				   const LaVectorDouble &dx);
#endif

#ifdef _LA_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT LaVectorDouble operator*(const LaLowerTriangMatDouble &A, 
				   const LaVectorDouble &dx);
#endif

#ifdef _LA_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT LaVectorDouble operator*(const LaUpperTriangMatDouble &A, 
				   const LaVectorDouble &dx);
#endif

#ifdef _LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT LaVectorDouble operator*(const LaUnitLowerTriangMatDouble &A,
				   const LaVectorDouble &dx);
#endif

#ifdef _LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT LaVectorDouble operator*(const LaUnitUpperTriangMatDouble &A,
				   const LaVectorDouble &dx);
#endif
//@}

//-------------------------------------
/// @name Matrix/Matrix operators (deprecated) 
//-------------------------------------
//@{
/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
DLLIMPORT LaGenMatDouble operator*(const LaGenMatDouble &A, 
				   const LaGenMatDouble &B);

#ifdef _LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT LaGenMatDouble operator*(const LaUnitLowerTriangMatDouble &A,
				   const LaGenMatDouble &B);
#endif

#ifdef _LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT LaGenMatDouble operator*(const LaUnitUpperTriangMatDouble &A,
				   const LaGenMatDouble &B);
#endif

#ifdef _LA_LOWER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT LaGenMatDouble operator*(const LaLowerTriangMatDouble &A,
				   const LaGenMatDouble &B);
#endif

#ifdef _LA_UPPER_TRIANG_MAT_DOUBLE_H_
DLLIMPORT LaGenMatDouble operator*(const LaUpperTriangMatDouble &A,
				   const LaGenMatDouble &B);
#endif

#ifdef _LA_SYMM_MAT_DOUBLE_H_
DLLIMPORT LaGenMatDouble operator*(const LaSymmMatDouble &A, 
				   const LaGenMatDouble &B);
#endif

#ifdef _LA_SYMM_TRIDIAG_MAT_DOUBLE_H_
DLLIMPORT LaVectorDouble operator*(const LaSymmTridiagMatDouble& A, 
				   const LaVectorDouble& X);
#endif

#ifdef  _LA_TRIDIAG_MAT_DOUBLE_H_
DLLIMPORT LaVectorDouble operator*(const LaTridiagMatDouble& A, 
				   const LaVectorDouble& X);
#endif

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
DLLIMPORT LaGenMatDouble operator-(const LaGenMatDouble &A, 
				   const LaGenMatDouble &B);

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
DLLIMPORT LaGenMatDouble operator+(const LaGenMatDouble &A, 
				   const LaGenMatDouble &B);

# ifdef LA_COMPLEX_SUPPORT
/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
DLLIMPORT LaGenMatComplex operator+(const LaGenMatComplex &A, 
				    const LaGenMatComplex &B);

/** DEPRECATED. Use the Blas functions from blas1pp.h, blas2pp.h and
 * blas3pp.h instead because they are much faster. These operators can
 * already be disabled when you #define LA_NO_DEPRECATED. */
DLLIMPORT LaGenMatComplex operator-(const LaGenMatComplex &A, 
				    const LaGenMatComplex &B);
//@}
# endif // LA_COMPLEX_SUPPORT

#endif // LA_NO_DEPRECATED

#endif  // _BLAS_PP_H_
