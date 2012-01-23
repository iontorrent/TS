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

/** @file blas1pp.h
 * @brief Blas Level-1 Routines: Vector-Scalar and
 * Vector-Vector Operations
 *
 * This file defines the basic operations on vectors itself, commonly
 * known as the Blas Level-1 routines.
 */


#ifndef _BLAS1_PP_H_
#define _BLAS1_PP_H_

#include "blas1.h"
#include "lafnames.h"
#include "arch.h" // needed for DLLIMPORT
#include LA_VECTOR_DOUBLE_H
#ifdef LA_COMPLEX_SUPPORT
# include LA_VECTOR_COMPLEX_H
#endif
#include <cmath>

#ifdef LA_COMPLEX_SUPPORT
/** @name Complex-valued vector operations */
//@{
/** @brief Combined vector scaling and addition (saxpy)
 *
 * Combined vector scaling and addition:  dy = dy + da * dx */
DLLIMPORT
void Blas_Add_Mult(LaVectorComplex &dy, COMPLEX da, const LaVectorComplex &dx);

/** @brief Vector scaling
 *
 * Vector scaling: dy = da * dx
 *
 * @note This function is quite dumb - it only sets dy to 0.0,
 * then calls daxpy i.e. Blas_Add_Mult. You should rather use
 * Blas_Add_Mult() or Blas_Scale() instead.
 */
DLLIMPORT
void Blas_Mult(LaVectorComplex &dy, COMPLEX da, const LaVectorComplex &dx);

/** @brief Non-conjugated dot product (very unusual!)
 *
 * Returns the dot product of two vectors x and y, which is
 * <code>Sum(x[i]*y[i])</code>, without taking any complex conjugate,
 * which is a difference to the usual definition of complex dot
 * products. This is probably @e not what you want, and you probably
 * want to use Blas_H_Dot_Prod() instead.
 *
 * @note Unlike Blas_H_Dot_Prod(), the vector x is @e not taken
 * complex conjugate, which is a difference from the usual definition
 * (Matlab-notation) <code>x'*y</code> . This function rather calculates
 * (Matlab-notation) <code>x.'*y</code> .  */
DLLIMPORT
COMPLEX Blas_U_Dot_Prod(const LaVectorComplex &cx, const LaVectorComplex &cy);

/** @brief Dot product
 *
 * Returns the dot product of two vectors conj(x) and y, where the
 * first vector is taken conjugate complex, which is
 * <code>Sum(conj(x[i])*y[i])</code>. Note: This is not commutative
 * any longer but rather complex conjugate commutative. But this is
 * the usual case for these complex vectors.  */
DLLIMPORT
COMPLEX Blas_H_Dot_Prod(const LaVectorComplex &cx, const LaVectorComplex &cy);

/** Vector assignment (copying): dy = dx */
DLLIMPORT
void Blas_Copy(const LaVectorComplex &dx, LaVectorComplex &dy);

/** Vector scaling: dx = da * dx */
DLLIMPORT
void Blas_Scale(COMPLEX da, LaVectorComplex &dx);

/** Swaps the elements of two vectors: dx <=> dy */
DLLIMPORT
void Blas_Swap(LaVectorComplex &dx, LaVectorComplex &dy);
//@}
#endif // LA_COMPLEX_SUPPORT





/** @name Real-valued vector operations */
//@{
/** @brief Combined vector scaling and addition (saxpy)
 *
 * Combined vector scaling and addition:  dy = dy + da * dx */
DLLIMPORT
void Blas_Add_Mult(LaVectorDouble &dy, double da, const LaVectorDouble &dx);

/** @brief Vector scaling
 *
 * Vector scaling: dy = da * dx
 *
 * @note This function is quite dumb - it only sets dy to 0.0,
 * then calls daxpy i.e. Blas_Add_Mult. You should rather use
 * Blas_Add_Mult() or Blas_Scale() instead.
 */
DLLIMPORT
void Blas_Mult(LaVectorDouble &dy, double da, const LaVectorDouble &dx);

/** @brief Dot product
 *
 * Returns the dot product of two vectors x and y, which is
 * Sum(x[i]*y[i]). */
DLLIMPORT
double Blas_Dot_Prod(const LaVectorDouble &dx, const LaVectorDouble &dy);

/** @brief Apply Givens plane rotation
 *
 * Applies a Givens plane rotation to (x,y): dx = c*dx + s*dy; dy =
 * c*dy - s*dx */
DLLIMPORT
void Blas_Apply_Plane_Rot(LaVectorDouble &dx, LaVectorDouble &dy, 
			  double &c, double &s);

/** @brief Construct Givens plane rotation
 *
 * Construct a Givens plane rotation for (a,b). da, db are the
 * rotational elimination parameters a,b. */
DLLIMPORT
void Blas_Gen_Plane_Rot(double &da, double &db, double &c, double &s);

/** Vector assignment (copying): dy = dx */
DLLIMPORT
void Blas_Copy(const LaVectorDouble &dx, LaVectorDouble &dy);

/** Vector scaling: dx = da * dx */
DLLIMPORT
void Blas_Scale(double da, LaVectorDouble &dx);

/** Swaps the elements of two vectors: dx <=> dy */
DLLIMPORT
void Blas_Swap(LaVectorDouble &dx, LaVectorDouble &dy);

//@}



/** @name Vector norms */
//@{

#ifdef LA_COMPLEX_SUPPORT
/** \brief 1-Norm
 *
 * Returns the sum of the absolute values: \f$|x|_1=\sum_i|x_i|\f$
 *
 * \see Eric W. Weisstein. "Vector Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/VectorNorm.html */
DLLIMPORT
double Blas_Norm1(const LaVectorComplex &dx);

/** \brief 2-Norm, Euclidean Norm
 *
 * Returns the euclidean norm of the vector:
 * \f$|x|_2=\sqrt{\sum_i|x_i|^2}\f$
 *
 * In other notation
 * <code>sqrt(conj(x')*x)</code> or in Matlab notation
 * <code>sqrt(x'*x)</code>
 *
 * \see Eric W. Weisstein. "Vector Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/VectorNorm.html */
DLLIMPORT
double Blas_Norm2(const LaVectorComplex &dx);

/** Returns the index of largest absolute value; 
 *  i such that |x[i]| == max(|x[0]|,|x[1]|,...) */
DLLIMPORT
int Blas_Index_Max(const LaVectorComplex &dx);

/** \brief Infinity-Norm
 *
 * Returns the Infinity norm of a vector, which is the absolute value
 * of its maximum element: \f$|x|_{\infty}=\max_i|x_i|\f$
 *
 * \see Eric W. Weisstein. "Vector Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/VectorNorm.html */
inline double Blas_Norm_Inf(const LaVectorComplex &x)
{   
    integer index = Blas_Index_Max(x);
    return la::abs(LaComplex(x(index)));
}
#endif // LA_COMPLEX_SUPPORT

/** \brief 1-Norm
 *
 * Returns the sum of the absolute values: \f$|x|_1=\sum_i|x_i|\f$
 *
 * \see Eric W. Weisstein. "Vector Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/VectorNorm.html */
DLLIMPORT
double Blas_Norm1(const LaVectorDouble &dx);

/** \brief 2-Norm, Euclidean Norm
 *
 * Returns the euclidean norm of the vector:
 * \f$|x|_2=\sqrt{\sum_i|x_i|^2}\f$
 *
 * In other notation
 * <code>sqrt(conj(x')*x)</code> or in Matlab notation
 * <code>sqrt(x'*x)</code>
 *
 * \see Eric W. Weisstein. "Vector Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/VectorNorm.html */
DLLIMPORT
double Blas_Norm2(const LaVectorDouble &dx);

/** Returns the index of largest absolute value; 
 *  i such that |x[i]| == max(|x[0]|,|x[1]|,...) */
DLLIMPORT
int Blas_Index_Max(const LaVectorDouble &dx);

/** \brief Infinity-Norm
 *
 * Returns the Infinity norm of a vector, which is the absolute value
 * of its maximum element: \f$|x|_{\infty}=\max_i|x_i|\f$
 *
 * \see Eric W. Weisstein. "Vector Norm." From MathWorld--A Wolfram
 * Web Resource. http://mathworld.wolfram.com/VectorNorm.html */
inline double Blas_Norm_Inf(const LaVectorDouble &x)
{   
    integer index = Blas_Index_Max(x);
    return std::fabs(x(index));
}

#ifndef DOXYGEN_IGNORE
# ifdef LA_COMPLEX_SUPPORT
/** DEPRECATED, use Blas_Norm_Inf instead. */
inline double Norm_Inf(const LaVectorComplex &A) { return Blas_Norm_Inf(A); }
# endif // LA_COMPLEX_SUPPORT
/** DEPRECATED, use Blas_Norm_Inf instead. */
inline double Norm_Inf(const LaVectorDouble &A) { return Blas_Norm_Inf(A); }
#endif // DOXYGEN_IGNORE

//@}


#endif  
    /* _BLAS1_PP_H_ */
