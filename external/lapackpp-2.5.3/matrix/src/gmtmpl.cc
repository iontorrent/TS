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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "arch.h"
#include "lafnames.h"
#ifdef LA_COMPLEX_SUPPORT
#  include LA_GEN_MAT_COMPLEX_H
#endif
#include LA_GEN_MAT_DOUBLE_H
#include LA_GEN_MAT_FLOAT_H
#include LA_GEN_MAT_INT_H
#include LA_GEN_MAT_LONG_INT_H
#include LA_EXCEPTION_H
#include LA_TEMPLATES_H
#include "mtmpl.h"

// This file contains those member methods of the matrix classes
// that are merely a wrapper to the template functions in
// latmpl.h.

// But in addition to those template-function-wrappers we also
// added those functions here that are identical in all matrix
// implementations, hence we save a lot of typing and make sure
// that optimizations will be used by all classes.

// Methods whose implementation is really identical for all five
// matrix classes
#define TEMPLATED_MEMBERS_ALL(matrix_type) \
bool matrix_type :: equal_to(const matrix_type& mat) const \
{ return la::equal(*this, mat); }  \
matrix_type matrix_type :: repmat (int M, int N) const \
{ return la::repmat (*this, M, N).shallow_assign(); }  \
matrix_type matrix_type :: diag () const      \
{ return la::diag (*this).shallow_assign(); } \
 \
matrix_type & matrix_type :: operator=(value_type s) \
{ return mtmpl::assign(*this, v, s); } \
matrix_type & matrix_type :: operator=(const matrix_type& s) \
{ return copy(s); } \
matrix_type matrix_type :: copy() const \
{ matrix_type result;  result.copy(*this);  return result; } \
matrix_type & matrix_type :: inject(const matrix_type & s) \
{ return mtmpl::inject(*this, v, s, s.v); } \
matrix_type & matrix_type :: resize(int m, int n) \
{ return mtmpl::resize(*this, m, n); } \
matrix_type & matrix_type :: resize(const matrix_type & A) \
{ return resize(A.size(0), A.size(1)); } \
matrix_type & matrix_type :: copy(const matrix_type & A) \
{ return mtmpl::copy(*this, A); } \
matrix_type & matrix_type :: add(value_type s) \
{ return (*this) += s; } \
 \
matrix_type matrix_type :: row (int k)        \
{ return (operator()(LaIndex(k), LaIndex())).shallow_assign(); } \
matrix_type matrix_type :: row (int k) const  \
{ return (operator()(LaIndex(k), LaIndex())).shallow_assign(); } \
matrix_type matrix_type :: col (int k)     \
{ return (operator()(LaIndex(), LaIndex(k))).shallow_assign(); } \
matrix_type matrix_type :: col (int k) const \
{ return (operator()(LaIndex(), LaIndex(k))).shallow_assign(); }


// Now instantiate these methods for the complex-valued class
#ifdef LA_COMPLEX_SUPPORT
TEMPLATED_MEMBERS_ALL(LaGenMatComplex)
#endif


// Methods that are identical for all four real-valued matrix
// classes
#define NONCPLX_TEMPLATED_MEMBERS(matrix_type) \
TEMPLATED_MEMBERS_ALL(matrix_type) \
matrix_type matrix_type :: zeros (int N, int M) \
{ return la::zeros < matrix_type > (N, M).shallow_assign(); } \
matrix_type matrix_type :: ones (int N, int M) \
{ return la::ones < matrix_type > (N, M).shallow_assign(); } \
matrix_type matrix_type :: eye (int N, int M) \
{ return la::eye < matrix_type > (N, M).shallow_assign(); } \
matrix_type matrix_type :: from_diag (const matrix_type &vect) \
{ return la::from_diag (vect).shallow_assign(); } \
 \
bool matrix_type :: is_zero() const \
{ return la::is_zero(*this); } \
matrix_type::value_type matrix_type :: trace () const \
{ return la::trace (*this); } \
matrix_type matrix_type :: linspace (matrix_type::value_type start, matrix_type::value_type end, int nr_points) \
{ return la::linspace < matrix_type > (start, end, nr_points).shallow_assign(); } \
 \
matrix_type & matrix_type :: operator+=(value_type s) \
{ return mtmpl::add_scalar(*this, v, s); } \
std::ostream & operator<<(std::ostream& s, const matrix_type & A) \
{ return mtmpl::print(s, A, A.info_); }


// Now instantiate these methods for the real-valued classes
NONCPLX_TEMPLATED_MEMBERS(LaGenMatDouble)
NONCPLX_TEMPLATED_MEMBERS(LaGenMatFloat)
NONCPLX_TEMPLATED_MEMBERS(LaGenMatInt)
NONCPLX_TEMPLATED_MEMBERS(LaGenMatLongInt)


// Note: we also keep these implementations below because it is
// too much of a hassle to include the LA_TEMPLATES_H header into
// the four matrix implementation files.
LaGenMatDouble LaGenMatDouble :: rand (int N, int M, LaGenMatDouble::value_type low, LaGenMatDouble::value_type high) 
{ return la::rand < LaGenMatDouble > (N, M, low, high).shallow_assign(); } 

LaGenMatFloat LaGenMatFloat :: rand (int N, int M, LaGenMatFloat::value_type low, LaGenMatFloat::value_type high) 
{ return la::rand < LaGenMatFloat > (N, M, low, high).shallow_assign(); } 

LaGenMatInt LaGenMatInt :: rand (int N, int M, LaGenMatInt::value_type low, LaGenMatInt::value_type high) 
{ return la::int_rand < LaGenMatInt > (N, M, low, high).shallow_assign(); } 

LaGenMatLongInt LaGenMatLongInt :: rand (int N, int M, LaGenMatLongInt::value_type low, LaGenMatLongInt::value_type high) 
{ return la::int_rand < LaGenMatLongInt > (N, M, low, high).shallow_assign(); } 
