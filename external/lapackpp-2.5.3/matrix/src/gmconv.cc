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

// This file contains the conversion member functions of the five
// different general matrix classes. It seems to be easier to collect
// them all here together in one file instead of all the different
// matrix files.

// In principle, all the matrix member functions are then defined as
// follows:
LaGenMatDouble LaGenMatFloat :: to_LaGenMatDouble() const {
   return la::convert_mat < LaGenMatDouble > (*this).shallow_assign(); }

// Usually macros are evil, but here they save the heck of a lot of
// typing
#define FROMTO(fromT, toT) LaGenMat##toT LaGenMat##fromT :: to_LaGenMat##toT () const {	\
      return la::convert_mat < LaGenMat##toT > (*this).shallow_assign(); }

//FROMTO(Float, Double); // defined above; uncomment for testing
FROMTO(Float, Int)
FROMTO(Float, LongInt)
FROMTO(Double, Float)
FROMTO(Double, Int)
FROMTO(Double, LongInt)
FROMTO(Int, Double)
FROMTO(Int, Float)
FROMTO(Int, LongInt)
FROMTO(LongInt, Int)
FROMTO(LongInt, Double)
FROMTO(LongInt, Float)
#undef FROMTO

#ifdef LA_COMPLEX_SUPPORT

// And now the same stuff for the conversion to the complex matrix;
// slightly more complicated, but not too much.
template<class srcT>
LaGenMatComplex convert_toC(const srcT& src)
{
   LaGenMatComplex res(src.size(0), src.size(1));
   // optimize later; for now use the correct but slow implementation
   int i, j,  M=src.size(0), N=src.size(1);
   for (j=0; j<N; ++j)
      for (i=0; i<M; ++i) 
      {
	 res(i, j).r = double ( src(i, j) );
	 res(i, j).i = 0.0;
      }
   return res.shallow_assign();
}

#define FROMTOC(fromT) LaGenMatComplex LaGenMat##fromT :: to_LaGenMatComplex () const {	\
      return convert_toC (*this).shallow_assign(); }

FROMTOC(Double)
FROMTOC(Float)
FROMTOC(Int)
FROMTOC(LongInt)

// And now the conversion from the complex matrix, real or imaginary
// part
template<class destT>
destT fromR_to(const LaGenMatComplex& src)
{
   destT res(src.size(0), src.size(1));
   // optimize later; for now use the correct but slow implementation
   int i, j,  M=src.size(0), N=src.size(1);
   for (j=0; j<N; ++j)
      for (i=0; i<M; ++i)
	 res(i, j) = typename destT::value_type ( src(i, j).r );
   return res.shallow_assign();
}
template<class destT>
destT fromI_to(const LaGenMatComplex& src)
{
   destT res(src.size(0), src.size(1));
   // optimize later; for now use the correct but slow implementation
   int i, j,  M=src.size(0), N=src.size(1);
   for (j=0; j<N; ++j)
      for (i=0; i<M; ++i)
	 res(i, j) = typename destT::value_type ( src(i, j).i );
   return res.shallow_assign();
}

#define FROMRTO(toT) LaGenMat##toT LaGenMatComplex :: real_to_LaGenMat##toT () const { \
      return fromR_to < LaGenMat##toT > (*this).shallow_assign(); }

#define FROMITO(toT) LaGenMat##toT LaGenMatComplex :: imag_to_LaGenMat##toT () const { \
      return fromI_to < LaGenMat##toT > (*this).shallow_assign(); }

FROMRTO(Double)
FROMRTO(Float)
FROMRTO(Int)
FROMRTO(LongInt)

FROMITO(Double)
FROMITO(Float)
FROMITO(Int)
FROMITO(LongInt)

#endif // LA_COMPLEX_SUPPORT
