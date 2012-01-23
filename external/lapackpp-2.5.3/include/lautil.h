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
 * @brief A few general utilities
 *
 * Some (very few) utility functions that should have been included in
 * C++ (sic?!)
 */

#ifndef _LA_UTIL_H_
#define _LA_UTIL_H_

#include "f2c.h"
#include "arch.h"
#include LA_GEN_MAT_DOUBLE_H
#ifdef LA_COMPLEX_SUPPORT
# include LA_GEN_MAT_COMPLEX_H
#endif
#include LA_VECTOR_LONG_INT_H

// only callable from C-Lapack due to added ftnlen parameters by f2c; 
extern "C" 
    int ilaenv_(int *i, const char *n, const char *opts, 
        int *n1, int *n2, int *n3, int *n4, 
        ftnlen n_len, ftnlen opts_len);


/** Performs a series of row interchanges on the matrix A. Uses \c
 * dlaswp internally.
 *
 * @param A The matrix A whose rows are interchanged.
 * @param ipiv The vector of pivot indices. ipiv(K)=L implies rows
 * K and L are interchanged.*/
DLLIMPORT
void LaSwap(LaGenMatDouble &A, LaVectorLongInt &ipiv);

DLLIMPORT
int LaEnvBlockSize(const char *fname, const LaGenMatDouble &A);

#ifdef LA_COMPLEX_SUPPORT
DLLIMPORT
int LaEnvBlockSize(const char *fname, const LaGenMatComplex &A);
#endif

#ifdef _LA_SYMM_MAT_DOUBLE_H_
DLLIMPORT
int LaEnvBlockSize(const char *fname, const LaSymmMatDouble &A);
#endif

/** Determine double precision machine parameters. Uses \c dlamch
 * internally. Returns the "relative machine precision". */
DLLIMPORT
double Mach_eps_double();

#endif 
    // _LA_UTIL_H_
