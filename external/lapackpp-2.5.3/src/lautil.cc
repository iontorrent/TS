//
//              LAPACK++ 1.1 Linear Algebra Package 1.1
//               University of Tennessee, Knoxvilee, TN.
//            Oak Ridge National Laboratory, Oak Ridge, TN.
//        Authors: J. J. Dongarra, E. Greaser, R. Pozo, D. Walker
//                 (C) 1992-1996 All Rights Reserved
//
//                             NOTICE
//
// Permission to use, copy, modify, and distribute this software and
// its documentation for any purpose and without fee is hereby granted
// provided that the above copyright notice appear in all copies and
// that both the copyright notice and this permission notice appear in
// supporting documentation.
//
// Neither the Institutions (University of Tennessee, and Oak Ridge National
// Laboratory) nor the Authors make any representations about the suitability 
// of this software for any purpose.  This software is provided ``as is'' 
// without express or implied warranty.
//
// LAPACK++ was funded in part by the U.S. Department of Energy, the
// National Science Foundation and the State of Tennessee.

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "lafnames.h"
#include "f2c.h"
#include LA_GEN_MAT_DOUBLE_H
#include LA_VECTOR_LONG_INT_H
#include LA_SYMM_MAT_DOUBLE_H
#ifdef LA_COMPLEX_SUPPORT
#  include LA_GEN_MAT_COMPLEX_H
#endif

#include LA_UTIL_H
#include "lapackd.h"
#include <string.h>     // strlen()

int LaEnvBlockSize(const char *fname, const LaGenMatDouble &A)
{
    const char *opts = "U";

    int one = 1;
    int M = A.size(0);
    int N = A.size(1);
    int junk = -1;

    return ilaenv_(&one, fname, opts, &M, &N, &junk , &junk,
        strlen(fname), strlen(opts));
}

int LaEnvBlockSize(const char *fname, const LaSymmMatDouble &A)
{
    const char *opts = "U";
    int one = 1;
    int M = A.size(0);
    int N = A.size(1);
    int junk = -1;

    return ilaenv_(&one, fname, opts, &M, &N, &junk , &junk,
        strlen(fname), strlen(opts));
}

#ifdef LA_COMPLEX_SUPPORT
int LaEnvBlockSize(const char *fname, const LaGenMatComplex &A)
{
    const char *opts = "U";
    int one = 1;
    int M = A.size(0);
    int N = A.size(1);
    int junk = -1;

    return ilaenv_(&one, fname, opts, &M, &N, &junk , &junk,
        strlen(fname), strlen(opts));
}
#endif


double Mach_eps_double()
{
    char e= 'e';
    return F77NAME(dlamch)(&e);
}


void LaSwap(LaGenMatDouble &A, LaVectorLongInt &ipiv)
{
    integer lda = A.gdim(0),  n = A.size(1);
    integer k1 = ipiv.start(), k2 = ipiv.end(), incx = ipiv.inc();

    F77NAME(dlaswp)(&n, &A(0,0), &lda, &k1, &k2, &ipiv(0), &incx);
}


