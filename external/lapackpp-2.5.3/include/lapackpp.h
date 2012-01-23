//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.

/** @file
 * @brief Header file to include all Lapack++ headers
 *
 * This header file includes all of the header files for LAPACK++.
 * IT'S LARGE! You should rather include the individual headers in the
 * places where you need them. This reduces your compile time and it
 * avoids unnecessary dependencies.
*/

#ifndef _LAPACKPP_H
#define _LAPACKPP_H

#include "lapack.h"

#include "lafnames.h"
#include LA_GEN_MAT_DOUBLE_H
#include LA_VECTOR_DOUBLE_H 
#include LA_GEN_FACT_DOUBLE_H

#ifdef LA_COMPLEX_SUPPORT
# include LA_GEN_MAT_COMPLEX_H
# include LA_VECTOR_COMPLEX_H
# include LA_GEN_QR_FACT_COMPLEX_H
#endif // LA_COMPLEX_SUPPORT

#include LA_BAND_MAT_DOUBLE_H
#include LA_BAND_FACT_DOUBLE_H

#include LA_TRIDIAG_MAT_DOUBLE_H
#include LA_TRIDIAG_FACT_DOUBLE_H
#include LA_SPD_MAT_DOUBLE_H
#include LA_SYMM_MAT_DOUBLE_H
#include LA_SYMM_FACT_DOUBLE_H
#include LA_SYMM_TRIDIAG_MAT_DOUBLE_H
#include LA_SYMM_BAND_MAT_DOUBLE_H
#include LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H
#include LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H

#include LA_SOLVE_DOUBLE_H
#include "lasvd.h"

#include LA_GENERATE_MAT_DOUBLE_H
#include LA_UTIL_H
#include LA_TEMPLATES_H

#include "arch.h"
#include "laexcp.h"
#include "lautil.h"
#include "blaspp.h"

#endif // _LAPACKPP_H
