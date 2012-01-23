/*-*-c++-*-****************************************************************
 *                     lacomplex.h Helper file for complex numbers 
                       -------------------
 begin                : 2004-01-14
 copyright            : (C) 2004 by Christian Stimming
 email                : stimming@tuhh.de
***************************************************************************/

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General
// Public License along with this library; see the file COPYING.
// If not, write to the Free Software Foundation, 59 Temple Place
// - Suite 330, Boston, MA 02111-1307, USA.

#ifndef _LACOMPLEX_H
#define _LACOMPLEX_H

/** @file
    
@brief Helper file for complex numbers

This file exists to be a replacement to Lapack++'s inclusion of
<complex.h>. 

@note Complex numbers are a difficult issue. This solution might be
non-trivial to understand, but please bear in mind that complex
numbers are really a nuisance in the C++ programming language.

This file is being used to help the automated conversion from
LAPACK++'s legacy type COMPLEX to the current up-to-date type
std::complex<double>. More information at the type definitions below.

*/

// Include the version number defines
#include <laversion.h>

// Include the FORTRAN definitions from LAPACK++
#include <f2c.h>


// ////////////////////////////////////////////////////////////

/** An application must define LA_COMPLEX_SUPPORT if it wants to use
 * complex numbers here. */
#ifdef LA_COMPLEX_SUPPORT

/** @brief Complex type that is used in LAPACK++ internally. 

The complex type inside LAPACK++ should be the FORTRAN type, since
LAPACK++ has to pass these values back and forth to the FORTRAN
functions. Don't use this data type anywhere outside anymore. */
typedef doublecomplex COMPLEX;

// As opposed to the FORTRAN "COMPLEX", include now the
// std::complex<double> type.
#include <complex>
// And finally include the la::complex<double> which is a copy of
// std::complex<double> with additional type conversions.
#include <lacomplex>

/** @brief Complex data type that can be used from the application.
 *
 * This type is used for passing scalars in and out of LAPACK++. It is
 * a copy of the \c std::complex<double> and it includes automatic
 * conversion from and to \c std::complex<double>. Additionally it
 * includes automatic conversion from and to the internal FORTRAN type
 * COMPLEX, which is why this class is needed to pass complex values
 * into Lapack++.
 *
 * Again: If you get errors when passing a \c std::complex<double>
 * into Lapack++, then you first need to convert your \c
 * std::complex<double> into this \c LaComplex value.
 */
typedef la::complex<double> LaComplex;

/** Additional operator to make stream output easier.
*/
inline std::ostream&
operator<<(std::ostream& __os, const COMPLEX& __x)
{
  return __os << LaComplex(__x);
}

/** Equality operator for \ref COMPLEX. (New in lapackpp-2.4.5) */
inline bool operator==(const COMPLEX& _a, const COMPLEX& _b)
{
    return _a.r == _b.r && _a.i == _b.i;
}
/** Inequality operator for \ref COMPLEX. (New in lapackpp-2.4.5) */
inline bool operator!=(const COMPLEX& _a, const COMPLEX& _b)
{
    return _a.r != _b.r || _a.i != _b.i;
}

#endif /* LA_COMPLEX_SUPPORT */

// ////////////////////////////////////////////////////////////

#endif // _LACOMPLEX_H
