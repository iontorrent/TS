/* -*-C-*- 

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
*/


/*  Linkage names between C, C++, and Fortran (platform dependent) */

/** @file
 * @brief Platform-dependent macro definitions
 */

#ifndef _ARCH_H_
#define _ARCH_H_


#if  defined(RIOS) && !defined(CLAPACK)
# define F77NAME(x) x
#else
# define F77NAME(x) x##_
#endif

#if defined(SGI) && !defined(SGI_DEC)
# define SGI_DEC
# ifdef __cplusplus
extern "C" {
# endif
	void mkidxname() {}
	void mkdatname() {}
# ifdef __cplusplus
}
# endif
#endif

/* Needed for windows DLLs */
#ifndef DLLIMPORT
#  if defined( __declspec ) | defined ( _MSC_VER )
/*     _MSC_VER checks for Microsoft Visual C++. */
/*      Microsoft Visual C++ 7.1  _MSC_VER = 1310 */
/*      Microsoft Visual C++ 7.0  _MSC_VER = 1300 */
/*      Microsoft Visual C++ 6.0  _MSC_VER = 1200 */
/*      Microsoft Visual C++ 5.0  _MSC_VER = 1100 */
#    if BUILDING_LAPACK_DLL
#      define DLLIMPORT __declspec (dllexport)
#    else     /* Not BUILDING_LAPACK_DLL */
#      define DLLIMPORT __declspec (dllimport)
#    endif    /* Not BUILDING_LAPACK_DLL */
#  else
#    define DLLIMPORT
#  endif    /* __declspec */
#endif  /* DLLIMPORT */

#endif /* _ARCH_H_ */
