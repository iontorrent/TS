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

// (obsolete comment:)
//  C++ exception handling is not currently supported by most compilers.
//  The macros below allow one to "mimic" the throw expression of
//  character strings.  Transfer of control does not correspond
//  to a "try" block, but rather exits the program with the proper
//  banner.  This is a similar behavior to the real exception handling
//  if there was no explicit user-supplied try block.
//

/** @file
 * @brief Exception class for Lapack++ exceptions
 */

#ifndef _LA_EXCEPTION_H_
#define _LA_EXCEPTION_H_

#include <iostream>
#include <cstdlib>

#include <stdexcept>
#include "arch.h"

#ifndef __ASSERT_H
# include <cassert>
# ifndef __ASSERT_H
#  define __ASSERT_H
# endif
#endif

// If you want to have exceptions thrown rather than failed
// assertions, uncomment the following lines. Then, in your
// application you have to make sure to include only this file,
// <laexcp.h>, and not <cassert>, and also to include this file
// first of all lapackpp headers.
/*
#undef assert
// This probably works for gcc 3.3.5 on Linux
#define assert(expr) { if (!(expr)) \
  { throw LaException(__STRING(expr), __FILE__, \
  __LINE__, __ASSERT_FUNCTION); } }
// This was seen to work for gcc 3.4.5 on Win32/mingw
//#define assert(expr) { if (!(expr)) { throw LaException(#expr, __FILE__, __LINE__, ""); } }
*/


/** @brief General exception class in Lapack++
 *
 * General exception class for when an exceptions occurs inside
 * Lapack++. */
class DLLIMPORT LaException : public std::runtime_error
{
   public:
      /** Empty constructor. */
      LaException();
      
      /** Constructor with text information. The text can be retrieved
       * by the what() method, i.e. 
\verbatim
  LaException e("myFunction()", "some error"); 
  std::cout << e.what();
\endverbatim
       *
       * If the static flag LaException::enablePrint() is enabled,
       * then this constructor also writes the given text to stderr.
       */
      LaException(const char *where, const char *what);

      /** Constructor with more text information, similar to the
       * assert(3) macro.
       *
       * If the static flag LaException::enablePrint() is enabled,
       * then this constructor also writes the given text to stderr.
       */
      LaException(const char *assertion, const char *file,
		  unsigned int line, const char *function);

      /** Static (i.e. library-wide) flag whether any new LaException
       * should also print its message to stderr. Disabled by
       * default. 
       *
       * @note Other static (i.e. library-wide) preferences are stored
       * in the @ref LaPreferences class. This flag is stored here
       * because it concerns only this LaException class. */
      static void enablePrint(bool enable);
   private:
      static bool _print;
};

#define LA_ASSERTZERO(infovar) if ((infovar) != 0) { \
    std::ostringstream temp; \
    temp << ":" << __LINE__ << ": Internal error in LAPACK function: Returned info=" << (infovar);	\
    if (infovar < 0) \
      temp << ". This means the "<< -(infovar) << "th argument has an invalid value."; \
    if (infovar > 0) \
      temp << ". This means the calculation did not converge. Maybe an input matrix was ill-conditioned, or any of the input values were NaN or inf."; \
    throw(LaException(__FILE__, temp.str().c_str())); \
  }


#endif  // _LA_EXCEPTION_H_

