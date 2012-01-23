// -*-C++-*- 

// Copyright (C) 2005
// Christian Stimming <stimming@tuhh.de>

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2, or (at
// your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along
// with this library; see the file COPYING.  If not, write to the Free
// Software Foundation, 59 Temple Place - Suite 330, Boston, MA 02111-1307,
// USA.

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "laexcp.h"
#include <string>
#include <iostream>

LaException::LaException()
    : std::runtime_error("LaException in LAPACK++") 
{
}

LaException::LaException(const char *where, const char *what)
    : std::runtime_error ( 
	std::string(where ? where : "") +
	std::string(what ? what : "") )
{
    if (_print)
	std::cerr << std::string("LaException: ")
		  << std::string(where ? where : "") 
		  << std::string(" : ")
		  << std::string(what ? what : "") << std::endl;
}

LaException::LaException(const char *assertion, const char *file,
			 unsigned int line, const char *function)
    : std::runtime_error (
	std::string(file ? file : "") + std::string(": ") +
	std::string(function ? function : "") + 
	std::string(": Assertion failed: ") +
	std::string(assertion ? assertion : "") )
{
    if (_print)
	std::cerr << std::string("LaException: ")
		  << std::string(file ? file : "") << std::string(":")
		  << line << std::string(": ")
		  << std::string(function ? function : "") 
		  << std::string(": Assertion failed: ")
		  << std::string(assertion ? assertion : "")
		  << std::endl;
}

bool LaException::_print = false;
void LaException::enablePrint(bool e)
{
    _print = e;
}
