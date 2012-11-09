/*
 *  Created on: 04-15-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 48965 $
 *  Last changed by:  $Author: moultoka $
 *  Last change date: $Date: 2010-08-30 09:53:49 -0700 (Mon, 30 Aug 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef XSQ_EXCEPTION_HPP_
#define XSQ_EXCEPTION_HPP_

#include <boost/cstdint.hpp>
#include <stdexcept>
#include <string>

/*!
 lifetechnologies namespace
 */
namespace lifetechnologies
{

/*! Generic exception for Xsq API */
class XsqException : public std::exception {
	std::string m_what;
public:
	XsqException() throw() {}
	XsqException(std::string const& what) throw() : m_what(what) {}
	XsqException& operator= (XsqException const& other) throw() { m_what = other.m_what; return *this; }
	~XsqException() throw() {}
	const char* what() const throw() { return m_what.c_str(); }
};

/*!
 Exception thrown when an index cannot be created.
 */
class io_exception: public std::exception
{
public:
	io_exception(std::string msg) :
        m_msg(msg) {}
    ~io_exception() throw () {}
    const char* what() const throw ()
    {
        std::ostringstream sstrm;
        sstrm << m_msg;
        return sstrm.str().c_str();
    }
private:
    std::string m_msg;
};

/*!
 * Exception thrown when a file cannot be read as in the specified format.
 */
class file_format_exception: public std::exception
{
public:
	file_format_exception(std::string filename, std::string format) :
        m_filename(filename), m_format(format) {}
    ~file_format_exception() throw () {}
    const char* what() const throw ()
    {
        std::ostringstream sstrm;
        sstrm << "File is not in " << m_format << " format: " << m_filename;
        return sstrm.str().c_str();
    }
private:
    std::string m_filename;
    std::string m_format;
};

/*!
 Exception thrown when an index cannot be created.
 */
class missing_data_exception: public std::exception
{
public:
    missing_data_exception(std::string msg) :
        m_msg(msg) {}
    ~missing_data_exception() throw () {}
    const char* what() const throw ()
    {
        std::ostringstream sstrm;
        sstrm << m_msg;
        return sstrm.str().c_str();
    }
private:
    std::string m_msg;
};

/*!
 Exception thrown when an string cannot be parsed.
 */
class parse_exception: public std::exception
{
public:
	parse_exception(std::string msg) :
        m_msg(msg) {}
    ~parse_exception() throw () {}
    const char* what() const throw ()
    {
        std::ostringstream sstrm;
        sstrm << m_msg;
        return sstrm.str().c_str();
    }
private:
    std::string m_msg;
};

/*!
 Exception thrown when an string cannot be parsed.
 */
class illegal_argument_exception: public std::exception
{
public:
	illegal_argument_exception(std::string msg) :
        m_msg(msg) {}
    ~illegal_argument_exception() throw () {}
    const char* what() const throw ()
    {
        std::ostringstream sstrm;
        sstrm << m_msg;
        return sstrm.str().c_str();
    }
private:
    std::string m_msg;
};

} //namespace lifetechnologies

#endif //LTS_EXCEPTION_HPP_
