/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#ifndef __streamwrap_h__
#define __streamwrap_h__

/// \file
/// defines StreamWrap class - the wrapper for the ostream object.
/// Instance of StreamWrap can accept same output (left-shift) operator 
/// and can be dynamically assigned the std::ostream instance.
/// Intended to use as "assignable reference" to ostream object

#include "nullstream.h" // it is essential that nullstream includes full <ostream>

/// Serves as assignable reference for std::ostream.
/// accepts same output as std::ostream. More convenient on the right side of output operator
/// then pointer (std::ostream* op), that requires clumsy dereferencing [ (*op) << ... ]
class StreamWrap
{
    std::ostream* o_;
public:
    /// Constructor - initializes instance to a passed std::ostream 
    StreamWrap (std::ostream& o)
    :
    o_ (&o)
    {
    }
    /// Default Constructor - initialized instance to NULL (causing nullstream to be used in output operations)
    StreamWrap ()
    :
    o_ (NULL)
    {
    }
    /// Assignment operator - sets instance to represent passed ostream
    StreamWrap& operator = (std::ostream& o)
    {
        o_ = &o;
        return *this;
    }
    /// Cast to std::ostream
    operator std::ostream& ()
    {
        return o_? *o_ : nullstream;
    }
};

/// output support
template <typename T>
StreamWrap& operator << (StreamWrap& s, const T& t)
{
    ((std::ostream&) s) << t;
    return s;
}
/// ostream manipulators support
inline StreamWrap& operator << (StreamWrap& s, std::ostream& (*op) (std::ostream&))
{
    ((std::ostream&) s)  << op;
    return s;
}


#endif // __streamwrap_h__
