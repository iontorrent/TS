/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __nullstream_h__
#define __nullstream_h__

/// \file
/// Defines Nullstream class - the std::ostream derivative that silently swallows any output;
/// also declares global nullstream instance.

#include <ostream>

/// std::ostream compatible class that siilently swallows any output
class Nullstream : public std::ostream 
{
public:
    Nullstream () 
    : 
    std::ios(0), 
    std::ostream(0) 
    {
    }
};

#ifndef __nullstream_cpp__
/// The global instance of the Nullstream
extern Nullstream nullstream;
#endif

#endif // __nullstream_h__
