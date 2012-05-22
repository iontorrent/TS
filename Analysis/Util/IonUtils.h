/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef IONUTILS_H
#define IONUTILS_H

#include <stdexcept>

namespace ION
{
// Uncomment the DBG_INFO compiler flag to enable source location information in the error output.
#define DBG_INFO

// Utility Macros
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT_FILE TOSTRING(__FILE__)
#define AT AT_FILE ", Line " TOSTRING(__LINE__)

#ifdef DBG_INFO
#define ION_THROW( str ) { throw runtime_error( std::string(AT) + ", " + __FUNCTION__ + "() : " + str ); }
#else
#define ION_THROW( str ) { throw runtime_error( str ); }
#endif

}
// END namespace ION


#endif  // IONUTILS_H
