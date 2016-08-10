/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#define __runtime_error_cpp__
#include "runtime_error.h"

RunTimeErrorStream ers;

std::ostream& operator << (std::ostream& o, const RunTimeError& runtime_error)
{
    o << (const char*) runtime_error;
    return o;
}
