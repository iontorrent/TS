/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#define __common_errors_cpp__

const char* ERR_NoMemory = "Unable to allocate memory";
const char* ERR_Internal = "Internal program error";
const char* ERR_FileNotFound = "File not found";
const char* ERR_OSError = "Operating system error";
const char* ERR_OutOfBounds = "Out of bounds error";

#include <cerrno>
#include <cstring>
#include <cstdio>
#include "runtime_error.h"

const char* OSError::get_err_str () const
{
    return strerror (errno);
}

static char errno_str_buf [8];
const char* OSError::get_errno_str () const
{
    sprintf (errno_str_buf, "%d", errno);
    return errno_str_buf;
}

