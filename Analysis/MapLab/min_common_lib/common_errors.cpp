/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
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
#include "rerror.h"

// Note that "common_errors.h" is nt included - rerror.h includes it at the end

const char* OSRerror::get_err_str () const
{
    return strerror (errno);
}

static char errno_str_buf [8];
const char* OSRerror::get_errno_str () const
{
    sprintf (errno_str_buf, "%d", errno);
    return errno_str_buf;
}

