/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#define __rerror_cpp__
#include "rerror.h"

ErrorStream ers;

std::ostream& operator << (std::ostream& o, const Rerror& rerror)
{
    o << (const char*) rerror;
    return o;
}
