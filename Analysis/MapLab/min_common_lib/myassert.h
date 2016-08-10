/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#include "runtime_error.h"

#ifndef NDEBUG

#define myassert(C) if (!(C)) ers << "Condition check for " #C " failed" << ThrowEx (InternalError);
#define myassertm(C,M) if (!(C)) ers << M << ": condition" #C  " is false" << ThrowEx (InternalError);
#define myassertc(C,X) if (!(C)) ers << "Condition check for " #C " failed" << ThrowEx (X);
#define myassertcm(C,X,M) if (!(C)) ers << M << ": condition " #C " failed" << ThrowEx (X);

#else

#define myassert(C)
#define myassertm(C,M)
#define myassertc(C,X)
#define myassertcm(C,X,M)

#endif
