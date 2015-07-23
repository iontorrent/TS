/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "rerror.h"

#ifndef NDEBUG

#define myassert(C) if (!(C)) ers << "Condition check for " #C " failed" << ThrowEx (InternalRerror);
#define myassertm(C,M) if (!(C)) ers << M << ": condition" #C  " is false" << ThrowEx (InternalRerror);
#define myassertc(C,X) if (!(C)) ers << "Condition check for " #C " failed" << ThrowEx (X);
#define myassertcm(C,X,M) if (!(C)) ers << M << ": condition " #C " failed" << ThrowEx (X);

#else

#define myassert(C)
#define myassertm(C,M)
#define myassertc(C,X)
#define myassertcm(C,X,M)

#endif
