/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __clreol_h__
#define __clreol_h__

#include "fileno.hpp"
#include <ios>

#define clreol_esc "\x1B[K"
class clreol_class
{
};
#ifndef __clreol_cpp__
extern clreol_class clreol;
#endif

inline std::ostream& operator << (std::ostream& ostr, const clreol_class&)
{
    int fh = fileno (ostr);
    if (isatty (fh))
        ostr << clreol_esc;
    return ostr;
}

#endif // __clreol_h__
