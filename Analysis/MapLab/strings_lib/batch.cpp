/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "batch.h"

namespace genstr 
{

std::ostream& operator << (std::ostream& o, const Batch& b)
{
    o << b.beg1 << ":" << b.beg2 << ":" << b.len;
    return o;
}
std::ostream& operator << (std::ostream& o, const Alignment& al)
{
    bool first = true;
    for (Alignment::const_iterator itr = al.begin (), sent = al.end (); itr != sent; ++itr, first = false)
    {
        if (!first) o << ";";
        o << *itr;
    }
    return o;
}


} // namespace genstr
