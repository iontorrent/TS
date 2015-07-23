/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __BATCH_H__
#define __BATCH_H__

#include <deque>
#include <ostream>
#include <tracer.h>

namespace genstr
{

struct Batch
{
    unsigned beg1;
    unsigned beg2;
    unsigned len;
    Batch (unsigned beg1 = 0, int beg2 = 0, int len = 0)
    {
        Batch::beg1 = beg1, Batch::beg2 = beg2, Batch::len = len;
    }
    Batch (const Batch& b)
    {
        operator = (b);
    }
    Batch& operator =  (const Batch& b)
    {
        beg1 = b.beg1, beg2 = b.beg2, len = b.len;
        return *this;
    }
};

typedef std::deque<Batch> Alignment;

std::ostream& operator << (std::ostream&, const Batch&);
std::ostream& operator << (std::ostream&, const Alignment&);
inline Logger& operator << (Logger& l, const Alignment& al)
{
    if (l.enabled ())
        l.o_ << al;
    return l;
}

} // namespace genstr

#endif // __BATCH_H__
