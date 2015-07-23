/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __align_batch_h__
#define __align_batch_h__

#include <compile_time_macro.h>
#include <tracer.h>


struct BATCH
{
    unsigned xpos;
    unsigned ypos;
    unsigned len;
    BATCH () : xpos ((unsigned) 0), ypos ((unsigned) 0), len ((unsigned) 0) {}
    BATCH (unsigned x, unsigned y, unsigned l) : xpos (x), ypos (y), len (l) {}
    bool x_overlaps (const BATCH& other) const
    {
        return (xpos < other.xpos + other.len) && (other.xpos < xpos + len);
    }
    bool y_overlaps (const BATCH& other) const
    {
        return (ypos < other.ypos + other.len) && (other.ypos < ypos + len);
    }
    bool overlaps (const BATCH& other) const
    {
        return x_overlaps (other) || y_overlaps (other);
    }
    bool above (const BATCH& other) const // if current diag is above other (has lower number) 
    {
        return (xpos - ypos < other.xpos - other.ypos);
    }
    bool intersects (const BATCH& other) const
    {
        // same diagonal and beg1 < end2 and beg2 < end1
        return (same_diag (other) && x_overlaps (other));
    }
    bool operator < (const BATCH& other) const // ordering by y, then x, then len
    {
        if (ypos != other.ypos)
            return ypos < other.ypos;
        if (xpos != other.xpos)
            return xpos < other.xpos;
        return len < other.len;
    }
    bool continuation (const BATCH& other, unsigned changed_last_y = 0) const // checks if this extends other
    {
        return same_diag (other) && (ypos >= other.ypos && ypos <= (changed_last_y ? changed_last_y : other.ypos + other.len));

    }
    bool same_diag (const BATCH& other) const
    {
        return (xpos - ypos == other.xpos - other.ypos);
    }
    unsigned diag_dist (const BATCH& other) const
    {
        int d1 = xpos - ypos;
        int d2 = other.xpos - other.ypos;
        if (d1 <= d2)
            return (unsigned) (d2 - d1);
        else
            return (unsigned) (d1 - d2);
    }
} __attribute__ ((packed)); // 96-bit value

inline std::ostream& operator << (std::ostream& o, const BATCH& b)
{
    o << "x_" << b.xpos << ":y_" << b.ypos << ":l_" << b.len;
    return o;
}

inline void batch_dbgout (std::ostream& o, const BATCH* b, int bno, const char delim = ';', unsigned margin = 0)
{
    while (bno)
    {
        if (margin)         
            o << std::setw (margin) << "" << std::setw (0);
        o << *b;
        b ++;
        bno --;
        if (bno) o << delim;
    }
}

inline void batch_dbgout (Logger& l, const BATCH* b, int bno, const char delim = ';', unsigned margin = 0)
{
    if (l.enabled ())
        batch_dbgout (l.o_, b, bno, delim, margin);
}

inline unsigned accum_batch_dist (const BATCH* b, unsigned bno)
{
    unsigned acc = 0;
    for (const BATCH *sent = b + bno, *prev = NULL; b != sent; prev = b++)
    {
        if (prev)
            acc += b->diag_dist (*prev);
    }
    return acc;
}

int align_score (const BATCH* bm, unsigned bno, const char* xseq, const char* yseq, int gip, int gep, int mat, int mis);

// adjusts the batch set containing duplicated zones (by any axis)
// to the straight (un-duplicated) alignment
// returns new number of batches (may shrink batches array
// WARNING! - this also removes (covers) any both-strand gaps!
int normalize_batches (BATCH* bptr, unsigned bno);

unsigned remove_batch_overlaps (const BATCH* src, unsigned bno, BATCH* dest);


#endif // __align_batch_h__
