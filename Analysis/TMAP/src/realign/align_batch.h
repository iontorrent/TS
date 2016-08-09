/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef ALIGN_BATCH_H
#define ALIGN_BATCH_H

#include <ostream>
#include <iomanip>

struct BATCH
{
    unsigned xpos;
    unsigned ypos;
    unsigned len;
    BATCH () : xpos ((unsigned) 0), ypos ((unsigned) 0), len ((unsigned) 0) {}
    BATCH (unsigned x, unsigned y, unsigned l) : xpos (x), ypos (y), len (l) {}
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


int align_score (const BATCH* bm, unsigned bno, const char* xseq, const char* yseq, int gip, int gep, int mat, int mis);

#endif // ALIGN_BATCH_H
