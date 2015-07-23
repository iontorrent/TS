/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef PRINT_BATCHES_H
#define PRINT_BATCHES_H

#include <ostream>
// #include "biosequence.h"
#include "align_batch.h"
// #include "weights.h"

struct PRAL // PRintable ALignment
{
    BATCH* b_;
    unsigned bno_;
    unsigned xoff_;
    unsigned yoff_;
    PRAL (BATCH* b, unsigned bno, unsigned xoff = 0, unsigned yoff = 0)
    :
    b_ (b),
    bno_ (bno),
    xoff_ (xoff),
    yoff_ (yoff)
    {
    }
};

std::ostream& operator << (std::ostream& ostr, const PRAL& pa);
void print_batches (const char* xseq, unsigned xlen, bool xrev, const char* yseq, unsigned ylen, bool yrev, const BATCH *b_ptr, int b_cnt, std::ostream& stream, bool unpack = true, unsigned xoff = 0, unsigned yoff = 0, unsigned margin = 0, unsigned width = 78, bool zero_based = false);
int print_batches_cigar (const BATCH *b_ptr, int b_cnt, char* dest, unsigned destlen);
#endif
