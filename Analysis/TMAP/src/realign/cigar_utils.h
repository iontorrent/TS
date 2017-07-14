/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CIGAR_UTILS_H
#define CIGAR_UTILS_H

#include <stdint.h>
#include <ostream>
#include "align_batch.h"  // seqdata library
#include "cigar_op.h"

struct EndClips
{
    uint32_t hard_beg_;
    uint32_t soft_beg_;
    uint32_t soft_end_;
    uint32_t hard_end_;
    void reset ()
    {
        hard_beg_ = soft_beg_ = soft_end_ = hard_end_ = 0;
    }
    EndClips ()
    {
        reset ();
    }
};

inline std::ostream& operator << (std::ostream& o, const EndClips& ec)
{
    o << "EndClips{H" << ec.hard_beg_ << " S" << ec.soft_beg_ << " S" << ec.soft_end_ << " H" << ec.hard_end_ << "}";
    return o;
}

// finds the beginning and end clip zones;
// returns pointer to the first unclipped base and fills len with the not-clipped area length
// saves exact clipping arrangement into clip_store
const char* clip_seq (const char* qry, const uint32_t* cigar, unsigned cigar_sz, uint32_t& len, EndClips& clip_store);
// creates cigar for the alignment, using clipping info from clip_store.
// starting insertion is added to soft clip
// starting deletion is subtracted from starting insertion and result is returned as reference shift

unsigned roll_cigar (uint32_t* cigar, unsigned max_cigar_len, unsigned& cigar_len, const BATCH* batches, unsigned bno, unsigned clean_len, EndClips& clip_store, unsigned& x_off, unsigned& y_off, unsigned& xlen, unsigned& ylen);

// conversion from cigar to batch format
unsigned cigar_to_batches (const uint32_t* cigar, unsigned cigar_sz, BATCH* batches, unsigned max_batches);

// get alignment band width
bool band_width (const uint32_t* cigar, unsigned cigar_sz, unsigned& qryins, unsigned& refins);

void cigar_out (std::ostream& o, const uint32_t* cigar, unsigned cigar_sz);

#endif // CIGAR_UTILS_H
