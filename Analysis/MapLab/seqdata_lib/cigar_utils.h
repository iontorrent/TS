/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __cigar_utils_h__
#define __cigar_utils_h__

#include <cstddef>

#include <batch.h>        // strings library
#include "align_batch.h"  // seqdata library

#include <CigarRoller.h>

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

// finds the beginning and end clip zones;
// returns pointer to the first unclipped base and fills len with the not-clipped area length
// saves exact clipping arrangement into clip_store
const char* clip_seq (const char* qry, const Cigar& cigar, uint32_t& len, EndClips& clip_store);
// creates cigar for the alignment, using clipping info from clip_store.
// starting insertion is added to soft clip
// starting deletion is subtracted from starting insertion and result is returned as reference shift
int roll_cigar (CigarRoller& roller, const genstr::Alignment& alignment, unsigned clean_len, EndClips& clip_store, unsigned& x_off, unsigned& y_off);
int roll_cigar (CigarRoller& roller, const BATCH* batches, unsigned bno, unsigned clean_len, EndClips& clip_store, unsigned& x_off, unsigned& y_off);

// conversion from cigar to batch format

unsigned cigar_to_batches (const Cigar& cigar, BATCH* batches, unsigned max_batches);
unsigned cigar_to_batches (const std::string cigar_str, BATCH* batches, unsigned max_batches);

// get alignment band width
bool band_width (const Cigar& cigar, unsigned& qryins, unsigned& refins);

#endif // __cigar_utils_h__
