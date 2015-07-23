/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "cigar_utils.h"
#include <rerror.h>
#include <tracer.h>
#include <myassert.h>

const char* clip_seq (const char* qry, const Cigar& cigar, uint32_t& len, EndClips& clip_store)
{
    bool in_tail_clip = false;
    uint32_t cur_p = 0, noclip_beg = 0, noclip_end = 0;
    clip_store.reset ();
    for (uint32_t i = 0, sent = cigar.size (); i != sent; ++ i)
    {
        const Cigar::CigarOperator& curop = cigar [i];
        switch (curop.operation)
        {
            case Cigar::hardClip:
                if (in_tail_clip)
                    noclip_end = cur_p, clip_store.hard_end_ = curop.count;
                else
                    clip_store.hard_beg_ = curop.count;
                break;
            case Cigar::softClip:
                if (in_tail_clip)
                    noclip_end = cur_p, clip_store.soft_end_ = curop.count;
                else
                    clip_store.soft_beg_ = curop.count;
                cur_p += curop.count;
                break;
            case Cigar::match:
            case Cigar::mismatch:
            case Cigar::insert:
                if (!in_tail_clip)
                    noclip_beg = cur_p, in_tail_clip = true;
                cur_p += curop.count;
                break;
            default:
                if (!in_tail_clip)
                    noclip_beg = cur_p, in_tail_clip = true;
        }
    }
    if (noclip_end == 0)
        noclip_end = cur_p;
    len = noclip_end - noclip_beg;
    return qry + noclip_beg;
}

// rolls the cigar, see cigar_utils_h for description
int roll_cigar (CigarRoller& roller, const genstr::Alignment& alignment, unsigned clean_qry_len, EndClips& clip_store, unsigned& x_off, unsigned& y_off)
{
    trc << "passed alignment is " << alignment;
    // add beginning clips if any
    // x is 1st is query, y is 2nd is reference
    if (clip_store.hard_beg_)
        roller.Add (Cigar::hardClip, clip_store.hard_beg_);
    // add alignment operations
    unsigned cur_x = 0, cur_y = 0, bcnt = 0;
    int x_shift = 0;
    for (genstr::Alignment::const_iterator itr = alignment.begin (), sent = alignment.end (); itr != sent; ++itr, ++bcnt)
    {
        const genstr::Batch& cb = *itr;
        // if (bcnt && cur_x != cb.beg1 && cur_y != cb.beg2) // gap over both sequences. Possible if alignment is not global.
        //     ers << "Gap over both aligned sequences detected" << Throw;
        if (!bcnt)
        {
            if (clip_store.soft_beg_ + cb.beg1)
                roller.Add (Cigar::softClip, clip_store.soft_beg_ + cb.beg1);
            x_shift = int (cb.beg2) - int (cb.beg1);
            x_off = cb.beg1;
            y_off = cb.beg2;
        }
        if (cur_x != cb.beg1) // extra bases (insertion) in query
        {
            myassert (cur_x < cb.beg1);
            if (bcnt)
                roller.Add (Cigar::insert, cb.beg1 -  cur_x);
            cur_x = cb.beg1;
        }
        if (cur_y != cb.beg2) // extra bases (insertion) in reference == deletion from query
        {
            myassert (cur_y < cb.beg2);
            if (bcnt)
                roller.Add (Cigar::del,  cb.beg2 - cur_y);
            cur_y = cb.beg2;
        }
        if (cb.len)
        {
            roller.Add (Cigar::match,  cb.len);
            cur_x += cb.len;
            cur_y += cb.len;
        }
    }
    if (cur_x > clean_qry_len)
        ers << "Aligned query portion extends over clipped query length" << Throw;

    // add ending clips if any
    if (clip_store.soft_end_ + (clean_qry_len - cur_x))
        roller.Add (Cigar::softClip, clip_store.soft_end_ + (clean_qry_len - cur_x));
    if (clip_store.hard_end_)
        roller.Add (Cigar::hardClip, clip_store.hard_end_);
    return x_shift;
}

// rolls the cigar, see cigar_utils_h for description
int roll_cigar (CigarRoller& roller, const BATCH* batches, unsigned bno, unsigned clean_qry_len, EndClips& clip_store, unsigned& x_off, unsigned& y_off)
{
    trclog << "Rolling cigar: ";
    batch_dbgout (trclog, batches, bno);
    trclog << "\n";
    // add beginning clips if any
    // x is 1st is query, y is 2nd is reference
    if (clip_store.hard_beg_)
        roller.Add (Cigar::hardClip, clip_store.hard_beg_);
    // add alignment operations
    unsigned cur_x = 0, cur_y = 0;
    int x_shift = 0;
    for (unsigned bcnt = 0;  bcnt != bno; ++bcnt)
    {
        const BATCH& cb = batches [bcnt];

        // if (bcnt && cur_x != cb.xpos && cur_y != cb.ypos) // gap over both sequences. Possible if alignment is not global.
        //     ers << "Gap over both aligned sequences detected" << Throw;
        if (!bcnt)
        {
            if (clip_store.soft_beg_ + cb.xpos)
                roller.Add (Cigar::softClip, clip_store.soft_beg_ + cb.xpos);
            x_shift = int (cb.ypos) - int (cb.xpos);
            x_off = cb.xpos;
            y_off = cb.ypos;
        }
        if (cur_x != cb.xpos) // extra bases (insertion) in query
        {
            myassert (cur_x < cb.xpos);
            if (bcnt)
                roller.Add (Cigar::insert, cb.xpos -  cur_x);
            cur_x = cb.xpos;
        }
        if (cur_y != cb.ypos) // extra bases (insertion) in reference == deletion from query
        {
            myassert (cur_y < cb.ypos);
            if (bcnt)
                roller.Add (Cigar::del,  cb.ypos - cur_y);
            cur_y = cb.ypos;
        }
        if (cb.len)
        {
            roller.Add (Cigar::match,  cb.len);
            cur_x += cb.len;
            cur_y += cb.len;
        }
    }
    if (cur_x > clean_qry_len)
        ers << "Aligned query portion extends over clipped query length" << Throw;

    // add ending clips if any
    if (clip_store.soft_end_ + (clean_qry_len - cur_x))
        roller.Add (Cigar::softClip, clip_store.soft_end_ + (clean_qry_len - cur_x));
    if (clip_store.hard_end_)
        roller.Add (Cigar::hardClip, clip_store.hard_end_);
    return x_shift;
}

unsigned cigar_to_batches (const Cigar& cigar, BATCH* batches, unsigned max_batches)
{
    unsigned xpos = 0, ypos = 0;  // x is query, y is reference
    BATCH* curb = batches;
    for (uint32_t i = 0, sent = cigar.size (); i != sent && curb - batches != max_batches; ++ i)
    {
        const Cigar::CigarOperator& curop = cigar [i];
        switch (curop.operation)
        {
            case Cigar::hardClip: // skip
            case Cigar::softClip: // skip
                break;
            case Cigar::match:
            case Cigar::mismatch:
                curb->xpos = xpos;
                curb->ypos = ypos;
                curb->len = curop.count;
                xpos += curop.count;
                ypos += curop.count;
                ++curb;
                break;
            case Cigar::insert:
                xpos += curop.count;
                break;
            case Cigar::del:
                ypos += curop.count;
                break;
            default:
                ;
        }
    }
    return curb - batches;
}
unsigned cigar_to_batches (const std::string cigar_str, BATCH* batches, unsigned max_batches)
{
    CigarRoller roller;
    roller.Set (cigar_str.c_str ());
    return cigar_to_batches (roller, batches, max_batches);
}

bool band_width (const Cigar& cigar, unsigned& qryins, unsigned& refins)
{
    qryins = refins = 0;
    for (uint32_t i = 0, sent = cigar.size (); i != sent; ++ i)
    {
        const Cigar::CigarOperator& curop = cigar [i];
        switch (curop.operation)
        {
            case Cigar::hardClip: // skip
            case Cigar::softClip: // skip
                break;
            case Cigar::match:
            case Cigar::mismatch:
                break;
            case Cigar::insert: // extra bases in query
                qryins += curop.count;
                break;
            case Cigar::del:  // extra bases in reference
                refins += curop.count;
                break;
            default:
                ;
        }
    }
    return true;
}
