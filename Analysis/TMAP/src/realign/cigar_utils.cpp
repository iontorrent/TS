/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#include "cigar_utils.h"

extern "C" {
#include "../samtools/bam.h"
#include "../util/tmap_error.h"
#include "../io/tmap_file.h"
}

#include <cassert>
#include <iostream>


const char* clip_seq (const char* qry, const uint32_t* cigar, unsigned cigar_sz, uint32_t& len, EndClips& clip_store)
{
    bool in_tail_clip = false;
    uint32_t cur_p = 0, noclip_beg = 0, noclip_end = 0;
    clip_store.reset ();

    unsigned oplen, constype;
    for (const uint32_t* elem = cigar, *sent = cigar + cigar_sz; elem != sent; ++elem)
    {
        oplen = bam_cigar_oplen (*elem);
        constype = bam_cigar_op (*elem);
        switch (constype)
        {
            case BAM_CHARD_CLIP:
                if (in_tail_clip)
                    noclip_end = cur_p, clip_store.hard_end_ = oplen;
                else
                    clip_store.hard_beg_ = oplen;
                break;
            case BAM_CSOFT_CLIP:
                if (in_tail_clip)
                    noclip_end = cur_p, clip_store.soft_end_ = oplen;
                else
                    clip_store.soft_beg_ = oplen;
                cur_p += oplen;
                break;
            case BAM_CMATCH:
            case BAM_CEQUAL:
            case BAM_CDIFF:
            case BAM_CINS:
                if (!in_tail_clip)
                    noclip_beg = cur_p, in_tail_clip = true;
                cur_p += oplen;
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

unsigned roll_cigar (uint32_t* cigar, unsigned max_cigar_len, unsigned& cigar_len, const BATCH* batches, unsigned bno, unsigned clean_qry_len, EndClips& clip_store, unsigned& x_off, unsigned& y_off, unsigned& xlen, unsigned& ylen)
{
    // add beginning clips if any
    // x is 1st is query, y is 2nd is reference
    cigar_len = 0;
    if (clip_store.hard_beg_)
    {
        assert (cigar_len < max_cigar_len);
        cigar [cigar_len++] = (clip_store.hard_beg_ << BAM_CIGAR_SHIFT)|BAM_CHARD_CLIP;
    }
    // add alignment operations
    unsigned cur_x = 0, cur_y = 0;
    int x_shift = 0;
    for (unsigned bcnt = 0;  bcnt != bno; ++bcnt)
    {
        const BATCH& cb = batches [bcnt];

        if (!bcnt)
        {
            if (clip_store.soft_beg_ + cb.xpos)
            {
                assert (cigar_len < max_cigar_len);
                cigar [cigar_len++] = ((clip_store.soft_beg_ + cb.xpos) << BAM_CIGAR_SHIFT)|BAM_CSOFT_CLIP;
            }
            x_shift = int (cb.ypos) - int (cb.xpos);
            x_off = cb.xpos;
            y_off = cb.ypos;
        }
        if (cur_x != cb.xpos) // extra bases (insertion) in query
        {
            assert (cur_x < cb.xpos);
            if (bcnt)
            {
                assert (cigar_len < max_cigar_len);
                cigar [cigar_len++] = ((cb.xpos -  cur_x) << BAM_CIGAR_SHIFT)|BAM_CINS;
            }
            cur_x = cb.xpos;
        }
        if (cur_y != cb.ypos) // extra bases (insertion) in reference == deletion from query
        {
            assert (cur_y < cb.ypos);
            if (bcnt)
            {
                assert (cigar_len < max_cigar_len);
                cigar [cigar_len++] = ((cb.ypos - cur_y) << BAM_CIGAR_SHIFT)|BAM_CDEL;
            }
            cur_y = cb.ypos;
        }
        if (cb.len)
        {
            assert (cigar_len < max_cigar_len);
            cigar [cigar_len++] = (cb.len << BAM_CIGAR_SHIFT)|BAM_CMATCH;
            cur_x += cb.len;
            cur_y += cb.len;
        }
    }
    if (cur_x > clean_qry_len)
    {
        tmap_file_fprintf (tmap_file_stderr, "Aligned query portion extends over clipped query length\n");
        tmap_bug ();
    }

    xlen = cur_x - x_off;
    ylen = cur_y - y_off;

    // shift x_off by the 5' soft clip 
    x_off += clip_store.soft_beg_;

    // add ending clips if any
    if (clip_store.soft_end_ + (clean_qry_len - cur_x))
    {
        assert (cigar_len < max_cigar_len);
        cigar [cigar_len++] = ((clip_store.soft_end_ + (clean_qry_len - cur_x)) << BAM_CIGAR_SHIFT)|BAM_CSOFT_CLIP;
    }
    if (clip_store.hard_end_)
    {
        assert (cigar_len < max_cigar_len);
        cigar [cigar_len++] = (clip_store.hard_end_ << BAM_CIGAR_SHIFT)|BAM_CHARD_CLIP;
    }
    return x_shift;
}

unsigned cigar_to_batches (const uint32_t* cigar, unsigned cigar_sz, BATCH* batches, unsigned max_batches)
{
    unsigned xpos = 0, ypos = 0;  // x is query, y is reference
    BATCH* curb = batches;
    for (const uint32_t* sent = cigar+cigar_sz; cigar != sent && curb - batches != max_batches; ++cigar)
    {
        uint32_t curop = bam_cigar_op (*cigar);
        uint32_t count = bam_cigar_oplen (*cigar);
        switch (curop)
        {
            case BAM_CHARD_CLIP: // skip
            case BAM_CSOFT_CLIP: // skip
                break;
            case BAM_CMATCH:
            case BAM_CEQUAL:
            case BAM_CDIFF:
                curb->xpos = xpos;
                curb->ypos = ypos;
                curb->len = count;
                xpos += count;
                ypos += count;
                ++curb;
                break;
            case BAM_CINS:
                xpos += count;
                break;
            case BAM_CDEL:
                ypos += count;
                break;
            default:
                ;
        }
    }
    return curb - batches;
}

bool band_width (const uint32_t* cigar, unsigned cigar_sz, unsigned& qryins, unsigned& refins)
{
    qryins = refins = 0;
    for (const uint32_t* sent = cigar+cigar_sz; cigar != sent; ++cigar)
    {
        uint32_t curop = bam_cigar_op (*cigar);
        uint32_t count = bam_cigar_oplen (*cigar);
        switch (curop)
        {
            case BAM_CHARD_CLIP: // skip
            case BAM_CSOFT_CLIP: // skip
                break;
            case BAM_CMATCH:
            case BAM_CEQUAL:
            case BAM_CDIFF:
                break;
            case BAM_CINS:
                qryins += count;
                break;
            case BAM_CDEL:
                refins += count;
                break;
            default:
                ;
        }
    }
    return true;
}

void cigar_out (std::ostream& o, const uint32_t* cigar, unsigned cigar_sz)
{
    const uint32_t* sent;
    for (sent = cigar+cigar_sz; cigar != sent; ++cigar)
    {
        uint32_t curop = bam_cigar_op (*cigar);
        uint32_t count = bam_cigar_oplen (*cigar);
        char schar;
        switch (curop)
        {
            case BAM_CHARD_CLIP:
                schar = 'H';
                break;
            case BAM_CSOFT_CLIP: // skip
                schar = 'S';
                break;
            case BAM_CMATCH:
                schar = 'M';
                break;
            case BAM_CEQUAL:
                schar = '=';
                break;
            case BAM_CDIFF:
                schar = '#';
                break;
            case BAM_CINS:
                schar = 'I';
                break;
            case BAM_CDEL:
                schar = 'I';
                break;
            default:
                schar = '?';
        }
        o << schar << count;
    }
}
