/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#include "realign_c_util.h"

#include <stdint.h>
#include "../util/tmap_cigar_util.h"

// #define CIGAR_ACCESS_REFACTORED

#ifndef CIGAR_ACCESS_REFACTORED
#include "samtools/bam.h"
#endif


unsigned seq_lens_from_bin_cigar (uint32_t* cigar_bin, unsigned cigar_bin_sz, unsigned* q_len, unsigned* r_len)
{
#ifdef CIGAR_ACCESS_REFACTORED
    uint32_t allen;
    cigar_footprint (cigar_bin, cigar_bin_sz, q_len, r_len, &allen, NULL, NULL);
    return allen;
#else
    unsigned oplen, constype;
    uint32_t *sent;
    *q_len = *r_len = 0;
    unsigned allen = 0;
    for (sent = cigar_bin + cigar_bin_sz; cigar_bin != sent; ++cigar_bin)
    {
        oplen = bam_cigar_oplen (*cigar_bin);
        constype = bam_cigar_type (bam_cigar_op (*cigar_bin));
        if (constype & CONSUME_QRY) *q_len += oplen;
        if (constype & CONSUME_REF) *r_len += oplen;
        allen += oplen;
    }
    return allen;
#endif
}

unsigned alignment_bounds_from_bin_cigar (uint32_t* cigar_bin, unsigned cigar_bin_sz, uint8_t forward, unsigned qry_len, unsigned* q_beg, unsigned* q_end, unsigned* r_beg, unsigned* r_end)
{
#ifdef CIGAR_ACCESS_REFACTORED
    uint32_t allen, q_len, r_len, clip_l, clip_r;
    cigar_footprint (cigar_bin, cigar_bin_sz, &q_len, &r_len, &allen, &clip_l, &clip_r);
    if (q_beg) *q_beg = clip_l;
    if (r_beg) *r_beg = 0;
    if (q_end) *q_end = clip_l + q_len;
    if (r_end) *r_end = r_len;
#else
    unsigned oplen, op, constype;
    uint32_t *sent;
    *q_beg = *q_end = *r_beg = *r_end = 0;
    unsigned allen = 0;
    uint32_t tail = 0;
    for (sent = cigar_bin + cigar_bin_sz; cigar_bin != sent; ++cigar_bin)
    {
        oplen = bam_cigar_oplen (*cigar_bin);
        op = bam_cigar_op (*cigar_bin);
        constype = bam_cigar_type (op);

        if (tail && (op == BAM_CHARD_CLIP || op == BAM_CSOFT_CLIP)) // the aligned zone ended, clip started. Note that tail indels are not valid, so we do not assume they are possible..
            break;

        if (op != BAM_CHARD_CLIP && op != BAM_CSOFT_CLIP)
            tail = 1;

        if (constype & CONSUME_QRY)
        {
            if (!tail) *q_beg += oplen;
            *q_end += oplen;
        }
        if (constype & CONSUME_REF) 
        {
            if (!tail) *r_beg += oplen;
            *r_end += oplen;
        }
        allen += oplen;
    }
#endif
    if (!forward && q_beg && q_end)
    {
        unsigned tmp = qry_len - *q_beg;
        *q_beg = qry_len - *q_end;
        *q_end = tmp;
    }
    return allen;

}

void cigar_print (FILE* f, uint32_t* cigar, unsigned cigar_sz)
{
#ifdef CIGAR_ACCESS_REFACTORED
    uint32_t cigar_buf_sz = compute_cigar_strlen (cigar, cigar_sz);
    char* cigar_str_buf = (char*) alloca (cigar_buf_sz);
    uint32_t written = cigar_to_string (cigar, cigar_sz, cigar_str_buf, cigar_buf_sz);
    assert (written == cigar_buf_sz);
    fprintf (f, cigar);
#else
    uint32_t* sent;
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
        fprintf (f, "%d%c", count, schar);
    }
#endif
}
