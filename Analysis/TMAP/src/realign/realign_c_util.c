/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#include "realign_c_util.h"

#include <stdint.h>
#include "../samtools/bam.h"

unsigned seq_lens_from_bin_cigar (uint32_t* cigar_bin, unsigned cigar_bin_sz, unsigned* q_len, unsigned* r_len)
{
    unsigned oplen, constype;
    uint32_t *sent;
    *q_len = *r_len = 0;
    unsigned allen = 0;
    for (sent = cigar_bin + cigar_bin_sz; cigar_bin != sent; ++cigar_bin)
    {
        oplen = bam_cigar_oplen (*cigar_bin);
        constype = bam_cigar_type (*cigar_bin);
        if (constype & CONSUME_QRY) *q_len += oplen;
        if (constype & CONSUME_REF) *r_len += oplen;
        allen += oplen;
    }
    return allen;
}

unsigned alignment_bounds_from_bin_cigar (uint32_t* cigar_bin, unsigned cigar_bin_sz, uint8_t forward, unsigned qry_len, unsigned* q_beg, unsigned* q_end, unsigned* r_beg, unsigned* r_end)
{
    unsigned oplen, op, constype;
    uint32_t *sent;
    *q_beg = *q_end = *r_beg = *r_end = 0;
    unsigned allen = 0;
    uint32_t tail = 0;
    for (sent = cigar_bin + cigar_bin_sz; cigar_bin != sent; ++cigar_bin)
    {
        oplen = bam_cigar_oplen (*cigar_bin);
        op = bam_cigar_op (*cigar_bin);
        constype = bam_cigar_type (*cigar_bin);

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
    if (!forward)
    {
        unsigned tmp = qry_len - *q_beg;
        *q_beg = qry_len - *q_end;
        *q_end = tmp;
    }
    return allen;
}

void cigar_print (FILE* f, uint32_t* cigar, unsigned cigar_sz)
{
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
}
