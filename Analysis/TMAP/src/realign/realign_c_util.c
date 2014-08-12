/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include "realign_c_util.h"

#include <stdint.h>
#include "../samtools/bam.h"

#define CONSUME_QRY 1
#define CONSUME_REF 2

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
