/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REALIGN_C_UTIL_H
#define REALIGN_C_UTIL_H

#include <stdint.h>
#include <stdio.h>

#if defined (__cplusplus)
#define LANGSPEC extern "C"
#define CONSTSPEC const
#else
#define LANGSPEC
#define CONSTSPEC
#endif

#define CONSUME_QRY 1
#define CONSUME_REF 2

// computes full alignment length and footprints on reference and query, including clips
LANGSPEC unsigned seq_lens_from_bin_cigar (CONSTSPEC uint32_t* cigar_bin, unsigned cigar_bin_sz, unsigned* q_len, unsigned* r_len);

// computes alignment length and reference/query footprints excluding tail clips
LANGSPEC unsigned alignment_bounds_from_bin_cigar (CONSTSPEC uint32_t* cigar_bin, unsigned cigar_bin_sz, uint8_t forward, unsigned qry_len, unsigned* q_beg, unsigned* q_end, unsigned* r_beg, unsigned* r_end);

void cigar_print (FILE* f, uint32_t* cigar, unsigned cigar_len);


#endif // REALIGN_C_UTIL_H
