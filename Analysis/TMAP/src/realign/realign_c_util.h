/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REALIGN_C_UTIL_H
#define REALIGN_C_UTIL_H

#include <stdint.h>

#if defined (__cplusplus)
#define LANGSPEC extern "C"
#define CONSTSPEC const
#else
#define LANGSPEC
#define CONSTSPEC
#endif


LANGSPEC unsigned seq_lens_from_bin_cigar (CONSTSPEC uint32_t* cigar_bin, unsigned cigar_bin_sz, unsigned* q_len, unsigned* r_len);

#endif // REALIGN_C_UTIL_H
