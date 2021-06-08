/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TMAP_CIGAR_UTIL_H
#define TMAP_CIGAR_UTIL_H

#include <stdint.h>
#include <stddef.h>

#if defined (__cplusplus)
extern "C"
{
#endif 

typedef struct
{
    unsigned xpos;
    unsigned ypos;
    unsigned len;
} 
AlBatch;

#define CIGAR_CONSUME_QRY 1
#define CIGAR_CONSUME_REF 2

// computes number of decimal positions the decimal representation if argument takes
uint32_t decimal_positions (uint32_t value);

// computes byte length needed to accommodate cigar string for the given binary cigar
// returns number of bytes needed (less terminating zero)
uint32_t compute_cigar_strlen
(
    const uint32_t* cigar,
    size_t cigar_sz
);

// computes and returns number of cigar operations encoded in cigar string
// assumes cigar string is properly structured, dose very minimal validation
// on error, returns UINT32_MAX
uint32_t compute_cigar_bin_size
(
    const char* cigar
);

// converts binary cigar to string; writes to buffer, including terminating zero
// up to bufsz characters are written (including terminating zero)
// returns number of characters written, excluding terminating zero.
// if not enogh space given to write entire cigae, the buffer contains incomplete cigar string, zero terminated
// in latter case if space allows the ellipsis is appended.
uint32_t cigar_to_string
(
    const uint32_t* cigar,
    size_t cigar_sz,
    char* buffer,
    size_t bufsz
);


// converts cigar string to binary cigar
// returns binary cigar size
uint32_t string_to_cigar
(
    const char* cigar_str,
    uint32_t* buf,
    size_t bufsz
);

// gets adjacent position in cigar
// modifies *op_idx, *op_off
// can adjust positions to sentinels: 0:-1 in reverse, ncigar-1:bam_cigar_oplen (cigar [ncigar-1] in forward direction
// returns 1 on success, 0 if adjacent position in given direction is not available (goes past centinel)
uint8_t next_cigar_pos
(
    const uint32_t* cigar,
    uint32_t ncigar,
    int32_t* op_idx,
    int32_t* op_off,
    int8_t increment // 1 or -1
);


// computes the associated lengths from he cigar string.
// returns them by writing to provided addresses
// ignores NULL addresses 
// returns 1 if no cigar structure errors were encountered, 0 otherwise
// (note: does not perform comprehensive validation, only some malformations are detected)
uint8_t cigar_footprint
(
    const uint32_t* cigar_bin,
    uint32_t cigar_bin_len,
    uint32_t* qlen,
    uint32_t* rlen,
    uint32_t* allen,
    uint32_t* left_clip,
    uint32_t* right_clip
);


#if defined (__cplusplus)
}
#endif 

#endif // TMAP_CIGAR_UTIL_H

