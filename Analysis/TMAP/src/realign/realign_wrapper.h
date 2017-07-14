/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REALIGN_WRAPPER_H
#define REALIGN_WRAPPER_H

// Provides C interface to the functionality of the RealignProxy class
#include <stdint.h>
#include "realign_cliptype.h"


#if defined (__cplusplus)
extern "C"
{
#endif

struct RealignProxy;

struct RealignProxy* realigner_create ();
struct RealignProxy* realigner_create_spec (unsigned reserve_size, unsigned clipping_size);

struct RealignProxy* context_aligner_create ();

void realigner_destroy (struct RealignProxy* r);

void realigner_set_verbose (struct RealignProxy* r, uint8_t verbose);
void realigner_set_debug (struct RealignProxy* r, uint8_t debug);
uint8_t realigner_invalid_input_cigar (struct RealignProxy* r);
void realigner_set_log (struct RealignProxy* r, int posix_file_handle);

// parameters setup
void realigner_set_scores (struct RealignProxy* r, double mat, double mis, double gip, double gep);
void realigner_set_bandwidth (struct RealignProxy* r, int bandwidth);
void realigner_set_clipping (struct RealignProxy* r, enum CLIPTYPE clipping);
void realigner_set_gap_scale_mode (struct RealignProxy* r, int scale_mode);

// alignment setup and run
uint8_t realigner_compute_alignment (
    struct RealignProxy* r, 
    const char* q_seq, 
    uint32_t q_len,
    const char* r_seq, 
    uint32_t r_len,
    int r_pos, 
    uint8_t forward, 
    const uint32_t* cigar, 
    unsigned cigar_sz, 
    uint32_t** cigar_dest, 
    unsigned* cigar_dest_sz, 
    unsigned* new_r_pos,
    unsigned* new_q_pos,
    unsigned* new_r_len,
    unsigned* new_q_len,
    uint64_t* num_realign_already_perfect,
    uint64_t* num_realign_not_clipped,
    uint64_t* num_realign_sw_failures,
    uint64_t* num_realign_unclip_failures);

// memory management helpers
char* qry_mem (struct RealignProxy* r, unsigned len);
char* ref_mem (struct RealignProxy* r, unsigned len);



#if defined (__cplusplus)  // closure of the 'extern "C"' scope
}
#endif


#endif // REALIGN_WRAPPER_H
