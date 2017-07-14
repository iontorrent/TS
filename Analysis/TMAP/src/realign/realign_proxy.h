/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REALIGN_PROXY_H
#define REALIGN_PROXY_H

// Defines abstract interfaces to SW realignment functionality 
// operating with the TMAP data structures

#include <stdint.h>
#include <ostream>
#include "realign_cliptype.h"

class RealignProxy
{
public:
    virtual ~RealignProxy ();

    // general setup
    virtual void set_verbose (bool verbose) = 0;
    virtual void set_debug (bool debug) = 0;
    virtual void set_log (int posix_handle) = 0;
    virtual bool invalid_input_cigar () const = 0;

    // parameters setup
    virtual void set_scores (double mat, double mis, double gip, double gep) = 0;
    virtual void set_bandwidth (int bandwidth) = 0;
    virtual void set_clipping (CLIPTYPE clipping) = 0;
    virtual void set_gap_scale_mode (int gap_scale_mode) = 0;

    // alignment setup and run
    virtual bool compute_alignment (const char* q_seq,
                                    unsigned q_len,
                                    const char* r_seq, 
                                    unsigned r_len,
                                    int r_pos, 
                                    bool forward, 
                                    const uint32_t* cigar, 
                                    unsigned cigar_sz, 
                                    uint32_t*& cigar_dest, 
                                    unsigned& cigar_dest_sz, 
                                    unsigned& new_r_pos,
                                    unsigned& new_q_pos,
                                    unsigned& new_r_len,
                                    unsigned& new_q_len,
                                    bool& already_perfect,
                                    bool& clip_failed,
                                    bool& alignment_failed,
                                    bool& unclip_failed) = 0;

    // resource management helpers
    virtual char* qry_buf (unsigned len) = 0;
    virtual char* ref_buf (unsigned len) = 0;
};

RealignProxy* createRealigner ();
RealignProxy* createRealigner (unsigned reserve_size, unsigned clipping_size);

RealignProxy* createContextAligner ();


#endif // REALIGN_PROXY_H
