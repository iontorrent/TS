/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REALIGN_WRAPPER_IMP_H
#define REALIGN_WRAPPER_IMP_H

#include "realign_proxy.h"
#include "Realign.h"

class RealignImp : public RealignProxy, protected Realigner
{
    CLIPTYPE cliptype_;

    // buffers for reference and inverse query sequences
    char* qry_buf_;
    unsigned qry_buf_len_;
    char* ref_buf_;
    unsigned ref_buf_len_;

    // clip zones 
    unsigned clip_beg_;
    unsigned clip_end_;

    unsigned len_roundup (unsigned len);

protected:
    RealignImp ();
    RealignImp (unsigned reserve_size, unsigned clipping_size);

public:
    ~RealignImp ();
    // general control
    void set_verbose (bool verbose);
    void set_debug (bool debug);
    void set_log (int posix_handle);
    bool invalid_input_cigar () const;

    // parameters setup
    void set_scores (int mat, int mis, int gip, int gep);
    void set_bandwidth (int bandwidth);
    void set_clipping (CLIPTYPE clipping);
    void set_gap_scale_mode (int) {}

    // alignment setup and run
    bool compute_alignment (const char* q_seq,
                            unsigned q_len,
                            const char* r_seq,
                            unsigned r_len,
                            int r_pos,
                            bool forward,
                            const uint32_t* cigar,
                            unsigned cigar_sz,
                            uint32_t*& cigar_dest,
                            unsigned& cigar_dest_sz,
                            int& new_pos,
                            bool& already_perfect,
                            bool& clip_failed,
                            bool& alignment_failed,
                            bool& unclip_failed);

    // resource management helpers
    char* qry_buf (unsigned len);
    char* ref_buf (unsigned len);

    friend RealignProxy* createRealigner (unsigned reserve_size, unsigned clipping_size);
    friend RealignProxy* createRealigner ();
    
};

#endif // REALIGN_WRAPPER_IMP_H
