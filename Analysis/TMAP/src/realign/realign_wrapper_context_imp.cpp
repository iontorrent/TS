/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#include "realign_wrapper_context_imp.h"

#include "realign_util.h"
#include "realign_c_util.h"
#include "cigar_utils.h"
#include "../util/tmap_alloc.h"
#include "../samtools/bam.h"

#include <iostream>
#include <cassert>

static const unsigned EXTRA_BAND_DEFAULT = 5;
static const unsigned MAX_SEQ_LEN = 1000;
static const unsigned MAX_BWID = 60;
static const int DEF_GIP = 5, DEF_GEP = 1, DEF_MAT = 1, DEX_MIS = 2;

ContAlignImp::ContAlignImp ()
:
qry_buf_ (NULL),
qry_buf_len_ (0),
ref_buf_ (NULL),
ref_buf_len_ (0),
extra_bandwidth_ (EXTRA_BAND_DEFAULT)
{
    init (MAX_SEQ_LEN, MAX_SEQ_LEN, MAX_SEQ_LEN*MAX_BWID, DEF_GIP, DEF_GEP, DEF_MAT, DEX_MIS);
}

ContAlignImp::~ContAlignImp ()
{
    if (qry_buf_) delete [] qry_buf_, qry_buf_ = NULL;
    if (ref_buf_) delete [] ref_buf_, ref_buf_ = NULL;
}

unsigned ContAlignImp::len_roundup (unsigned len)
{
    unsigned bb = (0x1u<<8);
    for (; bb != (0x1u<<31); bb <<= 1)
        if (bb > len)
            break;
    return bb;
}

char* ContAlignImp::qry_buf (unsigned len)
{
    if (!len)
        return NULL;
    if (len <= qry_buf_len_)
        return qry_buf_;
    qry_buf_len_ = len_roundup (len);
    if (qry_buf_)
        delete [] qry_buf_;
    // TODO: memory allocation error handling
    qry_buf_ = new char [qry_buf_len_];
    return qry_buf_;
}

char* ContAlignImp::ref_buf (unsigned len)
{
    if (!len)
        return NULL;
    if (len <= ref_buf_len_)
        return ref_buf_;
    ref_buf_len_ = len_roundup (len);
    if (ref_buf_)
        delete [] ref_buf_;
    // TODO: memory allocation error handling
    ref_buf_ = new char [ref_buf_len_];
    return ref_buf_;
}


void ContAlignImp::set_verbose (bool)
{
}

void ContAlignImp::set_debug (bool debug)
{
}

void ContAlignImp::set_log (int posix_handle)
{
    if (posix_handle >= 0)
        ContAlign::set_log (posix_handle);
    else
        ContAlign::reset_log ();
}

bool ContAlignImp::invalid_input_cigar () const
{
    return invalid_cigar_in_input_;
}

// parameters setup
void ContAlignImp::set_scores (double mat, double mis, double gip, double gep)
{
    ContAlign::set_scoring (gip, gep, mat, mis);
}

void ContAlignImp::set_bandwidth (int bandwidth)
{
    extra_bandwidth_ = bandwidth;
}

void ContAlignImp::set_clipping (CLIPTYPE)
{
}

void ContAlignImp::set_gap_scale_mode (int gap_scale_mode)
{
    ContAlign::set_scale ((ContAlign::SCALE_TYPE) gap_scale_mode);
}
// alignment setup and run    

// prepare internal structures for clipping and alignment
// returns true if realignment was performed
bool ContAlignImp::compute_alignment (
    const char* q_seq,
    unsigned q_len,
    const char* r_seq, 
    unsigned r_len,
    int r_pos, 
    bool forward, 
    const uint32_t* cigar, 
    unsigned cigar_sz, 
    uint32_t*& cigar_dest, 
    unsigned& cigar_dest_sz, 
    unsigned& new_ref_pos,
    unsigned& new_qry_pos,
    unsigned& new_ref_len,
    unsigned& new_qry_len,
    bool& already_perfect,
    bool& clip_failed,
    bool& alignment_failed,
    bool& unclip_failed)
{
    already_perfect = false;
    alignment_failed = false;
    unclip_failed = false;
    // unsigned oplen;

    //const char* q_seq_clipped = q_seq;
    //const uint32_t* cigar_clipped = cigar;
    //unsigned cigar_sz_clipped = cigar_sz;

    //unsigned sclip_q_len, sclip_r_len, sclip_al_len;

    assert (cigar_sz);

    // clip out the hard and soft clipping zones from 5" and 3"
    // The 'cut out' of the q_seq is done by switching to downstream pointer.
    uint32_t clean_len;
    EndClips clips;
    const char* clean_read = clip_seq (q_seq, cigar, cigar_sz, clean_len, clips);
    // clip reference accordingly
    //r_seq += clips.soft_beg_;
    //r_len -= clips.soft_beg_ + clips.soft_end_;

    if (clean_len > MAX_SEQ_LEN || r_len > MAX_SEQ_LEN)
    {
        // std::cerr << "Sequence is too long to fit into aligner (" << std::max (clean_len, r_len) << ", max is " << MAX_SEQ_LEN << ")" << std::endl;
#if 0
        std::cerr << "Cigar is ";
        cigar_out (std::cerr, cigar, cigar_sz);
        std::cerr << "\nClips: " << clips << std::endl;
#endif
        return false;
    }

    unsigned qry_ins; // extra bases in query     == width_left
    unsigned ref_ins; // extra bases in reference == width_right
    band_width (cigar, cigar_sz, qry_ins, ref_ins);

    if (!ContAlign::can_align (clean_len, r_len, qry_ins + extra_bandwidth_, ref_ins + extra_bandwidth_))
        return false;

    ContAlign::align_band (
        clean_read,                     // xseq
        clean_len,                      // xlen
        r_seq,                          // yseq
        r_len,                          // ylen
        0,                              // xpos
        0,                              // ypos
        std::max (clean_len, r_len),    // segment length
        qry_ins + extra_bandwidth_,     // width_left
        ref_ins + extra_bandwidth_,     // width_right - forces to width_left
        true,                           // to_beg
        true                            // to_end
        );
    unsigned bno = ContAlign::backtrace   
        (
            batches_,      // BATCH buffer
            MAX_BATCH_NO,  // size of BATCH buffer
            ref_ins + extra_bandwidth_        // width
        );
    // convert alignment to cigar
    unsigned ref_off;
    // int ref_shift = 
    roll_cigar (new_cigar_, MAX_CIGAR_SZ, cigar_dest_sz, batches_, bno, clean_len, clips, new_qry_pos, ref_off, new_qry_len, new_ref_len);
    new_ref_pos = r_pos + ref_off;
    cigar_dest = new_cigar_;

    return true;
}
