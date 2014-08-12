/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include "realign_wrapper_imp.h"
#include "realign_util.h"
#include "realign_c_util.h"
#include "Realign.h"
#include "../util/tmap_alloc.h"

#include "../samtools/bam.h"

#include <cassert>


RealignImp::RealignImp ()
{
}
RealignImp::RealignImp (unsigned reserve_size, unsigned clipping_size)
:
Realigner (reserve_size, clipping_size),
cliptype_ (NO_CLIP),
qry_buf_ (NULL),
qry_buf_len_ (0),
ref_buf_ (NULL),
ref_buf_len_ (0)
{
}

RealignImp::~RealignImp ()
{
    if (qry_buf_) delete [] qry_buf_, qry_buf_ = NULL;
    if (ref_buf_) delete [] ref_buf_, ref_buf_ = NULL;
}

unsigned RealignImp::len_roundup (unsigned len)
{
    unsigned bb = (0x1u<<8);
    for (; bb != (0x1u<<31); bb <<= 1)
        if (bb > len)
            break;
    return bb;
}

char* RealignImp::qry_buf (unsigned len)
{
    if (!len)
        return NULL;
    if (len <= qry_buf_len_)
        return qry_buf_;
    qry_buf_len_ = len_roundup (len);
    if (qry_buf_)
        delete qry_buf_;
    // TODO: memory allocation error handling
    qry_buf_ = new char [qry_buf_len_];
    return qry_buf_;
}

char* RealignImp::ref_buf (unsigned len)
{
    if (!len)
        return NULL;
    if (len <= ref_buf_len_)
        return ref_buf_;
    ref_buf_len_ = len_roundup (len);
    if (ref_buf_)
        delete ref_buf_;
    // TODO: memory allocation error handling
    ref_buf_ = new char [ref_buf_len_];
    return ref_buf_;
}


void RealignImp::set_verbose (bool verbose)
{
    Realigner::verbose_ = verbose;
}

void RealignImp::set_debug (bool debug)
{
    Realigner::debug_ = debug;
}

bool RealignImp::invalid_input_cigar () const
{
    return Realigner::invalid_cigar_in_input;
}

// parameters setup
void RealignImp::set_scores (int mat, int mis, int gip, int gep)
{
    std::vector<int> scores (4);
    scores [0] = mat, scores [1] = mis, scores [2] = gip, scores [3] = gep;
    SetScores (scores);
}

void RealignImp::set_bandwidth (int bandwidth)
{
    SetAlignmentBandwidth (bandwidth);
}

void RealignImp::set_clipping (CLIPTYPE clipping)
{
    cliptype_ = clipping;
}

// alignment setup and run    

// prepare internal structures for clipping and alignment
// returns true if realignment was performed
bool RealignImp::compute_alignment (
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
    int& new_pos,
    bool& already_perfect,
    bool& clip_failed,
    bool& alignment_failed,
    bool& unclip_failed)
{
    already_perfect = false;
    alignment_failed = false;
    unclip_failed = false;
    unsigned oplen;
    
    const char* q_seq_clipped = q_seq;
    const uint32_t* cigar_clipped = cigar;
    unsigned cigar_sz_clipped = cigar_sz;

    unsigned sclip_q_len, sclip_r_len, sclip_al_len;
    
    assert (cigar_sz);
    // reset realigner
    Reset ();

    // set clipping 
    SetClipping ((int) cliptype_, forward);
    
    // clip out the hard and soft clipping zones from 5" and 3"
    // The 'cut out' of the q_seq is done by switching to downstream pointer.
    if (bam_cigar_op (*cigar) == BAM_CSOFT_CLIP)
    {
        oplen = bam_cigar_oplen (*cigar);
        ClipStart (oplen);
        q_seq_clipped += oplen;
        ++cigar_clipped;
        --cigar_sz_clipped;
    }
    
    if (cigar_sz > 1 && bam_cigar_op (cigar [cigar_sz - 1]) == BAM_CSOFT_CLIP)
    {
        oplen = bam_cigar_oplen (cigar [cigar_sz - 1]);
        ClipEnd (oplen);
        --cigar_sz_clipped;
    }

    // cigar defines q_seq and t_seq lengths
    sclip_al_len = seq_lens_from_bin_cigar (cigar_clipped, cigar_sz_clipped, &sclip_q_len, &sclip_r_len);
    
    const std::string query (q_seq_clipped, sclip_q_len);
    const std::string target (r_seq, sclip_r_len);
    std::string pretty_al; pretty_al.reserve (sclip_al_len);
    
    pretty_al_from_bin_cigar (cigar_clipped, cigar_sz_clipped, q_seq_clipped, r_seq, pretty_al);
    
    // Realigner requires strings of proper size to be passed to SetSequences
    SetSequences (query, target, pretty_al, forward);

    if (!ClipAnchors (clip_failed))
    {
        already_perfect = true;
        return false; // alignment already good, no imperfect zone to realign found
    }

    // TODO avoid automatic vectors to prevent unneeded heap usage
    vector<MDelement> new_md_vec; 
    vector<CigarOp> new_cigar_vec;
    unsigned int start_pos_shift;
    
    if (!computeSWalignment(new_cigar_vec, new_md_vec, start_pos_shift))
    {
        alignment_failed = true;
        return false;
    }
    
    if (!addClippedBasesToTags(new_cigar_vec, new_md_vec, q_len))
    {
        unclip_failed = true;
        return false; // error adding back clipped out zones
    }
        
    if (!LeftAnchorClipped () && start_pos_shift != 0) 
    {
        // build cigar data only if it is needed
        // TODO avoid automatic vectors to prevent unneeded heap usage
        std::vector <CigarOp> cigar_vec;
        cigar_vector_from_bin (cigar, cigar_sz, cigar_vec);
        new_pos = updateReadPosition (cigar_vec, start_pos_shift, r_pos);
    }
    else
        new_pos = r_pos;
    
    // free (cigar_dest);
    // TODO: switch to better alignment memory management, avoid heap operations
    cigar_dest = (uint32_t*) tmap_malloc (sizeof (uint32_t) * new_cigar_vec.size (), "cigar_dest");
    cigar_dest_sz = new_cigar_vec.size ();
    cigar_vector_to_bin (new_cigar_vec, cigar_dest);
    
    return true;
}




