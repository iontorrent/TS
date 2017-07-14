/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#include "realign_wrapper_imp.h"
#include "realign_util.h"
#include "realign_c_util.h"
#include "../util/tmap_alloc.h"

#include "Realign.h"

#include "../samtools/bam.h"

#include <cassert>


RealignImp::RealignImp ()
:
Realigner (),
cliptype_ (NO_CLIP),
qry_buf_ (NULL),
qry_buf_len_ (0),
ref_buf_ (NULL),
ref_buf_len_ (0)
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
        delete [] qry_buf_;
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
        delete [] ref_buf_;
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

void RealignImp::set_log (int)
{
}

bool RealignImp::invalid_input_cigar () const
{
    return Realigner::invalid_cigar_in_input;
}

// parameters setup
void RealignImp::set_scores (double mat, double mis, double gip, double gep)
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

    if (!computeSWalignment (new_cigar_vec, new_md_vec, start_pos_shift))
    {
        alignment_failed = true;
        return false;
    }

    if (!addClippedBasesToTags (new_cigar_vec, new_md_vec, q_len))
    {
        unclip_failed = true;
        return false; // error adding back clipped out zones
    }

    if (!LeftAnchorClipped () && start_pos_shift != 0) 
    {
        // build cigar data only if it is needed
        // TODO avoid automatic vectors to prevent unneeded heap usage
        CigarVec cigar_vec;
        cigar_vector_from_bin (cigar, cigar_sz, cigar_vec);
        new_ref_pos = updateReadPosition (cigar_vec, start_pos_shift, r_pos);
    }
    else
        new_ref_pos = r_pos;

    // here we need to adjust for the possible weirdness of Realigner:
    // it may produce alignments starting / ending with indels.
    // They are to be converted to softclips or to ref shifts.
    // Namely:
    //    insertions at the start and at the end should be added to softclips
    //    deletions at the starts should be added to new_ref_pos
    //    deletions at the end should be subtracted from new_ref_len
    // we do processing here in order to avoid touching code inherited from Bamrealignment (can be re-considered later)
    // the 
    adjust_cigar (new_cigar_vec, new_ref_pos, new_ref_pos, new_qry_pos, new_ref_len, new_qry_len);

    // free (cigar_dest);
    // TODO: switch to better alignment memory management, avoid heap operations
    cigar_dest = (uint32_t*) tmap_malloc (sizeof (uint32_t) * new_cigar_vec.size (), "cigar_dest");
    cigar_dest_sz = new_cigar_vec.size ();
    cigar_vector_to_bin (new_cigar_vec, cigar_dest);

    return true;
}


void adjust_cigar (CigarVec& src, unsigned start_pos_shift, unsigned& new_ref_pos, unsigned& new_qry_pos, unsigned& new_ref_len, unsigned& new_qry_len)
{
    // walk from front to first aligned pair of bases. Count softclips and Is as softclip, Ds as reference shifts
    // it is safe to reset new_ref_pos since even if same var is passed as start_pos_shif and new_ref_pos, the former is copied on stack.
    new_ref_pos = new_qry_pos = new_ref_len = new_qry_len = 0;
    if (!src.size ())
    {
        new_ref_pos = start_pos_shift;
        return;
    }
    // walks toward the end to compute ref and lengths
    bool in_prefix = true;
    unsigned last_prefix_idx = 0;
    for (CigarVec::iterator ci = src.begin (); ci != src.end (); ++ci)
    {
        switch (ci->Type)
        {
            case 'M':  // aligned
            case 'X':  // mismatch
            case '=':  // match
                // both ref and qry consumed
                new_ref_len += ci->Length, new_qry_len += ci->Length;
                in_prefix = false; // stops accumulating left flank
                break;
            case 'I':  // insert
            case 'S':  // softclip
                // qry consumed, ref not
                if (in_prefix)
                    new_qry_pos += ci->Length, ++last_prefix_idx;
                else
                    new_qry_len += ci->Length;
                break;
            case 'D':  // delete
            case 'N':  // refskip
                // ref consumed, qry not
                if (in_prefix)
                    new_ref_pos += ci->Length, ++last_prefix_idx;
                else
                    new_ref_len += ci->Length;
                break;
            default:   // hardclip and pad
                // none consumed
                if (in_prefix)
                    ++last_prefix_idx;
                break;
        }
    }
    // now figure out the suffix - walk backward
    bool in_suffix = true;
    unsigned first_suffix_idx = src.size ();
    for (CigarVec::reverse_iterator ri = src.rbegin (); in_suffix && ri != src.rend (); ++ri)
    {
        switch (ri->Type)
        {
            case 'M':  // aligned
            case 'X':  // mismatch
            case '=':  // match
                // both ref and qry consumed
                in_suffix = false; // stops accumulating left flank
                break;
            case 'I':  // insert
            case 'S':  // softclip
                // qry consumed, ref not
                assert (new_qry_len >= ri->Length);
                assert (first_suffix_idx);
                new_qry_len -= ri->Length;
                --first_suffix_idx;
                break;
            case 'D':  // delete
            case 'N':  // refskip
                // ref consumed, qry not
                assert (new_ref_len >= ri->Length);
                assert (first_suffix_idx);
                new_ref_len -= ri->Length;
                --first_suffix_idx;
                break;
            default:   // hardclip and pad
                // none consumed
                assert (first_suffix_idx);
                --first_suffix_idx;
                break;
        }
    }
    assert (last_prefix_idx <= first_suffix_idx);
    // take the suffix out
    src.resize (first_suffix_idx);
    // replace prefix with just one softclip (assuming no hard clip encountered - TMAP does not produce one
    if (last_prefix_idx)
    {
        src.erase (src.begin (), src.begin () + (last_prefix_idx - 1)); // leave one position
        src.front ().Type = 'S';
        src.front ().Length = new_qry_pos;
    }
    // add prior start offset
    new_ref_pos += start_pos_shift;
}


