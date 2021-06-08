#include <cstdlib>
#include "alignment_generator.h"
#include "TMAP/src/util/tmap_definitions.h"
#include <cassert>

static unsigned simple_gap_size_freqs [] = {0, 3, 3, 10, 5, 5, 10, 10, 10, 8, 6, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // value for size 0 is ignored
static unsigned prepare_simple_gap_size_distr ()
{
    return std::accumulate (simple_gap_size_freqs, simple_gap_size_freqs + sizeof (simple_gap_size_freqs) / sizeof (*simple_gap_size_freqs), 0);
}
static unsigned simple_distr_sum = prepare_simple_gap_size_distr ();
static uint32_t simple_gap_size_generator ()
{
    uint32_t gs = 0;
    unsigned bound = unsigned (rand () * (double (simple_distr_sum) / double (RAND_MAX)));
    for (unsigned acc = 0; acc < bound && gs != sizeof (simple_gap_size_freqs) / sizeof (*simple_gap_size_freqs); acc += simple_gap_size_freqs [++gs]);
    assert (gs <= sizeof (simple_gap_size_freqs) / sizeof (*simple_gap_size_freqs));
    return gs;
}

const bool   AlignmentGenerator::Parameters::def_lclip     = false;
const bool   AlignmentGenerator::Parameters::def_rclip     = true;
const double AlignmentGenerator::Parameters::def_gc_freq   = .5;
const double AlignmentGenerator::Parameters::def_gc_skew   = .5;
const double AlignmentGenerator::Parameters::def_at_skew   = .5;
const double AlignmentGenerator::Parameters::def_gap_freq  = .04; // 1 per 25 bases
const double AlignmentGenerator::Parameters::def_ins_frac  = .5;
const double AlignmentGenerator::Parameters::def_mism_frac = .05; // 1 per 20 bases

uint32_t (*const AlignmentGenerator::Parameters::def_gap_size_generator) () = simple_gap_size_generator;

const AlignmentGenerator::Parameters AlignmentGenerator::default_gen_param;


AlignmentGenerator::Parameters::Parameters (bool lclip, bool rclip, double gc_freq, double gc_skew, double at_skew, double gap_freq, double ins_frac, double mism_frac, uint32_t (*gap_size_generator) ())
:
lclip (lclip),
rclip (rclip),
gc_freq (gc_freq),
gc_skew (gc_skew),
at_skew (at_skew),
gap_freq (gap_freq),
ins_frac (ins_frac),
mism_frac (mism_frac),
gap_size_generator (gap_size_generator)
{
}

AlignmentGenerator::AlignmentGenerator (const AlignmentGenerator::Parameters* gen_param, const tmap_sw_param_t* sw_param)
:
gen_param_ (gen_param),
sw_param_ (sw_param),
tot_steps (0)
{
    init_alignment_stats (&stats_);
}

AlignmentGenerator::AlignmentGenerator (const tmap_sw_param_t* sw_param)
:
gen_param_ (&default_gen_param),
sw_param_ (sw_param),
tot_steps (0)
{
    init_alignment_stats (&stats_);
}


void AlignmentGenerator::incr_qry (unsigned baseno)
{
    // generate sequence according to parameters
    while (baseno--)
    {
        add_qry_base ();
    }
}

char AlignmentGenerator::next_base ()
{
    char base;
    bool base_gc = (rand () < RAND_MAX * gen_param_->gc_freq);
    if (base_gc)
        base = (rand () < RAND_MAX * gen_param_->gc_skew) ? 'G' : 'C';
    else
        base = (rand () < RAND_MAX * gen_param_->at_skew) ? 'A' : 'T';
    return base;
}
char AlignmentGenerator::add_qry_base ()
{
    char base = next_base ();
    qseq_buf.push_back (base);
    return base;
}

void AlignmentGenerator::incr_ref (unsigned baseno)
{
    // generate sequence according to parameters
    while (baseno--)
    {
        add_ref_base ();
    }
}

char AlignmentGenerator::add_ref_base ()
{
    char base = next_base ();
    rseq_buf.push_back (base);
    return base;
}

void AlignmentGenerator::incr_both (unsigned baseno)
{
    while (baseno--)
    {
        char q_base = add_qry_base ();
        bool use_same = (rand () >= gen_param_->mism_frac * RAND_MAX);
        char r_base = q_base;
        if (!use_same)
            while (r_base == q_base)
                r_base = next_base ();
        rseq_buf.push_back (r_base);
        if (sw_param_)
            stats_.score += sw_param_->matrix [sw_param_->row * tmap_iupac_char_to_int [(size_t) q_base] + tmap_iupac_char_to_int [(size_t) r_base]];
        if (q_base == r_base) ++stats_.matches;
        else ++stats_.mismatches;
    }
}

// generates one additional cigar operation
void AlignmentGenerator::advance_head (bool last)
{
    if ((!cigar_buf.size () && gen_param_->lclip) || last)
    {
        // update cigar 
        uint32_t clip_size = gen_param_->gap_size_generator ();
        cigar_buf.push_back (bam_cigar_gen (clip_size, BAM_CSOFT_CLIP));
        // update query
        incr_qry (clip_size);
    }
    else
    {
        // decide match vs gap: add gap after match, match after any non-match
        bool add_gap = (cigar_buf.size () && bam_cigar_op (cigar_buf.back ()) == BAM_CMATCH);
        if (add_gap)
        {
            // update cigar
            uint32_t cigar_op = (rand () < RAND_MAX * gen_param_->ins_frac)?BAM_CINS:BAM_CDEL;
            uint32_t gap_size = gen_param_->gap_size_generator ();
            assert (gap_size);
            cigar_buf.push_back (bam_cigar_gen (gap_size, cigar_op));
            // update sequences
            if (cigar_op == BAM_CINS)
                incr_qry (gap_size);
            else
                incr_ref (gap_size);
            ++ stats_.gapcnt;
            stats_.gaplen += gap_size;
            if (sw_param_)
                stats_.score -= sw_param_->gap_open + sw_param_->gap_ext * (gap_size - 1) + ((sw_param_->gap_end)?(sw_param_->gap_end):(sw_param_->gap_ext));
        }
        else // add match
        {
            do 
            {
                // update cigar 
                if (cigar_buf.size () && bam_cigar_op (cigar_buf.back ()) == BAM_CMATCH)
                    cigar_buf.back () = bam_cigar_gen (bam_cigar_oplen (cigar_buf.back ()) + 1, BAM_CMATCH);
                else
                    cigar_buf.push_back (bam_cigar_gen (1, BAM_CMATCH));
                // update sequences and score
                incr_both ();
            }
            while (rand () > RAND_MAX * gen_param_->gap_freq);
        }
    }
}
// removes one cigar operation from the tail
void AlignmentGenerator::reduce_tail ()
{
    uint32_t code = cigar_buf.front ();
    cigar_buf.pop_front ();
    uint32_t op = bam_cigar_op (code);
    uint32_t oplen = bam_cigar_oplen (code);
    uint32_t adv_type = bam_cigar_type (op);
    if (op == BAM_CMATCH)
    {
        for (size_t pos = 0; pos != oplen; ++pos)
        {
            char q_base = qseq_buf [pos];
            char r_base = rseq_buf [pos];
            if (sw_param_)
                stats_.score -= sw_param_->matrix [sw_param_->row * tmap_iupac_char_to_int [(size_t) q_base] + tmap_iupac_char_to_int [(size_t) r_base]];
            if (q_base == r_base) --stats_.matches;
            else --stats_.mismatches;
        }
    }
    else if (op == BAM_CINS || op == BAM_CDEL)
    {
        -- stats_.gapcnt;
        stats_.gaplen -= oplen;
        if (sw_param_)
            stats_.score += sw_param_->gap_open + sw_param_->gap_ext * (oplen - 1) + ((sw_param_->gap_end)?(sw_param_->gap_end):(sw_param_->gap_ext));
    }
    if (adv_type & CIGAR_CONSUME_QRY)
        qseq_buf.erase (0, oplen);
    if (adv_type & CIGAR_CONSUME_REF)
        rseq_buf.erase (0, oplen);
}

// capture the alignment at current state. The actual data is copied into passed in containers, alignment is filled with the pointers to the data held in that containers
void AlignmentGenerator::capture (tmap_map_alignment& alignment, std::string& qseq, std::string& rseq, std::vector<uint32_t>& cigar) const
{
    qseq = qseq_buf;
    rseq = rseq_buf;
    cigar.clear ();
    std::copy (cigar_buf.begin (), cigar_buf.end (), std::back_inserter (cigar));
    alignment.cigar = &(cigar [0]);
    alignment.ncigar = cigar.size ();
    alignment.qseq = qseq.c_str ();
    alignment.qseq_len = qseq.length ();
    alignment.qseq_off = 0;
    alignment.q_int = 0;
    alignment.rseq = rseq.c_str ();
    alignment.rseq_len = rseq.length ();
    alignment.rseq_off = 0;
    alignment.r_int = 0;
}
