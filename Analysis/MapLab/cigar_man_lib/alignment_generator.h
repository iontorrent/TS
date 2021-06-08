#ifndef __alignment_generator_h__
#define __alignment_generator_h__

#include <string>
#include <deque>
#include <vector>
#include <numeric>
#include <samtools/bam.h>
#include <TMAP/src/util/tmap_cigar_util.h>
#include <TMAP/src/map/util/tmap_map_align_util.h>


// this class intended to generate square matrices of overlapping/nested alignments
class AlignmentGenerator
{
public:
    class Parameters
    {
    public:
        // parameter defaults
        static const bool   def_lclip;
        static const bool   def_rclip;
        static const double def_gc_freq;
        static const double def_gc_skew;
        static const double def_at_skew;
        static const double def_gap_freq;
        static const double def_ins_frac;
        static const double def_mism_frac;
        static uint32_t (*const def_gap_size_generator) ();

        // generation parameters
        bool   lclip;   // generate left clip (using gap zize generator)
        bool   rclip;   // generate left clip (using gap size generator)
        double gc_freq; // fraction of G+C bases
        double gc_skew; // fraction of Gs in G/C bases (strand G skew)
        double at_skew; // fraction of As in A/T bases (strand A skew)
        double gap_freq; // the frequency of gaps (average segment length between gaps)^-1
        double ins_frac; // ins/ins+del
        double mism_frac; // frequency of mismatches
        uint32_t (*gap_size_generator) (); // generator for a gap size using the chosen distribution
        // constructor (for convenience)
        Parameters (bool lclip = def_lclip, bool rclip = def_rclip, double gc_freq = def_gc_freq, double gc_skew = def_gc_skew, double at_skew = def_at_skew, double gap_freq = def_gap_freq, double ins_frac = def_ins_frac, double mism_frac = def_mism_frac, uint32_t (*gap_size_generator) () = def_gap_size_generator);
    };
private:
    const Parameters* gen_param_;
    const tmap_sw_param_t* sw_param_;
    static const Parameters default_gen_param;
    // structures to hold generated data
    std::deque<uint32_t> cigar_buf;
    size_t tot_steps;
    std::string qseq_buf;
    std::string rseq_buf;
    tmap_map_alignment_stats stats_;

    void incr_qry (unsigned baseno = 1);
    void incr_ref (unsigned baseno = 1);
    void incr_both (unsigned baseno = 1);
    char add_qry_base ();
    char add_ref_base ();
    char next_base ();


public:
    AlignmentGenerator (const AlignmentGenerator::Parameters* gen_param = &default_gen_param, const tmap_sw_param_t* sw_param = NULL);
    AlignmentGenerator (const tmap_sw_param_t* sw_param);

    void advance_head (bool last = false); // generates one additional cigar operation at the head. If last is given and right softclip is allowed, it is added
    void reduce_tail (); // removes one cigar operation from the tail

    const tmap_sw_param_t* sw_param () const { return sw_param_; }
    void sw_param (const tmap_sw_param_t* sw_param) { sw_param_ = sw_param;}
    const Parameters* gen_param () const { return gen_param_; }
    void gen_param (const Parameters* gen_param) { gen_param_ = gen_param ? gen_param : (&default_gen_param); }

    // current state
    size_t q_size () const { return qseq_buf.length (); }
    size_t r_size () const { return rseq_buf.length (); }
    size_t opno   () const { return cigar_buf.size (); }
    size_t stepno () const { return tot_steps; }
    const tmap_map_alignment_stats& stats () const { return stats_; }

    // capture the alignment at current state. The actual data is copied into passed in containers, alignment is filled with the pointers to the data held in that containers
    // NOTE both std::vector and std::string guarantee continuous storage, so they can be used as destination.
    //      the data pointers can be retrieved as (const char* std::string::c_str ()) and (const uint32_t* (&std::vectpor::operator [] (0)))
    void capture (tmap_map_alignment& alignment, std::string& qseq, std::string& rseq, std::vector<uint32_t>& cigar) const;
};

#endif // __alignment_generator_h__

