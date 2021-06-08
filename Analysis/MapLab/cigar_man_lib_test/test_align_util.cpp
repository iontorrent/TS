#include "test_align_util.h"
#include "TMAP/src/util/tmap_cigar_util.h"
#include "TMAP/src/map/util/tmap_map_align_util.h"
#include "TMAP/src/util/tmap_definitions.h"
#include <samtools/bam.h>
#include "swmatrix_facet.h"
#include <cstdio>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <alloca.h>

bool TestAlign::init ()
{
    // create and initialize common facets
    add_facet (new SwMatrixFacet ());
    return true;
};

TestValidateAlignment::TestValidateAlignment ()
:
Test ("ValidateAlignment")
{
}

bool TestValidateAlignment::process ()
{
    const char* test_cigar = "10S10M10I10M10S"; // query_len = 50, ref_len = 20
    const char* q_seq =  "CCATGCAAATGCATGCAAATGCAAATTTCCCGATGCAAATGCATGCAAATGC";
    const char* r_seq = "GGGATGCAAATGCATGCAAATGC";

    const char* q_seq_short =  "CCATGCAAATGCATGCAAATGCAAATTTCCCGATGC";
    const char* r_seq_short = "GGGATGCAAATGCATGCA";

    size_t bin_cigar_sz = compute_cigar_bin_size (test_cigar);
    uint32_t bin_cigar [bin_cigar_sz];
    string_to_cigar (test_cigar, bin_cigar, bin_cigar_sz);

    tmap_map_alignment al;
    init_alignment (&al,
        bin_cigar,
        bin_cigar_sz,
        q_seq,
        2,
        r_seq,
        3 );
    TEST_ASSERT (validate_alignment (&al));

    al.qseq_len = 30;
    TEST_ASSERT (validate_alignment (&al) == 0);

    al.qseq_len = 52;
    al.rseq_len = 19;
    TEST_ASSERT (validate_alignment (&al) == 0);
    return 1;
}


TestNormalizeSegment::TestNormalizeSegment ()
:
Test ("NormalizeSegment")
{
}

bool TestNormalizeSegment::process ()
{
    // const char* test_cigar = "3M";
    const char* test_cigar = "10S10M10I10M10S"; // query_len = 50, ref_len = 20
    const char* q_seq =  "CCATGCAAATGCATGCAAATGCAAATTTCCCGATGCAAATGCATGCAAATGC";
    const char* r_seq = "GGGATGCAAATGCATGCAAATGC";

    size_t bin_cigar_sz = compute_cigar_bin_size (test_cigar);
    uint32_t bin_cigar [bin_cigar_sz];
    string_to_cigar (test_cigar, bin_cigar, bin_cigar_sz);

    tmap_map_alignment al;
    init_alignment (&al,
        bin_cigar,
        bin_cigar_sz,
        q_seq,
        2,
        r_seq,
        3 );

    tmap_map_alignment_segment seg;
    seg.alignment = &al;
    init_segment (&seg);

    tmap_map_alignment_segment ref_seg = seg;

    // make footprintds underdefined
    seg.q_start = seg.q_end = seg.r_start = seg.r_end = -1;

    TEST_ASSERT (tmap_map_normalize_alignment_segment (&seg));

    TEST_ASSERT (segments_match (&ref_seg,  &seg));

    seg = ref_seg;
    // make bounds  underdefined
    seg.first_op = seg.last_op = seg.first_op_off = seg.sent_op_off = -1;

    TEST_ASSERT (tmap_map_normalize_alignment_segment (&seg));

    TEST_ASSERT (segments_match (&ref_seg,  &seg));

    return 1;
}

TestSegmentScore::TestSegmentScore ()
:
Test ("SegmentScore")
{
}

bool TestSegmentScore::process ()
{
    const char* cigar = "5M2D5M"; // score 5+5-5-2*2=1
    const char* ref = "ATGCATCGAGCT";
    const char* qry = "ATGCAGAGCT";

    size_t bin_cigar_sz = compute_cigar_bin_size (cigar);
    uint32_t bin_cigar [bin_cigar_sz];
    string_to_cigar (cigar, bin_cigar, bin_cigar_sz);


    SwMatrixFacet* sw = dynamic_cast<SwMatrixFacet*> (find_facet ("SwMatrix"));
    assert (sw);
    const tmap_sw_param_t* swpar = sw->sw_param ();

    tmap_map_alignment al;
    init_alignment (&al,
                    bin_cigar,
                    bin_cigar_sz,
                    qry,
                    0,
                    ref,
                    0);

    tmap_map_alignment_segment seg;
    seg.alignment = &al;
    init_segment (&seg);

    tmap_map_alignment_stats result_stats;

    int32_t score = tmap_map_alignment_segment_score (&seg, swpar, &result_stats);

    TEST_ASSERTX (score == 1, "score is %d", score);
    TEST_ASSERTX (result_stats.matches == 10, "result_stats.matches is %d", result_stats.matches);
    TEST_ASSERTX (result_stats.mismatches == 0, "result_stats.mismatches is %d", result_stats.mismatches);
    TEST_ASSERTX (result_stats.gapcnt == 1, "result_stats.gapcnt is %d", result_stats.gapcnt);
    TEST_ASSERTX (result_stats.gaplen == 2, "result_stats.gaplen is %d", result_stats.gaplen);
    TEST_ASSERTX (result_stats.score == 1, "result_stats.score is %d", result_stats.score);
    return 1;
}

TestWorstScorePosFinder::TestWorstScorePosFinder ()
:
Test ("WorstScorePosFinder")
{
}

static void alignment_from_string 
(
    tmap_map_alignment& dest,
    std::vector<uint32_t>& bin_cigar_storage,
    const char* cigar_str, 
    const char* qseq,
    const char* rseq
)
{
    dest.ncigar = compute_cigar_bin_size (cigar_str);
    bin_cigar_storage.resize (dest.ncigar);
    dest.cigar = &(bin_cigar_storage [0]);
    string_to_cigar (cigar_str, dest.cigar, dest.ncigar);
    dest.q_int = dest.r_int = 0;
    dest.qseq_off = dest.rseq_off = 0;
    dest.qseq = qseq;
    dest.rseq = rseq;
    dest.qseq_len = strlen (qseq);
    dest.rseq_len = strlen (rseq);
}

std::ostream& operator << (std::ostream& o, const tmap_map_alignment& s)
{
    size_t cigar_strlen = compute_cigar_strlen (s.cigar, s.ncigar);
    char cigar_str [cigar_strlen+1];
    size_t written = cigar_to_string (s.cigar, s.ncigar, cigar_str, cigar_strlen+1);
    assert (written == cigar_strlen);
    const char* qseq = s.qseq;
    char qseq_buf [s.q_int ? s.qseq_len + 1 : 0];
    if (s.q_int)
    {
        char* o = qseq_buf;
        for (char const *c = qseq, *sent = qseq + s.qseq_len; c != sent; ++c, ++o)
            *o = tmap_iupac_int_to_char [(size_t) *c];
        *o = 0;
        qseq = qseq_buf;
    }
    const char* rseq = s.rseq;
    char rseq_buf [s.r_int ? s.rseq_len + 1 : 0];
    if (s.r_int)
    {
        char* o = rseq_buf;
        for (char const *c = rseq, *sent = rseq + s.rseq_len; c != sent; ++c, ++o)
            *o = tmap_iupac_int_to_char [(size_t) *c];
        *o = 0;
        rseq = rseq_buf;
    }
    o << cigar_str << "{" << qseq << ":" << rseq << "}";
    return o;
}

std::ostream& out_short (std::ostream& o, const tmap_map_alignment& s)
{
    size_t cigar_strlen = compute_cigar_strlen (s.cigar, s.ncigar);
    char cigar_str [cigar_strlen+1];
    size_t written = cigar_to_string (s.cigar, s.ncigar, cigar_str, cigar_strlen+1);
    assert (written == cigar_strlen);
    o << cigar_str;
    return o;
}

std::ostream& operator << (std::ostream& o, const tmap_map_alignment_segment& s)
{
    o << "AlSeg:[" << s.first_op << ":" << s.first_op_off << "-" << s.last_op << ":" << s.sent_op_off << ";" << s.q_start << "-" << s.q_end << ";" << s.r_start << "-" << s.r_end << "]";
    return o;
}
std::ostream& operator << (std::ostream& o, const tmap_map_alignment_stats& s)
{
    o << "sc=" << s.score << ",m=" << s.matches << ",x=" << s.mismatches << ",g=" << s.gapcnt << ",l=" << s.gaplen;
    return o;
}


// model simple cases
struct WorstScoreCase
{
    const char* cigar;
    const char* qry;
    const char* ref;
    bool from_beg;
    uint8_t res;
    int32_t q_pos;
    int32_t r_pos;
    int32_t opno;
    int32_t opoff;
    int32_t score;
};
std::ostream& operator << (std::ostream& o, WorstScoreCase& c)
{
    o << "Case:" << c.cigar << "{" << c.qry << ":" << c.ref << "}" << (c.from_beg?"from_beg":"from_end") << ".exp:";
    if (c.res == 0)
        o << "FAIL  ";
    else
        o << "FOUND ";
    o << "[" << c.opno << ":" << c.opoff << ",q" << c.q_pos << ",r" << c.r_pos << ",sw=" << c.score << "]";
    return o;
}

bool TestWorstScorePosFinder::process ()
{
    WorstScoreCase cases [] = 
    {
        /* 1*/ {"1M",     "A", "A",    true,  0, 0, 0, 0, -1, 0}, 
        /* 2*/ {"1M",     "A", "A",    false, 0, 1, 1, 0, 1,  0},
        /* 3*/ {"1M",     "A", "T",    true,  1, 1, 1, 0, 0, -3},
        /* 4*/ {"1M",     "A", "T",    false, 1, 0, 0, 0, 0, -3},
        /* 5*/ {"1M1D1M", "GC", "GAC", true,  1, 1, 2, 1, 0, -6},
        /* 6*/ {"1M1D1M", "GC", "GAC", false, 1, 1, 1, 1, 0, -6},
        /* 7*/ {"1M2D1M", "GC", "GATC", true, 1, 1, 3, 1, 1, -8},
        /* 8*/ {"1M2D1M", "GC", "GATC", false, 1, 1, 1, 1, 0, -8},
        /* 9*/ {"3M2D4M4I4M", "ATGCAATGCGCATGC", "ATGCCCAATATGC", true,  1, 11, 9, 3, 3, -15}, // 3 - 5 - 2 - 2 + 4 - 5 - 2 - 2 - 2 - 2 = -15
        /*10*/ {"3M2D4M4I4M", "ATGCAATGCGCATGC", "ATGCCCAATATGC", false, 1,  3, 3, 1, 0, -14}, // 4 - 5 - 2 - 2 - 2 - 2 + 4 - 5 - 2 - 2 = -14
        /*11*/ {"10M2I10M", "AAAAAAAAAACCTTTTTTTTTT", "AAAAAAAAAATTTTTTTTTT", true,  0, 0,   0,  0, -1, 0},
        /*12*/ {"10M2I10M", "AAAAAAAAAACCTTTTTTTTTT", "AAAAAAAAAATTTTTTTTTT", false, 0, 22, 20,  2, 10, 0},
        /*13*/ {"5S10M2I10M", "GGGGGAAAAAAAAAACCTTTTTTTTTT", "AAAAAAAAAATTTTTTTTTT", true,  0, 0,   0,  0, -1, 0},
        /*14*/ {"5S10M2I10M", "GGGGGAAAAAAAAAACCTTTTTTTTTT", "AAAAAAAAAATTTTTTTTTT", false, 0, 27, 20,  3, 10, 0},
        /*15*/ {"2S3M2D4M4I4M7S", "GGATGCAATGCGCATGCGGGGGGG", "ATGCCCAATATGC", true,  1, 13, 9, 4, 3, -15}, // 3 - 5 - 2 - 2 + 4 - 5 - 2 - 2 - 2 - 2 = -15
        /*16*/ {"2S3M2D4M4I4M7S", "GGATGCAATGCGCATGCGGGGGGG", "ATGCCCAATATGC", false, 1,  5, 3, 2, 0, -14}, // 4 - 5 - 2 - 2 - 2 - 2 + 4 - 5 - 2 - 2 = -14
    };

    SwMatrixFacet* swf = dynamic_cast<SwMatrixFacet*> (find_facet ("SwMatrix"));
    assert (swf);
    const tmap_sw_param_t* swpar = swf->sw_param ();

    for (WorstScoreCase* c = cases, *sent = cases + sizeof (cases) / sizeof (*cases); c != sent; ++c)
    {
        // make alignment
        tmap_map_alignment al;
        std::vector <uint32_t> bin_cigar_storage;
        alignment_from_string (al, bin_cigar_storage, c->cigar, c->qry, c->ref);
        // structures to hold results
        tmap_map_alignment_segment cropped;
        tmap_map_alignment_segment clipped;
        tmap_map_alignment_stats cropped_stats;
        tmap_map_alignment_stats clip_stats;
        int32_t wqpos, wrpos, wopno, wopoff;
        // perform search
        uint8_t found = tmap_map_find_worst_score_pos_x (&al, swpar, c->from_beg, &cropped, &clipped, &cropped_stats, &clip_stats, &wopno, &wopoff, &wqpos, &wrpos);
        // print out results
        (std::ostream&) o_ << c - cases + 1 << ": " << *c << " got:" << (found?" FOUND ": " FAIL  ") << "[" << wopno << ":" << wopoff << ",q" << wqpos << ",r" << wrpos << ",sw=" << clip_stats.score << "]" << std::endl;
        (std::ostream&) o_ << std::setw (4) << "" << "Cropped: " << cropped << std::endl;
        (std::ostream&) o_ << std::setw (4) << "" << "Clipped: " << clipped << std::endl;
        (std::ostream&) o_ << std::setw (4) << "" << "Cropped stats: " << cropped_stats << std::endl;
        (std::ostream&) o_ << std::setw (4) << "" << "Clipped stats: " << clip_stats << std::endl;
        // check if matches expectations
        TEST_ASSERTX (found == c->res, "case %d: received %d, expected %d,  %d: %s  %s/%s", c - cases + 1, found, c->res, c - cases, c->cigar, c->qry, c->ref);
        //if (found)
        {
            TEST_ASSERTX (wopno == c->opno, "case %d: received %d, expected %d", c - cases + 1, wopno, c->opno);
            TEST_ASSERTX (wopoff == c->opoff, "case %d: received %d, expected %d", c - cases + 1, wopoff, c->opoff);
            TEST_ASSERTX (wqpos == c->q_pos, "case %d: received %d, expected %d", c - cases + 1, wqpos, c->q_pos);
            TEST_ASSERTX (wrpos == c->r_pos, "case %d: received %d, expected %d", c - cases + 1, wrpos, c->r_pos);
            TEST_ASSERTX (clip_stats.score == c->score, "case %d: received %d, expected %d", c - cases + 1, clip_stats.score, c->score);


            if (c->from_beg)
            {
                // cropped segment should start next position to the worst score one
                // sequence footprints start on the qpos/rpos positions (these are positions next to the one of the worst score
                int32_t exp_first_op = c->opno;
                int32_t exp_first_off = c->opoff;
                int32_t exp_last_op = ((int32_t) al.ncigar) - 1;
                int32_t exp_last_sent = (int32_t) bam_cigar_oplen (al.cigar [al.ncigar-1]);
                next_cigar_pos (al.cigar, al.ncigar, &exp_first_op, &exp_first_off, 1);

                TEST_ASSERTX (cropped.first_op == exp_first_op, "case %d: received %d, expected %d", c - cases + 1, cropped.first_op, exp_first_op);
                TEST_ASSERTX (cropped.last_op == exp_last_op, "case %d: received %d, expected %d", c - cases + 1, cropped.last_op, exp_last_op);
                TEST_ASSERTX (cropped.first_op_off == exp_first_off, "case %d: received %d, expected %d", c - cases + 1, cropped.first_op_off, exp_first_off);
                TEST_ASSERTX (cropped.sent_op_off == exp_last_sent, "case %d: received %d, expected %d", c - cases + 1, cropped.sent_op_off, exp_last_sent);
                TEST_ASSERTX (cropped.q_start == c->q_pos, "case %d: received %d, expected %d", c - cases + 1, cropped.q_start, c->q_pos);
                TEST_ASSERTX (cropped.q_end == (int) al.qseq_len, "case %d: received %d, expected %d", c - cases + 1, cropped.q_end, (int) al.qseq_len);
                TEST_ASSERTX (cropped.r_start == c->r_pos, "case %d: received %d, expected %d", c - cases + 1, cropped.r_start, c->r_pos);
                TEST_ASSERTX (cropped.r_end == (int) al.rseq_len, "case %d: received %d, expected %d", c - cases + 1, cropped.r_end, (int) al.rseq_len);

                TEST_ASSERTX (clipped.first_op == 0, "case %d: received %d, expected %d", c - cases + 1, clipped.first_op, 0);
                TEST_ASSERTX (clipped.last_op == exp_first_op, "case %d: received %d, expected %d", c - cases + 1, clipped.last_op, exp_first_op);
                TEST_ASSERTX (clipped.first_op_off == 0, "case %d: received %d, expected %d", c - cases + 1, clipped.first_op_off, 0);
                TEST_ASSERTX (clipped.sent_op_off == exp_first_off, "case %d: received %d, expected %d", c - cases + 1, clipped.sent_op_off, exp_first_off);
                TEST_ASSERTX (clipped.q_start == 0, "case %d: received %d, expected %d", c - cases + 1, clipped.q_start,0);
                TEST_ASSERTX (clipped.q_end == c->q_pos, "case %d: received %d, expected %d", c - cases + 1, clipped.q_end, c->q_pos);
                TEST_ASSERTX (clipped.r_start == 0, "case %d: received %d, expected %d", c - cases + 1, clipped.r_start, 0);
                TEST_ASSERTX (clipped.r_end == c->r_pos, "case %d: received %d, expected %d", c - cases + 1, clipped.r_end, c->r_pos);;


            }
            else
            {
                // cropped segment should end at the position to the worst score one (it should be sentinel)
                // sequence footprints should end at the ones of the worst score: the ((seg_len -1) - position) is how many bases were seen when worst score was achieved
                int32_t exp_first_op = 0;
                int32_t exp_first_off = 0;
                int32_t exp_last_op = c->opno;
                int32_t exp_last_sent = c->opoff;
                int32_t exp_clip_last_op = ((int32_t) al.ncigar) - 1;
                int32_t exp_clip_last_sent = (int32_t) bam_cigar_oplen (al.cigar [al.ncigar-1]);

                TEST_ASSERTX (cropped.first_op == exp_first_op, "case %d: received %d, expected %d", c - cases + 1, cropped.first_op, exp_first_op);
                TEST_ASSERTX (cropped.last_op == exp_last_op, "case %d: received %d, expected %d", c - cases + 1, cropped.last_op, exp_last_op);
                TEST_ASSERTX (cropped.first_op_off == exp_first_off, "case %d: received %d, expected %d", c - cases + 1, cropped.first_op_off, exp_first_off);
                TEST_ASSERTX (cropped.sent_op_off == exp_last_sent, "case %d: received %d, expected %d", c - cases + 1, cropped.sent_op_off, exp_last_sent);
                TEST_ASSERTX (cropped.q_start == 0, "case %d: received %d, expected %d", c - cases + 1, cropped.q_start, 0);
                TEST_ASSERTX (cropped.q_end == c->q_pos, "case %d: received %d, expected %d", c - cases + 1, cropped.q_end, c->q_pos);
                TEST_ASSERTX (cropped.r_start == 0, "case %d: received %d, expected %d", c - cases + 1, cropped.r_start, 0);
                TEST_ASSERTX (cropped.r_end == c->r_pos, "case %d: received %d, expected %d", c - cases + 1, cropped.r_end, c->r_pos);

                TEST_ASSERTX (clipped.first_op == exp_last_op, "case %d: received %d, expected %d", c - cases + 1, clipped.first_op, exp_last_op);
                TEST_ASSERTX (clipped.last_op == exp_clip_last_op, "case %d: received %d, expected %d", c - cases + 1, clipped.last_op, exp_clip_last_op);
                TEST_ASSERTX (clipped.first_op_off == exp_last_sent, "case %d: received %d, expected %d", c - cases + 1, clipped.first_op_off, exp_last_sent);
                TEST_ASSERTX (clipped.sent_op_off == exp_clip_last_sent, "case %d: received %d, expected %d", c - cases + 1, clipped.sent_op_off, exp_clip_last_sent);
                TEST_ASSERTX (clipped.q_start == c->q_pos, "case %d: received %d, expected %d", c - cases + 1, clipped.q_start, c->q_pos);
                TEST_ASSERTX (clipped.q_end == (int) al.qseq_len, "case %d: received %d, expected %d", c - cases + 1, clipped.q_end, al.qseq_len);
                TEST_ASSERTX (clipped.r_start == c->r_pos, "case %d: received %d, expected %d", c - cases + 1, clipped.r_start, c->r_pos);
                TEST_ASSERTX (clipped.r_end == (int) al.rseq_len, "case %d: received %d, expected %d", c - cases + 1, clipped.r_end, al.rseq_len);
            }
        }
    }
    return 1;
}

TestSegmentMoveBases::TestSegmentMoveBases ()
:
Test ("SegmentMoveBases")
{
};


std::string segment_str (const tmap_map_alignment_segment& seg)
{
    std::ostringstream oss;
    oss << seg;
    return oss.str ();
}

std::string alstat_str (const tmap_map_alignment_stats& stats)
{
    std::ostringstream oss;
    oss << stats;
    return oss.str ();
}


bool TestSegmentMoveBases::process ()
{
    //    =====  ==+++++++++++++++++
    // Q: GGATG--CAATGCGCATGCGGGGGGG
    //    ssmmmddmmmmiiiimmmmsssssss
    //    2 3  2 4   4   4   7         
    // R: ~~ATGCCCAAT----ATGC~~~~~~~
    //      =======++    ++++
    // score       = 3 - 5 - 4 + 4 - 5 - 8 + 4 = -11
    // = seg score = 3 - 5 - 4 + 2 = -4
    // + seg score = 2 - 5 - 8 + 4 = -7

    // make alignment
    char cigar [] = "2S3M2D4M4I4M7S";
    char qseq  [] = "GGATGCAATGCGCATGCGGGGGGG";
    char rseq  [] = "ATGCCCAATATGC";
    std::vector <uint32_t> bin_cigar_storage;
    tmap_map_alignment al;
    alignment_from_string (al, bin_cigar_storage, cigar, qseq, rseq);

    SwMatrixFacet* swf = dynamic_cast<SwMatrixFacet*> (find_facet ("SwMatrix"));
    assert (swf);
    const tmap_sw_param_t* swpar = swf->sw_param ();

    // define initial segments
    tmap_map_alignment_segment s1, s2;
    s1.alignment = s2.alignment = &al;
    init_segment (&s1); init_segment (&s2);
    s1.last_op = 3;
    s1.sent_op_off = 2;
    s1.q_end = 7;
    s1.r_end = 7;
    s2.first_op = 3;
    s2.first_op_off = 2;
    s2.q_start = 7;
    s2.r_start = 7;

    tmap_map_alignment_stats seg1_stats, seg2_stats;
    tmap_map_alignment_segment_score (&s1, swpar, &seg1_stats);
    tmap_map_alignment_segment_score (&s2, swpar, &seg2_stats);

    tmap_map_alignment_segment orig_s1 = s1, orig_s2 = s2;
    tmap_map_alignment_stats orig_st1 = seg1_stats, orig_st2 = seg2_stats;
    std::string orig_s1_s = segment_str (orig_s1);
    std::string orig_s2_s = segment_str (orig_s2);
    std::string orig_stat1_s = alstat_str (orig_st1);
    std::string orig_stat2_s = alstat_str (orig_st2);

    const int sw = 27, tw = 24;
    // rest reversability in both directions
    for (int32_t direction = -1; direction <= 1; direction += 2)
    {
        uint32_t steps_moved, rev_steps_moved;
        uint32_t steps = 1;
        do
        {
            (std::ostream&) o_<< "Performing " << steps << " steps back and forth, direction: " << direction << std::endl;
            // would love to use std::format of C++20 here (and in many other places)!!
            (std::ostream&) o_<< std::setw (4) << "" << "original: " << "(" << std::setw(4) << std::right << steps << ") " << std::left << std::setw (sw) << orig_s1_s << " " << std::left << std::setw (sw) << orig_s2_s << " " << std::left << std::setw (tw) << orig_stat1_s << "" << orig_stat2_s << std::endl;

            steps_moved = tmap_map_alignment_segment_move_bases (&s1, &s2, steps * direction, swpar, &seg1_stats, &seg2_stats);
            std::string s1_is = segment_str (s1);
            std::string s2_is = segment_str (s2);
            std::string stat1_is = alstat_str (seg1_stats);
            std::string stat2_is = alstat_str (seg2_stats);
            (std::ostream&) o_<< std::setw (4) << "" << "moved   : " << "(" << std::setw(4) << std::right << steps_moved << ") " << std::left << std::setw (sw) << s1_is << " " << std::left << std::setw (sw) << s2_is << " " << std::setw (tw) << stat1_is << "" << stat2_is << std::endl;

            rev_steps_moved = tmap_map_alignment_segment_move_bases (&s1, &s2, -steps_moved * direction, swpar, &seg1_stats, &seg2_stats);
            std::string s1_s = segment_str (s1);
            std::string s2_s = segment_str (s2);
            std::string stat1_s = alstat_str (seg1_stats);
            std::string stat2_s = alstat_str (seg2_stats);
            (std::ostream&) o_<< std::setw (4) << "" << "returned: " << "(" << std::setw(4) << std::right << rev_steps_moved << ") " << std::left << std::setw (sw) << s1_s << " " << std::left << std::setw (sw) << s2_s << " " << std::left << std::setw (tw) << stat1_s << "" << stat2_s << std::endl;

            TEST_ASSERTX (steps_moved == rev_steps_moved, "steps: %d, steps_moved = %d, rev_steps_moved = %d", steps, steps_moved, rev_steps_moved);
            TEST_ASSERTX (segments_match (&s1, &orig_s1), "steps: %d, orig_segment: %s, inter_seg: %s, new_segment: %s", steps, orig_s1_s.c_str (), s1_is.c_str (), s1_s.c_str ());
            TEST_ASSERTX (segments_match (&s2, &orig_s2), "steps: %d, orig_segment: %s, inter_seg: %s, , new_segment: %s", steps, orig_s2_s.c_str (), s2_is.c_str (), s2_s.c_str ());
            TEST_ASSERTX (segment_stats_match (&seg1_stats, &orig_st1), "steps: %d, orig_stats: %s, inter_stat: %s, , new_stats: %s", steps, orig_stat1_s.c_str (), stat1_is.c_str (), stat1_s.c_str ());
            TEST_ASSERTX (segment_stats_match (&seg2_stats, &orig_st2), "steps: %d, orig_stats: %s, inter_stat: %s, , new_stats: %s", steps, orig_stat2_s.c_str (), stat2_is.c_str (), stat2_s.c_str ());

        }
        while (steps_moved == steps++);
    }
    return 1;
}


TestClipBasesFromSegment::TestClipBasesFromSegment ()
:
Test ("ClipBasesFromSegment")
{
}

bool TestClipBasesFromSegment::process ()
{
    return 1;
}

TestClipSegmentByScore::TestClipSegmentByScore ()
:
Test ("ClipSegmentByScore")
{
}

bool TestClipSegmentByScore::process ()
{
    return 1;
}


TestClipSegmentToRefBase::TestClipSegmentToRefBase ()
:
Test ("ClipSegmentToRefBase")
{
}
bool TestClipSegmentToRefBase::deterministic_test ()
{
    bool rv = true;
    // make alignment
    char cigar [] = "2S3M2D4M4I4M7S";
    char qseq  [] = "GGATGCAATGCGCATGCGGGGGGG";
    char rseq  [] = "ATGCCCAATATGC";
    std::vector <uint32_t> bin_cigar_storage;
    tmap_map_alignment al;
    alignment_from_string (al, bin_cigar_storage, cigar, qseq, rseq);

    // define initial segments
    tmap_map_alignment_segment s;
    s.alignment = &al;
    init_segment (&s);
    s.first_op = 3;
    s.first_op_off = 2;
    s.q_start = 7;
    s.r_start = 7;

    std::string orig_seg = segment_str (s);

    for (int32_t rpos = 0; rpos != (int32_t) s.alignment->rseq_len + 1; ++rpos)
    {
        uint32_t from_beg = 1;
        do
        {
            (std::ostream&) o_ << "Clipping to position " << rpos << (from_beg?" from_beg":" from_end") << std::endl;
            tmap_map_alignment_segment t = s;
            uint8_t res = tmap_map_segment_clip_to_ref_base (&t, from_beg, rpos);
            std::string result_seg = segment_str (t);
            if ( ( from_beg && (rpos < s.r_start || rpos > s.r_end)) ||
                 (!from_beg && (rpos + 1 < s.r_start || rpos + 1 > s.r_end)) )
            {
                TEST_ASSERTX (res == 0, "from %s: %d in [%d:%d], not expected, got %s", (from_beg?"beg":"end"), rpos, s.r_start, s.r_end, result_seg.c_str());
            }
            else
            {
                (std::ostream&) o_ << "    " << orig_seg << " -> " << result_seg << std::endl;
                TEST_ASSERTX (res != 0, "from %s: %d in [%d:%d], expected OK, got FAILURE", (from_beg?"beg":"end"), rpos, s.r_start, s.r_end);
                if (from_beg)
                {
                    TEST_ASSERTX (t.r_start == s.r_start,           "from beg: %d in [%d:%d], got %s", rpos, s.r_start, s.r_end, result_seg.c_str());
                    TEST_ASSERTX (t.r_end == rpos,                  "from beg: %d in [%d:%d], got %s", rpos, s.r_start, s.r_end, result_seg.c_str());
                    TEST_ASSERTX (t.q_start == s.q_start,           "from beg: %d in [%d:%d], got %s", rpos, s.r_start, s.r_end, result_seg.c_str());
                    TEST_ASSERTX (t.first_op == s.first_op,         "from beg: %d in [%d:%d], got %s", rpos, s.r_start, s.r_end, result_seg.c_str());
                    TEST_ASSERTX (t.first_op_off == s.first_op_off, "from beg: %d in [%d:%d], got %s", rpos, s.r_start, s.r_end, result_seg.c_str());
                }
                else
                {
                    TEST_ASSERTX (t.r_start == rpos+1,              "from end: %d in [%d:%d], got %s", rpos, s.r_start, s.r_end, result_seg.c_str());
                    TEST_ASSERTX (t.r_end == s.r_end,               "from end: %d in [%d:%d], got %s", rpos, s.r_start, s.r_end, result_seg.c_str());
                    TEST_ASSERTX (t.q_end == s.q_end,               "from end: %d in [%d:%d], got %s", rpos, s.r_start, s.r_end, result_seg.c_str());
                    TEST_ASSERTX (t.last_op == s.last_op,           "from end: %d in [%d:%d], got %s", rpos, s.r_start, s.r_end, result_seg.c_str());
                    TEST_ASSERTX (t.sent_op_off == s.sent_op_off,   "from end: %d in [%d:%d], got %s", rpos, s.r_start, s.r_end, result_seg.c_str());
                }
            }
        }
        while (!(from_beg = !from_beg));
    }


    return rv;
}
bool TestClipSegmentToRefBase::random_test ()
{
    bool rv = true;
    return rv;
}
bool TestClipSegmentToRefBase::process ()
{
    bool rv = deterministic_test ();
    if (rv) rv = random_test ();
    return rv;
}
