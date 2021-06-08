#include "test_alignment_generator.h"
#include "swmatrix_facet.h"
#include "../cigar_man_lib/alignment_generator.h"
#include "TMAP/src/util/tmap_cigar_util.h"

#include <iostream>
#include <iomanip>

#define STEPNO 10

bool TestAlignmentGenerator::process ()
{
    tmap_map_alignment al;
    std::string qry, ref;
    std::vector <uint32_t> cigar;

    SwMatrixFacet* sw = dynamic_cast<SwMatrixFacet*> (find_facet ("SwMatrix"));
    assert (sw);
    const tmap_sw_param_t* swpar = sw->sw_param ();

    AlignmentGenerator generator (swpar);

    unsigned step = 0;
    (std::ostream&) o_ << "Growing" << std::endl;
    while (cigar.size () != STEPNO)
    {
        generator.advance_head ();
        generator.capture (al, qry, ref, cigar);
        {
            size_t cigar_strlen = compute_cigar_strlen (al.cigar, al.ncigar);
            char cigar_strbuf [cigar_strlen + 1];
            cigar_to_string (al.cigar, al.ncigar, cigar_strbuf, cigar_strlen + 1);
            (std::ostream&) o_ << step <<": " <<  cigar_strbuf << " ("<< al.ncigar << "), q=" <<  al.qseq << " (" << al.qseq_len << "), r=" << al.rseq << " (" << al.rseq_len << ")" << std::endl;
            const tmap_map_alignment_stats& s = generator.stats ();
            (std::ostream&) o_ << std::setw (5) << "" << "score: " << s.score << ", " << s.matches << " m, " << s.mismatches << " x, " << s.gapcnt << " g, " << s.gaplen << " gl" << std::endl;
        }
        ++step;
    }
    (std::ostream&) o_ << "Curtailing" << std::endl;
    {
        --step;
        while (cigar.size ())
        {
            generator.reduce_tail ();
            generator.capture (al, qry, ref, cigar);
            {
                size_t cigar_strlen = compute_cigar_strlen (al.cigar, al.ncigar) + 1;
                char cigar_strbuf [cigar_strlen + 1];
                cigar_to_string (al.cigar, al.ncigar, cigar_strbuf, cigar_strlen + 1);
                (std::ostream&) o_ << step <<": " <<  cigar_strbuf << " ("<< al.ncigar << "), q=" <<  al.qseq << " (" << al.qseq_len << "), r=" << al.rseq << " (" << al.rseq_len << ")" << std::endl;
                const tmap_map_alignment_stats& s = generator.stats ();
                (std::ostream&) o_ << std::setw (5) << "" << "score: " << s.score << ", "  << s.matches << " m, " << s.mismatches << " x, " << s.gapcnt << " g, " << s.gaplen << " gl" << std::endl;
            }
        }
    }
    return true;
}

