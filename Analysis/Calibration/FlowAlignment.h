/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef FLOWALIGNMENT_H
#define FLOWALIGNMENT_H

#include <string>
#include <vector>
#include "api/BamAlignment.h"

// ------------------------------------------------------------

const static int FROM_M = 0;            // The alignment was extended from a match.
const static int FROM_I = 1;            // The alignment was extended from an insertion.
const static int FROM_D = 2;            // The alignment was extended from an deletion.
const static int FROM_ME = 3;           // The alignment was extended from a match with an empty flow.
const static int FROM_IE = 4;           // The alignment was extended from an insertion with an empty flow.
const static int FROM_MP = 5;           // The alignment was extended from a phased match (skipping a phase).
const static int FROM_IP = 6;           // The alignment was extended from a phased insertion (skipping a phase).
const static int FROM_S = 7;            // The alignment was extended from an insertion.
const static int MINOR_INF = -1000000;  // The lower bound on the alignment score, or negative infinity.

const static char ALN_DEL = '-';        // A flow deletion in the alignment string.
const static char ALN_INS = '+';        // A flow insertion in the alignment string.
const static char ALN_MATCH = '|';      // A flow match in the alignment string.
const static char ALN_MISMATCH = ' ';   // A flow mismatch in the alignment string.

const static int PHASE_PENALTY = 1;     // Settable via args?

// ----------------------------------------------------------

bool IsInDelAlignSymbol(char symbol);

// ----------------------------------------------------------

void ReverseComplementInPlace (std::string& sequence);

void RetrieveBaseAlignment(
    // Inputs:
    const std::string&           alignment_query_bases,
    const std::vector<BamTools::CigarOp>&  alignment_cigar_data,
    const std::string&           md_tag,
    // Outputs:
    std::string&                 tseq_bases,
    std::string&                 qseq_bases,
    std::string&                 pretty_tseq,
    std::string&                 pretty_qseq,
    std::string&                 pretty_aln,
    unsigned int&                left_sc,
    unsigned int&                right_sc);

bool PerformFlowAlignment(
    // Inputs:
    const std::string&             tseq_bases,
    const std::string&             qseq_bases,
    const std::string&             main_flow_order,
    int                       first_useful_flow,
    //const std::vector<uint16_t>&   fz_tag,
    // Outputs:
    std::vector<char>&             flowOrder,
    std::vector<int>&              qseq,
    std::vector<int>&              tseq,
    std::vector<int>&              aln_flow_index,
    std::vector<char>&             aln);

#endif // FLOWALIGNMENT_H
