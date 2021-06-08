#ifndef TMAP_MAP_ALIGN_UTILS_H
#define TMAP_MAP_ALIGN_UTILS_H

#include <stdint.h>
#include "../../sw/tmap_sw.h"

#if defined (__cplusplus)
extern "C"
{
#endif

typedef struct __tmap_map_alignment_stats
{
    int32_t  score;
    uint32_t matches;
    uint32_t mismatches;
    uint32_t gapcnt;
    uint32_t gaplen;
}
tmap_map_alignment_stats;
// NOTE: agnostic to the direction of the actual mapped query. 

// representation of the alignment;
// (for reverse mappings, all the coordinates are done for the inverse-complemented queries, so that it does not make difference)
typedef struct __tmap_map_alignment
{
    uint32_t*   cigar;    // binary cigar
    uint32_t    ncigar;   // length of binary cigar
    uint8_t     q_int : 1;    // 1 if query is binary encoded, 0 if character string (AGCT)
    uint8_t     r_int : 1;    // 1 if reference is binary encoded, 0 if character string (AGCT)
    const char* qseq;     // query 
    uint32_t    qseq_len; // query sequence length (if binary encoded there is no null-termination)
    uint32_t    qseq_off; // query offset of the first cigar op, normally 0
    const char* rseq;     // pointer to reference sequence as byte array (binary or ascii, defined by flags.r_int)
    uint32_t    rseq_len; // reference sequence length (if binary encoded there is no null-termination)
    uint32_t    rseq_off; // reference offset of the first cigar op, normally 0
}
tmap_map_alignment;


// in the fields of the following structure, 
// the value of -1 indicates 'unspecified'
// and can be computed from others
// If the segment spans to the very end of cigar, both representation of the end are valid
typedef struct __tmap_map_alignment_segment
{
    const tmap_map_alignment* alignment; // the host alignment 
    int32_t q_start;               // segment start on query (in abs query coords)
    int32_t q_end;                 // segment end on query query. This position is not included in a segment (it is a sentinel, can be one past the end of the segment) (in abs query coords)
    int32_t r_start;               // segment start on reference (in abs reference fragment coords)
    int32_t r_end;                 // segment end on the refernece. This position is not included in a segment (it is a sentinel, can be one past the end of the segment) (in abs reference fragment coords)
    int32_t first_op;              // index of cigar op where segment starts
    int32_t last_op;               // index of cigar op where segment ends
    int32_t first_op_off;          // offset in the first operation of the first position belonging to a segment
    int32_t sent_op_off;           // offset of the position next to the last in te segment. This position does not belong to a segment (it is a sentinel, can be one past the end of the segment)
}
tmap_map_alignment_segment;

// structure holding the state of the alignment walker engine that is exposed to the step callback
typedef struct __alignment_walk_state
{
    uint8_t from_beg;
    uint8_t qbase;
    uint8_t rbase;
    int32_t q_pos;
    int32_t r_pos;
    int32_t op_idx;
    int32_t op_off;
    uint32_t op;
    uint32_t op_len;
    uint8_t advance_qry;
    uint8_t advance_ref;
    tmap_map_alignment_stats stats;
}
alignment_walk_state;

// alignment walking engine
// moves along cigar-encoded alignment path in given direction
// advances state (held in alignment_walker_position  and calls registered callback with the address of updated alignment_walk_position structure
// also passes the pointer to any auxillary structure passed in
// upon return the state structure is valid and holds state at termination. The positions may point past the end or before the beginning of the sequences / cigar / cigar segments
// returns 1 if walk is sucessfull, 0 if empty alignment was passed (ncigar equals 0).
// NOTE: does not do rigorous test of valididty of alignment, misformed one may lead to unpredictable behavior
uint8_t alignment_walker 
(
    const tmap_map_alignment* alignment,
    const tmap_sw_param_t* sw_params,
    uint8_t from_beg,
    uint8_t (*step_callback) (const alignment_walk_state* state, void* aux),
    alignment_walk_state* state,
    void* aux
);

// alignment segment walking engine
// similar to alignment_walker, but walk only the segment of the alignment
uint8_t alignment_segment_walker 
(
    const tmap_map_alignment_segment* segment,
    const tmap_sw_param_t* sw_params,
    uint8_t from_beg,
    uint8_t (*step_callback) (const alignment_walk_state* state, void* aux),
    alignment_walk_state* state,
    void* aux
);


// resets all members of stats to zeros
void init_alignment_stats (tmap_map_alignment_stats* stats);

// sets segment to match entire alignment it refers to
void init_segment (tmap_map_alignment_segment* segment);


// convenience function to populate alignment
// this can be used only for ASCII sequences (consisting of AGTCN...)
uint8_t init_alignment 
(
    tmap_map_alignment* dest,
    uint32_t*           cigar,
    uint32_t            ncigar,
    const char*         qseq,
    uint32_t            qseq_off,
    const char*         rseq,
    uint32_t            rseq_off
);

// populates alignment using binary converted sequences
uint8_t init_alignment_x
(
    tmap_map_alignment* dest,
    uint32_t*           cigar,
    uint32_t            ncigar,
    const uint8_t*      qseq,
    uint32_t            qseq_off,
    uint32_t            qseq_len,
    const uint8_t*      rseq,
    uint32_t            rseq_off,
    uint32_t            rseq_len
);

// validator for alignment
uint8_t validate_alignment
(
    tmap_map_alignment* al
);

// makes string representation of alignment segment
uint8_t segment_to_string 
(
    tmap_map_alignment_segment* segment, 
    char* buf, unsigned* bufsz
);

// writes alignment segment to tmap error stream


// computes values for the fields marked as 'unspecified' (holding values of -1)
uint8_t
tmap_map_normalize_alignment_segment
(
    tmap_map_alignment_segment* segment
);

// compares two segments, returns 1 if equal 0 if not.
uint8_t segments_match
(
    const tmap_map_alignment_segment* segment1,
    const tmap_map_alignment_segment* segment2
);

uint8_t segment_stats_match 
(
    const tmap_map_alignment_stats* s1,
    const tmap_map_alignment_stats* s2
);

// finds the worst (lowest) score position from given end;
// fills in:
//       resulting retained segment (the one AFTER the worst position, excluding one, if reached in the direction of search) 
//       stats for the resulting retained segment 
//       stats for the clipped patrs of the alignment
// returns true on success, false if there is no segment with score worser than entire alignment
uint8_t
tmap_map_find_worst_score_pos
(
    tmap_map_alignment* alignment,
    const tmap_sw_param_t* sw_params,
    uint8_t from_beg,
    tmap_map_alignment_segment* result,
    tmap_map_alignment_stats *result_stats,
    tmap_map_alignment_stats* clip_stats
);
// same as tmap_map_find_worst_score_pos 
// but also stores actual worst score position in passed in pointers
//
// *** correct interpretation of q/rpos:***
// (abs (seq_position - seq_scan_start_position)) is how many bases of the sequence were seen when worst score was achieved
// forward scan starts with qseq_off/rseq_off
// reverse scan starts with qseq_len/rseq_len (at sentinel coord)
// (thus, for reverse searches (seq_len - worst_position) is how many bases were seen when worst score was achieved)

uint8_t
tmap_map_find_worst_score_pos_x
(
    tmap_map_alignment* alignment,
    const tmap_sw_param_t* sw_params,
    uint8_t from_beg,
    tmap_map_alignment_segment* result,
    tmap_map_alignment_segment* clip,
    tmap_map_alignment_stats *result_stats,
    tmap_map_alignment_stats* clip_stats,
    int32_t* opno,
    int32_t* opoff,
    int32_t* qpos,
    int32_t* rpos
);

// move given number of query bases from one alignment segment to another
// segments have to be adjacent, seg2 must imeediately follow seg1
// returns number of bases moved
uint32_t
tmap_map_alignment_segment_move_bases
(
    tmap_map_alignment_segment* seg1,
    tmap_map_alignment_segment* seg2,
    int32_t bases, // negative value means move from seg2 to seg1
    const tmap_sw_param_t* sw_params,
    tmap_map_alignment_stats* seg1_stats,
    tmap_map_alignment_stats* seg2_stats
);


// clips additional bases from the alignment segment, until the alignment score of clipped part reaches given value
// optionally terminates when a given number of query bases are clipped
// fills in the resulting segment and the stats for the retained and for the clipped patrs of the alignment
// returns true on success, false if can not proceed (neither clip score nor max_clipped_query_bases can be achieved)
uint8_t
tmap_map_clip_alignment_segment_by_score
(
    tmap_map_alignment_segment* initial,
    const tmap_sw_param_t* sw_params,
    uint8_t from_beg,
    int32_t clip_score,
    int32_t max_clipped_qry_bases,
    tmap_map_alignment_segment* result,
    tmap_map_alignment_stats *result_stats,
    tmap_map_alignment_stats* clip_stats
);

// computes statistics on the given alignment segment
// returns SW score
int32_t tmap_map_alignment_segment_score
(
    tmap_map_alignment_segment* segment,
    const tmap_sw_param_t* sw_params,
    tmap_map_alignment_stats *result_stats
);

// modifies segment so that it extends over reference either from the beginning of given segment to the given centinel (when from_beg == 1),
// or from (given centinel+1) to the given segment's end (when from_beg == 0),
// the content of a segment is discarded
// returns 1 on success, 0 if target position does not lay within an alignment
int8_t tmap_map_segment_clip_to_ref_base 
(
    tmap_map_alignment_segment* segment, 
    int8_t from_beg, 
    int32_t ref_pos_sentinel
);


#if defined (__cplusplus)
}
#endif


#endif // TMAP_MAP_ALIGN_SEGMENT_H
