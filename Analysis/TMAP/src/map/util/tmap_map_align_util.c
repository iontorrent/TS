#include "tmap_map_align_util.h"

#include "../../util/tmap_error.h"
#include "../../util/tmap_definitions.h"
#include "../../util/tmap_cigar_util.h"
#include <samtools/bam.h>
#include <stdio.h>
#include <assert.h>

#if defined (NDEBUG)
#define ALIGN_SEG_ERROR_POLICY 0
#else
#define ALIGN_SEG_ERROR_POLICY T_REPORT_ERROR
#endif

static uint8_t align_seg_error_policy = ALIGN_SEG_ERROR_POLICY;

static inline uint8_t segment_check_normalized
(
    const tmap_map_alignment_segment* segment
)
{
    return (   segment->q_start      != -1
            && segment->r_end        != -1
            && segment->r_start      != -1
            && segment->r_end        != -1
            && segment->first_op     != -1
            && segment->first_op_off != -1
            && segment->last_op      != -1
            && segment->sent_op_off  != -1 );
}

static uint8_t validate_segment_align (tmap_map_alignment_segment* segment, uint8_t error_policy)
{
    uint8_t rv = 1;
    if (segment->first_op > segment->last_op)
        rv = 0;
    if (rv && segment->first_op == segment->last_op && segment->first_op_off > segment->sent_op_off)
        rv = 0;
    if (!rv)
        tmap_flagerr (error_policy, "Invalid cigar indexing in alignment segment");
    return rv;
}

static uint8_t validate_segment_footprint 
(
    tmap_map_alignment_segment* segment,
    uint8_t error_policy
)
{
    uint8_t rv = 1;
    if (segment->q_start > segment->alignment->qseq_len || segment->q_end > segment->alignment->qseq_len)
        rv = 0;
    if (rv && (segment->r_start > segment->alignment->rseq_len || segment->r_end > segment->alignment->rseq_len))
        rv = 0;
    if (!rv)
        tmap_flagerr (error_policy, "Segment footprint is invalid")
    return rv;
}

static void init_alignment_walk_state
(
    alignment_walk_state* aws,
    uint8_t from_beg,
    int32_t q_pos,
    int32_t r_pos,
    int32_t op_idx,
    int32_t op_off
)
{
    aws->from_beg = from_beg;
    aws->q_pos = q_pos; 
    aws->r_pos = r_pos; 
    aws->op_idx = op_idx;
    aws->op_off = op_off;
    init_alignment_stats (&aws->stats);
}

// the inner loop for alignment segment walker
// terminates returning 0 whenever callback returns 0
// otherwise returns 1

// the state structure passed to callback reflects the cigar position is ALREADY seen and evaluated. 
// the sequence positions reflects HOW MANY bases were seen, essentially pointing to a sentinel (not yet seen) base number 
// so qpos = 0 means no positions in query were seen so far (query position never advanced)
// In contrary, the cigar position passed in is NOT YET seen and evaluated; it is visited on first loop iteration (and passed to the callback in a state structure)
// After the loop terminates, the cigar position would be the one just beyond the valid range 
// (last_op:last_op_off+1) for forward and first_op:(first_op_off-1) for reverse walks) 
// NOTE: the callback is not called for the position BEFORE the start of the loop (when no positions were seen). 
//       it is up to the loop caller to implement logic evaluating this position if needed

static uint8_t alignment_walker_loop 
(
    const uint32_t* cigar,
    const char *qseq,
    const char *rseq,
    uint8_t q_int,
    uint8_t r_int,
    uint32_t qpos,   // offset of first query base to be seen by walker
    uint32_t rpos,   // offset of first reference base to be seen by walker
    int32_t op_first,
    int32_t op_first_start_off,
    int32_t op_last,
    int32_t op_last_end_off, // sentinel! (excluded)
    const tmap_sw_param_t* sw_params, // can be NULL, score will not be updated then
    uint8_t from_beg,
    uint8_t (*step_callback)(const alignment_walk_state* state, void* aux),
    alignment_walk_state* aws,
    void* aux
)
{
    // walk state is overwritten based on passed in data
    init_alignment_walk_state (aws, 
        from_beg,
        qpos,
        rpos,
        op_first,
        op_first_start_off );


    const uint32_t* first_op_p = cigar + op_first;
    const uint32_t* last_op_p = cigar + op_last;
    const uint32_t* sent_op_p = from_beg ? (last_op_p + 1) : (last_op_p - 1);
    int32_t incr = from_beg ? 1 : -1;

    const uint32_t* cigar_op_p;
    uint32_t advance_type;
    int32_t op_off_first, op_off_sent;
    // loop over the cigar operations
    for (cigar_op_p = first_op_p; cigar_op_p != sent_op_p;  cigar_op_p += incr, aws->op_idx += incr)
    {
        aws->op = bam_cigar_op (*cigar_op_p);
        aws->op_len = bam_cigar_oplen (*cigar_op_p);
        advance_type = bam_cigar_type (aws->op);
        aws->advance_qry = advance_type & CIGAR_CONSUME_QRY ? 1 : 0;
        aws->advance_ref = advance_type & CIGAR_CONSUME_REF ? 1 : 0;
        // set up op_off_first and op_off_sent(inel)
        if (cigar_op_p == first_op_p)
            op_off_first = op_first_start_off; // no need to flip - this is passed from caller, already correct // TODO: check if it would be cleaner to flip here
        else
            op_off_first = from_beg ? 0 : aws->op_len - 1; // here we do need to flip
        if (cigar_op_p == last_op_p)
            op_off_sent = op_last_end_off; // // no need to flip - this is passed from caller, already correct
        else
            op_off_sent = from_beg ? aws->op_len : -1; // here we do need to flip
        // loop over the segment positions
        for (aws->op_off = op_off_first; aws->op_off != op_off_sent; aws->op_off += incr)
        {
            // adjust score for current position
            switch (aws->op)
            {
                case BAM_CMATCH:
                case BAM_CEQUAL:  // do not use match/mismatch distinction, use score matrix instead
                case BAM_CDIFF:
                    // add corresponding score from weight matrix
                    aws->qbase = qseq [aws->q_pos];
                    if (!q_int)
                        aws->qbase = tmap_iupac_char_to_int [aws->qbase];
                    aws->rbase = rseq [aws->r_pos];
                    if (!r_int)
                        aws->rbase = tmap_iupac_char_to_int [aws->rbase];
                    if (sw_params)
                        aws->stats.score += sw_params->matrix [sw_params->row * aws->qbase + aws->rbase];
                    if (aws->qbase == aws->rbase) ++(aws->stats.matches); else ++(aws->stats.mismatches);
                    break;
                case BAM_CINS:
                case BAM_CDEL:
                    if (aws->op_off == op_off_first)
                    {
                        if (sw_params)
                            aws->stats.score -= sw_params->gap_open;
                        ++(aws->stats.gapcnt);
                    }
                    if (sw_params)
                    {
                        if (aws->op_off == op_off_sent - 1 && sw_params->gap_end)
                            aws->stats.score -= sw_params->gap_end;
                        else
                            aws->stats.score -= sw_params->gap_ext;
                    }
                    ++(aws->stats.gaplen);
                    break;
                default: // just to satisfy older gcc syntax analyser :)
                    break;
            }

            aws->q_pos += aws->advance_qry * incr;
            aws->r_pos += aws->advance_ref * incr;

            if (step_callback && !step_callback (aws, aux))
                return 0;
        }
    }
    return 1;
}

void init_alignment_stats 
(
    tmap_map_alignment_stats* stats
)
{
    memset (stats, 0, sizeof (tmap_map_alignment_stats));
}

void init_segment 
(
    tmap_map_alignment_segment* segment
)
{
    if (segment->alignment)
    {
        segment->q_start = segment->alignment->qseq_off;
        segment->q_end = segment->alignment->qseq_len;
        segment->r_start = segment->alignment->rseq_off;
        segment->r_end = segment->alignment->rseq_len;
        segment->first_op = 0;
        segment->last_op = segment->alignment->ncigar ? (segment->alignment->ncigar - 1) : 0;
        segment->first_op_off = 0;
        if (segment->alignment->ncigar)
            segment->sent_op_off = bam_cigar_oplen (segment->alignment->cigar [segment->alignment->ncigar - 1]);
        else
            segment->sent_op_off = 0;
    }
    else
    {
        memset (segment, 0, sizeof (tmap_map_alignment_segment));
    }
}

uint8_t init_alignment
(
    tmap_map_alignment* dest,
    uint32_t* cigar,
    uint32_t ncigar,
    const char* qseq,
    uint32_t qseq_off,
    const char* rseq,
    uint32_t rseq_off
)
{
    dest->cigar = cigar;
    dest->ncigar = ncigar;

    dest->qseq = qseq;
    dest->q_int = 0;
    dest->qseq_off = qseq_off;

    dest->rseq = rseq;
    dest->r_int = 0;
    dest->rseq_off = rseq_off;

    uint32_t fp_q, fp_r;
    cigar_footprint (cigar, ncigar, &fp_q, &fp_r, NULL, NULL, NULL);

    dest->qseq_len = qseq_off + fp_q;
    dest->rseq_len = rseq_off + fp_r;

    if (dest->qseq_len > strlen (qseq))
        return 0;
    if (dest->rseq_len > strlen (rseq))
        return 0;
    return 1;
}

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
)
{
    dest->cigar = cigar;
    dest->ncigar = ncigar;

    dest->qseq = (const char*) qseq;
    dest->q_int = 1;
    dest->qseq_off = qseq_off;
    dest->qseq_len = qseq_len;

    dest->rseq = (const char*) rseq;
    dest->r_int = 1;
    dest->rseq_off = rseq_off;
    dest->rseq_len = rseq_len;

#if !defined (NDEBUG)
    // validate 
    uint32_t fp_q, fp_r;
    cigar_footprint (cigar, ncigar, &fp_q, &fp_r, NULL, NULL, NULL);
    assert (dest->qseq_len == qseq_off + fp_q);
    assert (dest->rseq_len == rseq_off + fp_r);
#endif

    return 1;
}


uint8_t validate_alignment
(
    tmap_map_alignment* al
)
{
    // compute cigar q and r footprints and make sure qseq and rseq are long enough
    uint32_t qlen, rlen, allen, left_clip, right_clip;
    if (!cigar_footprint (al->cigar, al->ncigar, &qlen, &rlen, &allen, &left_clip, &right_clip))
        return 0;
    if (al->qseq_len < al->qseq_off + qlen)
        return 0;
    if (al->rseq_len < al->rseq_off + rlen)
        return 0;
    return 1;
}


// NOTE: does not use walker as segment is supposed to be incomlete, which causes malfunction of walker loop
static uint8_t compute_alignment_segment_footprints 
(
    tmap_map_alignment_segment* segment,
    uint8_t error_policy
)
{
    // initial footprint marks
    uint32_t q_start, q_end, r_start, r_end;
    q_start = q_end = r_start = r_end = UINT32_MAX;

    // validate segment correctness
    if (!validate_segment_align (segment, error_policy))
        return 0;

    // walk the alignment and fill in sequence coordinates
    uint32_t *cigar_op_p, *cigar_cent_p, advance_type, advance_qry, advance_ref;
    uint32_t op = 0, op_len = -1, op_off, op_idx;
    uint32_t q_pos = segment->alignment->qseq_off;
    uint32_t r_pos = segment->alignment->rseq_off;

    for (cigar_op_p = segment->alignment->cigar, cigar_cent_p = cigar_op_p + segment->alignment->ncigar, op_idx = 0; 
         cigar_op_p != cigar_cent_p; 
         ++cigar_op_p, ++op_idx)
    {
        op = bam_cigar_op (*cigar_op_p);
        op_len = bam_cigar_oplen (*cigar_op_p);
        advance_type = bam_cigar_type (op);
        advance_qry = advance_type & CIGAR_CONSUME_QRY ? 1 : 0;
        advance_ref = advance_type & CIGAR_CONSUME_REF ? 1 : 0;
        for (op_off = 0; op_off != op_len; ++op_off)
        {
            if (op_idx == segment->first_op && op_off == segment->first_op_off)
            {
                q_start = q_pos;
                r_start = r_pos;
            }
            if (op_idx == segment->last_op && op_off == segment->sent_op_off)
            {
                q_end = q_pos;
                r_end = r_pos;
            }
            q_pos += advance_qry;
            r_pos += advance_ref;
        }
        // here we are at the last position of segment: op_off == op_len
        // this is valid offset; check if we are at the segment end
        // NOTE: the segment start at this position is not valid, only the segment end
        if (op_idx == segment->last_op && op_off == segment->sent_op_off)
        {
            q_end = q_pos;
            r_end = r_pos;
        }
    }
    // check if we found what we wanted
    if (q_start == UINT32_MAX || q_end== UINT32_MAX || r_start == UINT32_MAX || r_end == UINT32_MAX)
    {
        tmap_flagerr (error_policy, "Alignment segment boundaries beyond alignment size");
        return 0;
    }
    else // modify data only if result passed validation
    {
        segment->q_start = q_start, segment->q_end = q_end, segment->r_start = r_start, segment->r_end = r_end;
    }
    return 1;
}

// NOTE: does not use walker as segment is supposed to be incomlete, which causes malfunction of walker loop
static uint8_t compute_alignment_bounds_by_footprint 
(
    tmap_map_alignment_segment* segment,
    uint8_t error_policy
)
{
    // validate footprints
    if (!validate_segment_footprint (segment, error_policy))
        return 0;

    // local alignment bounds; we do not modify the passed in object unless operation succeeds
    uint32_t first_op, last_op, first_op_off, sent_op_off;
    first_op = last_op = first_op_off = sent_op_off = UINT32_MAX;

    // walk the alignment and fill in segment bounds
    uint32_t *cigar_op_p, *cigar_cent_p, advance_type, advance_qry, advance_ref;;
    uint32_t op = 0, op_len = -1, op_off, op_idx;
    uint32_t q_pos = segment->alignment->qseq_off;
    uint32_t r_pos = segment->alignment->rseq_off;
    for (cigar_op_p = segment->alignment->cigar, cigar_cent_p = cigar_op_p + segment->alignment->ncigar, op_idx = 0; 
         cigar_op_p != cigar_cent_p; 
         ++cigar_op_p, ++op_idx)
    {
        op = bam_cigar_op (*cigar_op_p);
        op_len = bam_cigar_oplen (*cigar_op_p);
        advance_type = bam_cigar_type (op);
        advance_qry = advance_type & CIGAR_CONSUME_QRY ? 1 : 0;
        advance_ref = advance_type & CIGAR_CONSUME_REF ? 1 : 0;
        for (op_off = 0; op_off != op_len; ++op_off)
        {
            if (q_pos == segment->q_start && r_pos == segment->r_start)
            {
                first_op = op_idx;
                first_op_off = op_off;
            }
            if (op_off && q_pos == segment->q_end && r_pos == segment->r_end) // end can not appear at the position 0 of a segment
            {
                last_op = op_idx;
                sent_op_off = op_off;
            }
            q_pos += advance_qry;
            r_pos += advance_ref;
        }
        if (op_off && q_pos == segment->q_end && r_pos == segment->r_end) // check for non-zero op_off prevents ending in zero-length segments
        {
            last_op = op_idx;
            sent_op_off = op_off;
        }
    }
    // check if we got all 
    if (first_op == UINT32_MAX ||  last_op == UINT32_MAX || first_op_off == UINT32_MAX || sent_op_off == UINT32_MAX)
    {
        tmap_flagerr (error_policy, "passed footprint does not fit into alignment boundaries");
        return 0;
    }
    else
    {
        segment->first_op = first_op, segment->first_op_off = first_op_off, segment->last_op = last_op, segment->sent_op_off = sent_op_off;
    }
    return 1;
}


static uint8_t normalize_alignment_segment_r
(
    tmap_map_alignment_segment* segment,
    uint8_t error_policy
)
{

    // check if we need to do anything
    if (segment_check_normalized (segment))
        return 1; // nothing to do

    // figure out if definition is sufficient 
    // the following completions are supported:
    //   given q_start, r_start, q_len and r_len, compute first_op, first_op_off, last_op, sent_op_off
    //   given first_op, first_op_off, last_op, sent_op_off, compute q_start, r_start, q_len and r_len
    //   (in some segment arrangements it is possible to recover full info based on less information, or on other set of defined values. 
    //    For example, one of the projections length can be omitted if the alignment segment ends in a match.
    //   For now such scenarios are not supported)

    uint8_t footprint_underdef = (segment->q_start == -1 || segment->q_end == -1 || segment->r_start == -1 || segment->r_end == -1);
    uint8_t boundaries_underdef = (segment->first_op == -1 || segment->first_op_off == -1 || segment->last_op == -1 || segment->sent_op_off == -1);
    if (footprint_underdef && boundaries_underdef)    
    {
        tmap_flagerr (error_policy, "Cannot normalize alignment segment: not sufficiently defined");
        return 0;
    }
    if (footprint_underdef) 
        return compute_alignment_segment_footprints (segment, error_policy);
    else
        return compute_alignment_bounds_by_footprint (segment, error_policy);
}


// computes values for the fields marked as 'unspecified' (holding values of -1)
// returns true (1) on success, false (0) on failure
uint8_t
tmap_map_normalize_alignment_segment
(
    tmap_map_alignment_segment* segment
)
{
    return normalize_alignment_segment_r (segment, align_seg_error_policy);
}

uint8_t segments_match
(
    const tmap_map_alignment_segment* s1,
    const tmap_map_alignment_segment* s2
)
{
    return  s1->alignment == s2->alignment &&
            s1->first_op == s2->first_op &&
            s1->first_op_off == s2->first_op_off &&
            s1->last_op == s2->last_op &&
            s1->sent_op_off == s2->sent_op_off &&
            s1->q_start == s2->q_start &&
            s1->q_end == s2->q_end &&
            s1->r_start == s2->r_start &&
            s1->r_end == s2->r_end;
}

uint8_t segment_stats_match 
(
    const tmap_map_alignment_stats* s1,
    const tmap_map_alignment_stats* s2
)
{
    return  s1->matches == s2->matches &&
            s1->mismatches == s2->mismatches &&
            s1->gapcnt == s2->gapcnt &&
            s1->gaplen == s2->gaplen &&
            s1->score == s2->score;
}

// alignment walking engine
// moves along cigar-encoded alignment path in given direction 
// advances state variables and calls registered callback with alignment_walk_position structure

uint8_t
alignment_walker 
(
    const tmap_map_alignment* alignment,
    const tmap_sw_param_t* sw_params,
    uint8_t from_beg,
    uint8_t (*step_callback)(const alignment_walk_state* state, void* aux),
    alignment_walk_state* aws,
    void* aux
)
{
    // make a segment
    tmap_map_alignment_segment segment;
    segment.alignment = alignment;
    segment.q_start = alignment->qseq_off;
    segment.q_end = alignment->qseq_off + alignment->qseq_len;
    segment.r_start = alignment->rseq_off;
    segment.r_end = alignment->rseq_off + alignment->rseq_len;
    segment.first_op = 0;
    segment.last_op = alignment->ncigar - 1;
    segment.first_op_off = 0;
    segment.sent_op_off = bam_cigar_oplen (alignment->cigar [alignment->ncigar - 1]);

    return alignment_segment_walker
    (
        &segment,
        sw_params,
        from_beg,
        step_callback,
        aws,
        aux
    );
}

uint8_t
alignment_segment_walker
(
    const tmap_map_alignment_segment* segment,
    const tmap_sw_param_t* sw_params,
    uint8_t from_beg,
    uint8_t (*step_callback)(const alignment_walk_state* state, void* aux),
    alignment_walk_state* aws,
    void* aux
)
{
    if (!segment->alignment || !segment->alignment->ncigar)
        return 0;

    alignment_walker_loop 
    (
        segment->alignment->cigar,
        segment->alignment->qseq,
        segment->alignment->rseq,
        segment->alignment->q_int,
        segment->alignment->r_int,
        from_beg ? segment->q_start : segment->q_end - 1,
        from_beg ? segment->r_start : segment->r_end - 1,
        from_beg ? segment->first_op : segment->last_op,
        from_beg ? segment->first_op_off : segment->sent_op_off - 1,
        from_beg ? segment->last_op : segment->first_op,
        from_beg ? segment->sent_op_off : segment->first_op_off - 1,
        sw_params,
        from_beg,
        step_callback,
        aws,
        aux
    );
    return 1;
}

typedef struct __worst_score_trace
{
    tmap_map_alignment_stats worst;
    int32_t worst_score_q_pos, worst_score_r_pos, worst_score_cigar_op, worst_score_cigar_op_off;
    uint8_t q_incr, r_incr; // keep query/reference increments at recorded positions in order to compute retained segment's sentinels for from the end (reversed) scans
    // int32_t prev_score;
}
worst_score_trace;

static uint8_t worst_score_keeper (const alignment_walk_state* state, void* aux)
{
    worst_score_trace* trace = (worst_score_trace*) aux;
    if (state->stats.score < trace->worst.score)
    {
        trace->worst.score = state->stats.score;
        trace->worst_score_q_pos = state->q_pos, trace->worst_score_r_pos = state->r_pos;
        trace->worst_score_cigar_op = state->op_idx; trace->worst_score_cigar_op_off = state->op_off;
        trace->worst.matches = state->stats.matches, trace->worst.mismatches = state->stats.mismatches, trace->worst.gaplen = state->stats.gaplen, trace->worst.gapcnt = state->stats.gapcnt;
        trace->q_incr = state->advance_qry;
        trace->r_incr = state->advance_ref;
    }
    return 1;
}


// finds the worst (lowest) score position from given end;
// fills in:
//       resulting retained segment (the one AFTER the worst position, excluding one, if reached in the direction of search) 
//       stats for the resulting retained segment 
//       stats for the clipped patrs of the alignment
// returns true on success, false if there is no segment with score worser than entire alignment
uint8_t tmap_map_find_worst_score_pos
(
    tmap_map_alignment* alignment,
    const tmap_sw_param_t* sw_params,
    uint8_t from_beg,
    tmap_map_alignment_segment* result,
    tmap_map_alignment_stats *result_stats,
    tmap_map_alignment_stats* clip_stats
)
{
    return tmap_map_find_worst_score_pos_x 
    (
        alignment,
        sw_params,
        from_beg,
        result,
        NULL,
        result_stats, 
        clip_stats,
        NULL,
        NULL,
        NULL,
        NULL
    );
}
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
    int32_t* opno_p,
    int32_t* opoff_p,
    int32_t* qpos_p,
    int32_t* rpos_p
)
{
    uint8_t rv = 1;
    alignment_walk_state aws;
    worst_score_trace trace;
    trace.worst.matches = 0, trace.worst.mismatches = 0, trace.worst.gapcnt = 0, trace.worst.gaplen = 0;
    if (from_beg)
    {
        trace.worst_score_q_pos = 0, trace.worst_score_r_pos = 0; // no bases seen yet at the beginning
        trace.worst_score_cigar_op = 0, trace.worst_score_cigar_op_off = -1; // before the first cigar op
    }
    else
    {
        trace.worst_score_q_pos = alignment->qseq_len - 1, trace.worst_score_r_pos = alignment->rseq_len - 1; // no bases seen yet at the beginning: seqlen - seqpos == 0
        trace.worst_score_cigar_op = alignment->ncigar - 1, trace.worst_score_cigar_op_off = bam_cigar_oplen (alignment->cigar [alignment->ncigar - 1]); // before the first cigar op: segment_len - position == 0
    }
    trace.worst.score = 0; // zero score is at the position before first alignment op is evaluated.

    if (!alignment_walker (alignment, sw_params, from_beg, worst_score_keeper, &aws, &trace))
        return 0; // this never happens, as callback never returns 0

    // store results
    if (opno_p)
        *opno_p = trace.worst_score_cigar_op;
    if (opoff_p)
        *opoff_p = trace.worst_score_cigar_op_off;
    if (qpos_p)
        *qpos_p = trace.worst_score_q_pos + (from_beg?0:1);
    if (rpos_p)
        *rpos_p = trace.worst_score_r_pos + (from_beg?0:1);;
    // now create the segments
    if (result) // only do if passed non-zero pointer to the result
    {
        result->alignment = alignment;
        if (from_beg)
        {
            result->first_op = trace.worst_score_cigar_op;
            result->first_op_off = trace.worst_score_cigar_op_off;
            next_cigar_pos (alignment->cigar, alignment->ncigar, &(result->first_op), &(result->first_op_off), 1);
            result->last_op = alignment->ncigar - 1;
            result->sent_op_off = bam_cigar_oplen (alignment->cigar [alignment->ncigar - 1]);
            result->q_start = trace.worst_score_q_pos;
            result->q_end = alignment->qseq_len;
            result->r_start = trace.worst_score_r_pos;
            result->r_end = alignment->rseq_len;
        }
        else
        {
            result->first_op = 0;
            result->first_op_off = 0;
            result->last_op = trace.worst_score_cigar_op;
            result->sent_op_off = trace.worst_score_cigar_op_off; // no need to convert in this case
            result->q_start = alignment->qseq_off;
            result->q_end = trace.worst_score_q_pos + 1;
            result->r_start = alignment->rseq_off;
            result->r_end = trace.worst_score_r_pos + 1;
        }
    }
    if (clip) // only do if passed non-zero pointer to clip
    {
        clip->alignment = alignment;
        if (from_beg)
        {
            clip->first_op = 0; // trace.worst_score_cigar_op;
            clip->first_op_off = 0; // trace.worst_score_cigar_op_off;
            clip->last_op = trace.worst_score_cigar_op; // alignment->ncigar - 1;
            clip->sent_op_off = trace.worst_score_cigar_op_off; // bam_cigar_oplen (alignment->cigar [alignment->ncigar - 1]);
            next_cigar_pos (alignment->cigar, alignment->ncigar, &(clip->last_op), &(clip->sent_op_off), 1);
            clip->q_start = 0; // trace.worst_score_q_pos;
            clip->q_end = trace.worst_score_q_pos; // alignment->qseq_len;
            clip->r_start = 0; // trace.worst_score_r_pos;
            clip->r_end = trace.worst_score_r_pos; // alignment->rseq_len;
        }
        else
        {
            clip->first_op = trace.worst_score_cigar_op; // 0;
            clip->first_op_off = trace.worst_score_cigar_op_off; // 0;
            clip->last_op = alignment->ncigar - 1; // trace.worst_score_cigar_op;
            clip->sent_op_off = bam_cigar_oplen (alignment->cigar [alignment->ncigar - 1]); // trace.worst_score_cigar_op_off; // no need to convert in this case
            clip->q_start = trace.worst_score_q_pos + 1; // alignment->qseq_off;
            clip->q_end = alignment->qseq_len; // trace.worst_score_q_pos + 1;
            clip->r_start = trace.worst_score_r_pos + 1; // alignment->rseq_off;
            clip->r_end = alignment->rseq_len; // trace.worst_score_r_pos + 1;
        }
    }
    // fill in the stats on result and reminder
    if (result_stats)
    {
        result_stats->score = aws.stats.score - trace.worst.score;
        result_stats->matches = aws.stats.matches - trace.worst.matches;
        result_stats->mismatches = aws.stats.mismatches - trace.worst.mismatches;
        result_stats->gapcnt = aws.stats.gapcnt - trace.worst.gapcnt; // could be wrong!! (gap can split). Should check for worst score pos being inside gap. Not possible with Smit-Waterman scoring, so not relevant until other scores are implemented
        result_stats->gaplen = aws.stats.gaplen - trace.worst.gaplen;
    }
    if (clip_stats)
    {
        clip_stats->score = trace.worst.score;
        clip_stats->matches = trace.worst.matches;
        clip_stats->mismatches = trace.worst.mismatches;
        clip_stats->gapcnt = trace.worst.gapcnt;
        clip_stats->gaplen = trace.worst.gaplen;
    }
    rv = (trace.worst.score != 0); 
    return rv;
}


typedef struct __seg_move_state
{
    int32_t bases_left;
    tmap_map_alignment_segment* src;
    tmap_map_alignment_segment* dst;
    tmap_map_alignment_stats* src_stats;
    tmap_map_alignment_stats* dst_stats;
    const tmap_sw_param_t* sw_par;
}
SegMoveState;

static uint8_t seg_mover (const alignment_walk_state* state, void* aux)
{
    SegMoveState* st = (SegMoveState*) aux;
    const tmap_map_alignment* al = st->src->alignment;
    if (st->bases_left < 0)
    {
        // move one position from the beginning of src to the end of dest
        assert (st->src->q_start < st->src->q_end);

        // check if sync is mantained
        assert (st->src->first_op == state->op_idx && st->src->first_op_off == state->op_off);

        // advance start pos of src
        uint8_t res = next_cigar_pos (al->cigar, al->ncigar, &(st->src->first_op), &(st->src->first_op_off), 1);
        assert (res);
        st->src->q_start += state->advance_qry;
        st->src->r_start += state->advance_ref;

        // advance end pos of dest
        res = next_cigar_pos (al->cigar, al->ncigar, &(st->dst->last_op), &(st->dst->sent_op_off), 1);
        assert (res);
        st->dst->q_end += state->advance_qry;
        st->dst->r_end += state->advance_ref;

        st->bases_left += state->advance_qry;
    }
    else if (st->bases_left > 0)
    {
        // move one position from the end of src to the beginning of dest
        assert (st->src->q_start < st->src->q_end);

        // check if sync is mantained
#if !defined (NDEBUG)
        int32_t op_idx = st->src->last_op, op_off = st->src->sent_op_off;
        next_cigar_pos (al->cigar, al->ncigar, &op_idx, &op_off, -1);
        assert (op_idx == state->op_idx && op_off == state->op_off);
#endif

        // retract end pos of src
        uint8_t res = next_cigar_pos (al->cigar, al->ncigar, &(st->src->last_op), &(st->src->sent_op_off), -1);
        assert (res);
        st->src->q_end -= state->advance_qry;
        st->src->r_end -= state->advance_ref;

        // retract start pos of dest
        res = next_cigar_pos (al->cigar, al->ncigar, &(st->dst->first_op), &(st->dst->first_op_off), -1);
        assert (res);
        st->dst->q_start -= state->advance_qry;
        st->dst->r_start -= state->advance_ref;

        st->bases_left -= state->advance_qry;
    }
    // update stats
    if (st->src_stats || st->dst_stats)
    {
        int32_t gap_open = (state->op == BAM_CINS || state->op == BAM_CDEL)?((state->from_beg)?(state->op_off == 0):(state->op_off + 1 == state->op_len)):0; // do we want assymetry here?
        // int32_t gap_open = (state->op == BAM_CINS || state->op == BAM_CDEL)?(state->op_off == 0):0;
        int32_t gap_ext  = (state->op == BAM_CINS || state->op == BAM_CDEL); 
        int32_t gap_end  = (state->op == BAM_CINS || state->op == BAM_CDEL)?((state->from_beg)?(state->op_off + 1 == state->op_len):(state->op_off == 0)):0;
        // int32_t gap_end  = (state->op == BAM_CINS || state->op == BAM_CDEL)?(state->op_off + 1 == state->op_len):0;
        int32_t match    = (state->op == BAM_CEQUAL || (state->op == BAM_CMATCH && state->qbase == state->rbase));
        int32_t mism     = (state->op == BAM_CDIFF || (state->op == BAM_CMATCH && state->qbase != state->rbase));
        int32_t src_score_delta = 0;
        int32_t dst_score_delta = 0;
        int32_t s;
        if (st->sw_par)
        {
            if (gap_end) // gap disappears from src
            {
                src_score_delta += st->sw_par->gap_open; // gap disappears from src
                src_score_delta += ((st->sw_par->gap_end)?(st->sw_par->gap_end):(st->sw_par->gap_ext)); // gap disappears from src - subtract gap end score (add penalty)
            }
            else if (gap_ext && !gap_open && !gap_end) // inside a gap
            {
                src_score_delta += st->sw_par->gap_ext; // gap continues (and not ends) in src - subtract extension score (add penalty)
                dst_score_delta -= st->sw_par->gap_ext; // gap continues (and not ends) in dst - add extension score (subtract penalty)
            }
            if (gap_open) // gap appears in dst
            {
                dst_score_delta -= gap_open * st->sw_par->gap_open, // new gap opens in dst
                dst_score_delta -= gap_open * ((st->sw_par->gap_end)?(st->sw_par->gap_end):(st->sw_par->gap_ext)); // gap appears in dst - add gap end score (subtractpenalty)
            }
            if (match || mism) 
            {
                s = st->sw_par->matrix [st->sw_par->row * state->rbase + state->qbase];
                dst_score_delta += s;
                src_score_delta -= s;
            }
        }
        if (st->src_stats)
        {
            st->src_stats->gapcnt -= gap_end;
            st->src_stats->gaplen -= gap_ext;
            st->src_stats->matches -= match;
            st->src_stats->matches -= mism;
            st->src_stats->score += src_score_delta;
        }
        if (st->dst_stats)
        {
            st->dst_stats->gapcnt += gap_open;
            st->dst_stats->gaplen += gap_ext;
            st->dst_stats->matches += match;
            st->dst_stats->matches += mism;
            st->dst_stats->score += dst_score_delta;
        }
    }
    // check if we reached the end of query before advancing by needed number of positions
    if (st->src->q_start == st->src->q_end)
        return 0;
    // sanity (should not happen, query should exhaust as well)
    assert (st->src->first_op != st->src->last_op || st->src->first_op_off != st->src->sent_op_off);

    return (st->bases_left != 0);
}

// move given number of query bases from one alignment segment to another
// segments have to be adjacent, seg2 must imeediately follow seg1
// returns number of bases moved
// NOTE: it is safe to alter the segment being walked from walk callback, as far as the underlying alignment does not get changed.

uint32_t
tmap_map_alignment_segment_move_bases
(
    tmap_map_alignment_segment* seg1,
    tmap_map_alignment_segment* seg2,
    int32_t bases, // negative value means move from seg2 to seg1
    const tmap_sw_param_t* sw_params,     // optional, pass NULL to skip updating scores in stats
    tmap_map_alignment_stats* seg1_stats, // optional stats to update; should be pre-populated with stats for seg1 or NULL
    tmap_map_alignment_stats* seg2_stats  // optional stats to update; should be pre-populated with stats for seg2 or NULL
)
{
    assert (seg1->alignment == seg2->alignment);
    assert (seg1->last_op == seg2->first_op);
    assert (seg1->sent_op_off == seg2->first_op_off);
    assert (seg1->q_end == seg2->q_start);
    assert (seg1->r_end == seg2->r_start);

    // walk source segment, moving the steps from source into the destination
    // exit walker when desired number of query positions are copied

    // first, check if move is possible
    if (!bases)
        return 0;
    if (bases > 0 && seg1->q_end == seg1->q_start) // zero source footprint, no way to move
        return 0;
    if (bases < 0 && seg2->q_end == seg2->q_start) // zero source footprint, no way to move
        return 0;
    alignment_walk_state state;
    SegMoveState upd;
    upd.bases_left = bases;
    if (bases > 0)
        upd.src = seg1, upd.dst = seg2, upd.src_stats = seg1_stats, upd.dst_stats = seg2_stats;
    else
        upd.src = seg2, upd.dst = seg1, upd.src_stats = seg2_stats, upd.dst_stats = seg1_stats;
    upd.sw_par = sw_params;

    tmap_map_alignment_segment tpl = *upd.src; // walk the segment that is not getting modified in process - otherwise the gap opening position can be counted multiple times

    int32_t orig_seg1_len = seg1->q_end - seg1->q_start;
    alignment_segment_walker (
        &tpl, 
        sw_params, 
        (bases < 0), 
        seg_mover,
        &state,
        &upd);
    int32_t result_seg1_len = (seg1->q_end - seg1->q_start);

    return (bases > 0)?(orig_seg1_len - result_seg1_len):(result_seg1_len - orig_seg1_len);
}


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
)
{
    return 0;
}


// computes statistics on the given alignment segment
// returns SW score
int32_t tmap_map_alignment_segment_score
(
    tmap_map_alignment_segment* segment,
    const tmap_sw_param_t* sw_params,
    tmap_map_alignment_stats *result_stats
)
{
    assert (segment_check_normalized (segment));
    alignment_walk_state aws;
    alignment_segment_walker (segment, sw_params, 1, NULL, &aws, NULL);
    *result_stats = aws.stats;
    return aws.stats.score;
}


static uint8_t seg_sentry (const alignment_walk_state* state, void* aux) // would be fun to implement as lambda; but this is not c++11+
{
    const int32_t ref_sent = *(int32_t *) aux;
    return state->r_pos != ref_sent; // terminate walker loop when reference sentinel is reached
}

// modifies segment so that it extends over reference either from the beginning of given segment to the given sentinel (when from_beg == 1),
// or from (given sentinel+1) to the given segment's end (when from_beg == 0),
// the content of a segment is discarded
// returns 1 on success, 0 if target position does not lay within an alignment
int8_t tmap_map_segment_clip_to_ref_base 
(
    tmap_map_alignment_segment* segment, 
    int8_t from_beg, 
    int32_t ref_pos_sentinel
)
{
    // walk the segment using the walker until reference position reaches the centinel

    // first, check if move is possible
    if (from_beg && (ref_pos_sentinel < segment->r_start || ref_pos_sentinel > segment ->r_end))
        return 0;
    if (!from_beg && (ref_pos_sentinel + 1 < segment->r_start || ref_pos_sentinel + 1 > segment ->r_end)) 
        return 0;
    // now, check if move is needed
    if (from_beg && ref_pos_sentinel == segment->r_end)
        return 1;
    if (!from_beg && ref_pos_sentinel + 1 == segment ->r_start)
        return 1;
    // now, check if we arrived before making any steps
    if (from_beg && ref_pos_sentinel == segment->r_start)
    {
        // segment is zero length
        segment->r_end = segment->r_start;
        segment->q_end = segment->q_start;
        segment->last_op = segment->first_op;
        segment->sent_op_off = segment->first_op_off;
        return 1;
    }
    if (!from_beg && ref_pos_sentinel + 1 == segment->r_end)
    {
        // segment is zero length
        segment->r_start = segment->r_end;
        segment->q_start = segment->q_end;
        segment->first_op = segment->last_op;
        segment->first_op_off = segment->sent_op_off;
        return 1;
    }

    // now the move is possible and we need to move
    alignment_walk_state state;
    alignment_segment_walker (
        segment, 
        NULL, 
        from_beg, 
        seg_sentry,
        &state,
        &ref_pos_sentinel);

    assert (state.r_pos == ref_pos_sentinel);

    // update segment
    if (from_beg)
    {
        segment->q_end = state.q_pos;
        segment->r_end = state.r_pos;
        segment->last_op = state.op_idx;
        segment->sent_op_off = state.op_off;
        uint8_t res = next_cigar_pos (segment->alignment->cigar, segment->alignment->ncigar, &(segment->last_op), &(segment->sent_op_off), 1);
        assert (res);
    }
    else
    {
        segment->q_start = state.q_pos + 1;
        segment->r_start = state.r_pos + 1;
        segment->first_op = state.op_idx;
        segment->first_op_off = state.op_off;
    }
    return 1;

}
