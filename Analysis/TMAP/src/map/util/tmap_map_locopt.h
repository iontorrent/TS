#ifndef TMAP_MAP_LOCOPT_H
#define TMAP_MAP_LOCOPT_H

#include <sys/types.h>
#include "../../sw/tmap_sw.h"

#pragma pack (push, 1)

typedef struct __tmap_map_opt_ovr_i_t
{
    int32_t value;
    u_int8_t over;
}
tmap_map_opt_ovr_i_t;

typedef struct __tmap_map_opt_ovr_f_t
{
    double value;
    u_int8_t over;
}
tmap_map_opt_ovr_f_t;

typedef struct __tmap_map_stage_sw_params_ovr_t
{
    uint32_t stage;
    tmap_sw_param_t* sw_params;
} tmap_map_stage_sw_param_ovr_t;

// defines options that can be overriden based on read / base position on the reference genome
typedef struct __tmap_map_locopt_t
{
    tmap_map_opt_ovr_i_t use_bed_in_end_repair; /*!< use amplicon edge coordinates in end repair to find possible edge-bound long gaps or inserts */
    tmap_map_opt_ovr_i_t score_match; /*!< the match score (-A,--score-match) */
    tmap_map_opt_ovr_i_t pen_mm;  /*!< the mismatch penalty (-M,--pen-mismatch) */
    tmap_map_opt_ovr_i_t pen_gapo;  /*!< the indel open penalty (-O,--pen-gap-open) */
    tmap_map_opt_ovr_i_t pen_gape;  /*!< the indel extension penalty (-E,--pen-gap-extension) */
    tmap_map_opt_ovr_i_t pen_gapl;  /*!< the long indel penalty (-G,--pen-gap-long) */
    tmap_map_opt_ovr_i_t gapl_len;  /*!< the number of extra bases to add when searching for long indels (-K, --gap-long-length) */ 
    tmap_map_opt_ovr_i_t bw; /*!< the extra bases to add before and after the target during Smith-Waterman (-w,--band-width) */
    tmap_map_opt_ovr_i_t softclip_type; /*!< soft clip type (-g,--softclip-type) */
    tmap_map_opt_ovr_i_t end_repair; // /*!< specifies if and how to perform end repair on higher coordinate end of the amplicon (0 - disabled, 1 - prefer mismatches, 2 - prefer indels) (--end-repair) */
    tmap_map_opt_ovr_i_t end_repair_he; // /*!< specifies to perform 5' end repair on higher coordinate end of the amplicon (0 - disabled, 1 - prefer mismatches, 2 - prefer indels) (--end-repair) */
    tmap_map_opt_ovr_i_t end_repair_le; /*!< specifies to perform 5' end repair on lower coordinate end of the amplicon (0 - disabled, 1 - prefer mismatches, 2 - prefer indels) (--end-repair) */
    tmap_map_opt_ovr_i_t max_one_large_indel_rescue;  /*!< Try to rescue lone indel with amplicon info, largest gap to be rescued*/
    tmap_map_opt_ovr_i_t max_one_large_indel_rescue_he;  /*!< Try to rescue lone indel with amplicon info, largest gap to be rescued  on higher coordinate end of the amplicon*/
    tmap_map_opt_ovr_i_t max_one_large_indel_rescue_le;  /*!< Try to rescue lone indel with amplicon info, largest gap to be rescued  on lower coordinate end of the amplicon*/
    tmap_map_opt_ovr_i_t min_anchor_large_indel_rescue; /*!< minimum size of anchor needed to open one large gap*/
    tmap_map_opt_ovr_i_t min_anchor_large_indel_rescue_he; /*!< minimum size of anchor needed to open one large gap on higher coordinate end of the amplicon*/
    tmap_map_opt_ovr_i_t min_anchor_large_indel_rescue_le; /*!< minimum size of anchor needed to open one large gap on lower coordinate end of the amplicon*/
    tmap_map_opt_ovr_i_t max_amplicon_overrun_large_indel_rescue; /*!< maximum allowed alignment to overrun amplicon edge in one large indel alignment*/
    tmap_map_opt_ovr_i_t max_amplicon_overrun_large_indel_rescue_he; /*!< maximum allowed alignment to overrun amplicon edge in one large indel alignment on higher coordinate end of the amplicon*/
    tmap_map_opt_ovr_i_t max_amplicon_overrun_large_indel_rescue_le; /*!< maximum allowed alignment to overrun amplicon edge in one large indel alignment on lower coordinate end of the amplicon*/
    tmap_map_opt_ovr_i_t max_adapter_bases_for_soft_clipping; /*!< specifies to perform 3' soft-clipping (via -g) if at most this # of adapter bases were found (ZB tag) (--max-adapter-bases-for-soft-clipping) */ 
    tmap_map_opt_ovr_i_t max_adapter_bases_for_soft_clipping_he; /*!< specifies to perform 3' soft-clipping on higher coordinate end of the amplicon (via -g) if at most this # of adapter bases were found (ZB tag) (--max-adapter-bases-for-soft-clipping) */ 
    tmap_map_opt_ovr_i_t max_adapter_bases_for_soft_clipping_le; /*!< specifies to perform 3' soft-clipping on lower coordinate end of the amplicon (via -g) if at most this # of adapter bases were found (ZB tag) (--max-adapter-bases-for-soft-clipping) */ 
    tmap_map_opt_ovr_i_t end_repair_5_prime_softclip; /*!< end-repair is allowed to introduce 5' softclip*/
    tmap_map_opt_ovr_i_t end_repair_5_prime_softclip_he; /*!< end-repair is allowed to introduce 5' softclip on higher coordinate end of the amplicon*/
    tmap_map_opt_ovr_i_t end_repair_5_prime_softclip_le; /*!< end-repair is allowed to introduce 5' softclip on lower coordinate end of the amplicon*/
    tmap_map_opt_ovr_i_t do_realign; /*!< perform realignment after mapping */
    tmap_map_opt_ovr_i_t realign_mat_score; /*!< realignment match score */
    tmap_map_opt_ovr_i_t realign_mis_score; /*!< realignment mismatch score */
    tmap_map_opt_ovr_i_t realign_gip_score; /*!< realignment gap opening score */
    tmap_map_opt_ovr_i_t realign_gep_score; /*!< realignment gap extension score */
    tmap_map_opt_ovr_i_t realign_bandwidth; /*!< realignment DP matrix band width */
    tmap_map_opt_ovr_i_t realign_cliptype; /*!< realignment clipping type: 0: none, 1: semiglobal, 2: semiglobal+soft clip bead end, 3: semiglobal + soft clip key end, 4: local alignment */
    tmap_map_opt_ovr_i_t do_repeat_clip; /*!< clip tandem repeats at the alignment ends */
    tmap_map_opt_ovr_i_t repclip_continuation;  /*!< repeat clipping performed only if repeat continues past the end of the read into the reference by at least 1 period */
    tmap_map_opt_ovr_i_t cigar_sanity_check; /*!< check cigar conformity (detail levels 0(default) to 9(all checks)*/
    tmap_map_opt_ovr_i_t do_hp_weight; /*!< perform realignment with context-specific gap scores */
    tmap_map_opt_ovr_i_t gap_scale_mode; /*!< determines gap scaling mode (none, gep-only, gip-and-gep */
    tmap_map_opt_ovr_f_t context_mat_score; /*!< context realignment match score */
    tmap_map_opt_ovr_f_t context_mis_score; /*!< context realignment mismatch score */
    tmap_map_opt_ovr_f_t context_gip_score; /*!< context realignment gap opening score */
    tmap_map_opt_ovr_f_t context_gep_score; /*!< context realignment gap extension score */
    tmap_map_opt_ovr_i_t context_extra_bandwidth; /*!< context realignment DP matrix extra band width */
    tmap_map_opt_ovr_i_t specific_log; /*!< restricts logging only to amplicons with this override. If not specified for any amplicon, and logging enabled, logs everything */
    tmap_map_opt_ovr_i_t debug_log; /*!< output detailed log of match post-processing into a log file (designated by realign_log) */
    tmap_map_opt_ovr_i_t fscore;  /*!< the flow score penalty (-X,--pen-flow-error) */
    tmap_map_opt_ovr_i_t softclip_key; /*!< soft clip only the last base of the key (-y,--softclip-key) */
    tmap_map_opt_ovr_i_t ignore_flowgram;  /*!< specifies to ignore the flowgram if available (-S,--ignore-flowgram) */
    tmap_map_opt_ovr_i_t aln_flowspace; /*!< produce the final alignment in flow space (-F,--final-flowspace). */
    tmap_map_stage_sw_param_ovr_t *stage_sw_params; /*!< stage-specific overrides for SW parameters, or NULL */
    uint32_t stages_allocated;
    uint32_t stages_used;
}
tmap_map_locopt_t;

#pragma pack (pop)

void tmap_map_locopt_init (tmap_map_locopt_t* locopt);
void tmap_map_locopt_destroy (tmap_map_locopt_t* locopt);
tmap_sw_param_t* tmap_map_locopt_get_stage_sw_params (tmap_map_locopt_t* locopt, int32_t stage_ord);

#endif // TMAP_MAP_LOCOPT_H



