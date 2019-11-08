#include "tmap_map_locopt.h"
#include <stddef.h>
#include <stdlib.h>

void tmap_map_locopt_init (tmap_map_locopt_t* locopt)
{
    locopt->use_bed_in_end_repair.over = 0;
    locopt->score_match.over = 0;
    locopt->pen_mm.over = 0;
    locopt->pen_gapo.over = 0;
    locopt->pen_gape.over = 0;
    locopt->pen_gapl.over = 0;
    locopt->gapl_len.over = 0;
    locopt->bw.over = 0;
    locopt->softclip_type.over = 0;
    locopt->end_repair.over = 0;
    locopt->end_repair_he.over = 0;
    locopt->end_repair_le.over = 0;
    locopt->max_one_large_indel_rescue.over = 0;
    locopt->max_one_large_indel_rescue_he.over = 0;
    locopt->max_one_large_indel_rescue_le.over = 0;
    locopt->min_anchor_large_indel_rescue.over = 0;
    locopt->min_anchor_large_indel_rescue_he.over = 0;
    locopt->min_anchor_large_indel_rescue_le.over = 0;
    locopt->max_amplicon_overrun_large_indel_rescue.over = 0;
    locopt->max_amplicon_overrun_large_indel_rescue_he.over = 0;
    locopt->max_amplicon_overrun_large_indel_rescue_le.over = 0;
    locopt->max_adapter_bases_for_soft_clipping.over = 0;
    locopt->max_adapter_bases_for_soft_clipping_he.over = 0;
    locopt->max_adapter_bases_for_soft_clipping_le.over = 0;
    locopt->end_repair_5_prime_softclip.over = 0;
    locopt->end_repair_5_prime_softclip_he.over = 0;
    locopt->end_repair_5_prime_softclip_le.over = 0;
    locopt->do_realign.over = 0;
    locopt->realign_mat_score.over = 0;
    locopt->realign_mis_score.over = 0;
    locopt->realign_gip_score.over = 0;
    locopt->realign_gep_score.over = 0;
    locopt->realign_bandwidth.over = 0;
    locopt->realign_cliptype.over = 0;
    locopt->do_repeat_clip.over = 0;
    locopt->repclip_continuation.over = 0;
    locopt->cigar_sanity_check.over = 0;
    locopt->do_hp_weight.over = 0;
    locopt->gap_scale_mode.over = 0;
    locopt->context_mat_score.over = 0;
    locopt->context_mis_score.over = 0;
    locopt->context_gip_score.over = 0;
    locopt->context_gep_score.over = 0;
    locopt->context_extra_bandwidth.over = 0;
    locopt->specific_log.over = 0;
    locopt->debug_log.over = 0;
    locopt->fscore.over = 0;
    locopt->softclip_key.over = 0;
    locopt->ignore_flowgram.over = 0;
    locopt->aln_flowspace.over = 0;
    locopt->stage_sw_params = NULL;
    locopt->stages_allocated = 0;
    locopt->stages_used = 0;
}

void tmap_map_locopt_destroy (tmap_map_locopt_t* locopt)
{
    if (locopt && locopt->stage_sw_params)
    {
        tmap_map_stage_sw_param_ovr_t* p, *sent;
        for (p = locopt->stage_sw_params, sent = locopt->stage_sw_params + locopt->stages_used; p != sent; ++p)
            if (p->sw_params)
            {
                if (p->sw_params->matrix_owned)
                    free (p->sw_params->matrix);
                free (p->sw_params);
            }
        free (locopt->stage_sw_params);
        locopt->stage_sw_params = NULL; // for tracing
    }
}

tmap_sw_param_t* tmap_map_locopt_get_stage_sw_params (tmap_map_locopt_t* locopt, int32_t stage_ord)
{
    if (locopt && locopt->stage_sw_params) 
    {
        tmap_map_stage_sw_param_ovr_t* p, *sent;
        for (p = locopt->stage_sw_params, sent = locopt->stage_sw_params + locopt->stages_used; p != sent; ++p)
            if (p->stage == stage_ord)
                return p->sw_params;
    }
    return NULL;
}
