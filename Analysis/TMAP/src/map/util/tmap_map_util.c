/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <alloca.h>
#include "../../util/tmap_alloc.h"
#include "../../util/tmap_error.h"
#include "../../util/tmap_bsearch.h"
#include "../../util/tmap_histo.h"
#include "../../util/tmap_sam_convert.h"
#include "../../util/tmap_progress.h"
#include "../../util/tmap_sort.h"
#include "../../util/tmap_definitions.h"
#include "../../util/tmap_rand.h"
#include "../../seq/tmap_seq.h"
#include "../../index/tmap_refseq.h"
#include "../../index/tmap_bwt.h"
#include "../../index/tmap_sa.h"
#include "../../sw/tmap_sw.h"
#include "../../sw/tmap_fsw.h"
#include "../../sw/tmap_vsw.h"
#include "samtools/bam.h"
#include "tmap_map_opt.h"
#include "tmap_map_util.h"
#include "tmap_map_align_util.h"
#include "../../sw/tmap_sw.h"

// #define CONCURRENT_PARAMETERS_CACHE 1
#ifdef CONCURRENT_PARAMETERS_CACHE

#ifdef HAVE_LIBPTHREAD
#include <pthread.h>
#endif

#endif

#include "../../realign/realign_c_util.h"

extern int32_t local_ovrs;

#define tmap_map_util_reverse_query(_query, _ql, _i) \
  for(_i=0;_i<(_ql>>1);_i++) { \
      uint8_t _tmp = _query[_i]; \
      _query[_i] = _query[_ql-_i-1]; \
      _query[_ql-_i-1] = _tmp; \
  }


// #define WARN_END_REPAIR_FAILURE 1

// sort by strand, min-seqid, min-position
#define __tmap_map_sam_sort_coord_lt(a, b) (  ((a).strand < (b).strand) \
                                            || ( (a).strand == (b).strand && (a).seqid < (b).seqid) \
                                            || ( (a).strand == (b).strand && (a).seqid == (b).seqid && (a).pos < (b).pos ) \
                                            ? 1 : 0 )

#define __tmap_map_sam_sort_coord_end_lt(a, b) (  ((a).strand < (b).strand) \
                                            || ( (a).strand == (b).strand && (a).seqid < (b).seqid) \
                                            || ( (a).strand == (b).strand && (a).seqid == (b).seqid && (a).pos + (a).target_len < (b).pos + (b).target_len) \
                                            ? 1 : 0 )

// sort by strand, min-seqid, min-position, max score
#define __tmap_map_sam_sort_coord_score_lt(a, b) (  ((a).strand < (b).strand) \
                                            || ( (a).strand == (b).strand && (a).seqid < (b).seqid) \
                                            || ( (a).strand == (b).strand && (a).seqid == (b).seqid && (a).pos < (b).pos ) \
                                            || ( (a).strand == (b).strand && (a).seqid == (b).seqid && (a).pos == (b).pos && (a).score > (b).score) \
                                            ? 1 : 0 )

// sort by max-score, min-seqid, min-position, min-strand
#define __tmap_map_sam_sort_score_coord_lt(a, b) (  ((a).score > (b).score) \
                                            || ((a).score == (b).score && (a).strand < (b).strand) \
                                            || ((a).score == (b).score && (a).strand == (b).strand && (a).seqid < (b).seqid) \
                                            || ((a).score == (b).score && (a).strand == (b).strand && (a).seqid == (b).seqid && (a).pos < (b).pos) \
                                            ? 1 : 0 )

TMAP_SORT_INIT(tmap_map_sam_sort_coord, tmap_map_sam_t, __tmap_map_sam_sort_coord_lt)
TMAP_SORT_INIT(tmap_map_sam_sort_coord_end, tmap_map_sam_t, __tmap_map_sam_sort_coord_end_lt)
TMAP_SORT_INIT(tmap_map_sam_sort_coord_score, tmap_map_sam_t, __tmap_map_sam_sort_coord_score_lt)
TMAP_SORT_INIT(tmap_map_sam_sort_score_coord, tmap_map_sam_t, __tmap_map_sam_sort_score_coord_lt)

static void tmap_map_util_set_target_len( tmap_map_sam_t *s)
{
    s->target_len = 0;
    int j;
    for (j = 0; j < s->n_cigar; j++)
    {
        switch (TMAP_SW_CIGAR_OP (s->cigar [j]))
        {
            case BAM_CMATCH:
            case BAM_CDEL:
                s->target_len += TMAP_SW_CIGAR_LENGTH(s->cigar[j]);
                break;
            default:
                break;
        }
    }
}

int
tmap_map_get_amplicon
(
    tmap_refseq_t *refseq,
    int32_t seqid,
    uint32_t start,
    uint32_t end,
    uint32_t strand,
    uint32_t *ampl_start,
    uint32_t *ampl_end,
    tmap_map_locopt_t** locopt,
    tmap_map_endstat_p_t* read_ends
)
{
    uint32_t *srh;
    int32_t i, j;
    uint32_t key;
    if (!refseq->bed_exist || refseq->bednum [seqid] == 0)
        return 0;
    if (strand == 0)
    {
        srh = refseq->bedstart [seqid];
        key = start;
    }
    else
    {
        srh = refseq->bedend [seqid];
        key = end;
    }
    // binary search
    i = 0; j = refseq->bednum [seqid] - 1;
    while (i < j)
    {
            uint32_t m = (i + j) / 2;
            if (srh [m] == key)
            {
                i = j = m;
                break;
            }
            if (srh [m] < key)
                i = m + 1;
            else
                j = m;
    }
    --i;
    if (i < 0)
        i = 0;
    ++j;
    if (j >= refseq->bednum [seqid])
        j = refseq->bednum [seqid] - 1;
    // uint32_t best = abs (srh [i] - key), bi = i; // DK: this is rather dangerous, as (uint32_t (K) - uint32_t (K+1)) == UINT32_MAX!
    uint32_t best = abs (((int32_t) srh [i]) - ((int32_t) key)), bi = i; // as far as contigs are below 2 Gbases, this is Ok.
    for (i++; i <= j; ++i)
    {
        if (abs (((int32_t) srh [i]) - ((int32_t) key)) < best)
        {
            // best = abs (srh [i] - key);
            best = abs (((int32_t) srh [i]) - ((int32_t) key));
            bi = i;
        }
    }
    i = bi;
    // check for overlap
    if (refseq->bedstart [seqid][i] < end && start < refseq->bedend [seqid][i])
    {
        if (ampl_start)
            *ampl_start = refseq->bedstart [seqid][i];
        if (ampl_end)
            *ampl_end = refseq->bedend [seqid][i];
        if (locopt)
        {
            if (refseq->parovr && refseq->parovr [seqid] && refseq->parovr [seqid][i] != UINT32_MAX)
                *locopt = refseq->parmem + refseq->parovr [seqid][i];
            else
                *locopt = NULL;

        }
        if (read_ends)
        {
            if (refseq->read_ends && refseq->read_ends [seqid] && refseq->read_ends [seqid][i].index != UINT32_MAX)
            {
                tmap_map_endstat_t* src = refseq->read_ends [seqid] + i;
                read_ends->positions = refseq->endposmem + src->index;
                read_ends->starts_count = src->starts_count;
                read_ends->ends_count = src->ends_count;
            }
            else
            {
                read_ends->positions = NULL;
                read_ends->starts_count = 0;
                read_ends->ends_count = 0;
            }
        }
        return 1;
    }
    else
        return 0;
}

typedef struct __tmap_map_amplicon_info
{
    uint32_t ampl_start;
    uint32_t ampl_end;
    tmap_map_locopt_t* locopt;
    tmap_map_endstat_p_t read_ends;
}
tmap_map_amplicon_info;

typedef struct __candeval_overrides_s
{
    tmap_map_amplicon_info* source;
    int32_t softclip_type;
    int32_t max_adapter_bases_for_soft_clipping;
    tmap_vsw_opt_t vsw;
} 
candeval_overrides_s;

uint32_t candeval_overrides_same (candeval_overrides_s* p1, candeval_overrides_s* p2)
{
    return p1->softclip_type == p2->softclip_type &&
           p1->max_adapter_bases_for_soft_clipping == p2->max_adapter_bases_for_soft_clipping &&
           p1->vsw.score_match == p2->vsw.score_match &&
           p1->vsw.pen_mm == p2->vsw.pen_mm &&
           p1->vsw.pen_gapo == p2->vsw.pen_gapo &&
           p1->vsw.pen_gape == p2->vsw.pen_gape &&
           p1->vsw.score_thres == p2->vsw.score_thres;
}

#define AMPL_INFO_INIT_CAPACITY 0
static uint32_t capacity_snapper (uint32_t desired_capacity)
{
    // use closest higher degree of 2, but not less then AMPL_INFO_INIT_CAPACITY

    assert (desired_capacity <= INT32_MAX); // for the logic below to work, the upper bit should be 0
    uint8_t exp = 0;
    if (desired_capacity == 0)
        return 0;
    if (desired_capacity == 1)
        return 1;
    while (desired_capacity)
    {
        desired_capacity >>= 1;
        ++exp;
    }
    uint32_t capacity = 1 << exp;
#if AMPL_INFO_INIT_CAPACITY > 0
    if (capacity < AMPL_INFO_INIT_CAPACITY)
        capacity = AMPL_INFO_INIT_CAPACITY;
#endif
    return capacity;
}

static void init_amplicon_info (tmap_map_amplicon_info** ampl_info, uint32_t* capacity, uint32_t preallocate)
{
    assert (ampl_info);
    assert (capacity);
    preallocate = capacity_snapper (preallocate);
    if (preallocate)
        *ampl_info = (tmap_map_amplicon_info*) tmap_calloc (preallocate, sizeof (tmap_map_amplicon_info), "amplicon_info");
    else 
        *ampl_info = NULL;
    *capacity = preallocate;
}
static void manage_amplicon_info (tmap_map_amplicon_info** ampl_info, uint32_t* capacity, uint32_t needed_size)
{
    assert (ampl_info);
    assert (capacity);
    if (needed_size >= *capacity)
    {
        uint32_t new_capacity = capacity_snapper (needed_size);
        *ampl_info = tmap_realloc (*ampl_info, new_capacity * sizeof (tmap_map_amplicon_info), "amplicon_info");
    }
}
static void destroy_amplicon_info (tmap_map_amplicon_info** ampl_info, uint32_t* capacity)
{
    assert (ampl_info);
    assert (capacity);
    free (*ampl_info);
    *ampl_info = NULL;
    *capacity = 0;
}

// checks if amplicon overrides any of the candidate evaluation parameters;
// it yes, returns true and fills the passed in 'overrides' structure with the correct parameters (the non-overriden filled with default values)
static uint32_t ampl_candeval_parameters_from_locopt (tmap_map_opt_t* stage_opts, tmap_map_locopt_t* locopt, uint8_t strand,  candeval_overrides_s* overrides)
{
    uint32_t rv = 0;

    // preset to defaults
    overrides->softclip_type = stage_opts->softclip_type;
    overrides->max_adapter_bases_for_soft_clipping = stage_opts->max_adapter_bases_for_soft_clipping;
    overrides->vsw.score_match = stage_opts->score_match;
    overrides->vsw.pen_mm = stage_opts->pen_mm;
    overrides->vsw.pen_gapo = stage_opts->pen_gapo;
    overrides->vsw.pen_gape = stage_opts->pen_gape;
    overrides->vsw.score_thres = stage_opts->score_thr;
    overrides->source = NULL;

    if (!locopt)
        return rv;

    // softclip_type
    if (locopt->softclip_type.over && locopt->softclip_type.value != stage_opts->softclip_type)
    {
        overrides->softclip_type = locopt->softclip_type.value;
        rv = 1;
    }
    // "-J" parameter (--max-adapter-bases-for-soft-clipping) may be overriden for high or low amplicon end or for both. End-based spec takes over.
    if (locopt->max_adapter_bases_for_soft_clipping_he.over && strand == 0) // read's 3' mapped to the higher end and overriden
    {
        if (locopt->max_adapter_bases_for_soft_clipping_he.value != stage_opts->max_adapter_bases_for_soft_clipping)
        {
            overrides->max_adapter_bases_for_soft_clipping = locopt->max_adapter_bases_for_soft_clipping_he.value;
            rv = 1;
        }
    }
    else if (locopt->max_adapter_bases_for_soft_clipping_le.over && strand == 1) // read's 3' mapped to the lower end and overriden
    {
        if (locopt->max_adapter_bases_for_soft_clipping_le.value != stage_opts->max_adapter_bases_for_soft_clipping)
        {
            overrides->max_adapter_bases_for_soft_clipping = locopt->max_adapter_bases_for_soft_clipping_le.value;
            rv = 1;
        }
    }
    else if (locopt->max_adapter_bases_for_soft_clipping.over)
    {
        if (locopt->max_adapter_bases_for_soft_clipping.value != stage_opts->max_adapter_bases_for_soft_clipping)
        {
            overrides->max_adapter_bases_for_soft_clipping = locopt->max_adapter_bases_for_soft_clipping.value;
            rv = 1;
        }
    }
    // score_match
    if (locopt->score_match.over && locopt->score_match.value != stage_opts->score_match)
    {
        overrides->vsw.score_match = locopt->score_match.value;
        rv = 1;
    }
    // pen_mm
    if (locopt->pen_mm.over && locopt->pen_mm.value != stage_opts->pen_mm)
    {
        overrides->vsw.pen_mm = locopt->pen_mm.value;
        rv = 1;
    }
    // pen_gapo
    if (locopt->pen_gapo.over && locopt->pen_gapo.value != stage_opts->pen_gapo)
    {
        overrides->vsw.pen_gapo = locopt->pen_gapo.value;
        rv = 1;
    }
    // pen_gape
    if (locopt->pen_gape.over && locopt->pen_gape.value != stage_opts->pen_gape)
    {
        overrides->vsw.pen_gape = locopt->pen_gape.value;
        rv = 1;
    }
    // score_thres
    if (locopt->score_thr.over && locopt->score_thr.value != stage_opts->score_thr)
    {
        overrides->vsw.score_thres = locopt->score_thr.value;
        rv = 1;
    }
    return rv;
}

static uint32_t ampl_candeval_parameters (tmap_map_opt_t* stage_opts, tmap_map_amplicon_info* amplinfo, uint8_t strand,  candeval_overrides_s* overrides)
{
    uint32_t rv = ampl_candeval_parameters_from_locopt (stage_opts, amplinfo->locopt, strand,  overrides);
    if (rv)
        overrides->source = amplinfo;
    return rv;
}

// same as above using sam as a data source for overrides
static uint32_t ampl_candeval_parameters_from_sam (tmap_map_opt_t* stage_opts, tmap_map_sam_t* sam, candeval_overrides_s* overrides)
{
    return ampl_candeval_parameters_from_locopt (stage_opts, sam->param_ovr, sam->strand,  overrides);
}

// finds amplions intersecting with given start-end interval
// fills in the array of tmap_map_ampliocn_info at the address pointed by touched_ampls;
// allocates or re-allocates this array (updating address at touched_ampls), in this case updates touched_ampls_capacity
// returns number of touched_amplsactually filled in
// (always replaces whatever content is in the passed **touched_amplicons
int
tmap_map_get_overlapping_amplicons
(
    tmap_refseq_t *refseq, 
    int32_t seqid,
    uint32_t start,
    uint32_t end,
    uint32_t allowance, 
    tmap_map_amplicon_info** touched_ampls, // pointer to the array of ampliocn info structures, NULL if not allocated yest
    uint32_t* touched_ampls_capacity) // capacity (allocated size) of touched_ampls array (for ammortized re-allocation and re-use)
{
    if (!refseq->bed_exist || refseq->bednum [seqid] == 0)
        return 0;

    // find all amplions where ampl_ends are above start and ampl_starts are below end

    // WARNING: the code below assumes that both ampl_starts AND ampl_ends arrays are sorted
    //    This is true for current implementation: the BED loading code discards all fully contained amplicons, recording only the outermost ones.
    //    When we switch to handling inner amplicons as well, we'll need to add another level of indirection when searching for the ampl_ends, as the original bed-derived array may be unordered.

    start += allowance;
    end = (allowance > end)?0:(end - allowance);

    // find first amplicon with ampl_end equal or above start
    uint32_t* ampl_starts = refseq->bedstart [seqid];
    uint32_t* ampl_ends = refseq->bedend [seqid];
    uint32_t  ampl_no = refseq->bednum [seqid];

    uint32_t* first_ampl = (uint32_t*) tmap_binary_search (&start, ampl_ends, ampl_no, sizeof (*ampl_ends), lt_uint32);
    uint32_t first_idx = first_ampl - ampl_ends;
    if (first_idx == ampl_no)
        return 0;

    // find last amplicon with ampl_start equal or above end. This is a sentinel, excluded from overlapped list
    uint32_t* last_ampl = (uint32_t*) tmap_binary_search (&end, ampl_starts, ampl_no, sizeof (*ampl_starts), lt_uint32);
    uint32_t last_idx = last_ampl - ampl_starts;
    if (last_idx == 0)
        return 0;

    if (first_idx == last_idx)
        return 0;

    // reallocate result storage if needed
    manage_amplicon_info (touched_ampls, touched_ampls_capacity, last_idx - first_idx);

    // fill in results
    uint32_t ampl_idx;
    tmap_map_amplicon_info* cur_ainfo = *touched_ampls;
    for (ampl_idx = first_idx; ampl_idx != last_idx; ++ampl_idx, ++cur_ainfo)
    {
        cur_ainfo->ampl_start = ampl_starts [ampl_idx];
        cur_ainfo->ampl_end = ampl_ends [ampl_idx];
        if (refseq->parovr && refseq->parovr [seqid] && refseq->parovr [seqid][ampl_idx] != UINT32_MAX)
           cur_ainfo->locopt = refseq->parmem + refseq->parovr [seqid][ampl_idx];
        else
           cur_ainfo->locopt = NULL;
        if (refseq->read_ends && refseq->read_ends [seqid] && refseq->read_ends [seqid][ampl_idx].index != UINT32_MAX)
        {
            tmap_map_endstat_t* src = refseq->read_ends [seqid] + ampl_idx;
            cur_ainfo->read_ends.positions = refseq->endposmem + src->index;
            cur_ainfo->read_ends.starts_count = src->starts_count;
            cur_ainfo->read_ends.ends_count = src->ends_count;
        }
        else
        {
            cur_ainfo->read_ends.positions = NULL;
            cur_ainfo->read_ends.starts_count = 0;
            cur_ainfo->read_ends.ends_count = 0;
        }
    }
    return last_idx - first_idx;
}


// use softclip settings from tmap parameters
// disallow 3' softclip, if
//  -     adapter bases are present at 3' (sequencing reached the adapter)
//  - and there are more of them than max_adapter_bases_for_soft_clipping,
//  - and there is an insert (sequence between key abd barcode and before 3' adapter
//  - and sequence is more than 5 bases longer than the insert
static void
tmap_map_util_set_softclip
(
    int32_t softclip_type,
    tmap_seq_t *seq,
    int32_t max_adapter_bases_for_soft_clipping,
    int32_t *softclip_start,
    int32_t *softclip_end
)
{
    (*softclip_start) = (TMAP_MAP_OPT_SOFT_CLIP_LEFT == softclip_type || TMAP_MAP_OPT_SOFT_CLIP_ALL == softclip_type) ? 1 : 0;
    (*softclip_end) = (TMAP_MAP_OPT_SOFT_CLIP_RIGHT == softclip_type || TMAP_MAP_OPT_SOFT_CLIP_ALL == softclip_type) ? 1 : 0;
    // check if the ZB tag is present...
    if (TMAP_SEQ_TYPE_SAM == seq->type || TMAP_SEQ_TYPE_BAM == seq->type) // SAM/BAM
    {
        tmap_sam_t *sam = seq->data.sam;
        int32_t zb = tmap_sam_get_zb (sam);
        int32_t za = tmap_sam_get_za (sam);
        // keep 3' soft clipping on if zb tag does not exists or there are too few adapter bases.
        if (-1 != zb
            && max_adapter_bases_for_soft_clipping < zb
            && -1 != za
            && sam->seq->l >= za - 5)
            (*softclip_end) = 0;
    }
}

void
tmap_map_sam_init(tmap_map_sam_t *s)
{
  // set all to zero
  memset(s, 0, sizeof(tmap_map_sam_t));
  // nullify
  s->algo_id = TMAP_MAP_ALGO_NONE;
  s->ascore = s->score = INT32_MIN;
  //s->fivep_offset = 0;
}


void
tmap_map_sam_malloc_aux(tmap_map_sam_t *s)
{
  switch(s->algo_id) {
    case TMAP_MAP_ALGO_MAP1:
      s->aux.map1_aux = tmap_calloc(1, sizeof(tmap_map_map1_aux_t), "s->aux.map1_aux");
      break;
    case TMAP_MAP_ALGO_MAP2:
      s->aux.map2_aux = tmap_calloc(1, sizeof(tmap_map_map2_aux_t), "s->aux.map2_aux");
      break;
    case TMAP_MAP_ALGO_MAP3:
      s->aux.map3_aux = tmap_calloc(1, sizeof(tmap_map_map3_aux_t), "s->aux.map3_aux");
      break;
    case TMAP_MAP_ALGO_MAP4:
      s->aux.map4_aux = tmap_calloc(1, sizeof(tmap_map_map4_aux_t), "s->aux.map4_aux");
      break;
    case TMAP_MAP_ALGO_MAPVSW:
      s->aux.map_vsw_aux = tmap_calloc(1, sizeof(tmap_map_map_vsw_aux_t), "s->aux.map_vsw_aux");
      break;
    default:
      break;
  }
}

inline void
tmap_map_sam_destroy_aux(tmap_map_sam_t *s)
{
  switch(s->algo_id) {
    case TMAP_MAP_ALGO_MAP1:
      free(s->aux.map1_aux);
      s->aux.map1_aux = NULL;
      break;
    case TMAP_MAP_ALGO_MAP2:
      free(s->aux.map2_aux);
      s->aux.map2_aux = NULL;
      break;
    case TMAP_MAP_ALGO_MAP3:
      free(s->aux.map3_aux);
      s->aux.map3_aux = NULL;
      break;
    case TMAP_MAP_ALGO_MAP4:
      free(s->aux.map4_aux);
      s->aux.map4_aux = NULL;
      break;
    case TMAP_MAP_ALGO_MAPVSW:
      free(s->aux.map_vsw_aux);
      s->aux.map_vsw_aux = NULL;
      break;
    default:
      break;
  }
}

void
tmap_map_sam_destroy(tmap_map_sam_t *s)
{
  tmap_map_sam_destroy_aux(s);
  free(s->cigar);
  s->cigar = NULL;
  s->n_cigar = 0;
  free (s->orig_cigar);
  s->orig_cigar = NULL;
  s->n_orig_cigar = 0;

}

tmap_map_sams_t *
tmap_map_sams_init(tmap_map_sams_t *prev)
{
  tmap_map_sams_t *sams = tmap_calloc(1, sizeof(tmap_map_sams_t), "sams");
  sams->sams = NULL;
  sams->n = 0;
  if(NULL != prev) sams->max = prev->max;
  return sams;
}

void
tmap_map_sams_realloc(tmap_map_sams_t *s, int32_t n)
{
  int32_t i;
  if(n == s->n) return;
  for(i=n;i<s->n;i++) {
      tmap_map_sam_destroy(&s->sams[i]);
  }
  s->sams = tmap_realloc(s->sams, sizeof(tmap_map_sam_t) * n, "s->sams");
  for(i=s->n;i<n;i++) {
      // nullify
      tmap_map_sam_init(&s->sams[i]);
  }
  s->n = n;
}

void
tmap_map_sams_destroy(tmap_map_sams_t *s)
{
  int32_t i;
  if(NULL == s) return;
  for(i=0;i<s->n;i++) {
      tmap_map_sam_destroy(&s->sams[i]);
  }
  free(s->sams);
  free(s);
}

tmap_map_record_t*
tmap_map_record_init(int32_t num_ends)
{
  tmap_map_record_t *record = NULL;
  int32_t i;

  record = tmap_calloc(1, sizeof(tmap_map_record_t), "record");
  record->sams = tmap_calloc(num_ends, sizeof(tmap_map_sams_t*), "record->sams");
  record->n = num_ends;
  for(i=0;i<num_ends;i++) {
      record->sams[i] = tmap_map_sams_init(NULL);
  }
  return record;
}

tmap_map_record_t*
tmap_map_record_clone(tmap_map_record_t *src)
{
  int32_t i;
  tmap_map_record_t *dest = NULL;

  if(NULL == src) return NULL;

  // init
  dest = tmap_calloc(1, sizeof(tmap_map_record_t), "dest");
  dest->sams = tmap_calloc(src->n, sizeof(tmap_map_sams_t*), "dest->sams");
  dest->n = src->n;
  if(0 == src->n) return dest;

  // copy over data
  for(i=0;i<src->n;i++) {
      dest->sams[i] = tmap_map_sams_clone(src->sams[i]);
  }

  return dest;
}

void
tmap_map_record_merge(tmap_map_record_t *dest, tmap_map_record_t *src)
{
  int32_t i;
  if(NULL == src || 0 == src->n || src->n != dest->n) return;

  for(i=0;i<src->n;i++) {
      tmap_map_sams_merge(dest->sams[i], src->sams[i]);
  }
}

void
tmap_map_record_destroy(tmap_map_record_t *record)
{
  int32_t i;
  if(NULL == record) return;
  for(i=0;i<record->n;i++) {
      tmap_map_sams_destroy(record->sams[i]);
  }
  free(record->sams);
  free(record);
}

tmap_map_bam_t*
tmap_map_bam_init(int32_t n)
{
  tmap_map_bam_t *b = NULL;
  b = tmap_calloc(1, sizeof(tmap_map_bam_t), "b");
  if(0 < n) b->bams = tmap_calloc(n, sizeof(bam1_t*), "b->bams");
  b->n = n;
  return b;
}

void
tmap_map_bam_destroy(tmap_map_bam_t *b)
{
  int32_t i;
  if(NULL == b) return;
  for(i=0;i<b->n;i++) {
      bam_destroy1(b->bams[i]);
  }
  free(b->bams);
  free(b);
}

tmap_map_bams_t*
tmap_map_bams_init(int32_t n)
{
  tmap_map_bams_t *b = NULL;
  b = tmap_calloc(1, sizeof(tmap_map_bams_t), "b");
  if(0 < n) b->bams = tmap_calloc(n, sizeof(tmap_map_bam_t*), "b->bams");
  b->n = n;
  return b;
}

void
tmap_map_bams_destroy(tmap_map_bams_t *b)
{
  int32_t i;
  if(NULL == b) return;
  for(i=0;i<b->n;i++) {
      tmap_map_bam_destroy(b->bams[i]);
  }
  free(b->bams);
  free(b);
}

inline void
tmap_map_sam_copy(tmap_map_sam_t *dest, tmap_map_sam_t *src)
{
  int32_t i;
  // shallow copy
  (*dest) = (*src);
  // aux data
  tmap_map_sam_malloc_aux(dest);
  switch(src->algo_id) {
    case TMAP_MAP_ALGO_MAP1:
      (*dest->aux.map1_aux) = (*src->aux.map1_aux);
      break;
    case TMAP_MAP_ALGO_MAP2:
      (*dest->aux.map2_aux) = (*src->aux.map2_aux);
      break;
    case TMAP_MAP_ALGO_MAP3:
      (*dest->aux.map3_aux) = (*src->aux.map3_aux);
      break;
    case TMAP_MAP_ALGO_MAP4:
      (*dest->aux.map4_aux) = (*src->aux.map4_aux);
      break;
    case TMAP_MAP_ALGO_MAPVSW:
      (*dest->aux.map_vsw_aux) = (*src->aux.map_vsw_aux);
      break;
    default:
      break;
  }
  // cigar
  if(0 < src->n_cigar && NULL != src->cigar) {
      dest->cigar = tmap_malloc(sizeof(uint32_t) * (1 + dest->n_cigar), "dest->cigar");
      for(i=0;i<dest->n_cigar;i++) {
          dest->cigar[i] = src->cigar[i];
      }
  }
}
void
tmap_map_sams_merge(tmap_map_sams_t *dest, tmap_map_sams_t *src)
{
  int32_t i, j;
  if(NULL == src || 0 == src->n) return;

  j = dest->n;
  tmap_map_sams_realloc(dest, dest->n + src->n);
  for(i=0;i<src->n;i++,j++) {
      tmap_map_sam_copy(&dest->sams[j], &src->sams[i]);
  }
}

tmap_map_sams_t *
tmap_map_sams_clone(tmap_map_sams_t *src)
{
  int32_t i;
  tmap_map_sams_t *dest = NULL;

  // init
  dest = tmap_map_sams_init(src);
  if(0 == src->n) return dest;

  // realloc
  tmap_map_sams_realloc(dest, src->n);
  // copy over data
  for(i=0;i<src->n;i++) {
      tmap_map_sam_copy(&dest->sams[i], &src->sams[i]);
  }

  return dest;
}

void
tmap_map_sam_copy_and_nullify(tmap_map_sam_t *dest, tmap_map_sam_t *src)
{
  (*dest) = (*src);
  src->n_cigar = 0;
  src->cigar = NULL;
  switch(src->algo_id) {
    case TMAP_MAP_ALGO_MAP1:
      src->aux.map1_aux = NULL;
      break;
    case TMAP_MAP_ALGO_MAP2:
      src->aux.map2_aux = NULL;
      break;
    case TMAP_MAP_ALGO_MAP3:
      src->aux.map3_aux = NULL;
      break;
    case TMAP_MAP_ALGO_MAP4:
      src->aux.map4_aux = NULL;
      break;
    case TMAP_MAP_ALGO_MAPVSW:
      src->aux.map_vsw_aux = NULL;
      break;
    default:
      break;
  }
}

static bam1_t*
tmap_map_sam_print
(
    tmap_seq_t *seq,
    tmap_refseq_t *refseq,
    tmap_map_sam_t *sam,
    int32_t sam_flowspace_tags,
    int32_t bidirectional,
    int32_t seq_eq,
    int32_t nh,
    int32_t aln_num,
    int32_t end_num,
    int32_t mate_unmapped,
    tmap_map_sam_t *mate
)
{
  int64_t mate_strand, mate_seqid, mate_pos, mate_tlen;
  // mate info
  mate_strand = mate_seqid = mate_pos = mate_tlen = 0;
  if(NULL != mate && 0 == mate_unmapped) {
      mate_strand = mate->strand;
      mate_seqid = mate->seqid;
      mate_pos = mate->pos;
  }
  if(NULL == sam) { // unmapped
      return tmap_sam_convert_unmapped(seq, sam_flowspace_tags, bidirectional, refseq,
                              end_num, mate_unmapped, 0,
                              mate_strand, mate_seqid, mate_pos, NULL);
  }
  else {
      // Note: samtools does not like this value
      if(INT32_MIN == sam->score_subo) {
          sam->score_subo++;
      }
      // Set the mate tlen
      if(NULL != mate && 0 == mate_unmapped) {
          // NB: assumes 5'->3' ordering of the fragments
          if(mate->pos < sam->pos) {
              mate_tlen = sam->pos + sam->target_len - mate->pos;
          }
          else {
              mate_tlen = mate->pos + mate->target_len - sam->pos;
          }
          // NB: first fragment is always positive, the rest are always negative
          if(1 == end_num) { // first end
              if(mate_tlen < 0) mate_tlen = -mate_tlen;
          }
          else { // second end
              if(0 < mate_tlen) mate_tlen = -mate_tlen;
          }
      }
      switch(sam->algo_id) {
        case TMAP_MAP_ALGO_MAP1:
          return tmap_sam_convert_mapped(seq, sam_flowspace_tags, bidirectional, seq_eq, refseq,
                                         sam->strand, sam->seqid, sam->pos, sam->target_len,  aln_num,
                                         end_num, mate_unmapped, sam->proper_pair, sam->num_stds,
                                         mate_strand, mate_seqid, mate_pos, mate_tlen,
                                         sam->mapq, sam->cigar, sam->n_cigar,
                                         sam->score, sam->ascore, sam->pscore, nh, sam->algo_id, sam->algo_stage, "");
          break;
        case TMAP_MAP_ALGO_MAP2:
          if(0 < sam->aux.map2_aux->XI) {
              return tmap_sam_convert_mapped(seq, sam_flowspace_tags, bidirectional, seq_eq, refseq,
                                             sam->strand, sam->seqid, sam->pos, sam->target_len, aln_num,
                                             end_num, mate_unmapped, sam->proper_pair, sam->num_stds,
                                             mate_strand, mate_seqid, mate_pos, mate_tlen,
                                             sam->mapq, sam->cigar, sam->n_cigar,
                                             sam->score, sam->ascore, sam->pscore, nh, sam->algo_id, sam->algo_stage,
                                             "\tXS:i:%d\tXT:i:%d\t\tXF:i:%d\tXE:i:%d\tXI:i:%d",
                                             sam->score_subo,
                                             sam->n_seeds,
                                             sam->aux.map2_aux->XF, sam->aux.map2_aux->XE,
                                             sam->aux.map2_aux->XI);
          }
          else {
              return tmap_sam_convert_mapped(seq, sam_flowspace_tags, bidirectional, seq_eq, refseq,
                                             sam->strand, sam->seqid, sam->pos, sam->target_len, aln_num,
                                             end_num, mate_unmapped, sam->proper_pair, sam->num_stds,
                                             mate_strand, mate_seqid, mate_pos, mate_tlen,
                                             sam->mapq, sam->cigar, sam->n_cigar,
                                             sam->score, sam->ascore, sam->pscore, nh, sam->algo_id, sam->algo_stage,
                                             "\tXS:i:%d\tXT:i:%d\tXF:i:%d\tXE:i:%d",
                                             sam->score_subo,
                                             sam->n_seeds,
                                             sam->aux.map2_aux->XF, sam->aux.map2_aux->XE);
          }
          break;
        case TMAP_MAP_ALGO_MAP3:
          return tmap_sam_convert_mapped(seq, sam_flowspace_tags, bidirectional, seq_eq, refseq,
                                         sam->strand, sam->seqid, sam->pos, sam->target_len, aln_num,
                                         end_num, mate_unmapped, sam->proper_pair, sam->num_stds,
                                         mate_strand, mate_seqid, mate_pos, mate_tlen,
                                         sam->mapq, sam->cigar, sam->n_cigar,
                                         sam->score, sam->ascore, sam->pscore, nh, sam->algo_id, sam->algo_stage,
                                         "\tXS:i:%d\tXT:i:%d",
                                         sam->score_subo,
                                         sam->n_seeds);
          break;
        case TMAP_MAP_ALGO_MAP4:
          return tmap_sam_convert_mapped(seq, sam_flowspace_tags, bidirectional, seq_eq, refseq,
                                         sam->strand, sam->seqid, sam->pos, sam->target_len, aln_num,
                                         end_num, mate_unmapped, sam->proper_pair, sam->num_stds,
                                         mate_strand, mate_seqid, mate_pos, mate_tlen,
                                         sam->mapq, sam->cigar, sam->n_cigar,
                                         sam->score, sam->ascore, sam->pscore, nh, sam->algo_id, sam->algo_stage,
                                         "\tXS:i:%d\tXZ:i:%d",
                                         sam->score_subo,
					                     sam->fivep_offset);
          break;
        case TMAP_MAP_ALGO_MAPVSW:
          return tmap_sam_convert_mapped(seq, sam_flowspace_tags, bidirectional, seq_eq, refseq,
                                         sam->strand, sam->seqid, sam->pos, sam->target_len, aln_num,
                                         end_num, mate_unmapped, sam->proper_pair, sam->num_stds,
                                         mate_strand, mate_seqid, mate_pos, mate_tlen,
                                         sam->mapq, sam->cigar, sam->n_cigar,
                                         sam->score, sam->ascore, sam->pscore, nh, sam->algo_id, sam->algo_stage,
                                         "\tXS:i:%d",
                                         sam->score_subo);
          break;
      }
  }
  return NULL;
}

tmap_map_bam_t*
tmap_map_sams_print
(
    tmap_seq_t *seq,
    tmap_refseq_t *refseq,
    tmap_map_sams_t *sams,
    int32_t end_num,
    tmap_map_sams_t *mates,
    int32_t sam_flowspace_tags,
    int32_t bidirectional,
    int32_t seq_eq,
    int32_t min_al_len,
    double min_coverage,
    double min_identity,
    int32_t match_score,
    uint64_t* filtered
)
{
  int32_t i;
  tmap_map_sam_t *mate = NULL;
  int32_t mate_unmapped = 0;
  tmap_map_bam_t *bams = NULL;

  if(NULL != mates) {
      if(0 < mates->n) {
          // assumes the mates are sorted by their alignment score
          mate = &mates->sams[0];
      }
      else {
          mate_unmapped = 1;
      }
  }
  // filter alignments; store filtered out ones as unmapped
  int32_t mapped_cnt = 0;
  int8_t* passed = alloca (sams->n);

  for(i=0;i<sams->n;i++)
  {
    unsigned q_len_cigar, r_len_cigar;
    seq_lens_from_bin_cigar (sams->sams [i].cigar, sams->sams [i].n_cigar, &q_len_cigar, &r_len_cigar);
    int8_t mapped = 1;
#define FINAL_FILTER_QUICKFIX
#if defined (FINAL_FILTER_QUICKFIX) // fix for TS-18040
    if (mapped && min_al_len != MIN_AL_LEN_NOCHECK_SPECIAL && r_len_cigar < min_al_len)
        mapped = 0;
    if (mapped && min_coverage != MIN_AL_COVERAGE_NOCHECK_SPECIAL && seq->data.sam->seq->l != 0 && ((double) r_len_cigar) / seq->data.sam->seq->l < min_coverage)
        mapped = 0;
    if (mapped && min_identity != MIN_AL_IDENTITY_NOCHECK_SPECIAL && r_len_cigar != 0 && match_score != 0 && ((double) sams->sams [i].score) / (r_len_cigar*match_score) < min_identity )
        mapped = 0;
#else 
    if (mapped && r_len_cigar < min_al_len)
        mapped = 0;
    if (mapped && seq->data.sam->seq->l != 0 && ((double) r_len_cigar) / seq->data.sam->seq->l < min_coverage)
        mapped = 0;
    if (mapped && r_len_cigar != 0 && match_score != 0 && ((double) sams->sams [i].score) / (r_len_cigar*match_score) < min_identity )
        mapped = 0;
#endif
    passed [i] = mapped;
    if (mapped)
       ++ mapped_cnt;
    else
       ++ *filtered;
  }

  int32_t written_cnt = 0;
  if(mapped_cnt) {
      bams = tmap_map_bam_init(mapped_cnt);
      for(i=0;i<sams->n;i++)
      {
          if (passed [i])
             bams->bams[written_cnt++] = tmap_map_sam_print(seq, refseq, &sams->sams[i], sam_flowspace_tags, bidirectional, seq_eq, sams->max, i, end_num, mate_unmapped, mate);
      }
  }
  else {
      bams = tmap_map_bam_init(1);
      bams->bams[0] = tmap_map_sam_print(seq, refseq, NULL, sam_flowspace_tags, bidirectional, seq_eq, sams->max, 0, end_num, mate_unmapped, mate);
  }

  return bams;
}

void
tmap_map_util_keep_score (tmap_map_sams_t *sams, int32_t algo_id, int32_t score)
{
  int32_t i, j, cur_score;
  for(i=j=0;i<sams->n;i++) {
      if(TMAP_MAP_ALGO_NONE == algo_id
         || sams->sams[i].algo_id == algo_id) {
          cur_score = sams->sams[i].score;
          if(cur_score != score) { // not the best
              tmap_map_sam_destroy(&sams->sams[i]);
          }
          else {
              if(j < i) { // copy if we are not on the same index
                  tmap_map_sam_copy_and_nullify(&sams->sams[j], &sams->sams[i]);
              }
              j++;
          }
      }
      else {
          if(j < i) { // copy if we are not on the same index
              tmap_map_sam_copy_and_nullify(&sams->sams[j], &sams->sams[i]);
          }
          j++;
      }
  }
  // reallocate
  tmap_map_sams_realloc(sams, j);
}

void
tmap_map_sams_filter1 (tmap_map_sams_t *sams, int32_t aln_output_mode, int32_t algo_id, tmap_rand_t *rand)
{
  int32_t i, j, k;
  int32_t n_best = 0;
  int32_t best_score, cur_score;
  int32_t best_subo_score;

  if(sams->n <= 1) {
      return;
  }

  for(i=j=0;i<sams->n;i++) {
      if(TMAP_MAP_ALGO_NONE == algo_id
         || sams->sams[i].algo_id == algo_id) {
          j++;
      }
  }
  if(j <= 1) {
      return;
  }

  best_score = best_subo_score = INT32_MIN;
  n_best = 0;
  for(i=0;i<sams->n;i++) {
      if(TMAP_MAP_ALGO_NONE == algo_id
         || sams->sams[i].algo_id == algo_id) {
          cur_score = sams->sams[i].score;
          if(best_score < cur_score) {
              if(0 < n_best) {
                  best_subo_score = best_score;
              }
              best_score = cur_score;
              n_best = 1;
          }
          else if(!(cur_score < best_score)) { // equal
              best_subo_score = best_score; // more than one mapping
              n_best++;
          }
          else if(best_subo_score < cur_score) {
              best_subo_score = cur_score;
          }
          // check sub-optimal
          if(TMAP_MAP_ALGO_MAP2 == sams->sams[i].algo_id
             || TMAP_MAP_ALGO_MAP3 == sams->sams[i].algo_id) {
              cur_score = sams->sams[i].score_subo;
              if(best_subo_score < cur_score) {
                  best_subo_score = cur_score;
              }
          }

      }
  }

  // adjust mapping qualities
  if(1 < n_best) {
      for(i=0;i<sams->n;i++) {
          if(TMAP_MAP_ALGO_NONE == algo_id
             || sams->sams[i].algo_id == algo_id) {
              sams->sams[i].mapq = 0;
          }
      }
  }
  else {
      for(i=0;i<sams->n;i++) {
          if(TMAP_MAP_ALGO_NONE == algo_id
             || sams->sams[i].algo_id == algo_id) {
              cur_score = sams->sams[i].score;
              if(cur_score < best_score) { // not the best
                  sams->sams[i].mapq = 0;
              }
          }
      }
  }

  // adjust suboptimal
  if(TMAP_MAP_ALGO_NONE == algo_id) {
      for(i=0;i<sams->n;i++) {
          sams->sams[i].score_subo = best_subo_score;
      }
  }

  if(TMAP_MAP_OPT_ALN_MODE_ALL == aln_output_mode) {
      return;
  }

  // copy to the front
  if(n_best < sams->n) {
      tmap_map_util_keep_score(sams, algo_id, best_score);
  }

  if(TMAP_MAP_OPT_ALN_MODE_UNIQ_BEST == aln_output_mode) {
      if(1 < n_best) { // there can only be one
          if(TMAP_MAP_ALGO_NONE == algo_id) {
              tmap_map_sams_realloc(sams, 0);
          }
          else {
              // get rid of all of them
              for(i=j=0;i<sams->n;i++) {
                  if(sams->sams[i].algo_id == algo_id) {
                      tmap_map_sam_destroy(&sams->sams[i]);
                  }
                  else {
                      if(j < i) { // copy if we are not on the same index
                          tmap_map_sam_copy_and_nullify(&sams->sams[j], &sams->sams[i]);
                      }
                      j++;
                  }
              }
              tmap_map_sams_realloc(sams, j);
          }
      }
  }
  else if(TMAP_MAP_OPT_ALN_MODE_RAND_BEST == aln_output_mode) { // get a random
      int32_t r = (int32_t)(tmap_rand_get(rand) * n_best);

      // keep the rth one
      if(TMAP_MAP_ALGO_NONE == algo_id) {
          if(0 != r) {
              tmap_map_sam_destroy(&sams->sams[0]);
              tmap_map_sam_copy_and_nullify(&sams->sams[0], &sams->sams[r]);
          }
          // reallocate
          tmap_map_sams_realloc(sams, 1);
      }
      else {
          // keep the rth one
          for(i=j=k=0;i<sams->n;i++) {
              if(sams->sams[i].algo_id == algo_id) {
                  if(k == r) { // keep
                      if(j < i) { // copy if we are not on the same index
                          tmap_map_sam_copy_and_nullify(&sams->sams[j], &sams->sams[i]);
                      }
                      j++;
                  }
                  else { // free
                      tmap_map_sam_destroy(&sams->sams[i]);
                  }
                  k++;
              }
              else {
                  if(j < i) { // copy if we are not on the same index
                      tmap_map_sam_copy_and_nullify(&sams->sams[j], &sams->sams[i]);
                  }
                  j++;
              }
          }
          tmap_map_sams_realloc(sams, j);
      }
  }
  else if(TMAP_MAP_OPT_ALN_MODE_ALL_BEST == aln_output_mode) {
      // do nothing
  }
  else {
      tmap_bug();
  }
}

void
tmap_map_sams_filter2 (tmap_map_sams_t *sams, int32_t score_thr, int32_t mapq_thr)
{
  int32_t i, j;

  // filter based on score and mapping quality
  for(i=j=0;i<sams->n;i++) {
      if(sams->sams[i].score < score_thr || sams->sams[i].mapq < mapq_thr) {
          tmap_map_sam_destroy(&sams->sams[i]);
      }
      else {
          if(j < i) { // copy if we are not on the same index
              tmap_map_sam_copy_and_nullify(&sams->sams[j], &sams->sams[i]);
          }
          j++;
      }
  }
  tmap_map_sams_realloc(sams, j);
}

void
tmap_map_sams_filter (tmap_map_sams_t *sams, int32_t aln_output_mode, tmap_rand_t *rand)
{
  tmap_map_sams_filter1(sams, aln_output_mode, TMAP_MAP_ALGO_NONE, rand);
}

void
tmap_map_util_remove_duplicates (tmap_map_sams_t *sams, int32_t dup_window, tmap_rand_t *rand)
{
  int32_t i, next_i, j, k, end, best_score_i, best_score_n, best_score_subo;

  if(dup_window < 0 || sams->n <= 0) {
      return;
  }

  // sort
  // NB: since tmap_map_util_sw_gen_score only sets the end position of the
  // alignment, use that
  tmap_sort_introsort(tmap_map_sam_sort_coord_end, sams->n, sams->sams);

  // remove duplicates within a window
  for(i=j=0;i<sams->n;)
  {

      // get the change
      end = best_score_i = i;
      best_score_n = 0;
      best_score_subo = sams->sams[end].score_subo;
      while(end+1 < sams->n)
      {
          if (sams->sams[end].seqid == sams->sams[end+1].seqid
             && sams->sams[end].strand == sams->sams[end+1].strand
             && fabs ((sams->sams[end].pos + sams->sams[end].target_len) - (sams->sams[end+1].pos + sams->sams[end+1].target_len)) <= dup_window)
          {
              // track the best scoring
              if (sams->sams[best_score_i].score == sams->sams[end+1].score)
              {
                  best_score_i = end+1;
                  best_score_n++;
              }
              else if (sams->sams[best_score_i].score < sams->sams[end+1].score)
              {
                  best_score_i = end+1;
                  best_score_n = 1;
              }
              if (best_score_subo < sams->sams[end+1].score_subo)
                  best_score_subo = sams->sams[end+1].score_subo;
              end++;
          }
          else
              break;
      }
      next_i = end+1;

      // randomize the best scoring
      if(1 < best_score_n)
      {
          k = (int32_t)(best_score_n * tmap_rand_get(rand)); // make this zero-based
          best_score_n = 0; // make this one-based
          end = i;
          while(best_score_n <= k)
          { // this assumes we know there are at least "best_score
              if(sams->sams[best_score_i].score == sams->sams[end].score)
              {
                  best_score_i = end;
                  best_score_n++;
              }
              end++;
          }
      }

      // copy over the best
      if (j != best_score_i)
      {
          // destroy
          tmap_map_sam_destroy(&sams->sams[j]);
          // nullify
          tmap_map_sam_copy_and_nullify(&sams->sams[j], &sams->sams[best_score_i]);
      }

      // copy over sub-optimal score
      sams->sams[j].score_subo = best_score_subo;

      // next
      i = next_i;
      j++;
  }

  // resize
  tmap_map_sams_realloc(sams, j);
}

inline int32_t
tmap_map_util_mapq_score(int32_t seq_len, int32_t n_best, int32_t best_score, int32_t n_best_subo, int32_t best_subo_score, tmap_map_opt_t *opt)
{
    int32_t mapq;

    if(0 == n_best_subo) 
    {
        n_best_subo = 1;
        best_subo_score = opt->score_thr;
    }
    if (opt->use_new_QV == 1)
    {
        int32_t x = 11*opt->score_match;
        if (best_subo_score < x) {best_subo_score = x; n_best_subo= 1;}
        double sf = (double) (best_score - best_subo_score + 1 );
        if (sf < 0) return 1.0;
        sf /= ((double)opt->score_match+opt->pen_mm);
        sf *= 7.3;
        sf -= log(n_best_subo);
        mapq = (int32_t) (sf+0.9999);
        if (mapq < 0) 
            return 0;
        return mapq;
    }
    /*
        fprintf(stderr, "n_best=%d n_best_subo=%d\n",
        n_best, n_best_subo);
        fprintf(stderr, "best_score=%d best_subo_score=%d\n",
        best_score, best_subo_score);
        */
    // Note: this is the old calculationg, based on BWA-long
    //mapq = (int32_t)((n_best / (1.0 * n_best_subo)) * (best_score - best_subo) * (250.0 / best_score + 0.03 / opt->score_match) + .499);
    //
    double sf = 0.4; // initial scaling factor.  Note: 250 * sf is the maximum mapping quality.
    sf *= 250.0 / ((double)opt->score_match * seq_len); // scale based on the best possible alignment score
    sf *= (n_best / (1.0 * n_best_subo)); // scale based on number of sub-optimal mappings
    sf *= (double)(best_score - best_subo_score + 1 ); // scale based on distance to the sub-optimal mapping
    //sf *= (seq_len < 10) ? 1.0 : log10(seq_len); // scale based on longer reads having more information content
    mapq = (int32_t)(sf + 0.99999);
    if(mapq > 250) mapq = 250;
    if(mapq <= 0) mapq = 1;
    return mapq;
}

inline int32_t
tmap_map_util_mapq(tmap_map_sams_t *sams, int32_t seq_len, tmap_map_opt_t *opt, tmap_refseq_t *refseq)
{
    int32_t i;
    int32_t n_best = 0, n_best_subo = 0;
    int32_t best_score, cur_score, best_subo_score, best_subo_score2;
    int32_t mapq;
    int32_t algo_id = TMAP_MAP_ALGO_NONE;
    int32_t best_repetitive = 0;

    // estimate mapping quality TODO: this needs to be refined
    best_score = INT32_MIN;
    best_subo_score = best_subo_score2 = opt->score_thr;
    n_best = n_best_subo = 0;
    for(i=0;i<sams->n;i++) 
    {
        cur_score = sams->sams[i].score;
        if(best_score < cur_score) 
        {
            // save sub-optimal
            best_subo_score = best_score;
            n_best_subo = n_best;
            // update
            best_score = cur_score;
            n_best = 1;
            algo_id = (algo_id == TMAP_MAP_ALGO_NONE) ? sams->sams[i].algo_id : -1;
            if(sams->sams[i].algo_id == TMAP_MAP_ALGO_MAP2 && 1 == sams->sams[i].aux.map2_aux->flag) 
            {
                best_repetitive = 1;
            }
            else 
            {
                best_repetitive = 0;
            }
        }
        else if(cur_score == best_score) 
        { // qual
            if(sams->sams[i].algo_id == TMAP_MAP_ALGO_MAP2 && 1 == sams->sams[i].aux.map2_aux->flag) 
            {
                best_repetitive = 1;
            }
            n_best++;
        }
        else 
        {
            if(best_subo_score < cur_score) 
            {
                best_subo_score = cur_score;
                n_best_subo = 1;
            }
            else if(best_subo_score == cur_score) 
            {
                n_best_subo++;
            }
        }
        // get the best subo-optimal score
        cur_score = sams->sams[i].score_subo;
        if(INT32_MIN == cur_score) 
        {
            // ignore
        }
        else if(best_subo_score < cur_score) 
        {
            best_subo_score2 = cur_score;
        }
    }
    if(best_subo_score < best_subo_score2) 
    {
        best_subo_score = best_subo_score2;
        if(0 == n_best_subo) n_best_subo = 1;
    }
    if(1 < n_best || best_score < best_subo_score || 0 < best_repetitive) 
    {
        mapq = 0;
    }
    else 
    {
        mapq = tmap_map_util_mapq_score(seq_len, n_best, best_score, n_best_subo, best_subo_score, opt);
    }
    for(i=0;i<sams->n;i++) 
    {
        cur_score = sams->sams[i].score;
        if(cur_score == best_score) 
        {
            sams->sams[i].mapq = mapq;
        }
        else 
        {
            sams->sams[i].mapq = 0;
        }
    }

    if (mapq == 0 && refseq->bed_exist && opt->use_bed_in_mapq) 
    {
    int j = -1;
    // int overlap_s = 0, sec = 0;
    int sh_dis = 10000, sec_dis = 10000;
    for (i=0; i <sams->n; i++) 
    {
        tmap_map_sam_t tmp_sam = sams->sams[i];
        uint32_t start = tmp_sam.pos+1;
        uint32_t end = start+tmp_sam.target_len-1;
        uint32_t ampl_start, ampl_end;
        if (tmp_sam.score != best_score) continue;
        if (tmap_map_get_amplicon(refseq, tmp_sam.seqid, start, end, tmp_sam.strand, &ampl_start, &ampl_end, NULL, NULL)) 
        {
            /*
            if (abs(ampl_start-start) < 15 && abs(ampl_end-end) < 15) {
                fprintf(stderr, "%d %d %d %d %d\n", ampl_start, start, ampl_end, end, tmp_sam.seqid);
                if (j < 0) j = i;
                else {
                fprintf(stderr, "second one\n");
                return 0; // more than one hits match to an amplicon well
                }
            }
            */
            // use overlap to determine which amplicon to pick
            /*
            int ll = start;
            if (ll < ampl_start) ll = ampl_start;	
            int rr = end;
            if (rr > ampl_end) rr = ampl_end;
            int ov = rr -ll;
            if (ov > overlap_s) {
                sec = overlap_s;
                overlap_s = ov;
                j = i;
            } else if (ov > sec) {
                sec = ov;
            }
            */
            // Use the distance to 5' end start
            //fprintf(stderr, "found %d %d\n", ampl_start, ampl_end);
            int dis;
            if (tmp_sam.strand == 0) 
                dis = abs (((int32_t) ampl_start) - ((int32_t) start));  // DK unsigned subtraction of greater from smaller wraps by MAX_VALUE modulo ...
            else 
                dis = abs (((int32_t) ampl_end) - ((int32_t) end));
            if (dis < sh_dis) 
            {
                sec_dis = sh_dis;
                sh_dis = dis;
                j = i;
            } 
            else if (dis < sec_dis) 
            {
                sec_dis = dis;
            }
        }
    } // for
    if (j >= 0 && /*overlap_s - sec > 5*/ sec_dis - sh_dis > 1) 
    {
        //fprintf(stderr, "Bump to 12 %d\n", j);
        sams->sams[j].mapq = 12;
        // make the j unique best hit
        for (i = 0; i < sams->n; i++) 
        {
            if (i != j && sams->sams[i].score == best_score)  sams->sams[i].score = best_score-2;
        }
    }
    }
    return 0;
}

// ACGTNBDHKMRSVWYN
// This gives a mask for match iupac characters versus A/C/G/T/N
//  A  C  G  T   N  B  D  H   K  M  R  S   V  W  Y  N
static int32_t matrix_iupac_mask [IUPAC_MATRIX_SIZE] = {
    1, 0, 0, 0,  1, 0, 1, 1,  0, 1, 1, 0,  1, 1, 0, 1, // A
    0, 1, 0, 0,  1, 1, 0, 1,  0, 1, 0, 1,  1, 0, 1, 1, // C
    0, 0, 1, 0,  1, 1, 1, 0,  1, 0, 1, 1,  1, 0, 0, 1, // G
    0, 0, 0, 1,  1, 1, 1, 1,  1, 0, 0, 0,  0, 1, 1, 1, // T
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1  // N
};

/*
#define __map_util_gen_ap_iupac(par, opt) do { \
    int32_t i; \
    for(i=0;i<80;i++) { \
        if(0 < matrix_iupac_mask[i]) (par).matrix[i] = (opt)->score_match; \
        else (par).matrix[i] = -(opt)->pen_mm; \
    } \
    (par).gap_open = (opt)->pen_gapo; (par).gap_ext = (opt)->pen_gape; \
    (par).gap_end = (opt)->pen_gape; \
    (par).row = 16; \
    (par).band_width = (opt)->bw; \
} while(0)
*/
int32_t matrix_iupac [IUPAC_MATRIX_SIZE]; // this does not change, can be global

void tmap_map_util_populate_sw_par_iupac_direct (tmap_sw_param_t* par, int32_t score_match, int32_t pen_mm, int32_t pen_gapo, int32_t pen_gape, int32_t bw)
{
    int32_t i;
    for (i = 0; i < IUPAC_MATRIX_SIZE; ++i)
    {
        if (0 < matrix_iupac_mask [i]) par->matrix [i] = score_match;
        else par->matrix [i] = -pen_mm;
    }
    par->gap_open = pen_gapo;
    par->gap_ext =  pen_gape;
    par->gap_end =  pen_gape;
    par->row = IUPAC_MATRIX_ROWSIZE;
    par->band_width = bw;
}

void tmap_map_util_populate_sw_par_iupac (tmap_sw_param_t* par, tmap_map_opt_t* opt)
{
    int32_t i;
    for (i = 0; i < IUPAC_MATRIX_SIZE; ++i)
    {
        if (0 < matrix_iupac_mask [i]) par->matrix [i] = opt->score_match;
        else par->matrix [i] = -opt->pen_mm;
    }
    par->gap_open = opt->pen_gapo;
    par->gap_ext =  opt->pen_gape;
    par->gap_end =  opt->pen_gape;
    par->row = IUPAC_MATRIX_ROWSIZE;
    par->band_width = opt->bw;
}

void tmap_map_util_populate_stage_sw_par (tmap_sw_param_t* par, tmap_map_opt_t* opt)
{
    // it would be more efficient (space-wise and heap usage-wise) to check if matrix differes from global iupac one
    // and allocate stage-specific matrix only if it does; but the difference should be negligeble
    // (the only negative effect I can foresee is small additional factor to memory fragmentation)
    par->matrix = tmap_malloc (sizeof (int32_t) * IUPAC_MATRIX_SIZE, "SW substitution matrix for stage");
    par->matrix_owned = 1;
    tmap_map_util_populate_sw_par_iupac (par, opt);
}

// checks if any parameters is overriden,
// populates supplied tmap_sw_param_t structure if yes,
// filling in alternate substitution matrix if needed
// returns pointer to newly allocated and populated structure,
// or NULL if no SW overriding is needed

tmap_sw_param_t* tmap_map_util_populate_alt_sw_par (
    tmap_sw_param_t* def_sw_par,
    tmap_map_opt_t* opt,
    tmap_map_locopt_t* locopt)
{
    if (!opt->use_param_ovr || !locopt)
        return NULL;

    // check if any SW scoring parameters is overriden
    if (   (!locopt->score_match.over || locopt->score_match.value == opt->score_match)
        && (!locopt->pen_mm.over      || locopt->pen_mm.value      == opt->pen_mm)
        && (!locopt->bw.over          || locopt->bw.value          == opt->bw)
        && (!locopt->pen_gapo.over    || locopt->pen_gapo.value    == opt->pen_gapo)
        && (!locopt->pen_gape.over    || locopt->pen_gape.value    == opt->pen_gape))
        return NULL;

    tmap_sw_param_t* alt_sw_params = tmap_malloc (sizeof (tmap_sw_param_t), "SW parameters override");
    if (   (locopt->score_match.over && locopt->score_match.value != opt->score_match)
        || (locopt->pen_mm.over      && locopt->pen_mm.value      != opt->pen_mm))
    {
        alt_sw_params->matrix_owned = 1;
        alt_sw_params->matrix = tmap_malloc (sizeof (int32_t) * IUPAC_MATRIX_SIZE, "SW scoring matrix override");
        uint32_t i;
        for (i = 0; i != IUPAC_MATRIX_SIZE; ++i)
        {
            if (matrix_iupac_mask [i])
                alt_sw_params->matrix [i] = locopt->score_match.over ? locopt->score_match.value : opt->score_match;
            else
                alt_sw_params->matrix [i] = locopt->pen_mm.over ? locopt->pen_mm.value : opt->pen_mm;
        }
    }
    else
    {
        alt_sw_params->matrix_owned = 0;
        alt_sw_params->matrix = def_sw_par->matrix;
    }
    alt_sw_params->row = 16;
    alt_sw_params->band_width = locopt->bw.over       ? locopt->bw.value       : opt->bw;
    alt_sw_params->gap_open   = locopt->pen_gapo.over ? locopt->pen_gapo.value : opt->pen_gapo;
    alt_sw_params->gap_ext    = locopt->pen_gape.over ? locopt->pen_gape.value : opt->pen_gape;
    alt_sw_params->gap_end    = locopt->pen_gape.over ? locopt->pen_gape.value : opt->pen_gape;

    return alt_sw_params;
}

#ifdef CONCURRENT_PARAMETERS_CACHE

#ifdef HAVE_LIBPTHREAD
static pthread_mutex_t locopt_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

#endif

uint8_t cache_sw_overrides (
    tmap_map_locopt_t* locopt,
    int32_t stage_ord,
    tmap_map_opt_t* stage_opt,
    tmap_sw_param_t* def_sw_par)
{
    // if locopt not present for this amplicon, do nothing - local params will not be used anyway.
    if (!locopt)
        return 0;
    // check if stage sw params already cached in this locopt;
    // allocate memory if not yet or not enough allocated
    tmap_map_stage_sw_param_ovr_t *p;
    uint32_t allready_cached = 0;
#ifdef CONCURRENT_PARAMETERS_CACHE
    pthread_mutex_lock (&locopt_mutex);
#endif
    if (!locopt->stages_allocated)
    {
        locopt->stages_allocated = 1;
        locopt->stage_sw_params = tmap_malloc (sizeof (tmap_map_stage_sw_param_ovr_t), "amplicon/stage sw param overrides");
        memset (locopt->stage_sw_params, 0, sizeof (tmap_map_stage_sw_param_ovr_t));
    }
    else
    {
        // check if already cached
        tmap_map_stage_sw_param_ovr_t *sent;
        for (p = locopt->stage_sw_params, sent = locopt->stage_sw_params + locopt->stages_used; p != sent; ++p)
            if (p->stage == stage_ord)
            {
                allready_cached = 1;
                break;
            }

        // need to cache. Check if there is a slot to store cached sw params
        if (!allready_cached && locopt->stages_used == locopt->stages_allocated)
        {
            locopt->stages_allocated <<= 1; // duplicate
            locopt->stage_sw_params = tmap_realloc (locopt->stage_sw_params, locopt->stages_allocated * sizeof (tmap_map_stage_sw_param_ovr_t), "amplicon/stage sw param overrides");
            memset (locopt->stage_sw_params + locopt->stages_used, 0, locopt->stages_allocated - locopt->stages_used);
        }
    }
        // store
    if (!allready_cached)
    {
        p = locopt->stage_sw_params + locopt->stages_used;
        ++(locopt->stages_used);
        p->stage = stage_ord;
        p->sw_params = tmap_map_util_populate_alt_sw_par (def_sw_par, stage_opt, locopt);
    }
#ifdef CONCURRENT_PARAMETERS_CACHE
    pthread_mutex_unlock (&locopt_mutex);
#endif
    return !allready_cached;
}

static inline void
tmap_map_util_sw_get_start_and_end_pos
(
    tmap_map_sam_t *sam,
    int32_t seq_len,
    uint8_t strand,
    uint32_t *start_pos,
    uint32_t *end_pos
)
{
  if(strand == 0) {
      (*start_pos) = sam->pos + 1;
      (*end_pos) = sam->pos + sam->target_len;
  } else {
      (*start_pos) = sam->pos + 1;
      (*end_pos) = sam->pos + seq_len;
  }
}

// NB: this function unrolls banding in some cases
static int32_t
tmap_map_util_sw_gen_score_helper
(
    tmap_refseq_t *refseq,
    tmap_map_sams_t *sams,
    tmap_seq_t *origseq,
    tmap_seq_t *seq,
    tmap_map_sams_t *sams_tmp,
    int32_t *idx,
    int32_t start,
    int32_t end,
    uint8_t strand,
    tmap_vsw_t *vsw,
    int32_t seq_len,
    uint32_t start_pos,
    uint32_t end_pos,
    int32_t *target_mem,
    uint8_t **target,
    // int32_t softclip_start,
    // int32_t softclip_end,
    int32_t prev_n_best,
    int32_t max_seed_band, // NB: this may be modified as banding is unrolled
    int32_t prev_score, // NB: must be greater than or equal to the scoring threshold
    tmap_vsw_opt_t *vsw_opt,
    tmap_rand_t *rand,
    tmap_map_opt_t *opt,
    tmap_map_amplicon_info** amplicons,
    uint32_t* amplicons_capacity,
    tmap_map_stats_t *stat      // statistics

)
{
    tmap_map_sam_t tmp_sam;
    uint8_t *query;
    uint32_t qlen;
    int32_t tlen, overflow = 0, is_long_hit = 0;

    // choose a random one within the window
    if (start == end) 
    {
        tmp_sam = sams->sams [start];
    }
    else 
    {
        int32_t r = (int32_t) (tmap_rand_get (rand) * (end - start + 1));
        r += start;
        tmp_sam = sams->sams [r];
    }

    if (0 < opt->long_hit_mult && seq_len * opt->long_hit_mult <= end_pos - start_pos + 1) 
        is_long_hit = 1;

    // update the query sequence
    query = (uint8_t*) tmap_seq_get_bases (seq)->s;
    qlen = tmap_seq_get_bases_length (seq);

    // add in band width
    // one-based
    if (start_pos < opt->bw)
        start_pos = 1;
    else
        start_pos -= opt->bw - 1;

    end_pos += opt->bw - 1;
    if (refseq->annos [sams->sams [end].seqid].len < end_pos)
        end_pos = refseq->annos [sams->sams [end].seqid].len; // one-based

    // get the target sequence
    tlen = end_pos - start_pos + 1;
    if ((*target_mem) < tlen) 
    {  // more memory?
        (*target_mem) = tlen;
        tmap_roundup32 ((*target_mem));
        (*target) = tmap_realloc ((*target), sizeof (uint8_t)*(*target_mem), "target");
    }
    // NB: IUPAC codes are turned into mismatches
    if (NULL == tmap_refseq_subseq2 (refseq, sams->sams [end].seqid+1, start_pos, end_pos, (*target), 1, NULL)) 
    {
        tmap_error ("bug encountered", Exit, OutOfRange);
    }

    // reverse compliment the target
    if (1 == strand)
        tmap_reverse_compliment_int ((*target), tlen);


    // #define DEBUG_VSW_ASSYMETRY
    #ifdef DEBUG_VSW_ASSYMETRY
    {
    int j;
    // fprintf (stderr, "seqid:%u start_pos=%u end_pos=%u strand=%d\n", sams->sams [end].seqid+1, start_pos, end_pos, strand);
    for (j = 0; j < qlen; j++) 
    {
        if (query [j] >= 4)
        {
            tmap_warning ("QUERY [%d] = %d\n", j, query [j]);
            // query [j] = 0;
        }
        // fputc ("ACGTN" [query [j]], stderr);
    }
    // fputc ('\n', stderr);
    for (j = 0; j < tlen; j++) 
    {
        if ((*target) [j] >= 4)
        {
            tmap_warning ("TARGET [%d] = %d\n", j, (*target) [j]);
            // (*target) [j] = 0;
        }
        // fputc ("ACGTN" [(*target) [j]], stderr);
    }
    // fputc ('\n', stderr);
    }
    #endif

    // initialize the bounds
    tmp_sam.result.query_start = tmp_sam.result.query_end = 0;
    tmp_sam.result.target_start = tmp_sam.result.target_end = 0;

    /**
    * Discussion on choosing alignments.
    *
    * In general, we would like to choose the alignment using the *most* number
    * of query bases.  The complication comes when using soft-clipping.
    *
    * When no soft-clipping occurs, then the whole query will be used.
    *
    * When there is only soft-clipping at the 3' end of the query (end of the
    * read), we can search for the alignment with the greatest query end when
    * searching in the 5'->3' direction.  Since the query start is fixed, as there
    * is no 5' start-clipping, we are then guaranteed to find the alignment with
    * the most number of query bases if we fix the query end when we align in the
    * 3'->5' direction.
    *
    * When there is only soft-clipping at the 5' end of the query (start of the
    * read), we can search for the alignment with the greatest query end when
    * searching in the 5'->3' direction, as the query end is fixed.  Then, when
    * aligning in the 3'->5' direction, we can search for the alignment with the
    * smallest query start (which in the 3'->5' direction is the one with the one
    * with the largest query end).
    *
    * We cannot guarantee anything if we are allowing soft-clipping on both the
    * 5' and 3' ends.
    */

    // NB: this aligns in the sequencing direction
    // DVK: bug #11386
    {
        const int16_t TARGET_EDGE_ALLOWANCE = 4;
        const int16_t MAX_TARGET_LEN_FOR_VSW = 32000;
        int8_t target_causes_read_clipping;
        uint32_t clipping_extensions = 0;
        int8_t first_run = 1;
        int32_t score = 0, prev_score = 0;
        uint32_t new_end_pos, new_start_pos;
        tmap_vsw_result_t result;
        uint32_t amplicons_no = 0;
        uint32_t ampl_found = 0;
        tmap_map_amplicon_info best_ampl;
        memset (&best_ampl, 0, sizeof (best_ampl));
        do
        {
            // initial assumption
            target_causes_read_clipping = 0;
            ampl_found = 0;

            // initialize the bounds - why?
            result.query_start = result.query_end = 0;
            result.target_start = result.target_end = 0;

            // handle amplicons - find if parameters overriding is needed
            if (opt->use_param_ovr && opt->ovr_candeval)
            {
                // find amplicons
                amplicons_no = tmap_map_get_overlapping_amplicons (refseq, sams->sams [end].seqid, start_pos, end_pos, opt->amplicon_scope, amplicons, amplicons_capacity);
                ++stat->amplicon_searches;
                if (amplicons_no)
                    ++stat->amplicon_search_successes;
                if (amplicons_no > 1)
                    ++stat->amplicon_overlaps;
            }

            // make list of distinct overrides that do not match stage params
            candeval_overrides_s* overrides = (candeval_overrides_s*) alloca (sizeof (candeval_overrides_s) * amplicons_no); 
            int overrides_no = 0; 
            for (tmap_map_amplicon_info *ainfo = *amplicons, *sent = *amplicons + amplicons_no; ainfo != sent; ++ainfo)
            {
                // set parameters overrides: fill in first unoccupied slot in 'overrides'
                uint32_t differs = ampl_candeval_parameters (opt, ainfo, strand, overrides + overrides_no);
                if (!differs) // matched default and thus was not recorded
                    continue;
                // check if matches any prior one; increment overrides_no only if unique
                uint32_t match_found = 0;
                for (candeval_overrides_s* older_override = overrides, *older_sent = overrides + overrides_no; older_override != older_sent && ! match_found; ++older_override)
                    if (candeval_overrides_same (older_override, overrides + overrides_no))
                        match_found = 1;
                if (!match_found)
                    ++overrides_no;
            }
            // if no amplicons found (or no overriding ordered), use global params to compute score
            if (!overrides_no)
            {
                score = tmap_vsw_process_fwd (vsw, query, qlen, (*target), tlen,
                                        &result, &overflow, opt->score_thr, 1, opt->confirm_vsw_corr, opt->correct_failed_vsw, opt->use_nvsw_on_nonstd_bases, stat);
            }
            else // run for every override 
            {
                ++stat->candeval_overrides;
                if (overrides_no > 1)
                    ++stat->multiple_candeval_overrides;
                int32_t softclip_start, softclip_end;
                // save default parameters
                tmap_vsw_opt_t orig_vsw_opt = *(vsw->opt);
                int32_t orig_softclip_start = vsw->query_start_clip, orig_softclip_end = vsw->query_end_clip;
                // run for every override
                int32_t best_score = INT32_MIN;
                candeval_overrides_s* best_params = NULL;
                for (candeval_overrides_s* override = overrides, *ovr_sent = overrides + overrides_no; override != ovr_sent; ++override)
                {
                    tmap_map_util_set_softclip (override->softclip_type, origseq, override->max_adapter_bases_for_soft_clipping, &softclip_start, &softclip_end);
                    tmap_vsw_set_params (vsw, softclip_start, softclip_end, &(override->vsw));
                    score = tmap_vsw_process_fwd (vsw, query, qlen, (*target), tlen,
                                        &result, &overflow, opt->score_thr, 1, opt->confirm_vsw_corr, opt->correct_failed_vsw, opt->use_nvsw_on_nonstd_bases, stat);
                    if (score > best_score)
                    {
                        best_score = score;
                        best_params = override;
                        ampl_found = 1;
                    }
                }
                if (ampl_found)
                {
                    // pick the best (highest) one
                    score = best_score;
                    // remember what was the best amplicon (best_ampl)
                    best_ampl = *best_params->source;
                }
                // reset vsw parameters to orig
                tmap_vsw_set_params (vsw, orig_softclip_start, orig_softclip_end, &(orig_vsw_opt));
            }
            if (1 == overflow)
                return INT32_MIN;

            if (!first_run && score <= prev_score)  // check for zone edge crossing on first run always, then only if score is improved
                break; // if score did not improve, get out and keep last result/score in tmp_sam

            // query clip may be recoverable if it has 3' clip and alignment goes to target zone end
            new_end_pos = end_pos;
            new_start_pos = start_pos;
            if (tlen - result.target_end < TARGET_EDGE_ALLOWANCE && qlen > result.query_end)
            {
                // extend target if possible
                new_end_pos += qlen - result.query_end + TARGET_EDGE_ALLOWANCE;
                if (refseq->annos [sams->sams [end].seqid].len < new_end_pos)
                {
                    new_end_pos = refseq->annos [sams->sams [end].seqid].len; // one-based
                }
            }
            // similarly process 5' clips
            if (result.target_start < TARGET_EDGE_ALLOWANCE && result.query_start > 0)
            {
                if (result.query_start + TARGET_EDGE_ALLOWANCE > new_start_pos)
                    new_start_pos -= result.query_start + TARGET_EDGE_ALLOWANCE;
                else
                    new_start_pos = 1;
            }
            if (opt->candidate_ext && (new_end_pos > end_pos || new_start_pos < start_pos))
            {
                end_pos = new_end_pos;
                start_pos = new_start_pos;
                // get the target sequence
                tlen = end_pos - start_pos + 1;
                if ((*target_mem) < tlen)
                { // more memory?
                    (*target_mem) = tlen;
                    tmap_roundup32 ((*target_mem));
                    (*target) = tmap_realloc ((*target), sizeof (uint8_t) * (*target_mem), "target");
                }
                // NB: IUPAC codes are turned into mismatches
                if (NULL == tmap_refseq_subseq2 (refseq, sams->sams [end].seqid + 1, start_pos, end_pos, (*target), 1, NULL))
                    tmap_error ("bug encountered", Exit, OutOfRange);

                // reverse compliment the target
                if (1 == strand)
                    tmap_reverse_compliment_int ((*target), tlen);

                target_causes_read_clipping = 1;
                ++clipping_extensions;
            }
            prev_score = score;
            first_run = 0;
            if (start_pos + MAX_TARGET_LEN_FOR_VSW < end_pos)
                break;
        }
        while (target_causes_read_clipping);
        stat->read_clipping_extensions += clipping_extensions;
        tmp_sam.score = score;
        tmp_sam.result = result;
        if (ampl_found) // found the best scoring out of (one or more) overlapping amplicons
        {
            tmp_sam.ampl_start = best_ampl.ampl_start;
            tmp_sam.ampl_end = best_ampl.ampl_end;
            tmp_sam.param_ovr = best_ampl.locopt;
            tmp_sam.read_ends = best_ampl.read_ends;
        }
        else if (amplicons_no == 1) // regardless of how good/bad the match is and whether overrides were defined, if there is only one overlapping amplicon, bind it
        {
            tmp_sam.ampl_start = (*amplicons)->ampl_start;
            tmp_sam.ampl_end = (*amplicons)->ampl_end;
            tmp_sam.param_ovr = (*amplicons)->locopt;
            tmp_sam.read_ends = (*amplicons)->read_ends;
        }
        else
        {
            tmp_sam.ampl_start = 0;
            tmp_sam.ampl_end = 0;
            tmp_sam.param_ovr = NULL;
            tmp_sam.read_ends.positions = NULL, tmp_sam.read_ends.starts_count = 0, tmp_sam.read_ends.ends_count = 0;
        }
    }

    if (1 < tmp_sam.result.n_best)
    {
        // What happens if soft-clipping or not soft-clipping causes two
        // alignments to be equally likely? So while we could set the scores to be
        // similar, we can also down-weight the score a little bit.
        tmp_sam.score_subo = tmp_sam.score - opt->pen_gapo - opt->pen_gape;
        //tmp_sam.score_subo = tmp_sam.score - 1;
    }
    if (1 == is_long_hit)
    {
        // if we have many seeds that span a large tandem repeat, then we should
        // downweight the sub-optimal score
        tmp_sam.score_subo = tmp_sam.score - opt->pen_gapo - opt->pen_gape;
    }


    if (opt->score_thr <= tmp_sam.score) 
    { // NB: we could also specify 'prev_score <= tmp_sam.score'
        int32_t add_current = 1; // yes, by default

        // Save local variables
        // TODO: do we need to save this data in this fashion?
        int32_t t_end = end;
        int32_t t_start = start;
        int32_t t_end_pos = end_pos;
        uint32_t t_start_pos = start_pos;

        if (   1 == opt->unroll_banding // unrolling is turned on
            && 0 <= max_seed_band  // do we have enough bases to unrooll
            && 1 < end - start + 1 // are there enough seeds in this group
            && (   (prev_n_best < 0 && 1 < tmp_sam.result.n_best) // we need to do a first time unroll
                || (prev_n_best == tmp_sam.result.n_best))) 
        { // unroll was not successful, try again
            uint32_t start_pos_prev=0;
            uint32_t start_pos_cur=0, end_pos_cur=0;
            uint32_t unrolled = 0;

            while (0 == unrolled) 
            {
                // Reset local variables
                start = t_start;
                end = t_end;

                //fprintf (stderr, "max_seed_band=%d start=%d end=%d\n", max_seed_band, start, end);
                // NB: band based on EXACT start position
                int32_t n = end + 1;
                end = start;
                while (start < n) 
                {
                    int32_t cur_score;
                    // reset start and end position
                    if (start == end) 
                    {
                        tmap_map_util_sw_get_start_and_end_pos (&sams->sams [start], seq_len, strand, &start_pos, &end_pos);
                        start_pos_prev = start_pos;
                    }
                    if (end + 1 < n) 
                    {
                        if (   sams->sams [end].strand == sams->sams [end+1].strand
                            && sams->sams [end].seqid == sams->sams [end+1].seqid) 
                        {
                            tmap_map_util_sw_get_start_and_end_pos (&sams->sams [end+1], seq_len, strand, &start_pos_cur, &end_pos_cur);

                            // consider start positions only
                            if (start_pos_cur - start_pos_prev <= max_seed_band) 
                            { // consider start positions only
                                end++;
                                if (end_pos < end_pos_cur) 
                                    end_pos = end_pos_cur;
                                start_pos_prev = start_pos_cur;
                                continue; // there may be more to add
                            } 
                        } 
                    } // do not recurse if we did not unroll if (0 < max_seed_band && t_start == start && t_end == end) { // we did not unroll any max_seed_band = (max_seed_band >> 1); break; }
                    unrolled = 1; // we are going to unroll

                    // TODO: we could hash the alignment result, as unrolling may
                    // cause many more SWs

                    // recurse
                    cur_score = tmap_map_util_sw_gen_score_helper (refseq, sams, origseq, seq, sams_tmp, idx, start, end,
                                                                strand, vsw, seq_len, start_pos, end_pos,
                                                                target_mem, target,
                                                                // softclip_start, softclip_end,
                                                                tmp_sam.result.n_best,
                                                                (max_seed_band <= 0) ? -1 : (max_seed_band >> 1),
                                                                tmp_sam.score,
                                                                vsw_opt, rand, opt,
                                                                amplicons,
                                                                amplicons_capacity,
                                                                stat
                                                                );

                    if (cur_score == tmp_sam.score) 
                        add_current = 0; // do not add the current alignment, we found it during unrolling
                    // update start/end
                    end++;
                    start = end;
                }
            }
        }

        // Reset local variables
        // TODO: do we need to reset this data in this fashion?
        end = t_end;
        start = t_start;
        end_pos = t_end_pos;
        start_pos = t_start_pos;

        if (1 == add_current) 
        {
            tmap_map_sam_t *s = NULL;

            if (sams_tmp->n <= (*idx)) 
            {
                tmap_map_sams_realloc (sams_tmp, (*idx)+1);
                //tmap_error ("bug encountered", Exit, OutOfRange);
            }

            s = &sams_tmp->sams [(*idx)];
            // shallow copy previous data
            (*s) = tmp_sam;
            //s->result = tmp_sam.result;

            // nullify the cigar
            s->n_cigar = 0;
            s->cigar = NULL;

            // adjust target length and position NB: query length is implicitly
            // stored in s->query_end (consider on the next pass)
            s->pos = start_pos - 1; // zero-based
            s->target_len = s->result.target_end + 1;

            if (1 == strand) 
            {
                if (s->pos + tlen <= s->result.target_end) s->pos = 0;
                else s->pos += tlen - s->result.target_end - 1;
            }

            /*
            fprintf (stderr, "strand=%d pos=%d n_best=%d %d-%d %d-%d %d\n",
                    strand,
                    s->pos,
                    s->result.n_best,
                    s->query_start,
                    s->query_end,
                    s->target_start,
                    s->result.target_end,
                    s->target_len);
                    */

            // # of seeds
            s->n_seeds = (end - start + 1);
            // update aux data
            tmap_map_sam_malloc_aux (s);
            switch (s->algo_id) 
            {
            case TMAP_MAP_ALGO_MAP1:
                (*s->aux.map1_aux) = (*tmp_sam.aux.map1_aux);
                break;
            case TMAP_MAP_ALGO_MAP2:
                (*s->aux.map2_aux) = (*tmp_sam.aux.map2_aux);
                break;
            case TMAP_MAP_ALGO_MAP3:
                (*s->aux.map3_aux) = (*tmp_sam.aux.map3_aux);
                break;
            case TMAP_MAP_ALGO_MAP4:
                (*s->aux.map4_aux) = (*tmp_sam.aux.map4_aux);
                break;
            case TMAP_MAP_ALGO_MAPVSW:
                (*s->aux.map_vsw_aux) = (*tmp_sam.aux.map_vsw_aux);
                break;
            default:
                tmap_error ("bug encountered", Exit, OutOfRange);
                break;
            }
            (*idx)++;
        }
    }
    return tmp_sam.score;
}

typedef struct __tmap_map_util_gen_score_t
{
    uint32_t seqid;
    int8_t strand;
    int32_t start;
    int32_t end;
    uint32_t start_pos;
    uint32_t end_pos;
    int8_t filtered:4;
    int8_t repr_hit:4;
} tmap_map_util_gen_score_t;

tmap_map_sams_t *
tmap_map_util_sw_gen_score
(
    tmap_refseq_t *refseq,
    tmap_seq_t *seq,
    tmap_map_sams_t *sams,
    tmap_seq_t **seqs,
    tmap_rand_t *rand,
    tmap_map_opt_t *opt,
    int32_t *num_after_grouping,
    tmap_map_stats_t *stat      // statistics

)
{
  int32_t i, j;
  int32_t start, end;
  tmap_map_sams_t *sams_tmp = NULL;
  int32_t seq_len=0, target_mem=0;
  uint8_t *target=NULL;
  int32_t best_subo_score;
  tmap_vsw_t *vsw = NULL;
  tmap_vsw_opt_t *vsw_opt = NULL;
  uint32_t start_pos, end_pos;
  int32_t softclip_start, softclip_end;
  uint32_t end_pos_prev=0;
  uint32_t start_pos_cur=0, end_pos_cur=0;
  tmap_map_util_gen_score_t *groups = NULL;
  int32_t num_groups = 0, num_groups_filtered = 0;
  double stage_seed_freqc = opt->stage_seed_freqc;
  int32_t max_group_size = 0, repr_hit, filter_ok = 0;
  tmap_map_amplicon_info *amplicons = NULL;
  uint32_t amplicons_capacity = 0;
  // tmap_map_amplicon_info chosen_amplicon;

  if (NULL != num_after_grouping) (*num_after_grouping) = 0;

  if (0 == sams->n) {
      return sams;
  }
  // the final mappings will go here
  sams_tmp = tmap_map_sams_init (sams);
  tmap_map_sams_realloc (sams_tmp, sams->n);

  init_amplicon_info (&amplicons, &amplicons_capacity, 0);

  // sort by strand / chr/pos / score
  tmap_sort_introsort (tmap_map_sam_sort_coord, sams->n, sams->sams);

  // ! This may be re-set later (in _helper) for the hits located in the overriding amplions 
  tmap_map_util_set_softclip (opt->softclip_type, seq, opt->max_adapter_bases_for_soft_clipping, &softclip_start, &softclip_end);

  // initialize opt for initial vsw set up  - the parameters set here can be overriden later in tmap_map_util_sw_gen_score_helper, if the hit position appears to be within overriden amplicon
  vsw_opt = tmap_vsw_opt_init (opt->score_match, opt->pen_mm, opt->pen_gapo, opt->pen_gape, opt->score_thr);

  // init seqs
  seq_len = tmap_seq_get_bases_length (seqs [0]);
  // seq_len = tmap_seq_get_bases_length (seq);

  // forward
  vsw = tmap_vsw_init ((uint8_t*)tmap_seq_get_bases (seqs [0])->s, seq_len, softclip_start, softclip_end, opt->vsw_type, opt->vsw_fallback, vsw_opt);
  // vsw = tmap_vsw_init ((uint8_t*)tmap_seq_get_bases (seq)->s, seq_len, softclip_start, softclip_end, opt->vsw_type, vsw_opt);

  // pre-allocate groups
  groups = tmap_calloc (sams->n, sizeof (tmap_map_util_gen_score_t), "groups");

  // determine groups
  num_groups = num_groups_filtered = 0;
  start = end = 0;
  start_pos = end_pos = 0;
  best_subo_score = INT32_MIN; // track the best sub-optimal hit
  repr_hit = 0;
  while (end < sams->n) {
      uint32_t seqid;
      uint8_t strand;

      // get the strand / start/end positions
      seqid = sams->sams [end].seqid;
      strand = sams->sams [end].strand;
      //first pass, setup start and end
      if (start == end) {
          tmap_map_util_sw_get_start_and_end_pos (&sams->sams [start], seq_len, strand, &start_pos, &end_pos);
          end_pos_prev = end_pos;
          repr_hit = sams->sams [end].repr_hit;
      }

      // sub-optimal score
      if (best_subo_score < sams->sams [end].score_subo) {
          best_subo_score = sams->sams [end].score_subo;
      }

      // check if the hits can be banded
      //fprintf (stderr, "end=%d seqid: %d start: %d end: %d\n", end, sams->sams [end].seqid, sams->sams [end].pos, (sams->sams [end].pos + sams->sams [end].target_len));
      if (end + 1 < sams->n) {
          //fprintf (stderr, "%d seed start: %d end: %d next start: %d next end: %d\n", end, sams->sams [end].pos, (sams->sams [end].pos + sams->sams [end].target_len), sams->sams [end + 1].pos, (sams->sams [end + 1].pos + sams->sams [end + 1].target_len));
          if (sams->sams [end].strand == sams->sams [end + 1].strand
             && sams->sams [end].seqid == sams->sams [end + 1].seqid) {
              tmap_map_util_sw_get_start_and_end_pos (&sams->sams [end + 1], seq_len, strand, &start_pos_cur, &end_pos_cur);

              if (start_pos_cur <= end_pos_prev // NB: beware of unsigned int underflow
                 || (start_pos_cur - end_pos_prev) <= opt->max_seed_band) {
                  end++;
                  if (end_pos < end_pos_cur) {
                      end_pos = end_pos_cur;
                  }
                  end_pos_prev = end_pos_cur;
                  repr_hit |= sams->sams [end].repr_hit; // representitive hit
                  continue; // there may be more to add

              }
          }
      }

      // add a group
      num_groups++;
      // update
      groups [num_groups-1].seqid = seqid;
      groups [num_groups-1].strand = strand;
      groups [num_groups-1].start = start;
      groups [num_groups-1].end = end;
      groups [num_groups-1].start_pos = start_pos;
      groups [num_groups-1].end_pos = end_pos;
      groups [num_groups-1].filtered = 0; // assume not filtered, not guilty
      groups [num_groups-1].repr_hit = repr_hit;
      if (max_group_size < end - start + 1) {
          max_group_size = end - start + 1;
      }
      // update start / end
      end++;
      start = end;
  }

  // resize
  if (num_groups < sams->n) {
      groups = tmap_realloc (groups, num_groups * sizeof (tmap_map_util_gen_score_t), "groups");
  }

  // filter groups
  // NB: if we happen to filter all the groups, iteratively relax the filter
  // settings.
  do
  {
      num_groups_filtered = 0;

      for (i=0;i<num_groups;i++)
      {
          tmap_map_util_gen_score_t *group = &groups [i];

          // NB: if match / mismatch penalties are on the opposite strands, we may
          // have wrong score
          // NOTE:  end >(sams->n * opt->seed_freqc ) comes from
          /*
           * Anatomy of a hash-based long read sequence mapping algorithm for next
           * generation DNA sequencing
           * Sanchit Misra, Bioinformatics, 2011
           * "For each read, we find the maximum of the number of q-hits in
           * all regions, say C. We keep the cutoff as a fraction f of C. Hence,
           * if a region has ≥fC q-hits, only then it is processed further. "
           */
          /*
          fprintf (stderr, "repr_hit=%d size=%d freq=%lf\n",
                  group->repr_hit,
                  (group->end - group->start + 1),
                  ( sams->n * stage_seed_freqc));
          */

          if (0 == group->repr_hit && (group->end - group->start + 1) > ( sams->n * stage_seed_freqc) )
          {
              group->filtered = 0;
          }
          else
          {
              group->filtered = 1;
              num_groups_filtered++;
          }
      }

      /*
      fprintf (stderr, "stage_seed_freqc=%lf num_groups=%d num_groups_filtered=%d\n",
              stage_seed_freqc,
              num_groups,
              num_groups_filtered);
              */

      if (opt->stage_seed_freqc_min_groups <= num_groups
         && (num_groups - num_groups_filtered) < opt->stage_seed_freqc_min_groups) {
          stage_seed_freqc /= 2.0;
          // reset
          for (i=0;i<num_groups;i++)
          {
              tmap_map_util_gen_score_t *group = &groups [i];
              group->filtered = 0;
          }
          num_groups_filtered = 0;
          if (stage_seed_freqc < 1.0 / sams->n)
          { // the filter will accept all hits
              break;
          }
      }
      else
      { // the filter was succesfull
          filter_ok = 1;
          break;
      }
  }
  while (1);

  if (num_groups < opt->stage_seed_freqc_min_groups)
  { // if too few groups, always keep
      // reset
      for (i=0;i<num_groups;i++)
      {
          tmap_map_util_gen_score_t *group = &groups [i];
          group->filtered = 0;
      }
      num_groups_filtered = 0;
  }
  else if (0 == filter_ok)
  { // we could not find an effective filter
      int32_t cur_max;
      // Try filtering based on retaining the groups with most number of seeds.
      cur_max = (max_group_size * 2) - 1; // NB: this will be reduced
      // Assume all are filtered
      num_groups_filtered = num_groups;
      while (num_groups - num_groups_filtered < opt->stage_seed_freqc_min_groups)
      { // too many groups were filtered
          num_groups_filtered = 0;
          // halve the minimum group size
          cur_max /= 2;
          // go through the groups
          for (i=0;i<num_groups;i++)
          {
              tmap_map_util_gen_score_t *group = &groups [i];
              if (group->end - group->start + 1 < cur_max)
              { // filter if there were too few seeds
                  group->filtered = 1;
                  num_groups_filtered++;
              }
              else
              {
                  group->filtered = 0;
              }
          }
      }
  }

  /*
  for (i=j=0;i<num_groups;i++) { // go through each group
      tmap_map_util_gen_score_t *group = &groups [i];
      if (1 == group->filtered && 0 == group->repr_hit) continue;
      j++;
  }
  fprintf (stderr, "num_groups=%d num_groups_filtered=%d j=%d\n", num_groups, num_groups_filtered, j);
  */

  /*
  fprintf (stderr, "final sams->n=%d num_groups=%d num_groups_filtered=%d\n",
          sams->n,
          num_groups,
          num_groups_filtered);
          */

  // process unfiltered...
  for (i=j=0;i<num_groups;i++)
  { // go through each group
      tmap_map_util_gen_score_t *group = &groups [i];
      /*
      fprintf (stderr, "start=%d end=%d num=%d seqid=%u strand=%d start_pos=%u end_pos=%u repr_hit=%u filtered=%d\n",
              group->start,
              group->end,
              group->end - group->start + 1,
              group->seqid,
              group->strand,
              group->start_pos,
              group->end_pos,
              group->repr_hit,
              group->filtered);
              */
      if (1 == group->filtered && 0 == group->repr_hit) continue;
      /*
      fprintf (stderr, "start=%d end=%d num=%d seqid=%u strand=%d start_pos=%u end_pos=%u filtered=%d\n",
              group->start,
              group->end,
              group->end - group->start + 1,
              group->seqid,
              group->strand,
              group->start_pos,
              group->end_pos,
              group->filtered);
              */

      // generate the score
      tmap_map_util_sw_gen_score_helper (refseq, sams, seq, seqs [0], sams_tmp, &j, group->start, group->end,
                                        group->strand, vsw, seq_len, group->start_pos, group->end_pos,
                                        &target_mem, &target,
                                        // softclip_start, softclip_end,
                                        -1, // this is our first call
                                        opt->max_seed_band, // NB: this may be modified as banding is unrolled
                                        opt->score_thr-1,
                                        vsw_opt, rand, opt,
                                        &amplicons,
                                        &amplicons_capacity,
                                        stat
                                        );
      // save the number of groups
      if (NULL != num_after_grouping) (*num_after_grouping)++;
  }
  //fprintf (stderr, "unfiltered # = %d\n", j);

  // only if we applied the freqc filter
  if (0 < stage_seed_freqc && 0 < opt->stage_seed_freqc_rand_repr)
  {
      // check if we filtered too many groups, and so if we should keep
      // representative hits.
      /*
      fprintf (stderr, "Should we get representatives %lf < %lf\n",
              opt->stage_seed_freqc_group_frac,
              (double)num_groups_filtered / num_groups);
              */
      if (opt->stage_seed_freqc_group_frac <= (double)num_groups_filtered / num_groups)
      {
          int32_t best_score = INT32_MIN, best_score_filt = INT32_MIN;
          double pr = 0.0;
          int32_t k, l, c, n;

          // get the best score from a group that passed the filter
          for (k=0;k<j;k++)
          { // NB: keep k for later
              if (best_score < sams_tmp->sams [k].score)
              {
                  best_score = sams_tmp->sams [k].score;
              }
          }

          // the # added
          n = 0;

          // pick the one with the most # of seeds
          c = l = -1;
          for (i=0;i<num_groups;i++)
          {
              tmap_map_util_gen_score_t *group = &groups [i];
              if (0 == group->filtered) continue;
              if (c < group->end - group->start + 1)
              {
                  c = group->end - group->start + 1;
                  l = i;
              }
          }
          if (0 <= l)
          {
              tmap_map_util_gen_score_t *group = &groups [l];
              // generate the score
              tmap_map_util_sw_gen_score_helper (refseq, sams, seq, seqs [0], sams_tmp, &j, group->start, group->end,
                                                group->strand, vsw, seq_len, group->start_pos, group->end_pos,
                                                &target_mem, &target,
                                                // softclip_start, softclip_end,
                                                -1, // this is our first call
                                                opt->max_seed_band, // NB: this may be modified as banding is unrolled
                                                opt->score_thr-1,
                                                vsw_opt, rand, opt,
                                                &amplicons,
                                                &amplicons_capacity,
                                                stat
                                               );
              group->filtered = 0; // no longer filtered
          }

          // now, choose uniformly
          pr = num_groups / (1 + opt->stage_seed_freqc_rand_repr);
          for (i=c=n=0;i<num_groups;i++,c++)
          {
              tmap_map_util_gen_score_t *group = &groups [i];
              if (0 == group->filtered) continue;
              if (pr < c)
              {
                  if (n == opt->stage_seed_freqc_rand_repr) break;
                  // generate the score
                  tmap_map_util_sw_gen_score_helper (refseq, sams, seq, seqs [0], sams_tmp, &j, group->start, group->end,
                                                    group->strand, vsw, seq_len, group->start_pos, group->end_pos,
                                                    &target_mem, &target,
                                                    // softclip_start, softclip_end,
                                                    -1, // this is our first call
                                                    opt->max_seed_band, // NB: this may be modified as banding is unrolled
                                                    opt->score_thr-1,
                                                    vsw_opt, rand, opt,
                                                    &amplicons,
                                                    &amplicons_capacity,
                                                    stat
                                                   );
                  group->filtered = 0; // no longer filtered
                  n++;
                  // reset
                  c = 0;
              }
          }

          // get the best from the filtered
          for (;k<j;k++)
          { // NB: k should not have been modified
              if (best_score_filt < sams_tmp->sams [k].score)
              {
                  best_score_filt = sams_tmp->sams [k].score;
              }
          }

          /*
          fprintf (stderr, "best_score=%d best_score_filt=%d\n",
                  best_score,
                  best_score_filt);
                  */
          // check if we should redo ALL the filtered groups
          if (best_score < best_score_filt)
          {
              // NB: do not redo others...
              for (i=0;i<num_groups;i++)
              {
                  tmap_map_util_gen_score_t *group = &groups [i];
                  if (0 == group->filtered) continue;
                  // generate the score
                  tmap_map_util_sw_gen_score_helper (refseq, sams, seq, seqs [0], sams_tmp, &j, group->start, group->end,
                                                    group->strand, vsw, seq_len, group->start_pos, group->end_pos,
                                                    &target_mem, &target,
                                                    // softclip_start, softclip_end,
                                                    -1, // this is our first call
                                                    opt->max_seed_band, // NB: this may be modified as banding is unrolled
                                                    opt->score_thr-1,
                                                    vsw_opt, rand, opt,
                                                    &amplicons,
                                                    &amplicons_capacity,
                                                    stat
                                                   );
              }
          }
      }
  }

  // realloc
  tmap_map_sams_realloc (sams_tmp, j);

  // sub-optimal score
  for (i=0;i<sams_tmp->n;i++)
  {
      if (best_subo_score < sams_tmp->sams [i].score_subo)
      {
          best_subo_score = sams_tmp->sams [i].score_subo;
      }
  }
  // update the sub-optimal
  for (i=0;i<sams_tmp->n;i++)
  {
      sams_tmp->sams [i].score_subo = best_subo_score;
  }

  // free memory
  tmap_map_sams_destroy (sams);
  free (target);
  destroy_amplicon_info (&amplicons, &amplicons_capacity);
  tmap_vsw_opt_destroy (vsw_opt);
  tmap_vsw_destroy (vsw);
  free (groups);

  return sams_tmp;
}


static void
tmap_map_util_keytrim
(
    uint8_t *query, int32_t qlen,
    uint8_t *target, int32_t tlen,
    int8_t strand,
    uint8_t key_base,
    tmap_map_sam_t *s,
    tmap_map_stats_t* stat
)
{
  int32_t j, k, l;
  int32_t op, op_len;
  //NB: we may only need to look at the first cigar
  op = op_len = 0;
  if(0 == strand) { // forward
      for(j = k = l = 0; j < s->n_cigar; j++) {
          // get the cigar
          op = TMAP_SW_CIGAR_OP(s->cigar[j]);
          op_len = TMAP_SW_CIGAR_LENGTH(s->cigar[j]);
          if(op == BAM_CDEL) break; // looking for mismatch/insertion
          if(op == BAM_CSOFT_CLIP) break; // already start trimming

          while(0 < op_len) {
              if(query[k] != key_base) break;
              if(op == BAM_CINS) {
              }
              else if(op == BAM_CMATCH && target[l] != query[k]) {
                  l++;
              }
              else {
                  break;
              }
              op_len--;
              k++; // since we can only have match/mismatch/insertion
          }
          if(0 == k) {
              // no trimming
              break;
          }
          else if(0 < op_len) {
              if(j == 0) {
                  // reallocate
                  s->n_cigar++;
                  s->cigar = tmap_realloc(s->cigar, sizeof(uint32_t)*s->n_cigar, "s->cigar");
                  for(k=s->n_cigar-1;0<k;k--) { // shift up
                      s->cigar[k] = s->cigar[k-1];
                  }
                  s->cigar[0] = 0;
                  j++; // reflect the shift in j
              }
              // NB: 0 < j
              // add to the leading soft-clip
              TMAP_SW_CIGAR_STORE(s->cigar[0], BAM_CSOFT_CLIP, TMAP_SW_CIGAR_LENGTH(s->cigar[j]) - op_len + TMAP_SW_CIGAR_LENGTH(s->cigar[0]));
              // reduce the cigar length
              TMAP_SW_CIGAR_STORE(s->cigar[j], op, op_len);
              break;
          }
          else { // NB: the full op was removed
              if(0 == j) {
                  TMAP_SW_CIGAR_STORE(s->cigar[0], BAM_CSOFT_CLIP, TMAP_SW_CIGAR_LENGTH(s->cigar[0]));
              }
              else {
                  // add to the leading soft-clip
                  TMAP_SW_CIGAR_STORE(s->cigar[0], BAM_CSOFT_CLIP, TMAP_SW_CIGAR_LENGTH(s->cigar[j]) + TMAP_SW_CIGAR_LENGTH(s->cigar[0]));
                  // shift down and overwrite the current cigar
                  for(k=j;k<s->n_cigar-1;k++) {
                      s->cigar[k] = s->cigar[k+1];
                  }
                  s->n_cigar--;
                  j--; // since j is incremented later
              }
          }
      }
      // update the position based on the number of reference bases we
      // skipped
      s->pos += l;
  }
  else { // reverse
      for(j = s->n_cigar-1, k = qlen-1, l = tlen-1; 0 <= j; j--) {
          // get the cigar
          op = TMAP_SW_CIGAR_OP(s->cigar[j]);
          op_len = TMAP_SW_CIGAR_LENGTH(s->cigar[j]);
          if(op == BAM_CDEL) break; // looking for mismatch/insertion
          if(op == BAM_CSOFT_CLIP) break; // already start trimming
          while(0 < op_len) {
              if(query[k] != (3 - key_base)) break;
              if(op == BAM_CINS) {
              }
              else if(op == BAM_CMATCH && target[l] != query[k]) {
                  l--;
              }
              else {
                  break;
              }
              op_len--;
              k--; // since we can only have match/mismatch/insertion
          }
          if(qlen-1 == k) {
              // no trimming
              break;
          }
          else if(0 < op_len) {
              if(j == s->n_cigar-1) {
                  // reallocate
                  s->n_cigar++;
                  s->cigar = tmap_realloc(s->cigar, sizeof(uint32_t)*s->n_cigar, "s->cigar");
                  s->cigar[s->n_cigar-1] = 0;
              }
              // add to the ending soft-clip
              TMAP_SW_CIGAR_STORE(s->cigar[s->n_cigar-1], BAM_CSOFT_CLIP, TMAP_SW_CIGAR_LENGTH(s->cigar[j]) - op_len + TMAP_SW_CIGAR_LENGTH(s->cigar[s->n_cigar-1]));
              // reduce the cigar length
              TMAP_SW_CIGAR_STORE(s->cigar[j], op, op_len);
              break;
          }
          else { // NB: the full op was removed
              if(j == s->n_cigar-1) {
                  TMAP_SW_CIGAR_STORE(s->cigar[s->n_cigar-1], BAM_CSOFT_CLIP, TMAP_SW_CIGAR_LENGTH(s->cigar[s->n_cigar-1]));
              }
              else {
                  // add to the ending soft-clip
                  TMAP_SW_CIGAR_STORE(s->cigar[s->n_cigar-1], BAM_CSOFT_CLIP, TMAP_SW_CIGAR_LENGTH(s->cigar[s->n_cigar-1]));
                  // shift down and overwrite the current cigar
                  for(k=j;k<s->n_cigar-1;k++) {
                      s->cigar[k] = s->cigar[k+1];
                  }
                  s->n_cigar--;
                  j++; // since j is decremented later
              }
          }
      }
  }
}

// DK note: implementation is very inefficient, could be done in linear time

static void
tmap_map_util_merge_adjacent_cigar_operations (tmap_map_sam_t *s)
{
    int32_t i, j;
    // merge adjacent cigar operations that have the same value
    for (i = s->n_cigar-2; 0 <= i; i--)
    {
        if (TMAP_SW_CIGAR_OP (s->cigar [i]) == TMAP_SW_CIGAR_OP (s->cigar [i+1]))
        {
            TMAP_SW_CIGAR_STORE (s->cigar[i], TMAP_SW_CIGAR_OP(s->cigar[i]), TMAP_SW_CIGAR_LENGTH(s->cigar[i]) + TMAP_SW_CIGAR_LENGTH(s->cigar[i+1]));
            // shift down
            for (j = i + 1; j < s->n_cigar - 1; j++)
                s->cigar [j] = s->cigar [j + 1];
            s->n_cigar--;
        }
    }
}

// ZZ: finding #S#M#I or #S#M#D alignment for the two sequences allowing at most M mismatch.
// DK: returns 1 if found, 0 if not
// DK: above cigars are for reverse orientation (at the 3' of reverse - oriented read), appearing at the beginning of the whole alignment
// DK: for forward - oriented reads, that would be #I#M#S and #D#M#S, appearing on the end of the whole alignment

static int
tmap_map_util_one_gap 
(
    uint8_t *query,   // query sequence (numeric-encoded as [ACGT], 1 byte per base)
    int32_t qlen,     // query length
    uint8_t *target,  // target sequence (numeric-encoded as [ACGT], 1 byte per base)
    int32_t tlen,     // target length
    int32_t min_size, // min anchor size (min length of M segment)
    int max_q,        // max softclip in query
    int max_t,        // max clip (skipped bases, or shift) in target. This
    int max_MM,       // maximum allowed mismatches in the searched structure
    int max_indel,    // maximal number of indels (length of I or D)
    int8_t strand,    // strand (== orientation): 0 - forward, 1 - reverse
    int *n_mismatch,  // found number of mismatches
    int *indel_size,  // number of bases in Indel segment
    int *is_ins,      // 1 == Ins, 0 == Del
    int *softclip,    // length of prefix softclip
    int pad,          // the extension on either size of target buffer that is safe to access. This function checks this many bases as continuation of M segment as if no indel was present
    int matched_bases, 
    int *extra_match  // number of consecutive matches after M segment end (check relation with pad. Are these bases in the pad??)
)
{
    int i, j, q, qq, t, tt, inc;
    int qini, tini;
    if (strand == 0)
    {
        qini = qlen-1;
        tini = tlen-1;
        inc = -1;
    }
    else
    {
        qini = tini = 0;
        inc = 1;
    }
    /*  can still push the indel to middle
    if (qlen < min_size || tlen < min_size)
        return 0;
    */
    if (qlen - min_size < max_q)
        max_q = qlen - min_size;
    //if (tlen - min_size < max_t)
        //max_t = tlen - min_size;   // allow less anchor to find duplicate
    //if (qlen*3 < max_indel) max_indel = qlen*3;
    if (/*max_MM > 1 &&*/ qlen - max_q + 1 < 8 * max_MM)
        max_MM = (qlen - max_q + 1) / 8; // 7-14 bp allow 1 MM, 15 allow 2 etc.
    //fprintf(stderr, "qt=%d %d max_t=%d max_indel=%d MM=%d\n", qlen, tlen, max_t, max_indel, max_MM); 
    if (qlen - tlen > max_indel)
        return 0;
    if (tlen - max_t - qlen > max_indel)
        return 0;
    //debug
    /*
    char code[] = "ACGT";
    fprintf(stderr, "tlen=%d qlen=%d\n", tlen, qlen);
    for (i = 0; i < tlen; i++)
        fprintf(stderr, "%c", code[target[i]]);
    fprintf(stderr, "\n");
    for (i = 0; i < qlen; i++)
        fprintf(stderr,"%c", code[query[i]]);
    fprintf(stderr, "\n");
    */	
    for (i = 0, t = tini; i < max_t; i++, t+= inc)
    {
        int last_M = -1;  // last_mismatch
        int nM = 0;       // number_of_mismatches
        if (tlen - i < 5)
        {
            max_q = 0;
            max_MM = 0;
        }
        else if (tlen - i - max_q < 8 * max_MM)
        {
            max_MM = (tlen - i - max_q) / 8;
        }

        for (j = i, q = 0, qq = qini, tt = t; j < tlen && q < qlen; j++, q++, qq += inc, tt+= inc)
        {
            if (query [qq] == target [tt])
                continue;
            if (q < max_q)
                last_M = q;
            else
            {
                nM++;
                if (nM > max_MM)
                    break;
            }
        }
        if (j == tlen || q == qlen)
        {
            // find one
            *softclip = last_M + 1;
            *n_mismatch = nM;
            *extra_match = 0;
            if (j == tlen)
            {
                *indel_size = qlen-q;
                *is_ins = 1;
            }
            else
            {
                *indel_size = tlen - j;
                *is_ins = 0;
            }
            if (*indel_size > max_indel)
                return 0;
            if (*indel_size == 0)
                return 0;
            // here for strand == 0 we may go into negative index, intentionally. careful!
            // extend beyond the parallel
            // DK: for this to work correctly, passed pad should always be confined within the passed query and target buffers, with corrections for clips and indels. Is this the case - need to check calling code.!?
            //fprintf(stderr, "find gap strand %d pad %d %d %d %d\n", strand, pad, j, tlen, qlen);
            for (j = 0; j < pad; j++, qq += inc, tt += inc)
            {
                //fprintf(stderr, "%d %d %d %d\n", qq, tt, query[qq], target[tt]);
                if (query [qq] != target [tt])
                    break;
                (*extra_match) += 1;
            }
	    if (*extra_match >= matched_bases) return 0; // cannot have leading indel, total duplication or no extra anchor, TS-18181
	    int total_anchor = ((qlen < tlen-i)? qlen: tlen-i)  + (*extra_match);
	    if (total_anchor < min_size) continue;
            if (total_anchor < 15 && *indel_size > 3 * (qlen + (*extra_match)))  // reject indels 3 or more times longer than query + matching 'pad'
                return 0;
            return 1;
        }
    }
    return 0;
}
// swaps two integers (passed by pointers)
static void exchange(int32_t *s, int32_t *t)
{
    int32_t temp = *s; *s = *t; *t = temp;
}

static int
trim_read_bases 
(
    tmap_map_sam_t *s,
    int extra_match,
    int from_head,
    int *is_ins,
    int *indel_len
)
{
    if (extra_match <= 0)
        return 0;
    int32_t indel_type, other_type;

    indel_type = BAM_CDEL; other_type = BAM_CINS;
    if (!(*is_ins))
        exchange(&indel_type, &other_type);

    int inc, init, i, j;
    int dec = 0;
    int in_len = *indel_len;

    if (from_head)
    {
        init = 1;
        inc = 1;
    }
    else
    {
        init = s->n_cigar-2;
        inc = -1;
    }
    int32_t op, op_len;
    for (i = 0, j = init; extra_match > 0 && i < s->n_cigar-1; i++, j += inc)
    {
        op = TMAP_SW_CIGAR_OP (s->cigar[j]);
        op_len = TMAP_SW_CIGAR_LENGTH (s->cigar[j]);
        //fprintf(stderr, "%d %d extra_match=%d j=%d indel=%d other=%d\n", op, op_len, extra_match, j, indel_type, other_type);
        if (op == other_type)
        {
            in_len += op_len;
            dec++;
        }
        else
        {
            //if (op == BAM_CMATCH || op == indel_type || op == BAM_CSOFT_CLIP) are the only choice
            if (op == indel_type)
            {
                int x = extra_match > op_len ? op_len : extra_match;
                if (x > in_len)
                {
                    extra_match -= in_len;
                    in_len = op_len - in_len;
                    exchange (&indel_type, &other_type); // switch indel type now, tricky! ZZ
                    dec++; // put the indel as the ending one. so remove this op.
                    continue;
                }
                in_len -= x;
            }
            if (extra_match >= op_len)
            {
                extra_match -= op_len;
                dec++;
            }
            else
            {
                op_len -= extra_match;
                TMAP_SW_CIGAR_STORE(s->cigar[j], op, op_len);
                break;
            }
    	}
    }
    if (i >=  s->n_cigar-1)
        tmap_error("extra match exceed cigar length", Exit, OutOfRange);
    /* may not need the following
    op = TMAP_SW_CIGAR_OP(s->cigar[j]);
    op_len = TMAP_SW_CIGAR_LENGTH(s->cigar[j]);

    if (op == other_type) {dec++; in_len += op_len;}
    */
    *is_ins = (indel_type == BAM_CINS)? 0:1;
    *indel_len = in_len;
    return dec;
}

static uint32_t
tmap_map_util_end_repair 
(
    tmap_seq_t *seq, uint8_t *query, int32_t qlen,
    uint8_t *target_prev, int32_t tlen,
    int8_t strand,
    tmap_sw_path_t **path_buf,
    int32_t* path_buf_sz,
    tmap_refseq_t *refseq,
    tmap_map_sam_t *s,
    tmap_map_opt_t *opt,
    int repair5p,
    tmap_map_stats_t* stat
)
{
    int32_t i, cigar_i;
    int32_t op, op_len, cur_len;
    int32_t cur_op, cur_op_len, cur_cigar_i, cur_cur_len, target_adj;
    int32_t softclip_start, softclip_end;
    int32_t old_score, new_score;
    int32_t start_pos, end_pos;
    // scoring matrix
    int32_t matrix [25], matrix_iupac [80];
    tmap_sw_param_t par, par_iupac;
    par.matrix_owned = par_iupac.matrix_owned = 0;
    int32_t path_len = 0;
    uint32_t *cigar = NULL;
    int32_t n_cigar = 0;
    int32_t found = 0;

    // TODO: use external shared (thread local) target memory pool to reduce memory management overhead
    uint8_t *target = NULL;
    int32_t target_mem = 0;
    int32_t conv = 0;

    // PARAMETERS
    int32_t max_match_len = 5; // the maximum # of query bases to consider for the sub-alignment
    int32_t num_prev_bases = 2; // the maximum number of bases prior to the alignment to include in the target

    // side and orientation for stats
    int32_t sori = repair5p?(s->strand?R5P:F5P):(s->strand?R3P:F3P);
    int32_t repaired = 0; // return value: 0 for no change, 1 for softclip only, 2 for indel salvage.

    // local parameter override
    // use le (low end) for F5P and R3P; use he (high end) for R5P and F3P
    uint32_t use_le = (sori == F5P || sori == R3P);
    int32_t end_repair = opt->end_repair;
    int32_t use_bed_in_end_repair = opt->use_bed_in_end_repair;
    int32_t score_match = opt->score_match;
    int32_t pen_mm = opt->pen_mm;
    int32_t pen_gapo = opt->pen_gapo;
    int32_t pen_gape = opt->pen_gape;
    int32_t bw = opt->bw;
    int32_t min_anchor_large_indel_rescue = opt->min_anchor_large_indel_rescue;
    int32_t amplicon_overrun = opt->amplicon_overrun;
    int32_t max_one_large_indel_rescue = opt->max_one_large_indel_rescue;
    int32_t max_adapter_bases_for_soft_clipping = opt->max_adapter_bases_for_soft_clipping;
    if (s->param_ovr)
    {
        if (s->param_ovr->score_match.over)
            score_match = s->param_ovr->score_match.value;
        if (s->param_ovr->pen_mm.over)
            pen_mm = s->param_ovr->pen_mm.value;
        if (s->param_ovr->pen_gapo.over)
            pen_gapo = s->param_ovr->pen_gapo.value;
        if (s->param_ovr->pen_gape.over)
            pen_gape = s->param_ovr->pen_gape.value;
        if (s->param_ovr->bw.over)
            pen_gape = s->param_ovr->bw.value;
        if (s->param_ovr->use_bed_in_end_repair.over)
            use_bed_in_end_repair = s->param_ovr->use_bed_in_end_repair.value;
        if (use_le) // lower-coordinate end of the amplicon
        {
            if (s->param_ovr->end_repair_le.over)
                end_repair = s->param_ovr->end_repair_le.value;
            else if (s->param_ovr->end_repair.over)
                end_repair = s->param_ovr->end_repair.value;

            if (s->param_ovr->min_anchor_large_indel_rescue_le.over)
                min_anchor_large_indel_rescue = s->param_ovr->min_anchor_large_indel_rescue_le.value;
            else if (s->param_ovr->min_anchor_large_indel_rescue.over)
                min_anchor_large_indel_rescue = s->param_ovr->min_anchor_large_indel_rescue.value;

            if (s->param_ovr->max_amplicon_overrun_large_indel_rescue_le.over)
                amplicon_overrun = s->param_ovr->max_amplicon_overrun_large_indel_rescue_le.value;
            else if (s->param_ovr->max_amplicon_overrun_large_indel_rescue.over)
                amplicon_overrun = s->param_ovr->max_amplicon_overrun_large_indel_rescue.value;

            if (s->param_ovr->max_one_large_indel_rescue_le.over)
                max_one_large_indel_rescue = s->param_ovr->max_one_large_indel_rescue_le.value;
            else if (s->param_ovr->max_one_large_indel_rescue.over)
                max_one_large_indel_rescue = s->param_ovr->max_one_large_indel_rescue.value;

//#if 0
            // NOTE: not messing with max_adapter_bases_for_soft_clipping now, too dangerous: end_repair requires that softclip_start and softclip_end
            // determined by calls to tmap_map_util_set_softclip in calling function and below match. If they are not, end_repair does not work (may hang or segfault).
            if (s->param_ovr->max_adapter_bases_for_soft_clipping_le.over)
                max_adapter_bases_for_soft_clipping = s->param_ovr->max_adapter_bases_for_soft_clipping_le.value;
            else if (s->param_ovr->max_adapter_bases_for_soft_clipping.over)
                max_adapter_bases_for_soft_clipping = s->param_ovr->max_adapter_bases_for_soft_clipping.value;
//#endif
        }
        else // higher-coordinate end of the amplicon
        {
            if (s->param_ovr->end_repair_he.over)
                end_repair = s->param_ovr->end_repair_he.value;
            else if (s->param_ovr->end_repair.over)
                end_repair = s->param_ovr->end_repair.value;

            if (s->param_ovr->min_anchor_large_indel_rescue_he.over)
                min_anchor_large_indel_rescue = s->param_ovr->min_anchor_large_indel_rescue_he.value;
            else if (s->param_ovr->min_anchor_large_indel_rescue.over)
                min_anchor_large_indel_rescue = s->param_ovr->min_anchor_large_indel_rescue.value;

            if (s->param_ovr->max_amplicon_overrun_large_indel_rescue_he.over)
                amplicon_overrun = s->param_ovr->max_amplicon_overrun_large_indel_rescue_he.value;
            else if (s->param_ovr->max_amplicon_overrun_large_indel_rescue.over)
                amplicon_overrun = s->param_ovr->max_amplicon_overrun_large_indel_rescue.value;

            if (s->param_ovr->max_one_large_indel_rescue_he.over)
                max_one_large_indel_rescue = s->param_ovr->max_one_large_indel_rescue_he.value;
            else if (s->param_ovr->max_one_large_indel_rescue.over)
                max_one_large_indel_rescue = s->param_ovr->max_one_large_indel_rescue.value;

//#if 0
            // NOTE: not messing with max_adapter_bases_for_soft_clipping now, too dangerous: end_repair requires that softclip_start and softclip_end
            // determined by calls to tmap_map_util_set_softclip in calling function and below match. If they are not, end_repair does not work (may hang or segfault).
            if (s->param_ovr->max_adapter_bases_for_soft_clipping_he.over)
                max_adapter_bases_for_soft_clipping = s->param_ovr->max_adapter_bases_for_soft_clipping_he.value;
            else if (s->param_ovr->max_adapter_bases_for_soft_clipping.over)
                max_adapter_bases_for_soft_clipping = s->param_ovr->max_adapter_bases_for_soft_clipping.value;
//#endif
        }
    }
    if (!end_repair) // not enabled for a stage or overriden to disabled for this end of amplicon or for entire amplicon
        return repaired;

    // check if we allow soft-clipping on the 5' end
    tmap_map_util_set_softclip (opt->softclip_type, seq, max_adapter_bases_for_soft_clipping, &softclip_start, &softclip_end);
    int8_t o_strand = strand;
    if (end_repair > 2)  // Modern End-Repair, invoked when --end-repair parameter is above 2 (and denotes percent of mismatches above which to trim the alignment)
    {
        if (repair5p == 0 && 1 == softclip_end)
            return repaired; // allow softclip, then no need to repair.
        if (repair5p == 1 && 1 == softclip_start)
            return repaired;
        //fprintf(stderr, "repair5p=%d %d\n", repair5p, strand);
        if (repair5p)
            strand = (!strand);
        old_score = 0; // score of original (passed in) alignment
        found = 0; // number of non-matches (mismatches, inserts, deletes) found in alignment
        int worst = 0; // worst alignment score encountered along the alignment
        int ind = 0; // index of base within the cigar operation where worst alignment score is achieved
        int nMM = 0; // number of mismatches in the part of alignment before worst score position
        int nqb = 0; // number of query bases seen
        int ntb = 0; // number of target bases seen
        int nCC;     // (?) NuCleotide Count - 'average' of target and query lengths involved in (old, passed in) alignment
        int target_red = 0; // Target Reduction: the offset on target to the position of worst score
        int tb, qb, inc;  // target position, query position, increment (+1 | -1) for walking along target / query
        int i_c; // index of the cigar operation
        int i;   // index of base within cigar operation
        int nn_cigar = 0;  // cigar operation index where worst alignment score is achieved
        int total_scl = 0; // total soft clip (?) - actually the offset on query to the position of worst score
        int max_gap = 0, num_gap = 0; // ZZ, for TS-10540

        // remove softclip, shall not be necessary, remove later?? ZZ

        if  (strand == 0)
        {
            op = TMAP_SW_CIGAR_OP (s->cigar [s->n_cigar - 1]);
            op_len = TMAP_SW_CIGAR_LENGTH (s->cigar [s->n_cigar - 1]);
            if (op == BAM_CSOFT_CLIP)
            {
                op = TMAP_SW_CIGAR_OP (s->cigar [s->n_cigar - 2]);
                int32_t op_len1 = TMAP_SW_CIGAR_LENGTH (s->cigar [s->n_cigar - 2]);
                if (op ==  BAM_CMATCH)
                    TMAP_SW_CIGAR_STORE (s->cigar [s->n_cigar - 2], op, op_len + op_len1);
                else
                    return repaired;
                s->n_cigar--;
                tmap_map_util_set_target_len (s);
                qlen += op_len;
                tlen = s->target_len;
                repaired = 1;
            }
        }
        else
        {
            op = TMAP_SW_CIGAR_OP (s->cigar [0]);
            op_len = TMAP_SW_CIGAR_LENGTH (s->cigar [0]);
            if (op == BAM_CSOFT_CLIP)
            {
                op = TMAP_SW_CIGAR_OP (s->cigar [1]);
                int32_t op_len1 = TMAP_SW_CIGAR_LENGTH (s->cigar [1]);
                if (op ==  BAM_CMATCH)
                {
                    TMAP_SW_CIGAR_STORE (s->cigar[1], op, op_len+op_len1);
                    s->pos -= op_len;
                }
                else
                    return repaired;
                for (i = 0; i < s->n_cigar-1; i++)
                    s->cigar [i] = s->cigar [i+1];
                s->n_cigar--;
                tmap_map_util_set_target_len (s);
                query -= op_len;
                target_prev -= op_len;
                qlen += op_len;
                tlen = s->target_len;
                repaired = 1;
            }
        }

        // evaluate if repairing makes sense
        if (0 != strand)
        {
            tb = qb = 0;
            inc = 1;
        }
        else
        {
            tb = tlen-1;
            qb = qlen-1;
            inc = -1;
        }
        for (i_c = 0; i_c < s->n_cigar; i_c++)
        {
            cigar_i = (0 != strand) ? i_c : s->n_cigar-1-i_c;
            op = TMAP_SW_CIGAR_OP(s->cigar[cigar_i]);
            op_len = TMAP_SW_CIGAR_LENGTH(s->cigar[cigar_i]);
            if (op == BAM_CMATCH)
            {
                for (i = 0; i < op_len; i++, qb += inc, tb += inc, nqb++, ntb++)
                {
                    if (qb == -1 || tb == -1)
                        tmap_bug ();
                    if (query [qb] == target_prev [tb])
                        old_score += score_match;
                    else
                    {
                        old_score -= pen_mm;
                        found++;
                    }
                    if (old_score < worst)
                    {
                        worst = old_score;
                        ind = i + 1;
                        nMM = found;
                        target_red = ntb + 1;
                        nn_cigar = i_c;
                        total_scl = nqb + 1;
                    }
                }
            }
            else
            {
                if (op == BAM_CDEL)
                {
                    ntb += op_len;
                    tb += inc * op_len;
                }
                else if (op == BAM_CINS)
                {
                    nqb += op_len;
                    qb  += inc*op_len;
                }
                else
                    break;
                old_score -= pen_gapo + pen_gape * op_len;
                found += op_len;
                if (old_score < worst)
                {
                    worst = old_score;
                    ind = 0;
                    nMM = found;
                    target_red = ntb;
                    nn_cigar = i_c;
                    total_scl = nqb;
                    num_gap++;
                    if (op_len > max_gap)
                        max_gap = op_len; // ZZ, for TS-10540
                }
            }
            if (old_score > 6 * score_match)
                break; // no need go further.
        }
        if (worst >= 0)
            return repaired;
        nCC = target_red + total_scl + 1;
        if (num_gap < 3 && max_gap >= 3)
        {
            int del = max_gap / 15; // increase the error by 1 for each 15 bp of the gap.
            int adj = max_gap - 1 - del;
            /* int adj = max_gap-1;*/
            nMM -= adj;
            nCC -= adj;
        }
        nCC /= 2;
        int maxMM = (nCC - 2) * end_repair / 100 + 2;  // maximal tolerable number of mismatches // DK: integer arithmetics !
        if (maxMM < 2)
            maxMM = 2; // no need? always at least 2
        if (nCC < maxMM)
            maxMM = nCC;
        if (nMM >= maxMM) // observed (and adjusted) number of mismatches BEFORE THE WORST SCORE POSITION is over maximal tolerable number of mismatches
        {
            s->score -= worst;
            int start_o = s->pos, end_o = s->pos + s->target_len;

            // introduce softclip from worst alignment score position to the end (proper one, depending on passed in alignment 'end' for repair and on read strand)
            cigar_i = (0 != strand) ? nn_cigar : s->n_cigar - 1 - nn_cigar;
            op_len = TMAP_SW_CIGAR_LENGTH (s->cigar [cigar_i]);
            if (ind == op_len || ind == 0)
            {
                if (nn_cigar == s->n_cigar - 1)
                    return repaired; // all are removed, something not quite right. Give up
                if (nn_cigar == s->n_cigar - 2)
                {
                    if (TMAP_SW_CIGAR_OP (s->cigar [cigar_i + inc]) == BAM_CSOFT_CLIP)
                        return repaired; // something wrong, will become XSYS??
                }
                TMAP_SW_CIGAR_STORE (s->cigar [cigar_i], BAM_CSOFT_CLIP, total_scl);
                s->n_cigar -= nn_cigar;
            }
            else
            {
                TMAP_SW_CIGAR_STORE (s->cigar [cigar_i], BAM_CMATCH, op_len - ind);
                if (nn_cigar == 0)
                {
                    s->cigar = tmap_realloc (s->cigar, sizeof (uint32_t) * (1 + s->n_cigar), "s->cigar");
                    if (0 != strand)
                    {
                        for (i = s->n_cigar - 1; i >= 0; i--)
                            s->cigar [i+1] = s->cigar [i];
                        TMAP_SW_CIGAR_STORE (s->cigar [0], BAM_CSOFT_CLIP, total_scl);
                    }
                    else
                    {
                        TMAP_SW_CIGAR_STORE (s->cigar [s->n_cigar],  BAM_CSOFT_CLIP, total_scl);
                    }
                    s->n_cigar++;
                }
                else
                {
                    cigar_i -= inc; // move back one
                    TMAP_SW_CIGAR_STORE (s->cigar [cigar_i], BAM_CSOFT_CLIP, total_scl);
                    s->n_cigar -= nn_cigar - 1;
                }
            }
            // record in stat that we clipped the alignment
            stat->num_end_repair_clipped [sori] ++;
            stat->bases_end_repair_clipped [sori] += total_scl;
            if (0 != strand && cigar_i > 0)
            {
                for (i = 0; i < s->n_cigar; i++)
                {
                    s->cigar [i] = s->cigar [i+cigar_i];
                }
            }
            // adjust target position for skipped target bases. DK: Is this right for 5' forward (strand replaced to reverse...)? - yes seems Ok.
            if (0 != strand)
                s->pos += target_red;
            s->target_len -= target_red;

            if (repair5p)
                s->fivep_offset = target_red;

            // do one large gap label:oneGapZZ
            // uint32_t ampl_start, ampl_end;
            int min_anchor = min_anchor_large_indel_rescue;
            int half_anchor = min_anchor / 2;
            if (refseq->bed_exist && use_bed_in_end_repair && total_scl >= half_anchor && s->ampl_start != 0)
                // tmap_map_get_amplicon (refseq, s->seqid, start_o, end_o, o_strand, &ampl_start, &ampl_end, NULL)) // DK: if bed exists and coords use in end repair enabled, the amplicon coords are cached within *s (tmap_map_sam_t)
            {
                //total_scl is desired to be larger than 5
                //fprintf(stderr, "%d %d %d qlen=%d\n", ampl_start, ampl_end, total_scl, qlen);
                int max_q = 2;
                int original_total_scl = total_scl;
                int s_pos_adj = 0;
                if (total_scl < min_anchor)
                {
                    total_scl = min_anchor;
                    s_pos_adj = total_scl - original_total_scl;  /* Trimming bases from alignment to get 6 in case of repeat ZZ */
                }
                int need_do = 1;
                int half_buffer = /*6*/ amplicon_overrun; // outside freedom.
                if (half_buffer == 0) { // user set zero to allow no limit on this, but of course we do not (cannot) find indel outside amplicon
                    half_buffer = total_scl; // at most all bases are matched to outside
                }
                int buffer = half_buffer + 3;
                uint32_t start, end;
                uint32_t pad_s, pad_e, pad;
		uint32_t matched_bases;
                pad_s = pad_e = pad = 0;	
                if (strand == 0)
                {
                    start = s->pos + s->target_len + 1 - s_pos_adj;
                    end = s->ampl_end + half_buffer;
		    matched_bases = start-s->pos-1;
                    pad_s = 100;
                    if (pad_s > qlen - total_scl)
                        pad_s = qlen - total_scl;
                    if (end > refseq->annos [s->seqid].len)
                        end = refseq->annos [s->seqid].len;
                    query += qlen-total_scl;
                    pad = pad_s;
                }
                else
                {
                    if (s->ampl_start <= half_buffer)
                        start = 1;
                    else
                        start = s->ampl_start - half_buffer;
                    end = s->pos + s_pos_adj;
		    matched_bases = s->pos+s->target_len-end;
                    pad_e = 100;
                    if (pad_e > qlen - total_scl)
                        pad_e = qlen - total_scl;
                    if (pad_e + end > refseq->annos [s->seqid].len)
                        pad_e = refseq->annos [s->seqid].len - end;
                    pad = pad_e;
                }
		// effectively matched_bases = s->target_len-s_pos_adj; The target_len is the reference length after softclipping
                tlen = end - start + 1;
                //fprintf(stderr, "%d %d %d %d %d\n", start, end, tlen, min_anchor, pad);
                //if (tlen >= min_anchor)
		if (tlen >= 1) 
                {
                    if (target_mem < tlen + pad)
                    { // more memory?
                        target_mem = tlen + pad;
                        tmap_roundup32 (target_mem);
                        target = tmap_realloc (target, sizeof (uint8_t) * target_mem, "target");
                    }
                    if (NULL == tmap_refseq_subseq2 (refseq, s->seqid + 1, start - pad_s, end + pad_e, target, 1, NULL))
                        tmap_bug ();
                }
                else
                    need_do = 0;
                int softclip, target_adj, n_mismatch, indel_size, is_ins, extra_match = 0;
                if (need_do)
                {
                    if (tmap_map_util_one_gap (query,          // query
                                               total_scl,      // q_len
                                               target + pad_s, // target
                                               tlen,           // t_len
                                               min_anchor_large_indel_rescue /*anchor size*/, // min_size - a
                                               max_q,          // max softclip in query
                                               buffer /*int max_t*/, // max shift on target before M segment
                                               3 /*maxNM*/,    // max allowed mismatches in the resulting structure
                                               max_one_large_indel_rescue, /*int max_indel*/ // max ins / del size
                                               strand,        //  strand (0 forward 1 reverse)
                                               &n_mismatch,   // number of mismatches in resulting structure
                                               &indel_size,   // size of the indel in resulting structure
                                               &is_ins,       // 1==I, 0==D
                                               &softclip,     // size of softclip
                                               pad,           // size of pad on either side of target // DK: what if assymetric, like at the end of chromosome?
					       matched_bases, // tell the rountine how many bases the rest of cigar matched to the ref
                                               &extra_match)) //
                    {
                        free (target);
                        //adjust cigar
                        total_scl = original_total_scl; // adjust back
                        int match_size;
                        int target_adj; // how many additional target bases aligned.
                        if (is_ins)
                            match_size = total_scl - softclip - indel_size;
                        else
                            match_size = total_scl - softclip;
                        target_adj = match_size;
                        if (!is_ins)
                            target_adj += indel_size;
                        int s_adj = (match_size - n_mismatch) * score_match - n_mismatch * pen_mm - pen_gapo - pen_gape;
                        s->score += s_adj;
                        // adjust match size
                        int inc_cigar = 1;
                        if (softclip > 0)
                            inc_cigar++;
                        int dec_cigar =  trim_read_bases (s, extra_match, strand, &is_ins, &indel_size);
                        if (inc_cigar > dec_cigar)
                            s->cigar = tmap_realloc (s->cigar, sizeof (uint32_t) * (inc_cigar - dec_cigar + s->n_cigar), "s->cigar");
                        match_size += extra_match;
                        if (strand == 0)
                        {
                            if (dec_cigar > 0)
                            {
                                s->n_cigar -= dec_cigar;
                            }
                            i = s->n_cigar - 1;
                            if (indel_size > 0)
                                TMAP_SW_CIGAR_STORE (s->cigar [i++], is_ins ? BAM_CINS : BAM_CDEL, indel_size);
                            TMAP_SW_CIGAR_STORE (s->cigar [i++], BAM_CMATCH, match_size);
                            if (softclip > 0)
                            {
                                TMAP_SW_CIGAR_STORE (s->cigar [i++],  BAM_CSOFT_CLIP, softclip);
                            }
                            s->n_cigar = i;
                        }
                        else
                        {
                            int move = inc_cigar - dec_cigar;
                            if (indel_size == 0)
                                move--;
                            if (move > 0)
                            {
                                for (i = s->n_cigar-1; i >= dec_cigar; i--)
                                    s->cigar [i + move] = s->cigar [i];
                            }
                            else if (move < 0)
                            {
                                for (i = dec_cigar; i < s->n_cigar; i++)
                                    s->cigar [i + move] = s->cigar [i];
                            }
                            s->n_cigar += move;
                            i = 0;
                            if (softclip > 0)
                                TMAP_SW_CIGAR_STORE (s->cigar [i++],  BAM_CSOFT_CLIP, softclip);
                            TMAP_SW_CIGAR_STORE (s->cigar [i++], BAM_CMATCH, match_size);
                            if (indel_size > 0)
                                TMAP_SW_CIGAR_STORE (s->cigar [i++], is_ins? BAM_CINS: BAM_CDEL , indel_size);
                            //fprintf(stderr, "match_size=%d indel_size=%d is_ins=%d extra_match=%d\n", match_size, indel_size, is_ins, extra_match);
                            s->pos -= target_adj;
                        }
                        s->target_len += target_adj;
                        if (repair5p)
                            s->fivep_offset = softclip; // assume here are all mismatches ZZ.

                        // DK: quick fix. merge adjacent cigar operations that have the same type
                        tmap_map_util_merge_adjacent_cigar_operations(s);

                        // record stats
                        stat->num_end_repair_extended [sori] ++;
                        stat->bases_end_repair_extended [sori] += total_scl - softclip;
                        stat->total_end_repair_indel [sori] += indel_size;

                        repaired = 2;
                    }
                    else
                        free (target);
                }       
            }
        }
        return repaired;
    }
    // do not perform if so
    if (repair5p)
        return repaired;
    if(1 == softclip_start)
        return repaired;

    // check the first cigar op
    cigar_i = (0 == strand) ? 0 : (s->n_cigar - 1);
    op = TMAP_SW_CIGAR_OP(s->cigar[cigar_i]);
    op_len = TMAP_SW_CIGAR_LENGTH(s->cigar[cigar_i]);
    if(op != BAM_CMATCH)
        return repaired; // ignore, since we care only about turning mismatches into indels

    // get the amount of bases to adjust
    cur_len = (max_match_len < op_len) ? max_match_len : op_len;
    // NB: must include full HPs
    while(cur_len < qlen && query[cur_len-1] == query[cur_len])
    { // TODO: what about reference HPs?
        cur_len++;
    }
    if(cur_len < qlen)
    {
        cur_len++; // NB: include one more base to get left-justification correct
    }

    // compute the old score of this sub-alignment
    old_score = 0;
    // NB: assumes the op is a match/mismatch
    found = 0; // did we find mismatches?
    if(0 == strand)
    { // forward
        for(i=0;i<cur_len;i++)
        { // go through the match/mismatches
            if(query[i] == target_prev[i])
                old_score += score_match;
            else
                old_score -= pen_mm, found = 1;
        }
    }
    else
    { // reverse
        for(i=0;i<cur_len;i++)
        { // go through the match/mismatches
            if(query[qlen-i-1] == target_prev[tlen-i-1])
                old_score += score_match;
            else
                old_score -= pen_mm, found = 1;
        }
    }
    if(0 == found)
        return repaired; // no mismatches, so what are you worrying about?

    // see how many bases we should include from the reference
    target_adj = 0;
    cur_cur_len = cur_len;
    cur_cigar_i = (0 == strand) ? 0 : (s->n_cigar - 1);
    cur_op = TMAP_SW_CIGAR_OP(s->cigar[cur_cigar_i]);
    cur_op_len = TMAP_SW_CIGAR_LENGTH(s->cigar[cur_cigar_i]);
    while(0 < cur_cur_len)
    { // while read bases
        if(0 == cur_op_len)
        { // update cigar
            if(0 == strand)
            {
                cur_cigar_i++;
            }
            else
            {
                cur_cigar_i--;
            }
            cur_op = TMAP_SW_CIGAR_OP(s->cigar[cur_cigar_i]);
            cur_op_len = TMAP_SW_CIGAR_LENGTH(s->cigar[cur_cigar_i]);
        }

        switch(cur_op)
        {
            case BAM_CMATCH:
                cur_cur_len--;
                target_adj++;
                break;
            case BAM_CINS:
                cur_cur_len--;
                break;
            case BAM_CDEL:
                target_adj++;
                break;
            default:
                tmap_bug();
                break;
        }
        cur_op_len--;
    }

    // get more target upstream
    if(0 == strand)
    { // forward
        start_pos = s->pos + 1; // one-based
        if(start_pos < num_prev_bases)
            start_pos = 1;
        else
            start_pos -= num_prev_bases;
        end_pos = s->pos + target_adj; // one-based
    }
    else
    {
        start_pos = s->pos + s->target_len; // 1-based
        if(start_pos <= target_adj)
            start_pos = 1;
        else
            start_pos -= target_adj - 1;
        end_pos = s->pos + s->target_len + num_prev_bases;
        if(refseq->annos[s->seqid].len < end_pos)
            end_pos = refseq->annos[s->seqid].len; // bound
    }
    tlen = end_pos - start_pos + 1;
    if(target_mem < tlen)
    { // more memory?
        target_mem = tlen;
        tmap_roundup32(target_mem);
        target = tmap_realloc(target, sizeof(uint8_t)*target_mem, "target");
    }
    // NB: IUPAC codes are turned into mismatches
    if(NULL == tmap_refseq_subseq2(refseq, s->seqid+1, start_pos, end_pos, target, 0, &conv))
    {
        tmap_bug();
    }

    // set the scoring parameters
    if(0 < conv)
    {
        par_iupac.matrix = matrix_iupac;
        for(i=0;i<80;i++)
        {
            if(0 < matrix_iupac_mask[i])
                (par_iupac).matrix[i] = (opt)->score_match;
            else
                (par_iupac).matrix[i] = -cur_len * (opt)->pen_mm;
        }
        (par_iupac).row = 16;
        (par_iupac).band_width = (opt)->bw;

        (par_iupac).gap_open = 0;
        (par_iupac).gap_ext = (opt)->pen_mm;
        (par_iupac).gap_end = (opt)->pen_mm;
    }
    else
    {
        par.matrix = matrix;
        __map_util_gen_ap(par, opt);
        for(i=0;i<25;i++)
        {
            (par).matrix[i] = -cur_len * pen_mm; // TODO: is this enough?
        }
        for(i=0;i<4;i++)
        {
            (par).matrix[i*5+i] = (opt)->score_match;
        }
        (par).row = 5;
        (par).band_width = (opt)->bw;
        (par).gap_open = 0;
        (par).gap_ext = (opt)->pen_mm;
        (par).gap_end = (opt)->pen_mm;
    }

    // adjust query for the reverse strand
    if(1 == strand)
    { // reverse
        query = query + qlen - cur_len; // NB: do not use this otherwise
    }

    /*
    for(i=0;i<cur_len;i++) {
        fputc("ACGTN"[query[i]], stderr);
    }
    fputc('\n', stderr);
    for(i=0;i<tlen;i++) {
        fputc("ACGTN"[target[i]], stderr);
    }
    fputc('\n', stderr);
    */

    // map the target against the the query
    // path memory
    if (*path_buf_sz <= tlen + cur_len)
    {   // lengthen the path
        *path_buf_sz = tlen + cur_len;
        tmap_roundup32 (*path_buf_sz);
        *path_buf = tmap_realloc (*path_buf, sizeof (tmap_sw_path_t) * *path_buf_sz, "path_buf");
    }

    if(0 == strand)
    {
        new_score = tmap_sw_clipping_core2(target, tlen, query, cur_len,
                                            (0 < conv) ? &par_iupac : &par,
                                            1, 0, // do not skip the end of the target
                                            0, 0, // no soft-clipping
                                            *path_buf, &path_len, 0);
    }
    else
    {
        new_score = tmap_sw_clipping_core2(target, tlen, query, cur_len,
                                            (0 < conv) ? &par_iupac : &par,
                                            0, 1, // do not skip the start of the target
                                            0, 0, // no soft-clipping
                                            *path_buf, &path_len, 0);
    }
    //fprintf(stderr, "new_score=%d old_score=%d\n", new_score, old_score);

    found = 0; // reset
    if(old_score < new_score || (2 == end_repair && old_score == new_score))
    { // update
        // get the cigar
        cigar = tmap_sw_path2cigar(*path_buf, path_len, &n_cigar);
        if(0 == n_cigar)
        {
            tmap_bug();
        }
        // TODO: what if it starts with an indel

        /*
        if(0 == strand) { // reverse the cigar
            for(i=0;i<n_cigar>>1;i++) {
                int32_t j = cigar[i];
                cigar[i] = cigar[n_cigar-i-1];
                cigar[n_cigar-i-1] = j;
            }
        }
        */

        // check if the cigars match
        if(1 == n_cigar && TMAP_SW_CIGAR_OP(cigar[0]) == BAM_CMATCH)
            found = 0;
        else
            found = 1; // do not match, hmmn
    }

    // should we update?
    if(1 == found)
    {
        /*
        fprintf(stderr, "strand=%d\n", strand);
        fprintf(stderr, "cur_len=%d\n", cur_len);
        fprintf(stderr, "OLD:\n");
        for(i=0;i<s->n_cigar;i++) { // old
            fprintf(stderr, "i=%d %d%c\n", i,
                    bam_cigar_oplen(s->cigar[i]),
                    bam_cigar_opchr(s->cigar[i]));
        }
        fprintf(stderr, "ADDITION:\n");
        for(i=0;i<n_cigar;i++) { // addition
            fprintf(stderr, "i=%d %d%c\n", i,
                    bam_cigar_oplen(cigar[i]),
                    bam_cigar_opchr(cigar[i]));
        }
        */
        // update the previous cigar operators
        repaired = 2;
        int32_t sum = 0, n = 0;
        int32_t num_ref_added = 0, num_ref_removed = 0;
        i = (0 == strand) ? 0 : s->n_cigar-1;
        while((0 == strand && i < s->n_cigar) || (1 == strand && 0 <= i))
        { // gets the number to delete, and modifies the last one if necessary in-place
            // query
            switch(TMAP_SW_CIGAR_OP(s->cigar[i]))
            {
                case BAM_CMATCH:
                case BAM_CINS:
                    sum += bam_cigar_oplen(s->cigar[i]);
                default:
                    break;
            }
            // target
            switch(TMAP_SW_CIGAR_OP(s->cigar[i]))
            {
                case BAM_CMATCH:
                case BAM_CDEL:
                    if(cur_len <= sum) num_ref_removed += (cur_len - (sum - bam_cigar_oplen(s->cigar[i])));
                    else num_ref_removed += bam_cigar_oplen(s->cigar[i]);
                default:
                    break;
            }

            //fprintf(stderr, "i=%d cur_len=%d sum=%d\n", i, cur_len, sum);
            if(cur_len <= sum)
            {
                if(cur_len < sum)
                { // adjust current cigar
                    TMAP_SW_CIGAR_STORE(s->cigar[i], TMAP_SW_CIGAR_OP(s->cigar[i]), sum - cur_len); // store the difference
                    i = (0 == strand) ? (i-1) : (i+1);
                }
                break;
            }
            i = (0 == strand) ? (i+1) : (i-1);
        }
        n = (0 == strand) ? (i+1) : (s->n_cigar - i); // the number of cigar operations to delete
        //fprintf(stderr, "to delete: %d\n", n);
        if(0 < n)
        { // delete the cigars
            if(0 == strand)
            { // shift down, delete the first cigar operator
                for(i=0;i<s->n_cigar-n;i++)
                {
                    s->cigar[i] = s->cigar[i+n];
                }
            }
            s->n_cigar -= n;
            s->cigar = tmap_realloc(s->cigar, sizeof(uint32_t)*s->n_cigar, "s->cigar"); // reduce the size
        }
        /*
        fprintf(stderr, "CUT DOWN:\n");
        for(i=0;i<s->n_cigar;i++) { // old
            fprintf(stderr, "i=%d %d%c\n", i,
                    bam_cigar_oplen(s->cigar[i]),
                    bam_cigar_opchr(s->cigar[i]));
        }
        */

        // get more cigar
        s->cigar = tmap_realloc(s->cigar, sizeof(uint32_t)*(n_cigar+s->n_cigar), "s->cigar");
        if(0 == strand)
        { // forward
            // shift up
            for(i=s->n_cigar-1;0<=i;i--)
            {
                s->cigar[i+n_cigar] = s->cigar[i];
            }
        } // otherwise append to the end

        // add new cigar and update the position
        for(i=0;i<n_cigar;i++)
        {
            if(0 == strand)
            { // forward
                s->cigar[i] = cigar[i];
            }
            else
            { // reverse
                s->cigar[i + s->n_cigar] = cigar[i];
            }
            switch(TMAP_SW_CIGAR_OP(cigar[i]))
            {
                case BAM_CMATCH:
                case BAM_CDEL:
                    num_ref_added += TMAP_SW_CIGAR_LENGTH(cigar[i]);
                default:
                    break;
            }
            /*
            switch(TMAP_SW_CIGAR_OP(cigar[i])) {
            case BAM_CDEL:
                s->target_len -= TMAP_SW_CIGAR_LENGTH(cigar[i]);
                if(0 == strand) s->pos -= TMAP_SW_CIGAR_LENGTH(cigar[i]);
                break;
            case BAM_CINS:
                s->target_len += TMAP_SW_CIGAR_LENGTH(cigar[i]);
                if(0 == strand) s->pos += TMAP_SW_CIGAR_LENGTH(cigar[i]);
                break;
            default:
                break;
            }
            */
        }
        s->n_cigar += n_cigar;
        s->target_len += (num_ref_added - num_ref_removed);
        if(0 == strand)
            s->pos += (num_ref_removed - num_ref_added);

        // merge adjacent cigar operations that have the same value
        tmap_map_util_merge_adjacent_cigar_operations(s);
        /*
        for(i=0;i<s->n_cigar;i++) {
            fprintf(stderr, "i=%d %d%c\n", i,
                    bam_cigar_oplen(s->cigar[i]),
                    bam_cigar_opchr(s->cigar[i]));
        }
        */

        // update the score
        // TODO: is this correct?
        s->score += new_score;
        s->score -= old_score;

        /*
        fprintf(stderr, "NEW:\n");
        for(i=0;i<s->n_cigar;i++) { // old
            fprintf(stderr, "i=%d %d%c\n", i,
                    bam_cigar_oplen(s->cigar[i]),
                    bam_cigar_opchr(s->cigar[i]));
        }
        */
        // Check that the cigar is valid
        for(i=cur_len=0;i<s->n_cigar;i++)
        {
            switch(TMAP_SW_CIGAR_OP(s->cigar[i]))
            { // NB: do not include soft-clip bases
                case BAM_CMATCH:
                case BAM_CINS:
                    cur_len += TMAP_SW_CIGAR_LENGTH(s->cigar[i]);
                default:
                    break;
            }
        }
        //fprintf(stderr, "cur_len=%d qlen=%d\n", cur_len, qlen);
        if(cur_len != qlen)
            tmap_bug();
    }

    // free
    free(cigar);
    free(target);
    return repaired;
}

typedef struct __REPAiR_params
{
    // general alignment scoring
    tmap_sw_param_t* sw_param;
    // selection
    int32_t repair_min_freq; // presently not overridable (does not make sense to override as relates to BED stored value, not to anything that comes from alignment)
    int32_t repair_min_count; // presently not overridable
    int32_t repair_min_adapter;
    int32_t repair_max_overhang;
    int32_t max_adapter_bases_for_soft_clipping;
    double  repair_identity_drop_limit;
    int32_t repair_max_primer_zone_dist;
    int32_t repair_clip_ext;
}
REPAiR_params;

// fills in parameters used in REPAiR considering global/stage opts and local overrides
// returns 0 if REPAiR is not enabled
static uint32_t evalualte_REPAiR_params (tmap_map_opt_t* stage_opt, tmap_sw_param_t* stage_swpar, tmap_map_locopt_t* loc_opt, int32_t stage_ord, int32_t use_le, REPAiR_params* dest)
{
    // first check if repair is needed at all (skip the rest of parameters preparation if not)
    int32_t repair = stage_opt->use_bed_read_ends_stat;
    if (loc_opt)
    {
        if (loc_opt->repair.over)
            repair = loc_opt->repair.value;
        if (use_le)
        {
            if (loc_opt->repair_le.over)
                repair = loc_opt->repair_le.value;
        }
        else
        {
            if (loc_opt->repair_he.over)
                repair = loc_opt->repair_he.value;
        }
    }
    if (!repair)
        return repair;

    // we are here if repair is enabled for this 
    // general alignment scoring
    dest->sw_param = stage_swpar;
    // copy global values from stage opt
    dest->repair_min_freq = stage_opt->repair_min_freq;
    dest->repair_min_count = stage_opt->repair_min_count;
    dest->repair_min_adapter = stage_opt->repair_min_adapter;
    dest->repair_max_overhang = stage_opt->repair_max_overhang;
    dest->max_adapter_bases_for_soft_clipping = stage_opt->max_adapter_bases_for_soft_clipping;
    dest->repair_identity_drop_limit = stage_opt->repair_identity_drop_limit;
    dest->repair_max_primer_zone_dist = stage_opt->repair_max_primer_zone_dist;
    dest->repair_clip_ext = stage_opt->repair_clip_ext;
    if (loc_opt)
    {
        if (loc_opt->repair.over)
            repair = loc_opt->repair.value;
        dest->sw_param = tmap_map_locopt_get_stage_sw_params (loc_opt, stage_ord);
        if (!dest->sw_param)
            dest->sw_param = stage_swpar;
        // if (local_sw_param !!!

        if (use_le) // lower-coordinate end of the amplicon
        {
            if (loc_opt->repair_le.over)
                repair = loc_opt->repair_le.value;
            else if (loc_opt->repair.over)
                repair = loc_opt->repair.value;
            if (repair)
            {
                if (loc_opt->repair_min_adapter_le.over)
                    dest->repair_min_adapter = loc_opt->repair_min_adapter_le.value;
                else if (loc_opt->repair_min_adapter.over)
                    dest->repair_min_adapter = loc_opt->repair_min_adapter.value;

                if (loc_opt->repair_max_overhang_le.over)
                    dest->repair_max_overhang = loc_opt->repair_max_overhang_le.value;
                else if (loc_opt->repair_max_overhang.over)
                    dest->repair_max_overhang = loc_opt->repair_max_overhang.value;

                if (loc_opt->max_adapter_bases_for_soft_clipping_le.over)
                    dest->max_adapter_bases_for_soft_clipping = loc_opt->max_adapter_bases_for_soft_clipping_le.value;
                else if (loc_opt->max_adapter_bases_for_soft_clipping.over)
                    dest->max_adapter_bases_for_soft_clipping = loc_opt->max_adapter_bases_for_soft_clipping.value;

                if (loc_opt->repair_identity_drop_limit_le.over)
                    dest->repair_identity_drop_limit = loc_opt->repair_identity_drop_limit_le.value;
                else if (loc_opt->repair_identity_drop_limit.over)
                    dest->repair_identity_drop_limit = loc_opt->repair_identity_drop_limit.value;

                if (loc_opt->repair_max_primer_zone_dist_le.over)
                    dest->repair_max_primer_zone_dist = loc_opt->repair_max_primer_zone_dist_le.value;
                else if (loc_opt->repair_max_primer_zone_dist.over)
                    dest->repair_max_primer_zone_dist = loc_opt->repair_max_primer_zone_dist.value;

                if (loc_opt->repair_clip_ext_le.over)
                    dest->repair_clip_ext = loc_opt->repair_clip_ext_le.value;
                else if (loc_opt->repair_clip_ext.over)
                    dest->repair_clip_ext = loc_opt->repair_clip_ext.value;

            }
        }
        else // higher-coordinate end of the amplicon
        {
            if (loc_opt->repair_he.over)
                repair = loc_opt->repair_he.value;
            else if (loc_opt->repair.over)
                repair = loc_opt->repair.value;

            if (repair)
            {
                if (loc_opt->repair_min_adapter_he.over)
                    dest->repair_min_adapter = loc_opt->repair_min_adapter_he.value;
                else if (loc_opt->repair_min_adapter.over)
                    dest->repair_min_adapter = loc_opt->repair_min_adapter.value;

                if (loc_opt->repair_max_overhang_he.over)
                    dest->repair_max_overhang = loc_opt->repair_max_overhang_he.value;
                else if (loc_opt->repair_max_overhang.over)
                    dest->repair_max_overhang = loc_opt->repair_max_overhang.value;

                if (loc_opt->max_adapter_bases_for_soft_clipping_he.over)
                    dest->max_adapter_bases_for_soft_clipping = loc_opt->max_adapter_bases_for_soft_clipping_he.value;
                else if (loc_opt->max_adapter_bases_for_soft_clipping.over)
                    dest->max_adapter_bases_for_soft_clipping = loc_opt->max_adapter_bases_for_soft_clipping.value;

                if (loc_opt->repair_identity_drop_limit_he.over)
                    dest->repair_identity_drop_limit = loc_opt->repair_identity_drop_limit_he.value;
                else if (loc_opt->repair_identity_drop_limit.over)
                    dest->repair_identity_drop_limit = loc_opt->repair_identity_drop_limit.value;

                if (loc_opt->repair_max_primer_zone_dist_he.over)
                    dest->repair_max_primer_zone_dist = loc_opt->repair_max_primer_zone_dist_he.value;
                else if (loc_opt->repair_max_primer_zone_dist.over)
                    dest->repair_max_primer_zone_dist = loc_opt->repair_max_primer_zone_dist.value;

                if (loc_opt->repair_clip_ext_he.over)
                    dest->repair_clip_ext = loc_opt->repair_clip_ext_he.value;
                else if (loc_opt->repair_clip_ext.over)
                    dest->repair_clip_ext = loc_opt->repair_clip_ext.value;
            }
        }
    }
    return repair;
}

#if 0
/*
 * computes position of worst alignment score starting from the end of the alignment
 * returns position on query, 
 * fills in:
 *   cigar_op_index
 *   cigar_op_offset
 *   clip_diff
 *   clip_len
 *   target_red
 * returns new SW score
 */
static 
int32_t 
worst_score_pos_in_alignment
(
    const uint32_t* cigar,
    uint32_t cigar_len,
    uint32_t score,
    uint8_t strand,
    const uint8_t* query,
    uint32_t query_len,
    const uint8_t* ref,
    uint32_t ref_len,
    tmap_sw_param_t* sw_param,
    uint8_t softclip_as_insertion, // true for REPAiR, false for (possible future implementation) regular end-repair (presently regular end repair does not properly treat the softclips, works only if they are passed as INSs.)
    int32_t* cigar_op_index,  // index of cigar operation where worst alignment score position is located
    int32_t* cigar_op_offset, // offset into cigar batch to the worst alignment score position
    uint32_t* clip_diff,      // number of edit operations in the zone of alignment before the position of worst score
    uint32_t* clip_len,       // offset on query to the position of worst score
    uint32_t* target_red,      // offset on target to the position of worst score (target reduction)
    int32_t* clip_score
)
{
    int32_t i, cigar_i;
    int32_t op, op_len, cur_len;
    int32_t cur_op, cur_op_len, cur_cigar_i, cur_cur_len, target_adj;
    int32_t softclip_start, softclip_end;

    int32_t start_pos, end_pos;


    int tb, qb, inc;  // target position, query position, increment (+1 | -1) for walking along target / query
    int i_c; // index of the cigar operation
    // int i;   // index of base within cigar operation
    *cigar_op_index = -1;  // cigar operation index where worst alignment score is achieved
    *cigar_op_offset = 0;  // index of base within the cigar operation where worst alignment score is achieved
    *clip_diff = 0; // number of edit operations (mismatches, indels) in the part of alignment before worst score position
    *clip_len = 0;  // offset on query to the position of worst score
    *target_red = 0; // offset on target to the position of worst score (target reduction)
    *clip_score = 0;
    int found = 0; // number of non-matches (mismatches, inserts, deletes) found in alignment
    int worst = 0; // worst alignment score encountered along the alignment
    // int ind = 0; // index of base within the cigar operation where worst alignment score is achieved
    int nqb = 0; // number of query bases seen
    int ntb = 0; // number of target bases seen

    int cur_clip_score = 0;

    // Getting match and mismatch scores:
    // proper way (inplemented):
    //     compare the query to ref base using the matrix. For that, the inconsistency related to assymery of the IUPAC matrix should be corrected
    //     (the sequences whose bases are used as horisontal matrix coordinate allow IUPAC bases, while on the vertical one only standard + 'N' are allowed.)
    //     It is the reference that is considered having nono-standard bases in TMAP regular alignment. Keeping this assumption here.
    //     Thus, the formula to retrieve score is (read_base_num * row_len + ref_base_num)
    // Alternates are: 
    // 1) pass match/mismatch as separate fields in the sw_params (modify tmap_sw_param_t): lot of hassle, little sense
    // 2) use the following is hack (seemingly safe one)
    // to get mathch and mismatch score, respectfully, use the first and second elements in the SW scoring matrix.
    //    possible drawback: if 'iupac_matrix' used for matrix population is very weird, say, declares A/C as match, this will fail
    //    int32_t match = sw_param->matrix [0], mism = -sw_param->matrix [1];

    if (0 != strand) // strand: 0 == forward, 1 == reverse
    {   // reverse
        tb = qb = 0;
        inc = 1;
    }
    else
    {   // forward
        tb = ref_len - 1;
        qb = query_len - 1;
        inc = -1;
    }
    for (i_c = 0; i_c < cigar_len; i_c++)
    {
        cigar_i = strand ? i_c : cigar_len - 1 - i_c;
        op = TMAP_SW_CIGAR_OP (cigar [cigar_i]);
        op_len = TMAP_SW_CIGAR_LENGTH (cigar [cigar_i]);

        if (op == BAM_CMATCH)
        {
            for (i = 0; i < op_len; ++i, qb += inc, tb += inc, nqb++, ntb++)
            {
                if (qb == -1 || tb == -1)
                    tmap_bug ();

                cur_clip_score += sw_param->matrix [query [qb] * sw_param->row + ref [tb]];
                if (query [qb] != ref [tb])
                    found++;
                if (cur_clip_score <= worst) // use farthest of the equally bad positions
                {
                    worst = cur_clip_score;
                    *cigar_op_offset = i + 1;
                    *clip_diff = found;
                    *target_red = ntb + 1;
                    *cigar_op_index = i_c;
                    *clip_len = nqb + 1;
                }
            }
        }
        else
        {
            if (op == BAM_CDEL)
            {
                ntb += op_len;
                tb += inc * op_len;
            }
            else if (op == BAM_CINS || (softclip_as_insertion && op == BAM_CSOFT_CLIP)) // count soft clip as insertion. 
            {
                nqb += op_len;
                qb  += inc * op_len;
            }
            else
                break;
            cur_clip_score -= sw_param->gap_open + sw_param->gap_ext * op_len;
            found += op_len;
            if (cur_clip_score < worst)
            {
                worst = cur_clip_score;
                *cigar_op_offset = 0;
                *clip_diff = found;
                *target_red = ntb;
                *cigar_op_index = i_c;
                *clip_len = nqb;
            }
        }
    }
    *clip_score = worst;
    return score - worst;
}
#endif
// computes SW score on the alignment segment starting at a given position of given length along the query sequence
// the parameters are redundant, as this is a helper function that is called when any needed values are already pre-calculated
// returns the SW score computed according to a passed in SW parameters
uint32_t
tmap_util_cigar_segment_score
(
    const uint32_t* cigar,
    uint32_t cigar_len,
    uint8_t strand,   //? DO I NEED TO INVERSE/COMPLEMENT THE QUERY FOR strand == 1?
    const uint8_t* query,
    uint32_t query_len,
    const uint8_t* ref,
    uint32_t ref_len,
    int32_t query_off, // -1 to compute from the hints
    int32_t query_segment_len, // set to -1 to calc score to the end
    int32_t ref_off,  // set to -1 to compute from the other parameters
    int32_t ref_segment_len, // set to -1 to calc score to the end
    uint32_t from_beg, // if set to fasle, walks alignment in reverse direction. In this case, the query_off, ref_off and cigar index/offset are for the last position in the alignment segment
    int32_t cigar_op_index, // set to -1 to ignore; trusted if provided
    int32_t cigar_op_offset, // set to -1 to ignore; trusted if provided
    uint32_t clip_as_ins, // treat softclips as inserts
    tmap_sw_param_t* sw_param, // parameters scoring scheme
    uint32_t* edit_ops, // if not NULL, receiver for the number of edit operations. The number of mismatches = edit_ops - indels_length
    uint32_t* indels_count, // if not NULL, receiver for the count of indels (gap openings
    uint32_t* indels_length // if not NULL, receiver for the total length of indels 
)
{
    int32_t q_pos, r_pos, cigar_op_cur, cigar_seg_cur;
    int32_t q_sent, r_sent, cigar_op_sent, cigar_seg_sent;
    int32_t q_low = 0, q_high = query_len - 1, r_low = 0, r_high = ref_len - 1;
    // uint32_t hints_complete = 1;
    int32_t incr = from_beg?1:-1;
    uint32_t seen_query_bases = 0, seen_ref_bases = 0;
    // if offset hints not fully specified (at least one of the hints is not defined), then re-calculate them
    if (cigar_op_index == -1 || cigar_op_offset == -1 || ref_off == -1 || query_off == -1)
    {
        q_pos  = from_beg?0:(query_len-1);
        q_sent = from_beg?query_len:-1;
        r_pos  = from_beg?0:(ref_len-1); 
        r_sent = from_beg?ref_len:-1;
        cigar_op_cur = from_beg?0:(cigar_len-1);
        cigar_op_sent = from_beg?cigar_len:-1;
        cigar_seg_cur = from_beg?0:(TMAP_SW_CIGAR_LENGTH (cigar [cigar_op_cur]) - 1);
        // hints_complete = 0;
    }
    else // all hints are defined
    {
        q_pos = query_off;
        q_sent = (query_segment_len == -1) ? (from_beg ? query_len : -1) : (query_off + incr * query_segment_len);
        r_pos = ref_off; 
        r_sent = from_beg ? ref_len : -1;
        cigar_op_cur = cigar_op_index;
        cigar_op_sent = from_beg ? cigar_len : -1;;
        cigar_seg_cur = cigar_op_offset;
        // hints_complete = 1;
    }

    if (edit_ops) *edit_ops = 0;
    if (indels_count) *indels_count = 0;
    if (indels_length) *indels_length = 0;

    uint32_t in_cigar_seg = 0, done = 0;
    int32_t sw_score = 0;
    for (; cigar_op_cur != cigar_op_sent && !done; cigar_op_cur += incr)
    {
        int32_t cigar_opcode = TMAP_SW_CIGAR_OP (cigar [cigar_op_cur]);
        int32_t cigar_oplen = TMAP_SW_CIGAR_LENGTH (cigar [cigar_op_cur]);
        cigar_seg_sent = from_beg?cigar_oplen:-1;
        int32_t cigar_seg_beg = cigar_seg_cur;
        for (; cigar_seg_cur != cigar_seg_sent && !done; cigar_seg_cur += incr)
        {
            if (!in_cigar_seg && cigar_op_cur == cigar_op_index && cigar_seg_cur == cigar_op_offset) 
            {   // never happens if one of cigar_op_index or cigar_op_offset is passed as -1. As it should be.
                in_cigar_seg = 1;
            }

            switch (cigar_opcode)
            {
                case BAM_CMATCH:
                    if (seen_query_bases || seen_ref_bases || q_pos == query_off || r_pos == ref_off || in_cigar_seg)
                        // either segment being scored started earlier (seen_query_bases || seen_ref_bases), or this is the start (q_pos == query_off || r_pos == ref_off || in_cigar_seg)
                        ++seen_query_bases, ++seen_ref_bases;
                    if (seen_query_bases)
                    {
                        sw_score += sw_param->matrix [query [q_pos] * sw_param->row + ref [r_pos]];
                        if (query [q_pos] != ref [r_pos])
                            ++edit_ops;
                    }
                    ++q_pos;
                    ++r_pos;
                    break;
                case BAM_CDEL:
                    if (seen_query_bases || seen_ref_bases || q_pos == query_off || r_pos == ref_off || in_cigar_seg)
                        // either segment being scored started earlier (seen_query_bases || seen_ref_bases), or this is the start (q_pos == query_off || r_pos == ref_off || in_cigar_seg)
                        ++seen_ref_bases;
                    // if we are in segment being scored
                    if (seen_ref_bases)
                    {
                        // if this is the first step in CDEL segment, add GIP (negated)
                        if (cigar_seg_beg == cigar_seg_cur)
                            ++indels_count, sw_score -= sw_param->gap_open;
                        // add GEP (negated)
                        sw_score -= sw_param->gap_ext;
                        ++indels_length;
                    }
                    ++ref_off;
                    break;
                case BAM_CINS:
                    if (seen_query_bases || seen_ref_bases || q_pos == query_off || r_pos == ref_off || in_cigar_seg)
                        // either segment being scored started earlier (seen_query_bases || seen_ref_bases), or this is the start (q_pos == query_off || r_pos == ref_off || in_cigar_seg)
                        ++seen_query_bases;
                    // if we are in segment being scored
                    if (seen_query_bases)
                    {
                        // if this is the first step in CINS segment, add GIP (negated)
                        if (cigar_seg_beg == cigar_seg_cur)
                            ++indels_count , sw_score -= sw_param->gap_open;
                        // add GEP (negated)
                        sw_score -= sw_param->gap_ext;
                        ++indels_length;
                    }
                    ++ ref_off;
                    break;
                case BAM_CSOFT_CLIP:
                    if (seen_query_bases || seen_ref_bases || q_pos == query_off || r_pos == ref_off || in_cigar_seg)
                        // either segment being scored started earlier (seen_query_bases || seen_ref_bases), or this is the start (q_pos == query_off || r_pos == ref_off || in_cigar_seg)
                        ++seen_query_bases;
                    // if we are in segment being scored
                    if (seen_query_bases && clip_as_ins)
                    {
                        // if this is the first step in CSOFT_CLIPL segment, add GIP (negated)
                        if (cigar_seg_beg == cigar_seg_cur)
                            ++indels_count, sw_score -= sw_param->gap_open;
                        // add GEP (negated)
                        sw_score -= sw_param->gap_ext;
                        ++indels_length;
                    }
                    ++ ref_off;
                    break;
                default:
                    ;
            }
            if (seen_query_bases == query_segment_len 
                || seen_ref_bases == ref_segment_len
                || q_pos == q_sent
                || r_pos == r_sent)
                done = 1;
        }
        cigar_seg_cur = 0;
    }

    return sw_score;
}

#if 0
// OLDWAY REPAiR
// actually performs REPAiR (sunf the prepared parametrization and cached reference sequence)
// returns 0 if no repair performed, 1 if original alignment zone clipped, 2 if original alignment zone extended
// #pragma GCC diagnostic ignored "-Wunused_function"
static uint32_t
tmap_map_util_REPAiR
(
    tmap_seq_t *seq, 
    uint8_t *query, 
    int32_t qlen,
    ref_buf_t* target,
    int8_t strand,
    tmap_sw_path_t **path_buf,
    int32_t* path_buf_sz,
    tmap_map_sam_t *s,
    REPAiR_params* par,
    tmap_map_stats_t* stat
)
{
    int32_t repaired = 0; // return value: 0 for no change, 1 for softclip only, 2 for indel salvage.


    // perform REPAiR

    // Jingwei's proposal:
    // If one end does not fit any of the defined anchor sites (including S and L in addition to O), 
    // try placing the end at each of the anchor sites, then extends inwards into the amplicon.
    // primer portion bases have to perfectly match reference, 
    // extend further until a mismatch is found, or extend further by allowing up to 1 mismatch/indel every 7 base (calculated by --end-repair 15) for the anchor bases (except primer portion where no mismatch is allowed).

    // measurable heuristics:
    // extend the soft clip same way as Zheng's end_repair does - to the position of the worst alignment score from the end being repaired
    // for each end position:
    //    perform true global (Needleman-Wunch-like) alignment with negative gap extansion scoring, gap score zero-bound (starts with GIP then reduces by gep every base until reached zero?)

    // try to repair even if softclip is allowed (end-repair tries to modify sofclipped alignment for a better one)


    // we need to remove softclip, as prior alignment post-processing could have introduced it
    // @ remove softclip
        // in Zheng's end_repair, softclip is replaced by M. From SW and Denis's preparation for end_repair, 
        // softclip is replaced by I. Seems like both are legit? 
        // - check the calculation of the worst alignment score point in Zheng's end_repair
        //   - Zheng's method od S->I conversion is never actually invoked, and is unsafe with the assumption of reference (target) buffer. There was a bug (fixed in 2017(?)) manifested when softclipped read was passed to end_repair. 
        //   : need to use INS. The calculation for the worst point will be different for these two methods, but Ins is what is always used => assume it is Ok.
        //   - also need to figure out how to cut the alignment with respect to read end positions that are to be tried.
        //     : no need to actually remove. Just count it az zero cost in calculation of worst score alignment point

    // find the position in the alignment where the SW score reaches minimum. 
    int32_t cigar_op_idx; // cigar operation index where worst alignment score is achieved 
    int32_t cigar_op_off; // index of base within the cigar operation where worst alignment score is achieved
    uint32_t clip_diff;   // number of edit operations (mismatches, indels) in the part of alignment before worst score position
    uint32_t clip_len;    // offset on query to the position of worst score
    uint32_t target_red;  // offset on target to the position of worst score (target reduction)
    int32_t clip_score;  // the score of the clipped zone
    int32_t new_score = worst_score_pos_in_alignment 
        (
            s->cigar,
            s->n_cigar,
            s->score,
            strand,
            query,
            qlen,
            target->data + (s->pos - target->position),
            s->target_len,
            par->sw_param,
            1, // softclip_as_insertion
            &cigar_op_idx,
            &cigar_op_off,
            &clip_diff,
            &clip_len,  // this is offset from the read's 3'.
            &target_red, // this is offset from the target end
            &clip_score
        );

    // add requested number of bases to the clip


    add_to_hist64i (stat->repair_clip_histo, REPAIR_CLIP_HIST_BINNO, repair_clip_hist_lowerb, clip_len);
    if (clip_len  == 0)
        return 0;

    double clip_identity = ((double) (clip_score)) / (clip_len * par->sw_param->matrix [0]);

    // operate on the read/reference zones downstream of the location of worst alignment score. 
    // perform "global" alignment from the worst alignment zone position on 5' to the factual read end paired to each read end position on reference
    // (alternatively when the read end is beyond the amplicon end, we may first enforce the perfect match within the primer - but this could turn out to be just arbitrary restrictive)

    // reserve space for best result
    uint32_t* best_addition_cigar = NULL;
    uint32_t best_addition_cigar_len = 0;
    uint32_t best_addition_score = UINT32_MAX;
    uint32_t best_read_end_pos = UINT32_MAX;

    // read_end current pointer and sentinel
    tmap_map_endpos_t *re_p = s->read_ends.positions;
    tmap_map_endpos_t *re_sent = re_p;
    if (s->strand) 
        re_sent += s->read_ends.starts_count;
    else 
        re_p += s->read_ends.starts_count,
        re_sent += s->read_ends.starts_count + s->read_ends.ends_count;

    // loop over read ends
    for (; re_p != re_sent; ++re_p)
    {
        /// check if selection conditions are satisfied
        if ((re_p->fraction * 100) <= par->repair_min_freq)
            continue;
        if (re_p->count <= par->repair_min_count)
            continue;

        /// prepare coordinates for SW
        int32_t read_beg, read_end, ref_beg, ref_end;
        if (strand == 0)
        {
            read_beg = qlen - clip_len;
            read_end = qlen;
            ref_beg  = s->pos + s->target_len - target_red;
            ref_end  = re_p->coord - 1; // one-based!
        }
        else
        {
            read_beg = 0;
            read_end = clip_len;
            ref_beg  = re_p->coord - 1; // one-based!
            ref_end  = s->pos + target_red;
        }
        if (ref_end <= ref_beg) // we should still treat this as a valid case and try to realign, with longer read/target zone. later.
            continue;
        if (read_end <= read_beg)
            continue;

        assert (target->position <= ref_beg);
        int32_t target_beg = ref_beg - target->position;
        assert (target->position <= ref_end);
        int32_t target_end = ref_end - target->position;
        assert (target_end > target_beg);
        assert (read_end > read_beg);

        // make sure enough memory is allocated for the path
        int32_t plen_needed = (target_end - target_beg) + (read_end - read_beg);
        if (*path_buf_sz <= plen_needed)
        {   // lengthen the path
            *path_buf_sz = plen_needed;
            tmap_roundup32 (*path_buf_sz);
            *path_buf = tmap_realloc (*path_buf, sizeof (tmap_sw_path_t) * *path_buf_sz, "path_buf");
        }
        /// call global alignment routine
        int32_t path_len;
        int32_t add_score = tmap_sw_clipping_core2 (
                                                    target->data + target_beg,
                                                    target_end - target_beg,
                                                    query + read_beg,
                                                    read_end - read_beg,
                                                    par->sw_param,
                                                    0, 0, // do not allow either end of the target to be skipped
                                                    0, 0, // do not allow any softclips
                                                    *path_buf,
                                                    &path_len,
                                                    0
                                                   );
        if (path_len == 0) // alignment failure - REPAiR attempt failed at this read_end. 
            continue;

        if (!best_addition_cigar || add_score > best_addition_score)  // do we need to restrict to positive scores only? or to scores over the score of removed portion? (the latter sounds appropriate)
        {
            // perform checks on suitability of the alignment
            // check if this score makes sense: compare identities
            double addition_identity = ((double) add_score) / (read_end - read_beg);
            if (addition_identity < clip_identity * par->repair_identity_drop_limit)
                continue;

            // build the cigar
            int32_t addition_len;
            uint32_t* addition = tmap_sw_path2cigar (*path_buf, path_len, &addition_len);
            if (addition_len == 0) // alignment failure - REPAiR attempt failed. This happens if aligner sets path_len to 0
                continue;
            assert (addition);

            // check if the primer zone (amplicon end to read end position) aligns well enough
            if (strand == 0)
            {
                if (s->ampl_end < re_p->coord) // both are 1 - based
                {
                    // starting from the end of alignment to the amplicon end going downward by the cigar string,
                    // count the number of edit operations;
                    // if going over the MAX_PRIMER_ZONE_DIST, abandon this read end
                    int32_t op_idx, op_off, op, ol;
                    int32_t tpos = re_p->coord - 1 - target->position, qpos = qlen - 1; 
                    uint32_t num_edits = 0;
                    for (op_idx = addition_len - 1; op_idx > 0 && num_edits <= par->repair_max_primer_zone_dist && target->position + tpos > s->ampl_end - 1; --op_idx)
                    {
                        op = TMAP_SW_CIGAR_OP (addition [op_idx]);
                        ol = TMAP_SW_CIGAR_LENGTH (addition [op_idx]);
                        if (op == BAM_CINS)
                        {
                            num_edits += ol;
                            qpos -= ol;
                        }
                        else 
                        {
                            for (op_off = ol - 1; op_off > 0 && num_edits <= par->repair_max_primer_zone_dist && target->position + tpos > s->ampl_end - 1; --op_off)
                            {
                                switch (op)
                                {
                                    case BAM_CDEL:
                                        ++ num_edits;
                                        -- tpos;
                                        break;
                                    case BAM_CMATCH:
                                        num_edits += (query [qpos] != target->data [tpos]);
                                        -- tpos;
                                        -- qpos;
                                }
                            }
                        }
                    }
                    if (num_edits > par->repair_max_primer_zone_dist)
                    {
                        free (addition);
                        continue;
                    }
                }
            }
            else
            {
                if (re_p->coord < s->ampl_start) // both are 1 - based
                {
                    // starting from the start of the alignment to the amplicon start, going forward by the cigar string,
                    // count the number of edit operations;
                    // if going over the MAX_PRIMER_ZONE_DIST, abandon this read end
                    int32_t op_idx, op_off, op, ol;
                    int32_t tpos = re_p->coord - 1 - target->position, qpos = 0; 
                    uint32_t num_edits = 0;
                    for (op_idx = 0; op_idx != addition_len && num_edits <= par->repair_max_primer_zone_dist && target->position + tpos < s->ampl_start - 1; ++op_idx)
                    {
                        op = TMAP_SW_CIGAR_OP (addition [op_idx]);
                        ol = TMAP_SW_CIGAR_LENGTH (addition [op_idx]);
                        if (op == BAM_CINS)
                        {
                            num_edits += ol;
                            qpos += ol;
                        }
                        else 
                        {
                            for (op_off = 0; (op_off != ol - 1) && num_edits <= par->repair_max_primer_zone_dist && target->position + tpos < s->ampl_start - 1; ++op_off)
                            {
                                switch (op)
                                {
                                    case BAM_CDEL:
                                        ++ num_edits;
                                        ++ tpos;
                                        break;
                                    case BAM_CMATCH:
                                        num_edits += (query [qpos] != target->data [tpos]);
                                        ++ tpos;
                                        ++ qpos;
                                }
                            }
                        }
                    }
                    if (num_edits > par->repair_max_primer_zone_dist)
                    {
                        free (addition);
                        continue;
                    }
                }
            }

            if (best_addition_cigar)
                free (best_addition_cigar);
            best_addition_score = add_score;
            best_addition_cigar = addition;
            best_addition_cigar_len = addition_len;
            best_read_end_pos = re_p->coord;
        }
    }
    if (best_addition_cigar == NULL)
        return 0;

    // merge the addition into original cigar and adjust the score
    if (strand == 0)  // original then addition
    {
        int32_t addition_first_op = TMAP_SW_CIGAR_OP (best_addition_cigar [0]);
        int32_t last_orig_op = TMAP_SW_CIGAR_OP (s->cigar [cigar_op_idx]);
        int32_t last_orig_op_len = TMAP_SW_CIGAR_LENGTH (s->cigar [cigar_op_idx]);

        if (addition_first_op == last_orig_op)
        {
            int32_t merged_len = TMAP_SW_CIGAR_LENGTH (best_addition_cigar [0]) + TMAP_SW_CIGAR_LENGTH (s->cigar [cigar_op_idx]);
            TMAP_SW_CIGAR_STORE (best_addition_cigar [0], addition_first_op, merged_len);
            --cigar_op_idx; // last operation from original cigar now merged into the addition
            if (addition_first_op == BAM_CINS || addition_first_op == BAM_CDEL) // cannot be sofclip by the parameters to sw_clipping_core2
                best_addition_score -= par->sw_param->gap_open; // no opening, indel operation continues
        }
        int32_t new_cigar_len = cigar_op_idx + 1 + best_addition_cigar_len;
        if (new_cigar_len > s->n_cigar)
            s->cigar = tmap_realloc (s->cigar, new_cigar_len * sizeof (*(s->cigar)), "cigar");
        memcpy (s->cigar + cigar_op_idx + 1, best_addition_cigar, best_addition_cigar_len * sizeof (*(s->cigar)));
        s->n_cigar = new_cigar_len;
        s->score = new_score + best_addition_score;
        free (best_addition_cigar);

        // adjust target coords, len and alignment box.
        // starts do not change; target end changes to the best read end position; query end changes to the very end of read
        repaired = (best_read_end_pos - s->pos - s->target_len > 0)?2:1;
        s->target_len = best_read_end_pos - s->pos;
        s->result.target_end = s->target_len; // a bit of a hack, the convention is that target_beg is always zero
        s->result.query_end = qlen;
    }
    else  // addition then original
    {
        // check if last op of addition matches first (remaining) op of original cigar; merge ops if yes
        int32_t orig_shift = best_addition_cigar_len - cigar_op_idx;
        int32_t addition_last_op = TMAP_SW_CIGAR_OP (best_addition_cigar [best_addition_cigar_len - 1]);
        int32_t first_orig_op = TMAP_SW_CIGAR_OP (s->cigar [cigar_op_idx]);
        if (addition_last_op == first_orig_op) // merge into addidion's last op
        {
            int32_t merged_len = TMAP_SW_CIGAR_LENGTH (best_addition_cigar [best_addition_cigar_len - 1]) + TMAP_SW_CIGAR_LENGTH (s->cigar [cigar_op_idx]);
            TMAP_SW_CIGAR_STORE (best_addition_cigar [best_addition_cigar_len - 1], addition_last_op, merged_len);
            --orig_shift;
            if (addition_last_op == BAM_CINS || addition_last_op == BAM_CDEL) // cannot be sofclip by the parameters to sw_clipping_core2
                best_addition_score -= par->sw_param->gap_open; // no opening, indel operation continues
        }
        int32_t new_cigar_len = s->n_cigar + orig_shift;
        if (new_cigar_len > s->n_cigar)
            s->cigar = tmap_realloc (s->cigar, new_cigar_len * sizeof (*(s->cigar)), "cigar");
        if (orig_shift > 0)
            memmove (s->cigar + orig_shift, s->cigar, s->n_cigar * sizeof (*(s->cigar)));
        if (orig_shift < 0)
            memmove (s->cigar, s->cigar - orig_shift, (s->n_cigar + orig_shift) * sizeof (*(s->cigar)));
        memcpy (s->cigar, best_addition_cigar, best_addition_cigar_len * sizeof (*(s->cigar)));
        s->n_cigar = new_cigar_len;
        s->score = new_score + best_addition_score;
        free (best_addition_cigar);

        // adjust target coords, len and alignment box. 
        // ends do not change; target start changes to the best read end position; query start changes zero
        repaired = (s->pos + s->target_len - best_read_end_pos - s->target_len > 0)?2:1;
        s->target_len = s->pos + s->target_len - best_read_end_pos;
        s->pos = best_read_end_pos;
        s->result.query_start = 0;
    }

    return repaired; // for now ignored by the caller's caller
}
#endif

// actually performs REPAiR (sunf the prepared parametrization and cached reference sequence)
// returns 0 if no repair performed, 1 if original alignment zone clipped, 2 if original alignment zone extended

static uint32_t
tmap_map_util_REPAiR_new
(
    tmap_seq_t *seq, 
    uint8_t *query, 
    int32_t qlen,
    ref_buf_t* target,
    int8_t strand,
    tmap_sw_path_t **path_buf,
    int32_t* path_buf_sz,
    tmap_map_sam_t *s,
    REPAiR_params* par,
    tmap_map_stats_t* stat
)
{
    int32_t repaired = 0; // return value: 0 for no change, 1 for softclip only, 2 for indel salvage.

    // check if there is end stats data in needed direction. (This actually already checked by the caller)
    if (s->strand && !s->read_ends.ends_count)
        return 0;
    if (!s->strand && !s->read_ends.starts_count)
        return 0;

    // check if alignment already ends at one of the read ends
    tmap_map_endpos_t *re_p = s->read_ends.positions;
    tmap_map_endpos_t *re_sent = re_p;
    if (s->strand == 0) // forward read mapped to the forward ref strand. The alignment is to "read ends"
    {
        assert (s->n_cigar);
        if (bam_cigar_op (s->cigar [s->n_cigar - 1]) != BAM_CSOFT_CLIP)
        {
            uint32_t end_coord = s->pos + s->target_len;
            re_p += s->read_ends.starts_count,
            re_sent += s->read_ends.starts_count + s->read_ends.ends_count;
            for (; re_p != re_sent; ++re_p)
            {
                // check if aligned (not softclipped) zone ends at this read-end position
                if ( re_p->coord /*+1(read-end is inclusive but end_coord is not) -1(one-based)*/ == end_coord) // assuming end position is inclusive (position where last read base maps) and one-based
                    return 0;                      // TODO: check specs, by examining data or with Jingwei
                                                   // TODO: check if adding some allowance interval here makes sense
            }
        }
    }
    else // reverse read mapped to the forward ref strand: actual 3' is on the left. The alignment is to "read starts"
    {
        assert (s->n_cigar);
        if (bam_cigar_op (s->cigar [0]) != BAM_CSOFT_CLIP)
        {
            uint32_t beg_coord = s->pos;
            re_sent += s->read_ends.starts_count;
            for (; re_p != re_sent; ++re_p)
            {
                // check if aligned (not softclipped) zone ends at this read-end position
                if ( re_p->coord - 1/*one-based*/ == beg_coord) // assuming start position is inclusive (position where last base in the reverse read maps to the reference) and one-based
                    return 0;                  // TODO: see forward chain case
            }
        }
    }
    // we are here means we need to try to realign

    // find the position in the alignment where the SW score reaches minimum. 
    // if no such zone exists, clip the given number of bases

    tmap_map_alignment al;
    init_alignment_x 
    (
        &al,
        s->cigar,
        s->n_cigar,
        query,
        0,
        qlen,
        target->data + (s->pos - target->position),
        0,
        s->result.target_end - s->result.target_start + 1 // target_end is inclusize
    );
    tmap_map_alignment_segment cropped_al;
    tmap_map_alignment_segment clip_al;
    tmap_map_alignment_stats cropped_stats;
    tmap_map_alignment_stats clip_stats;

    // from_beg == strand: strand 0(fwd)=>from_beg == false, strand 1(rev)=>from_beg == true
    uint8_t worst_found = tmap_map_find_worst_score_pos_x 
    (
        &al, 
        par->sw_param,
        strand, 
        &cropped_al,
        &clip_al,
        &cropped_stats,
        &clip_stats,
        NULL,NULL,NULL,NULL
    );

    int32_t clip_len = clip_al.q_end - clip_al.q_start;
    add_to_hist64i (stat->repair_clip_histo, REPAIR_CLIP_HIST_BINNO, repair_clip_hist_lowerb, clip_len);

    if (par->repair_clip_ext && clip_len < par->repair_clip_ext)
    {
        int32_t add_bases = par->repair_clip_ext - clip_len;
        // int32_t bases_added;
        // add more space for the realignment: move bases from cropped into clip
        if (strand) // reverse: clip then cropped
            tmap_map_alignment_segment_move_bases (&clip_al, &cropped_al, -add_bases, par->sw_param, &cropped_stats, &clip_stats);
        else // forward: cropped then clip
            tmap_map_alignment_segment_move_bases (&cropped_al, &clip_al, add_bases, par->sw_param, &cropped_stats, &clip_stats);
        clip_len = clip_al.q_end - clip_al.q_start;
        // TODO: check validity of the bases_added?
    }
    if (!clip_len)
        return 0;

    double clip_identity = ((double) (clip_stats.score)) / (clip_len * par->sw_param->matrix [0]); // A:A match score :) We may want to actually compute self score, but in TMAP it does not make sense - all 'real' matches are generated equal

    // operate on the read/reference zones downstream of the location of worst alignment score. 
    // perform "global" alignment from the worst alignment zone position on 5' to the factual read end paired to each read end position on reference
    // (alternatively when the read end is beyond the amplicon end, we may first enforce the perfect match within the primer - but this could turn out to be just arbitrary restrictive)

    // reserve space for best result
    uint32_t* best_addition_cigar = NULL;
    uint32_t best_addition_cigar_len = 0;
    uint32_t best_addition_score = UINT32_MAX;
    uint32_t best_read_end_pos = UINT32_MAX;

    // read_end current pointer and sentinel
    re_p = s->read_ends.positions;
    re_sent = re_p;
    if (s->strand) // reverse - check starts
        re_sent += s->read_ends.starts_count;
    else // forward - check ends
        re_p += s->read_ends.starts_count,
        re_sent += s->read_ends.starts_count + s->read_ends.ends_count;

    // loop over end positions
    for (; re_p != re_sent; ++re_p)
    {
        /// check if selection conditions are satisfied
        if ((re_p->fraction * 100) <= par->repair_min_freq)
            continue;
        if (re_p->count <= par->repair_min_count)
            continue;

        /// prepare coordinates for SW
        int32_t read_beg, read_end, ref_beg, ref_end;
        read_beg = clip_al.q_start;
        read_end = clip_al.q_end;
        if (strand == 0) // forward => look in higher coord zone (clip after cropped)
        {
            ref_beg  = s->pos + clip_al.r_start; 
            ref_end  = re_p->coord - 1; // read end coordinates are one-based, so subtract 1
        }
        else // reverse =? look in lower coord zone (clip before cropped)
        {
            ref_beg  = re_p->coord - 1; // read end coordinates are one-based, so subtract 1
            ref_end  = s->pos + clip_al.r_end;
        }
        if (ref_end <= ref_beg) // we should still treat this as a valid case and try to realign, with longer read/target zone. later.
            continue;

        assert (target->position <= ref_beg);
        int32_t target_beg = ref_beg - target->position;
        assert (target->position <= ref_end);
        int32_t target_end = ref_end - target->position;
        assert (target_end > target_beg);
        assert (read_end > read_beg);

        // make sure enough memory is allocated for the path
        int32_t plen_needed = (target_end - target_beg) + (read_end - read_beg);
        if (*path_buf_sz <= plen_needed)
        {   // lengthen the path
            *path_buf_sz = plen_needed;
            tmap_roundup32 (*path_buf_sz);
            *path_buf = tmap_realloc (*path_buf, sizeof (tmap_sw_path_t) * *path_buf_sz, "path_buf");
        }
        /// call global alignment routine
        int32_t path_len;
#if 0
        // tmap_sw_clipping_core2 does not disallow the clipping of the sequence1 (reference) when instructed to do so.
        int32_t add_score = tmap_sw_clipping_core2 (
                                                    target->data + target_beg,
                                                    target_end - target_beg,
                                                    query + read_beg,
                                                    read_end - read_beg,
                                                    par->sw_param,
                                                    0, 0, // do not allow either end of the target to be skipped
                                                    0, 0, // do not allow any softclips
                                                    *path_buf,
                                                    &path_len,
                                                    0
                                                   );
#endif
        int32_t add_score = tmap_sw_global_core (
                                                    target->data + target_beg,
                                                    target_end - target_beg,
                                                    query + read_beg,
                                                    read_end - read_beg,
                                                    par->sw_param,
                                                    *path_buf,
                                                    &path_len,
                                                    0
                                                   );
        if (path_len == 0) // alignment failure - REPAiR attempt failed at this read_end. 
            continue;

        if (!best_addition_cigar || add_score > best_addition_score)  // do we need to restrict to positive scores only? or to scores over the score of removed portion? (the latter sounds appropriate)
        {
            // perform checks on suitability of the alignment
            // check if this score makes sense: compare identities
            double addition_identity = ((double) add_score) / (read_end - read_beg);
            if (addition_identity < clip_identity * par->repair_identity_drop_limit)
                continue;

            // build the cigar
            int32_t addition_len;
            uint32_t* addition = tmap_sw_path2cigar (*path_buf, path_len, &addition_len);
            if (addition_len == 0) // alignment failure - REPAiR attempt failed. This happens if aligner sets path_len to 0
                continue;
            assert (addition);

            // check if the primer zone (amplicon end to read end position) aligns well enough

            // make alignment
            tmap_map_alignment candidate;
            init_alignment_x (&candidate, addition, addition_len, query + read_beg, 0, read_end - read_beg, target->data + target_beg,  0, target_end - target_beg);
            // make primer match == segment between alignment end and amplicon end
            tmap_map_alignment_segment primer_match;
            primer_match.alignment = &candidate;
            init_segment (&primer_match); // segment includes entire alignment

            if (strand == 0)
            {
                // check if amplicon end is within (re)aligned zone
                if (target->position + primer_match.r_start <= s->ampl_end - 1 && s->ampl_end - 1 < target->position + primer_match.r_end)
                {
                    // isolate out the primer match zone (between ampl end and alignment end)
                    int32_t ampl_end_pos = s->ampl_end - 1 - target->position;
                    tmap_map_segment_clip_to_ref_base (&primer_match, 1, ampl_end_pos);
                    // compute stats on the primer match zone (no sw score)
                    tmap_map_alignment_stats stats;
                    tmap_map_alignment_segment_score (&primer_match, NULL, &stats);
                    uint32_t num_edits = stats.mismatches + stats.gaplen;
                    if (num_edits > par->repair_max_primer_zone_dist)
                    {
                        free (addition);
                        continue;
                    }
                }
            }
            else
            {
                // check if amplicon end is within (re)aligned zone
                if (target->position + primer_match.r_start <= s->ampl_start - 1 && s->ampl_start - 1 < target->position + primer_match.r_end)
                {
                    // isolate out the primer match zone (between ampl end and alignment end)
                    int32_t ampl_start_pos = s->ampl_start - 1 - target->position;
                    tmap_map_segment_clip_to_ref_base (&primer_match, 0, ampl_start_pos);
                    // compute stats on the primer match zone (no sw score)
                    tmap_map_alignment_stats stats;
                    tmap_map_alignment_segment_score (&primer_match, NULL, &stats);
                    uint32_t num_edits = stats.mismatches + stats.gaplen;
                    if (num_edits > par->repair_max_primer_zone_dist)
                    {
                        free (addition);
                        continue;
                    }
                }
            }

            if (best_addition_cigar)
                free (best_addition_cigar);
            best_addition_score = add_score;
            best_addition_cigar = addition;
            best_addition_cigar_len = addition_len;
            best_read_end_pos = re_p->coord;
        }
    }
    if (best_addition_cigar == NULL)
        return 0;

    int32_t cigar_op_idx = (strand == 0) ? cropped_al.last_op : cropped_al.first_op;
    int32_t new_score = cropped_stats.score;

    // merge the addition into original cigar and adjust the score
    if (strand == 0)  // original then addition
    {
        int32_t addition_first_op = TMAP_SW_CIGAR_OP (best_addition_cigar [0]);
        int32_t last_orig_op = TMAP_SW_CIGAR_OP (s->cigar [cigar_op_idx]);
        int32_t last_orig_op_len = TMAP_SW_CIGAR_LENGTH (s->cigar [cigar_op_idx]);

        if (addition_first_op == last_orig_op)
        {
            int32_t merged_len = TMAP_SW_CIGAR_LENGTH (best_addition_cigar [0]) + TMAP_SW_CIGAR_LENGTH (s->cigar [cigar_op_idx]);
            TMAP_SW_CIGAR_STORE (best_addition_cigar [0], addition_first_op, merged_len);
            --cigar_op_idx; // last operation from original cigar now merged into the addition
            if (addition_first_op == BAM_CINS || addition_first_op == BAM_CDEL) // cannot be sofclip by the parameters to sw_clipping_core2
                best_addition_score -= par->sw_param->gap_open; // no opening, indel operation continues
        }
        int32_t new_cigar_len = cigar_op_idx + 1 + best_addition_cigar_len;
        if (new_cigar_len > s->n_cigar)
            s->cigar = tmap_realloc (s->cigar, new_cigar_len * sizeof (*(s->cigar)), "cigar");
        memcpy (s->cigar + cigar_op_idx + 1, best_addition_cigar, best_addition_cigar_len * sizeof (*(s->cigar)));
        s->n_cigar = new_cigar_len;
        s->score = new_score + best_addition_score;
        free (best_addition_cigar);

        // adjust target coords, len and alignment box.
        // starts do not change; target end changes to the best read end position; query end changes to the very end of read
        repaired = (best_read_end_pos - s->pos - s->target_len > 0)?2:1;
        s->target_len = best_read_end_pos - s->pos;
        s->result.target_end = s->target_len; // a bit of a hack, the convention is that target_beg is always zero
        s->result.query_end = qlen;
    }
    else  // addition then original
    {
        // check if last op of addition matches first (remaining) op of original cigar; merge ops if yes
        int32_t orig_shift = best_addition_cigar_len - cigar_op_idx;
        int32_t addition_last_op = TMAP_SW_CIGAR_OP (best_addition_cigar [best_addition_cigar_len - 1]);
        int32_t first_orig_op = TMAP_SW_CIGAR_OP (s->cigar [cigar_op_idx]);
        if (addition_last_op == first_orig_op) // merge into addidion's last op
        {
            int32_t merged_len = TMAP_SW_CIGAR_LENGTH (best_addition_cigar [best_addition_cigar_len - 1]) + TMAP_SW_CIGAR_LENGTH (s->cigar [cigar_op_idx]);
            TMAP_SW_CIGAR_STORE (best_addition_cigar [best_addition_cigar_len - 1], addition_last_op, merged_len);
            --orig_shift;
            if (addition_last_op == BAM_CINS || addition_last_op == BAM_CDEL) // cannot be sofclip by the parameters to sw_clipping_core2
                best_addition_score -= par->sw_param->gap_open; // no opening, indel operation continues
        }
        int32_t new_cigar_len = s->n_cigar + orig_shift;
        if (new_cigar_len > s->n_cigar)
            s->cigar = tmap_realloc (s->cigar, new_cigar_len * sizeof (*(s->cigar)), "cigar");
        if (orig_shift > 0)
            memmove (s->cigar + orig_shift, s->cigar, s->n_cigar * sizeof (*(s->cigar)));
        if (orig_shift < 0)
            memmove (s->cigar, s->cigar - orig_shift, (s->n_cigar + orig_shift) * sizeof (*(s->cigar)));
        memcpy (s->cigar, best_addition_cigar, best_addition_cigar_len * sizeof (*(s->cigar)));
        s->n_cigar = new_cigar_len;
        s->score = new_score + best_addition_score;
        free (best_addition_cigar);

        // adjust target coords, len and alignment box. 
        // ends do not change; target start changes to the best read end position; query start changes zero
        repaired = (s->pos + s->target_len - best_read_end_pos - s->target_len > 0)?2:1;
        s->target_len = s->pos + s->target_len - best_read_end_pos;
        s->pos = best_read_end_pos;
        s->result.query_start = 0;
    }

    return repaired; // for now ignored by the caller's caller
}

void target_cache_init (ref_buf_t* target)
{
    target->buf = NULL;
    target->buf_sz = 0;
    target->data = NULL;
    target->data_len = 0;
    target->position = 0;
    target->seqid = 0; // for reference access the sequence ids are 1-based, so 0 is always invalid
    target->seq_start = 0xFFFFFFFF;
    target->seq_end = 0xFFFFFFFF;
}
void target_cache_free (ref_buf_t* target)
{
    free (target->buf);
    target_cache_init (target);
}

void cache_target (ref_buf_t* target, tmap_refseq_t *refseq, uint32_t seqid, uint32_t seq_start, uint32_t seq_end)
{
    uint32_t tlen = seq_end - seq_start + 1;
    // check if already in buffer
    if (seqid == target->seqid && seq_start >= target->seq_start && seq_end <= target->seq_end)
    {
        target->data = target->buf + (seq_start - target->seq_start);
        target->data_len = tlen;
        target->position = seq_start - 1;
    }
    else
    {
        if (target->buf_sz < tlen)
        {   // more memory?
            target->buf_sz = tlen;
            tmap_roundup32 (target->buf_sz);
            target->buf = tmap_realloc (target->buf, sizeof (uint8_t) * (target->buf_sz), "target->buf");
        }
        if (!tmap_refseq_subseq2 (refseq, seqid, seq_start, seq_end, target->buf, 1, NULL))
        {
            uint32_t rl = refseq->annos[seqid-1].len;
            tmap_progress_print2 ("\n seqid = %d, seq_start = %d, seq_end = %d, Ref seq len = %d", seqid, seq_start, seq_end, rl);
            tmap_bug ();
        }
        target->seqid = seqid;
        target->seq_start = seq_start;
        target->seq_end = seq_end;
        target->data = target->buf;
        target->data_len = tlen;
        target->position = seq_start - 1;
    }
}

int find_alignment_start
(
    tmap_map_sam_t* src_sam,    // source: raw (position-only) mapping
    tmap_map_sam_t* dest_sam,   // destination: refined (aligned) mapping
    tmap_seq_t* seq,            // // needed to set overriden softclip
    tmap_seq_t** seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_refseq_t* refseq,      // reference server
    int32_t softclip_start,     // is 5' softclip allowed
    int32_t softclip_end,       // is 3' softclip allowed
    tmap_map_opt_t* opt,        // tmap parameters
    ref_buf_t* target,          // reference data
    tmap_vsw_t *vsw,            // vectorized aligner object
    tmap_map_stats_t * stat      // statistics
)
// returns 1 on success, 0 on failure
{
    // get the target sequence
    uint32_t start_pos = src_sam->pos + 1;
    uint32_t end_pos = start_pos + src_sam->target_len - 1;
    // int32_t tlen = src_sam->result.target_end + 1; // adjust based on the target end
    uint8_t* orig_query = (uint8_t*) tmap_seq_get_bases (seqs [src_sam->strand])->s; // forward genomic strand
    int32_t orig_query_len = tmap_seq_get_bases_length (seqs [0]);

    // cache data
    cache_target (target, refseq, src_sam->seqid + 1, start_pos, end_pos);

    // retrieve the reverse complimented query sequence
    uint8_t *query_rc = (uint8_t*) tmap_seq_get_bases (seqs [1])->s;
    query_rc += orig_query_len - src_sam->result.query_end - 1; // offset query

    // uint8_t *query_rc = NULL;
    int32_t qlen = src_sam->result.query_end + 1; // adjust based on the query end;
// #define DEBUG_VSW_ASSYMETRY
#ifdef DEBUG_VSW_ASSYMETRY
    {
    for (int dd = 0; dd != qlen; ++dd)
        if (query_rc [dd] >= 4)
        {
            tmap_warning ("QUERY Base [%d] is %d\n", dd, query_rc [dd]);
        }
    }
#endif
    // do not band when generating the cigar
    tmap_map_sam_t tmp_sam; // TODO: actually, only the .pos, .score and .result members change; it would be cleaner to use only them
    tmp_sam = *src_sam;

    // save the position and target len as passed from mapper for end_repair
    tmp_sam.mapper_pos = tmp_sam.pos;
    tmp_sam.mapper_tlen = tmp_sam.target_len;

    // reverse compliment the target_buf
    if(0 == src_sam->strand)
        tmap_reverse_compliment_int (target->data, target->data_len);
#ifdef DEBUG_VSW_ASSYMETRY
    {
    for (int dd = 0; dd != target->data_len; ++dd)
        if (target->data [dd] >= 4)
        {
            tmap_warning ("TAGET Base [%d] is %d\n", dd, target->data [dd]);
        }
    }
#endif

    // check if the amplicon was found and has overrides
    candeval_overrides_s override;
    ampl_candeval_parameters_from_sam (opt, src_sam, &override);
    // save default parameters
    tmap_vsw_opt_t orig_vsw_opt = *(vsw->opt);
    int32_t orig_softclip_start = vsw->query_start_clip, orig_softclip_end = vsw->query_end_clip;

    tmap_map_util_set_softclip (override.softclip_type, seq, override.max_adapter_bases_for_soft_clipping, &softclip_start, &softclip_end);
    // tmap_vsw_set_params (vsw, softclip_start, softclip_end, &(override.vsw));

    //DK: since we compute in reverse direction, softclip_end is at the start and softclip_start is at the end of the alignment!
    tmap_vsw_set_params (vsw, softclip_end, softclip_start, &(override.vsw));


    // NB: if match/mismatch penalties are on the opposite strands, we may
    // have wrong scores
    // NB: this aligns in the opposite direction than sequencing (3'->5')
    int32_t overflow;
    int32_t orig_score = tmp_sam.score;
    tmp_sam.score = tmap_vsw_process_rev (vsw, query_rc, qlen, target->data, target->data_len,
                                    &tmp_sam.result, &overflow, opt->score_thr,
                                    (1 ==  softclip_end && 1 == softclip_start) ? 0 : 1, opt->confirm_vsw_corr, opt->correct_failed_vsw, opt->use_nvsw_on_nonstd_bases, stat); // NB: to guarantee correct soft-clipping if both ends are clipped

    if(1 == overflow)
        tmap_bug ();

    // reset vsw parameters to orig
    tmap_vsw_set_params (vsw, orig_softclip_start, orig_softclip_end, &(orig_vsw_opt));

    // reverse-compliment the target_buf back
    if(0 == tmp_sam.strand)
        tmap_reverse_compliment_int (target->data, target->data_len);

    if (tmp_sam.score < opt->score_thr) // this could happen if VSW fails.
        // The score error is reported in tmap_vsw_process_rev, no need to repeat it here (the overflows are not reported on stderr but reported in stats)
        return 0; 

    if (0 == tmp_sam.strand)
    {
        tmp_sam.pos += tmp_sam.result.target_start; // keep it zero based
    }
    else
    {
        int32_t query_start = orig_query_len - tmp_sam.result.query_end - 1;
        int32_t query_end = orig_query_len - tmp_sam.result.query_start - 1;
        tmp_sam.result.query_start = query_start;
        tmp_sam.result.query_end = query_end;
    }
    tmp_sam.result.target_end -= tmp_sam.result.target_start;
    tmp_sam.result.target_start = 0;
    tmp_sam.target_len = tmp_sam.result.target_end;

    *dest_sam = tmp_sam;


    // update aux data
    tmap_map_sam_malloc_aux (dest_sam);
    switch (dest_sam->algo_id)
    {
        case TMAP_MAP_ALGO_MAP1:
            (*dest_sam->aux.map1_aux) = (*tmp_sam.aux.map1_aux);
            break;
        case TMAP_MAP_ALGO_MAP2:
            (*dest_sam->aux.map2_aux) = (*tmp_sam.aux.map2_aux);
            break;
        case TMAP_MAP_ALGO_MAP3:
            (*dest_sam->aux.map3_aux) = (*tmp_sam.aux.map3_aux);
            break;
        case TMAP_MAP_ALGO_MAP4:
            (*dest_sam->aux.map4_aux) = (*tmp_sam.aux.map4_aux);
            break;
        case TMAP_MAP_ALGO_MAPVSW:
            (*dest_sam->aux.map_vsw_aux) = (*tmp_sam.aux.map_vsw_aux);
            break;
        default:
            tmap_bug ();
            break;
    }
    return 1;
}

static void extend_softclips_to_read_edges (
    tmap_map_sam_t* dest_sam,    // mapping being adjusted
    tmap_seq_t** seqs  // 4-element array containing fwd, rec, compl and rev/compl read sequence
)
{
    int32_t orig_query_len = tmap_seq_get_bases_length (seqs [0]);

    // add needed soft-clips (may be adjusted by salvage)
    // Already converted in place, in sams.result (by find_alignment_start)
    // int32_t query_start = dest_sam->strand?(orig_query_len - dest_sam->result.query_end - 1):dest_sam->result.query_start;
    // int32_t query_end = dest_sam->strand?(orig_query_len - dest_sam->result.query_start - 1):dest_sam->result.query_end;

    // add soft clipping
    if(0 < dest_sam->result.query_start)
    {
        int32_t j;
        // soft clip the front of the read
        dest_sam->cigar = tmap_realloc (dest_sam->cigar, sizeof (uint32_t) * (1 + dest_sam->n_cigar), "dest_sam->cigar");
        for (j = dest_sam->n_cigar - 1; 0 <= j; j--) // shift up
            dest_sam->cigar [j + 1] = dest_sam->cigar [j];
        TMAP_SW_CIGAR_STORE (dest_sam->cigar [0], BAM_CSOFT_CLIP, dest_sam->result.query_start);
        dest_sam->n_cigar++;
    }
    if (dest_sam->result.query_end < orig_query_len - 1)
    {
        // soft clip the end of the read
        dest_sam->cigar = tmap_realloc (dest_sam->cigar, sizeof (uint32_t) * (1 + dest_sam->n_cigar), "dest_sam->cigar");
        TMAP_SW_CIGAR_STORE (dest_sam->cigar [dest_sam->n_cigar], BAM_CSOFT_CLIP, orig_query_len - dest_sam->result.query_end - 1);
        dest_sam->n_cigar++;
    }
}

void compute_alignment (
    tmap_map_sam_t* dest_sam,   // mapping to update with cigar
    tmap_seq_t** seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_refseq_t* refseq,      // reference server
    ref_buf_t* target,          // reference data
    tmap_sw_path_t** path_buf,  // buffer for traceback path
    int32_t* path_buf_sz,       // used portion and allocated size of traceback path.
    tmap_sw_param_t* par,       // Smith-Waterman scoring parameters
    tmap_map_stats_t* stat      // statistics
)
{
    uint8_t* orig_query = (uint8_t*) tmap_seq_get_bases (seqs [dest_sam->strand])->s; // forward genomic strand
    int32_t orig_query_len = tmap_seq_get_bases_length (seqs [0]);
    // uint8_t* adjusted_query = orig_query + (dest_sam->strand?(orig_query_len - dest_sam->result.query_end - 1):(dest_sam->result.query_start)); // offset query
    uint8_t* adjusted_query = orig_query + dest_sam->result.query_start; // offset query; the result.query_start is already properly adjusted depending on a strand

    // uint8_t* adjusted_target = (*target_buf) + (dest_sam->strand?0:dest_sam->result.target_start);
    uint32_t start_pos = dest_sam->pos + dest_sam->result.target_start + 1; // addition of result.target_start is obsolete, as it always should be reset to 0 by find_alignment_start
    uint32_t end_pos = dest_sam->pos + dest_sam->result.target_end + 1;
    // cache data
    cache_target (target, refseq, dest_sam->seqid + 1, start_pos, end_pos);


    int32_t qlen = dest_sam->result.query_end - dest_sam->result.query_start + 1; // update query length
    // int32_t tlen = dest_sam->result.target_end - dest_sam->result.target_start + 1;
    int32_t path_len;
    int32_t new_score;

    // path memory
    if (*path_buf_sz <= target->data_len + orig_query_len)
    {   // lengthen the path
        *path_buf_sz = target->data_len + orig_query_len;
        tmap_roundup32 (*path_buf_sz);
        *path_buf = tmap_realloc (*path_buf, sizeof (tmap_sw_path_t) * *path_buf_sz, "path_buf");
    }

    // Smith Waterman with banding
    // NB: we store the score from the banded version, which does not allow ins then del, or del then ins. The vectorized version does.
    // NB: left genomic indel justification is facilitated by always using the forward strand target/query combination.
    new_score = tmap_sw_global_banded_core (target->data, target->data_len, adjusted_query, qlen, par,
                                            dest_sam->result.score_fwd, *path_buf, &path_len, 0);

    // ZZ: There can be case when the global score is the same as the vectorized version
    // whileas the alignments are different. We are not addressing all of them, but the
    // cases where the resulting alignment starts or ends with deletion shall be avoided.
    if (   (new_score != dest_sam->score && 1 < dest_sam->result.n_best)
        || (path_len > 0 && (TMAP_SW_FROM_D == (*path_buf) [0].ctype || TMAP_SW_FROM_D == (*path_buf) [path_len - 1].ctype)))
        // explicitly fit the query into the target
        new_score = tmap_sw_fitting_core (target->data, target->data_len, adjusted_query, qlen, par, *path_buf, &path_len, 0);
    // update score
    dest_sam->score = new_score;
    // adjust position to forward alignment
    int32_t path_ref_off = (*path_buf) [path_len - 1].i - 1; // path_ref_off is zero-based
    dest_sam->result.target_start += path_ref_off; // target_start is zero-based, path coords are one-based
    dest_sam->pos += dest_sam->result.target_start; // move pos to the (updated) target_start

    if ((*path_buf) [path_len-1].ctype == TMAP_SW_FROM_I) // don't quite understand this ZZ
        dest_sam->pos++; // if alignment starts with INS in query, this ins is placed at NEXT base in rev reference - alignment start should be one base to the right DK


    // actually convert SW path to cigar
    dest_sam->cigar = tmap_sw_path2cigar (*path_buf, path_len, &(dest_sam->n_cigar));
    if (0 == dest_sam->n_cigar)
        tmap_bug ();

    // compute target (alignment on the reference) len
    dest_sam->target_len = 0;
    uint32_t* cigar_pos;
    int32_t j;
    for (j = 0, cigar_pos = dest_sam->cigar; j != dest_sam->n_cigar; ++j, ++cigar_pos)
    {
        if (bam_cigar_type (bam_cigar_op (*cigar_pos)) & CONSUME_REF) // operation consumes reference
            dest_sam->target_len += bam_cigar_oplen (*cigar_pos);
    }

    // from now on, the box is not used in original code - all positions are based on pos and target_len for target, and softclips for query
    // we use it for everything, so we'll update the box here
    // The query positions is in the direction of strands aligning with FORWARD target:
    // : q_start is from the beginning of REVERSE query for reverse matches
    //           and from the beginning of FORWARD query for fwd matches
    // target always mean same
    dest_sam->result.query_end = dest_sam->result.query_start + (*path_buf) [0].j - 1; // TODO: check if adjustment for DEL at the alignment end needed
    dest_sam->result.query_start += (*path_buf) [path_len - 1].j - 1; // one-based in path, zero-based in box
    if ((*path_buf) [path_len-1].ctype == TMAP_SW_FROM_D) // should this ever happen?
        dest_sam->result.query_start++;

    dest_sam->result.target_start = 0;
    dest_sam->result.target_end = dest_sam->target_len - 1; // TODO: check if adjustment for INS at the alignment end needed

    // cure softclips
    extend_softclips_to_read_edges (dest_sam, seqs);

    // remember original alignment
    dest_sam->n_orig_cigar = dest_sam->n_cigar;
    dest_sam->orig_cigar = tmap_calloc (dest_sam->n_orig_cigar, sizeof (*(dest_sam->orig_cigar)), "orig_cigar");
    dest_sam->orig_pos = dest_sam->pos;
    memcpy (dest_sam->orig_cigar, dest_sam->cigar, dest_sam->n_orig_cigar * sizeof (*(dest_sam->orig_cigar)));
}

void salvage_long_indel_at_edges (
    tmap_map_sam_t* dest_sam,   // destination: refined (aligned) mapping
    tmap_seq_t** seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_refseq_t* refseq,      // reference server
    ref_buf_t* target,          // reference data
    tmap_sw_path_t** path_buf,  // buffer for traceback path
    int32_t* path_buf_sz,       // used portion and allocated size of traceback path.
    // tmap_map_opt_t* opt,     // tmap options
    int32_t gapl_len,           // the long indel penalty
    int32_t pen_gapl,           // the number of extra bases to add when searching for long indels
    tmap_sw_param_t* par,       // Smith-Waterman scoring parameters
    tmap_map_stats_t *stat      // statistics
)
{
    // long indel prefixes and suffixes
    // needs:
    //   tmap_map_sam_t* dest_sam,   // destination: refined (aligned) mapping
    //   tmap_map_opt_t* opt,        // tmap parameters
    //   tmap_seq_t** seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    //   tmap_refseq_t* refseq,      // reference server
    //   uint8_t** target_buf,       // pointer to the address of the memory buffer for unpacked reference sequence
    //   int32_t* target_buf_sz,     // pointer to the variable contining presently allocated size of target_buf
    //   tmap_sw_path_t *path_buf,   // buffer for traceback path
    //   int32_t path_buf_sz         // used portion and allocated size of traceback path.
    //   tmap_map_opt_t* opt         // tmap options
    //   tmap_sw_param_t* par;       // Smith-Waterman scoring parameters

    uint32_t ampl_start = 0, ampl_end = 0;
   //  uint32_t ampl_exist = 0; //DK: removing redundant variable

    uint8_t* orig_query = (uint8_t*) tmap_seq_get_bases (seqs [dest_sam->strand])->s; // forward genomic strand
    int32_t orig_query_len = tmap_seq_get_bases_length (seqs [0]);

    int32_t query_start = dest_sam->result.query_start;
    int32_t query_end = dest_sam->result.query_end;

    // DK: The code below is derived from original Niels's edge indel salvage code
    // It assumes cigar string is stripped of softclips.
    // We need to save them and then put back.
    unsigned beg_softclip = 0, end_softclip = 0;
    // int8_t cigar_buf_reallocated = 0;
    if (dest_sam->n_cigar > 1 && bam_cigar_op (dest_sam->cigar [dest_sam->n_cigar - 1]) == BAM_CSOFT_CLIP)
    {
        end_softclip = bam_cigar_oplen (dest_sam->cigar [dest_sam->n_cigar - 1]);
        if (0 == end_softclip)
            tmap_bug ();
        dest_sam->n_cigar --;
    }
    if (dest_sam->n_cigar && bam_cigar_op (dest_sam->cigar [0]) == BAM_CSOFT_CLIP)
    {
        beg_softclip = bam_cigar_oplen (dest_sam->cigar [0]);
        if (0 == beg_softclip)
            tmap_bug ();
        dest_sam->n_cigar --;
        memmove (dest_sam->cigar, dest_sam->cigar + 1, dest_sam->n_cigar * sizeof (uint32_t));
    }

    /**
    * Try to detect large indels present as soft-clipped prefixes of the
    * alignment.
    */

    uint32_t salvaged = 0;
    while(1) // NB: so we can break out at any time
    {
        if (0 < query_start)  // beginning of the alignment (3' for reverse, 5' for forward strand)
        { // start of the alignment
            // ZZ:We may check amplicon here as well. May try this only when it is near start of an amplicon.
            // Here the reference mapping position is known by s->pos to s->pos+s->target_len
            // We can check the amplicon with these.
            //   A. When bed file is not present, current logic holds.
            //   B.i. If bed file is given, when an amplicon is present, try use all the bases to the begin/end of amplicon, very small or no long gap penalty.
            //   B.ii.If bed file is given and the hit position is not in an amplicon, normal logic apply.

            int32_t del_score = 0;
            int32_t ins_score = 0;
            uint32_t op, op_len;
            int32_t new_score;
            uint32_t *cigar = NULL;
            int32_t n_cigar;
            int32_t pos_adj = 0;
            int32_t start_pos, end_pos;
            int32_t path_len;

            // query = (uint8_t*)tmap_seq_get_bases(seqs[tmp_sam.strand])->s; // forward genomic strand
            // DK: this is thr orig_query

            // get the target sequence before the start of the alignment
            int32_t tglen = query_start + gapl_len;

            if (dest_sam->pos <= tglen)
                start_pos = 1;
            else
                start_pos = dest_sam->pos - tglen + 1;
            end_pos = dest_sam->pos;
            if (end_pos < start_pos)
                break; // exit the loop
            int32_t long_gap = pen_gapl;
            //ZZ: check amplicon
            // uint32_t ampl_start = 0, ampl_end = 0;
            // if (tmap_map_get_amplicon (refseq, dest_sam->seqid, dest_sam->pos, dest_sam->pos + dest_sam->target_len, dest_sam->strand, &ampl_start, &ampl_end, NULL))
            if (dest_sam->ampl_end != 0)
            {
                // ampl_exist = 1;
                if (dest_sam->ampl_start >= dest_sam->pos)
                    break;
                if (start_pos > dest_sam->ampl_start)
                    start_pos = dest_sam->ampl_start;
                long_gap = 0;
            }

            // DK - do not reset; use target_buf, it is kept intact
            // target mem
            // target = tmp_target; // reset target in memory
            cache_target (target, refseq, dest_sam->seqid + 1, start_pos, end_pos);

            // try to add a larger deletion
            del_score = tmap_sw_clipping_core2 (target->data, target->data_len, orig_query, query_start, par,
                                            1, 1, // allow the start and end of the target to be skipped
                                            1, 0, // deletions
                                            NULL, 0, 0);

            // try to add a larger insertion
            // TODO: how do we enforce that the target starts immediately where
            // we want it to?
            ins_score = tmap_sw_clipping_core2 (target->data, target->data_len, orig_query, query_start, par,
                                            1, 0, // only allow the start of the target to be skipped
                                            1, 1, // insertions,
                                            NULL, 0, 0);
            // reset start/end position
            start_pos = dest_sam->pos + 1;
            end_pos = dest_sam->pos + dest_sam->target_len;

            if (del_score <= 0 && ins_score <= 0)
                // do nothing
                new_score = -1;
            else
            {
                // path memory
                if (*path_buf_sz <= target->data_len + query_start)
                {   // lengthen the path
                    *path_buf_sz = target->data_len + query_start;
                    tmap_roundup32 (*path_buf_sz);
                    *path_buf = tmap_realloc (*path_buf, sizeof (tmap_sw_path_t) * *path_buf_sz, "path_buf");
                }
                if (del_score < ins_score)
                {
                    // get the path
                    tmap_sw_clipping_core2 (target->data, target->data_len, orig_query, query_start,
                                        par,
                                        1, 0, // only allow the start of the target to be skipped
                                        1, 1, // insertions,
                                        *path_buf, &path_len, 0);
                    op = BAM_CINS;
                    op_len = query_start - (*path_buf) [0].j;
                    new_score = ins_score;
                    pos_adj = 0;
                }
                else
                {
                    // get the path
                    tmap_sw_clipping_core2 (target->data, target->data_len, orig_query, query_start,
                                        par,
                                        1, 1, // allow the start and end of the target to be skipped
                                        1, 0, // deletions
                                        *path_buf, &path_len, 0);
                    op = BAM_CDEL;
                    op_len = target->data_len - (*path_buf) [0].i;
                    new_score = del_score;
                    pos_adj = op_len;
                }
                if ((*path_buf) [path_len - 1].ctype == TMAP_SW_FROM_I)
                    pos_adj++; // TODO: is this correct?
            }
            if(0 < new_score && 0 <= new_score - long_gap) // ?? Redundant condition (DK)
            {
                int32_t n_cigar_op = (0 < op_len) ? 1 : 0;
                int32_t j;
                // get the cigar
                cigar = tmap_sw_path2cigar (*path_buf, path_len, &n_cigar);
                if (0 == n_cigar)
                    tmap_bug ();
                // re-allocate the cigar
                dest_sam->cigar = tmap_realloc (dest_sam->cigar, sizeof (uint32_t) * (n_cigar_op + n_cigar + dest_sam->n_cigar), "dest_sam->cigar");
                // cigar_buf_reallocated = 1;
                // shift up
                for (j = dest_sam->n_cigar - 1; 0 <= j; j--)
                    dest_sam->cigar [j + n_cigar_op + n_cigar] = dest_sam->cigar [j];
                // add the operation
                if (1 == n_cigar_op) // 0 < op_len
                    TMAP_SW_CIGAR_STORE (dest_sam->cigar [n_cigar], op, op_len);
                // add the rest of the cigar
                for(j = 0; j < n_cigar; j++)
                    dest_sam->cigar [j] = cigar [j];

                uint32_t added_ref = 0, added_qry = 0;
                dest_sam->n_cigar += n_cigar + n_cigar_op;
                if (op == BAM_CDEL)
                {
                    dest_sam->target_len += op_len;
                    added_ref += op_len;
                }
                else
                    added_qry += op_len;

                for (j = 0; j < n_cigar; j++)
                {
                    int32_t oplen = TMAP_SW_CIGAR_LENGTH (cigar [j]);
// #pragma GCC diagnostic push // unmatched pop will restore command line option
// #pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
                    switch (TMAP_SW_CIGAR_OP (cigar [j]))
                    {
                        case BAM_CMATCH:
                                    added_qry += oplen;
                                    // __attribute__ ((fallthrough)); // this is not available for compilers below version 7
                                    // below is a marker coment recognized by newer versions of GCC. Do not remove line that follows.
                                    /* FALLTHRU */
                        case BAM_CDEL: // fallthrough is intentional
                                    added_ref += oplen;
                                    dest_sam->target_len += oplen;
                                    dest_sam->pos -= oplen; // NB: must adjust the position at the start of the alignment
                                    break;
                        case BAM_CINS:
                        case BAM_CSOFT_CLIP:
                                    added_qry += oplen;
                        default:
                                    break;
                    }
// #pragma GCC diagnostic pop
                }
                free (cigar); cigar = NULL;
                // update the query end
                int32_t orig_qry_start = dest_sam->result.query_start;
                dest_sam->result.query_start = (*path_buf) [path_len - 1].j - 1; // query_start is one-based | DK: this should read path.j is one-based
                if (query_start < 0)
                    tmap_bug();
                dest_sam->score += new_score - long_gap;
                dest_sam->pos -= pos_adj; // adjust the position further if there was a deletion
                // merge adjacent cigar operations
                tmap_map_util_merge_adjacent_cigar_operations (dest_sam);

                // update alignment box
                // query_end does not change
                assert (dest_sam->result.query_start + added_qry == orig_qry_start);
                dest_sam->result.target_end += added_ref;
                uint32_t read_side = (dest_sam->strand==0)?F5P:R3P;
                stat->num_salvaged [read_side] ++;
                stat->bases_salvaged_qry [read_side] += added_qry;
                stat->bases_salvaged_ref [read_side] += added_ref;
                stat->score_salvaged_total [read_side] += new_score - long_gap;
                salvaged = 1;
            }
        }
        break;
    }

    /**
    * Try to detect large indels present as soft-clipped suffixes of the
    * alignment.
    */
    while(1)
    { // NB: so we can break out at any time
        if (query_end < orig_query_len - 1) // end of the alignment (5' for reverse, 3' for forward strand)
        {
            int32_t del_score = 0;
            int32_t ins_score = 0;
            uint32_t op, op_len;
            int32_t new_score = 0;
            uint32_t *cigar = NULL;
            int32_t n_cigar;
            int32_t start_pos, end_pos;
            int32_t path_len;


            // get the target sequence after the end of the alignment.
            int32_t tglen = orig_query_len - 1 - query_end + gapl_len;
            start_pos = dest_sam->pos + dest_sam->target_len + 1;
            end_pos = start_pos + tglen - 1;
            if (refseq->annos [dest_sam->seqid].len < end_pos) // bound
                end_pos = refseq->annos [dest_sam->seqid].len; // one-based

            if (end_pos < start_pos)
                break; // exit the loop
            int32_t long_gap = pen_gapl;

            //ZZ: check amplicon
            // uint32_t ampl_start = 0, ampl_end = 0;
            // DK : if both suffix and prefix branches are executed, and the read hits the amplicon, the ampl_exist is 1 here
            //      so tmap_map_get_amplicon is not executed and 0 is used as the value of ampl_end 
            //      Removing ampl_exist check clause from here.
            // if (ampl_exist || tmap_map_get_amplicon (refseq, dest_sam->seqid, dest_sam->pos, dest_sam->pos + dest_sam->target_len, dest_sam->strand, &ampl_start, &ampl_end, NULL))
            if (dest_sam->ampl_end != 0)
            {
                    // ampl_exist = 1;
                    if (dest_sam->ampl_end <= dest_sam->pos + dest_sam->target_len + 1)
                        break;
                    if (end_pos < dest_sam->ampl_end)
                        end_pos = dest_sam->ampl_end;
                    long_gap = 0;
            }

            cache_target (target, refseq, dest_sam->seqid + 1, start_pos, end_pos);

            // try to add a larger deletion
            // DK: BUG (in original code): this works only if the prefix salvage procedure above did not succeed. Otherwise, the query points to the new start of aligned region,
            // so that query + seq_len points beyond the end of valid sequence (and possibly beyond the end of the buffer)

            del_score = tmap_sw_clipping_core2 (target->data, target->data_len, orig_query + query_end + 1, orig_query_len - query_end - 1,
                                            par,
                                            1, 1, // allow the start and end of the target to be skipped
                                            0, 1, // deletions
                                            NULL, 0, 0);


            // try to add a larger insertion
            ins_score = tmap_sw_clipping_core2 (target->data, target->data_len, orig_query + query_end + 1, orig_query_len - query_end - 1,
                                            par,
                                            0, 1, // only allow the end of the target to be skipped
                                            1, 1, // insertions,
                                            NULL, 0, 0);

            if(del_score <= 0 && ins_score <= 0)
                // do nothing
                new_score = -1;
            else
            {
                // path memory
                if (*path_buf_sz <= target->data_len + orig_query_len - query_end - 1)
                {   // lengthen the path
                    *path_buf_sz = target->data_len + orig_query_len - query_end - 1;
                    tmap_roundup32 (*path_buf_sz);
                    *path_buf = tmap_realloc (*path_buf, sizeof (tmap_sw_path_t) * *path_buf_sz, "path_buf");
                }
                if (del_score < ins_score)
                {
                    // get the path
                    tmap_sw_clipping_core2 (target->data, target->data_len, orig_query + query_end + 1, orig_query_len - query_end - 1,
                                        par,
                                        0, 1, // only allow the end of the target to be skipped
                                        1, 1, // insertions,
                                        *path_buf, &path_len, 0);
                    op = BAM_CINS;
                    op_len = (*path_buf) [path_len - 1].j - 1;
                    new_score = ins_score;
                }
                else
                {
                    // get the path
                    tmap_sw_clipping_core2 (target->data, target->data_len, orig_query + query_end + 1, orig_query_len - query_end - 1,
                                        par,
                                        1, 1, // allow the start and end of the target to be skipped
                                        0, 1, // deletions
                                        *path_buf, &path_len, 0);
                    op = BAM_CDEL;
                    op_len = (*path_buf) [path_len - 1].i - 1;
                    new_score = del_score;
                }
            }
            if(0 < new_score && 0 <= new_score - long_gap) //?? redundant condition (DK)
            {
                int32_t n_cigar_op = (0 < op_len) ? 1 : 0;
                int32_t j;
                // get the cigar
                cigar = tmap_sw_path2cigar (*path_buf, path_len, &n_cigar);
                if (0 == n_cigar)
                    tmap_bug ();
                // re-allocate the cigar
                dest_sam->cigar = tmap_realloc (dest_sam->cigar, sizeof (uint32_t) * (n_cigar_op + n_cigar + dest_sam->n_cigar), "dest_sam->cigar");
                // cigar_buf_reallocated = 1;
                // add the operation
                uint32_t added_ref = 0, added_qry = 0;
                if (1 == n_cigar_op) // 0 < op_len
                {
                    TMAP_SW_CIGAR_STORE (dest_sam->cigar [dest_sam->n_cigar], op, op_len);
                    if (op == BAM_CDEL)
                    {
                        dest_sam->target_len += op_len;
                        added_ref += op_len;
                    }
                    else
                        added_qry += op_len;

                }
                // add the rest of the cigar
                for (j = 0; j < n_cigar; j++)
                    dest_sam->cigar [j + dest_sam->n_cigar + n_cigar_op] = cigar [j];

                dest_sam->n_cigar += n_cigar + n_cigar_op;
                for (j = 0; j < n_cigar; j++)
                {
                    int32_t oplen = TMAP_SW_CIGAR_LENGTH (cigar [j]);
// #pragma GCC diagnostic push // unmatched pop will restore command line option
// #pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
                    switch (TMAP_SW_CIGAR_OP (cigar [j]))
                    {
                        case BAM_CMATCH:
                                    added_qry += oplen;
                                    // __attribute__ ((fallthrough)); // this is not available for compilers below version 7
                                    // below is a marker coment recognized by newer versions of GCC. Do not remove line that follows.
                                    /* FALLTHRU */
                        case BAM_CDEL:
                                    added_ref += oplen;
                                    dest_sam->target_len += oplen;
                                    break;
                        case BAM_CINS:
                        case BAM_CSOFT_CLIP:
                                    added_qry += oplen;
                        default:
                                    break;
                    }
 // #pragma GCC diagnostic pop
                }
                free (cigar); cigar = NULL;
                // update the query end
                query_end += (*path_buf) [0].j; // query_end is zero-based
                if (orig_query_len <= query_end)
                    tmap_bug();
                dest_sam->score += new_score - long_gap;
                // merge adjacent cigar operations
                tmap_map_util_merge_adjacent_cigar_operations (dest_sam);
                // update alignment box

                dest_sam->result.query_end += added_qry;
                dest_sam->result.target_end += added_ref;
                if (dest_sam->result.query_end != query_end)
                    tmap_bug ();
                if (dest_sam->result.target_end != dest_sam->target_len - 1)
                    tmap_bug ();

                uint32_t read_side = (dest_sam->strand==0)?F3P:R5P;
                stat->num_salvaged [read_side] ++;
                stat->bases_salvaged_qry [read_side] += added_qry;
                stat->bases_salvaged_ref [read_side] += added_ref;
                stat->score_salvaged_total [read_side] += new_score - long_gap;
                salvaged = 1;            }
        }
        break;
    }
    extend_softclips_to_read_edges (dest_sam, seqs);
    stat->reads_salvaged += salvaged;
}


static void update_align_box_and_tlen (tmap_map_sam_t* dest_sam)
{
    // assumes pos and cigar are correct
    // update box from cigar
    dest_sam->result.target_start = dest_sam->result.target_end = 0;
    dest_sam->result.query_start = dest_sam->result.query_end = 0;
    uint32_t *cigar_el_p, *cent;
    for (cigar_el_p = dest_sam->cigar, cent = dest_sam->cigar + dest_sam->n_cigar; cigar_el_p != cent; ++ cigar_el_p )
    {
        uint32_t op = TMAP_SW_CIGAR_OP (*cigar_el_p);
        uint32_t oplen = TMAP_SW_CIGAR_LENGTH (*cigar_el_p);
        switch (op)
        {
            case BAM_CMATCH:
                dest_sam->result.query_end += oplen;
                dest_sam->result.target_end += oplen;
                break;
            case BAM_CDEL:
                dest_sam->result.target_end += oplen;
                break;
            case BAM_CINS:
                dest_sam->result.query_end += oplen;
                break;
            case BAM_CSOFT_CLIP:
                if (cigar_el_p == dest_sam->cigar) // very first operation
                    dest_sam->result.query_start = dest_sam->result.query_end = oplen;
                break;
            default:
                tmap_bug ();
        }
    }
    dest_sam->target_len = dest_sam->result.target_end;
    // adjust for inclusivity
    if (dest_sam->result.query_end != dest_sam->result.query_start)
        --dest_sam->result.query_end;
    if (dest_sam->result.target_end != dest_sam->result.target_start)
        --dest_sam->result.target_end;
}

void key_trim_alignment (
    tmap_map_sam_t* dest_sam, // mapping being trimmed
    tmap_seq_t** seqs,        // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_seq_t* seq,          // read
    tmap_refseq_t* refseq,    // reference server
    ref_buf_t* target,        // reference data
    tmap_map_stats_t *stat      // statistics
)
{

    // get first base
    uint8_t key_base = INT8_MAX;
    if (seq->ks)
        key_base = tmap_nt_char_to_int [(uint8_t) seq->ks [strlen (seq->ks) - 1]]; // NB: last base of the key
    // get the new target
    // cache data
    if (INT8_MAX != key_base)
    {
        int32_t qlen = dest_sam->result.query_end - dest_sam->result.query_start; // update query length
        uint8_t* orig_query = (uint8_t*) tmap_seq_get_bases (seqs [dest_sam->strand])->s; // forward genomic strand
        uint8_t* adjusted_query = orig_query + dest_sam->result.query_start;
        uint32_t start_pos = dest_sam->pos + 1;
        uint32_t end_pos = start_pos + dest_sam->target_len - 1;
        cache_target (target, refseq, dest_sam->seqid + 1, start_pos, end_pos);
        tmap_map_util_keytrim (adjusted_query, qlen, target->data, target->data_len, dest_sam->strand, key_base, dest_sam, stat);
        // only pos, cigar and n_cigar are updated by tmap_map_util_keytrim
        // for end-repair, we need target_len and the alignment box also updated
        // update them here
        update_align_box_and_tlen (dest_sam);
    }
}

// The 3' end-repair may alter the alignment and then 5' end-repair should get buffers adjusted accordingly.
// The IR-28816 reveals condition where both 3' and 5' are repairable, and 5' repair used unajusted positions, causing buffer overrun.
// This function serves as a wrapper arounf both 3' and 5' end-repair, correcting SCs and adjusting coordinates and buffers independently.
// returns code for stats collection: 0 for no adjustment, 1 for softclip, 2 for extension
// note: A bit redundant function - it could do a little less if softclip to insert conversion is done only for one proper end of the read.
// Not too much extra work though, and the logic is slightly cleaner.

static int32_t end_repair_helper
(
    tmap_map_sam_t* dest_sam,     // mapping being trimmed
    tmap_seq_t** seqs,            // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_seq_t* seq,              // read
    tmap_refseq_t* refseq,        // reference server
    ref_buf_t* target,            // reference data
    tmap_sw_path_t** path_buf,    // buffer for traceback path
    int32_t* path_buf_sz,         // used portion and allocated size of traceback path.
    int32_t softclip_start,       // softclip allowed at read 5'
    int32_t softclip_end,         // softclip allowed at read 3'
    tmap_map_opt_t* opt,          // tmap options
    tmap_map_stats_t *stat,       // statistics
    int32_t five_prime            // 1 to repair 5', 0 to repairt 3'
)
{
    uint8_t* orig_query = (uint8_t*) tmap_seq_get_bases (seqs [dest_sam->strand])->s; // forward genomic strand
    int32_t orig_qlen = tmap_seq_get_bases_length (seqs [0]);
    int32_t qlen = dest_sam->result.query_end - dest_sam->result.query_start + 1; // update query length
    uint8_t* adjusted_query = orig_query + dest_sam->result.query_start;

    // tmap_map_util_end_repair expects no softclips if they are not explicitely allowed;
    // if there are such, replace them with INSs
    // TS-15331: only change S->I on the read side that is being repaired in this call. Otherwise, lead/trail softclips introduced in first call get replaced with INS during second call.
    if (TMAP_SW_CIGAR_OP (dest_sam->cigar [0]) == BAM_CSOFT_CLIP && // the cigar starts with softclip
        ((softclip_start == 0 && dest_sam->strand == 0 && five_prime != 0) || // (reparing 5') and (5' softclip is not allowed) && (cigar start is 5' end, as the read is forward)
         (softclip_end   == 0 && dest_sam->strand == 1 && five_prime == 0)))  // (reparing 3') and (3' softclip is not allowed) && (cigar start is 3' end, as the read is reverse)
    {
        // replace leading softclip with INS
        int32_t op_len = TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [0]);
        TMAP_SW_CIGAR_STORE (dest_sam->cigar [0], BAM_CINS, op_len);
        qlen += op_len;
        adjusted_query -= op_len;
    }
    if (TMAP_SW_CIGAR_OP (dest_sam->cigar [dest_sam->n_cigar-1]) == BAM_CSOFT_CLIP && // the cigar ends with softclip
        ((softclip_end   == 0 && dest_sam->strand == 0 && five_prime == 0) || // (reparing 3') and (3' softclip is not allowed) && ((cigar end is 3', as the read is forward
         (softclip_start == 0 && dest_sam->strand == 1 && five_prime != 0)))  // (reparing 5') and (5' softclip is not allowed) && ((cigar end is 5', as the read is reverse
    {
        // replace trailing softclip with INS
        int32_t op_len = TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [dest_sam->n_cigar - 1]);
        TMAP_SW_CIGAR_STORE (dest_sam->cigar [dest_sam->n_cigar - 1], BAM_CINS, op_len);
        qlen += op_len;
    }

    // extend cached reference enough to support convertion of softclips to matches (tmap_map_util_end_repair assumes they are in)
    uint32_t sc5 = 0, sc3  = 0;
    if (dest_sam->n_cigar && TMAP_SW_CIGAR_OP (dest_sam->cigar [0]) == BAM_CSOFT_CLIP)
        sc5 = TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [0]);
    if (dest_sam->n_cigar > 1 && TMAP_SW_CIGAR_OP (dest_sam->cigar [dest_sam->n_cigar - 1]) == BAM_CSOFT_CLIP)
        sc3 = TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [dest_sam->n_cigar - 1]);

    // the cigar is in the query direction, convert 5' - 3' to target coords // DK 11/2019 : Is the cigar really is in query direction? It should be in reference one
    // DK 12/30//2019: It is in reference direction (in case of reverse strand, in the direction of inverse-complemented query, which is same as refetence's, 
    //   as invers-complemented query is aligned to forward reference
    // so for reverse we need to swap 5' and 3': 5' is at the left side on the alignment
    if (dest_sam->strand == 1)
    {
        uint32_t t = sc5;
        sc5 = sc3;
        sc3 = t;
    }
    // do not go below 0 or over contig length
    uint32_t start_pos, end_pos;
    if (dest_sam->pos + dest_sam->result.target_start < sc5)
        start_pos = 1;
    else
        start_pos = dest_sam->pos + dest_sam->result.target_start - sc5 + 1;
    end_pos = dest_sam->pos + dest_sam->result.target_end + sc3 + 1;
    if (end_pos > refseq->annos [dest_sam->seqid].len)
        end_pos = refseq->annos [dest_sam->seqid].len;
    // cache the data
    cache_target (target, refseq, dest_sam->seqid + 1, start_pos, end_pos);

    // massage target cache internals. This would not affect integrity of cache
    uint8_t* orig_data = target->data;
    target->data = orig_data + (dest_sam->pos + 1 - start_pos);
    target->data_len = dest_sam->target_len;

    int32_t result = tmap_map_util_end_repair (seq, adjusted_query, qlen, target->data, target->data_len, dest_sam->strand, path_buf, path_buf_sz, refseq, dest_sam, opt, /* stage_swpar, */ five_prime, stat);

    assert (dest_sam->n_cigar);
    dest_sam->result.query_start = (TMAP_SW_CIGAR_OP (dest_sam->cigar [0]) == BAM_CSOFT_CLIP) ? (TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [0])) : 0;
    dest_sam->result.query_end = (TMAP_SW_CIGAR_OP (dest_sam->cigar [dest_sam->n_cigar - 1]) == BAM_CSOFT_CLIP) ? (orig_qlen - TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [dest_sam->n_cigar - 1]) - 1) : orig_qlen - 1;
    dest_sam->result.target_start = 0;
    dest_sam->result.target_end = dest_sam->target_len - 1;

    return result;
}

#define smaller_number(X,Y) (((X)<(Y))?(X):(Y))
#define larger_number(X,Y) (((X)>(Y))?(X):(Y))

// prepares target buffers so that they contain enough reference sequence to do a repair for all relevant start or end positions
// as a 'side product' evaluates if there are read ends for the repair, if not, just returns 0 ('no repair performed')
static int32_t REPAiR_helper
(
    tmap_map_sam_t* dest_sam,     // mapping being trimmed
    tmap_seq_t** seqs,            // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_seq_t* seq,              // read
    tmap_refseq_t* refseq,        // reference server
    ref_buf_t* target,            // reference data
    tmap_sw_path_t** path_buf,    // buffer for traceback path
    int32_t* path_buf_sz,         // used portion and allocated size of traceback path.
    REPAiR_params* par,           // repair parameters, adjusted for any override
    tmap_map_stats_t *stat       // statistics
)
{
    int32_t result = 0;
    uint8_t* orig_query = (uint8_t*) tmap_seq_get_bases (seqs [dest_sam->strand])->s; // forward genomic strand
    int32_t orig_qlen = tmap_seq_get_bases_length (seqs [0]);
    // int32_t qlen = dest_sam->result.query_end - dest_sam->result.query_start + 1; // update query length
    // uint8_t* adjusted_query = orig_query + dest_sam->result.query_start;

    // extend cached reference enough to support convertion of softclips to matches (tmap_map_util_end_repair assumes they are in)
    // and extending to the end of the amplicon or the farthest read end in the read's 3' direction
    uint32_t sc_low = 0, sc_high  = 0;
    if (dest_sam->n_cigar && TMAP_SW_CIGAR_OP (dest_sam->cigar [0]) == BAM_CSOFT_CLIP)
        sc_low = TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [0]);
    if (dest_sam->n_cigar > 1 && TMAP_SW_CIGAR_OP (dest_sam->cigar [dest_sam->n_cigar - 1]) == BAM_CSOFT_CLIP)
        sc_high = TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [dest_sam->n_cigar - 1]);
    // so for reverse we need to swap 5' and 3': 5' is at the left side on the alignment

    // find if read_start or read_end applies 
    // assuming read_start applies when the read is aligned in reverse direction, so 3' points towards left end of the amplicon, and vice versa
    uint32_t pos_low, pos_high;
    if (dest_sam->strand == 1)
    {
        if (dest_sam->read_ends.starts_count == 0) // no read starts - no REPAiR
            return result;
        pos_low = (dest_sam->pos < sc_low)?0:(dest_sam->pos - sc_low);
        pos_high = dest_sam->pos + dest_sam->result.target_end;
        tmap_map_endpos_t* re_position = dest_sam->read_ends.positions;
        tmap_map_endpos_t* re_sent = re_position + dest_sam->read_ends.starts_count;
        for (; re_position != re_sent; ++re_position)
            pos_low = smaller_number (re_position->coord - 1, pos_low);
    }
    else
    {
        if (dest_sam->read_ends.ends_count == 0) // no read ends - no REPAiR
            return result;
        pos_low = dest_sam->pos;
        pos_high = smaller_number (dest_sam->pos + dest_sam->result.target_end + sc_high, refseq->annos [dest_sam->seqid].len);
        tmap_map_endpos_t* re_position = dest_sam->read_ends.positions + dest_sam->read_ends.starts_count;
        tmap_map_endpos_t* re_sent = re_position + dest_sam->read_ends.ends_count;
        for (; re_position != re_sent; ++re_position)
            pos_high = larger_number (re_position->coord - 1, pos_high);
    }
    cache_target (target, refseq, dest_sam->seqid + 1, pos_low + 1, pos_high); // inclusive one-based. Isn't that a bit dull?

    result =
    tmap_map_util_REPAiR_new 
    (
        seq, 
        orig_query, 
        orig_qlen, 
        target,
        dest_sam->strand, 
        path_buf, 
        path_buf_sz, 
        dest_sam, 
        par,
        stat
    );

    assert (dest_sam->n_cigar);
    dest_sam->result.query_start = (TMAP_SW_CIGAR_OP (dest_sam->cigar [0]) == BAM_CSOFT_CLIP) ? (TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [0])) : 0;
    dest_sam->result.query_end = (TMAP_SW_CIGAR_OP (dest_sam->cigar [dest_sam->n_cigar - 1]) == BAM_CSOFT_CLIP) ? (orig_qlen - TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [dest_sam->n_cigar - 1]) - 1) : orig_qlen - 1;
    dest_sam->result.target_start = 0;
    dest_sam->result.target_end = dest_sam->target_len - 1;

    return result;
}

void end_repair (
    tmap_map_sam_t* dest_sam,     // mapping being trimmed
    tmap_seq_t** seqs,            // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_seq_t* seq,              // read
    tmap_refseq_t* refseq,        // reference server
    tmap_map_opt_t* opt,          // tmap options
    ref_buf_t* target,            // reference data
    tmap_sw_path_t** path_buf,    // buffer for traceback path
    int32_t* path_buf_sz,         // used portion and allocated size of traceback path.
    tmap_map_stats_t *stat        // statistics
)
{
    int32_t max_adapter_bases_for_soft_clipping;
    int32_t softclip_start, softclip_end;

    // we set softclp separately for 3' and 5' mappings, as max 3' adapter lenghth for softclipping may be specified differently for lower and higher coordinate ends of the amplicon.

    // determine max_adapter_bases_for_soft_clipping
    // the following code replicates the way of determination of max_adapter_bases_for_soft_clipping from tmap_map_util_end_repair
    max_adapter_bases_for_soft_clipping = opt->max_adapter_bases_for_soft_clipping;
    if (dest_sam->param_ovr)
    {
        if (dest_sam->strand) // for reverse mappings, 3' is at lower coord. end of amplicon - use lower end parameters if provided
        {
            if (dest_sam->param_ovr->max_adapter_bases_for_soft_clipping_le.over)
                max_adapter_bases_for_soft_clipping = dest_sam->param_ovr->max_adapter_bases_for_soft_clipping_le.value;
            else if (dest_sam->param_ovr->max_adapter_bases_for_soft_clipping.over)
                max_adapter_bases_for_soft_clipping = dest_sam->param_ovr->max_adapter_bases_for_soft_clipping.value;
        }
        else // for forward mappings, 3' is at higher coord. end of amplicon - use higher end parameters if provided
        {
            if (dest_sam->param_ovr->max_adapter_bases_for_soft_clipping_he.over)
                max_adapter_bases_for_soft_clipping = dest_sam->param_ovr->max_adapter_bases_for_soft_clipping_he.value;
            else if (dest_sam->param_ovr->max_adapter_bases_for_soft_clipping.over)
                max_adapter_bases_for_soft_clipping = dest_sam->param_ovr->max_adapter_bases_for_soft_clipping.value;
        }
    }
    tmap_map_util_set_softclip (opt->softclip_type, seq, max_adapter_bases_for_soft_clipping, &softclip_start, &softclip_end); // this is done within end_repair anyway; doing redundant call here to check for ins instead of softclp inend_repair_helper
    int32_t r3 = end_repair_helper (dest_sam, seqs, seq, refseq, target, path_buf, path_buf_sz, softclip_start, softclip_end, opt, /* stage_swpar, */ stat, 0);

    max_adapter_bases_for_soft_clipping = opt->max_adapter_bases_for_soft_clipping;
    if (dest_sam->param_ovr)
    {
        if (!dest_sam->strand) // for forward mapping, 5' is at lower end of amplicon - use lower end parameters if provided
        {
            if (dest_sam->param_ovr->max_adapter_bases_for_soft_clipping_le.over)
                max_adapter_bases_for_soft_clipping = dest_sam->param_ovr->max_adapter_bases_for_soft_clipping_le.value;
            else if (dest_sam->param_ovr->max_adapter_bases_for_soft_clipping.over)
                max_adapter_bases_for_soft_clipping = dest_sam->param_ovr->max_adapter_bases_for_soft_clipping.value;
        }
        else // for reverse mapping, 5' is at higher end of amplicon - use higher end parameters if provided
        {
            if (dest_sam->param_ovr->max_adapter_bases_for_soft_clipping_he.over)
                max_adapter_bases_for_soft_clipping = dest_sam->param_ovr->max_adapter_bases_for_soft_clipping_he.value;
            else if (dest_sam->param_ovr->max_adapter_bases_for_soft_clipping.over)
                max_adapter_bases_for_soft_clipping = dest_sam->param_ovr->max_adapter_bases_for_soft_clipping.value;
        }
    }
    tmap_map_util_set_softclip (opt->softclip_type, seq, max_adapter_bases_for_soft_clipping, &softclip_start, &softclip_end); // this is done within end_repair anyway; doing redundant call here to check for ins instead of softclp inend_repair_helper
    int32_t r5 = end_repair_helper (dest_sam, seqs, seq, refseq, target, path_buf, path_buf_sz, softclip_start, softclip_end, opt, /* stage_swpar, */ stat, 1);

    // update alignment box (dest_sam->result), as it may be used downstream
    // assume alignment is correct here - covers entire read and softclips are properly extended
    if (r3 != 0 || r5 != 0)
        stat->reads_end_repair_clipped++;
    if (r3 == 2 || r5 == 2)
        stat->reads_end_repair_extended++;
}

// checks if repair is needed on a single mapped location; calls REPAiR_helper if yes
int32_t REPAiR (
    tmap_map_sam_t* dest_sam,     // mapping being trimmed
    tmap_seq_t** seqs,            // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_seq_t* seq,              // read
    tmap_refseq_t* refseq,        // reference server
    int32_t stage_ord,            // processing stage index
    tmap_sw_param_t* stage_swpar, // stage sw parameters (not overriden)
    tmap_map_opt_t* opt,          // tmap options
    ref_buf_t* target,            // reference data
    tmap_sw_path_t** path_buf,    // buffer for traceback path
    int32_t* path_buf_sz,         // used portion and allocated size of traceback path.
    tmap_map_stats_t *stat        // statistics
)
{
    // check if read ends data is present for this side of the amplicon
    if (!dest_sam->read_ends.positions)
        return 0;
    if (!dest_sam->read_ends.starts_count && dest_sam->strand == 1)
        return 0;
    if (!dest_sam->read_ends.ends_count && dest_sam->strand == 0)
        return 0;

    // collect overrides for repair (for the end where read's 3' is situated)
    REPAiR_params par;
    if (!evalualte_REPAiR_params (opt, stage_swpar, dest_sam->param_ovr, stage_ord, dest_sam->strand, &par))
        return 0; // repair is disabled after override

    // check read-based filters
    tmap_sam_t *sam = seq->data.sam;
    int32_t zb = tmap_sam_get_zb (sam);
    int32_t za = tmap_sam_get_za (sam);
    int32_t overhang = dest_sam->ampl_end - (dest_sam->ampl_start - 1) - za;
    if (zb < par.repair_min_adapter)
        return 0;
    if (overhang > par.repair_max_overhang)
        return 0;

    // we are to do REPAiR in general (if any of the end positions pass filter)
    int32_t r = REPAiR_helper 
    (
        dest_sam,
        seqs,
        seq,
        refseq,
        target,
        path_buf,
        path_buf_sz,
        &par,
        stat
    );
    if (r)
        stat->ends_REPAiRed [dest_sam->strand]++;
    if (r == 1)
        stat->ends_REPAiR_clipped [dest_sam->strand]++;
    if (r == 2)
        stat->ends_REPAiR_extended [dest_sam->strand]++;
    return r;
}

tmap_map_sams_t*
tmap_map_util_find_align_starts (
    tmap_refseq_t *refseq,      // reference server
    tmap_map_sams_t *sams,      // initial rough mapping
    tmap_seq_t *seq,            // read
    tmap_seq_t **seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_map_opt_t *opt,        // tmap parameters
    ref_buf_t* target,          // reference data
    tmap_map_stats_t* stat       // statistics

)
{
    int32_t end;  // index for the alignment being evaluated
    int32_t i;    // index into results buffer;

    tmap_map_sams_t *sams_tmp = NULL; // buffer to receive the alignments
    int32_t orig_query_len = 0; // current read size

    tmap_vsw_t *vsw = NULL;
    tmap_vsw_opt_t *vsw_opt = NULL;

    int32_t softclip_start, softclip_end;

    sams_tmp = tmap_map_sams_init (sams);
    tmap_map_sams_realloc (sams_tmp, sams->n); // pre-allocate for same number of mappings

    // set softclip flags, rule-based
    tmap_map_util_set_softclip (opt->softclip_type, seq, opt->max_adapter_bases_for_soft_clipping, &softclip_start, &softclip_end);

    // initialize opt
    vsw_opt = tmap_vsw_opt_init (opt->score_match, opt->pen_mm, opt->pen_gapo, opt->pen_gape, opt->score_thr);

    // init seqs
    orig_query_len = tmap_seq_get_bases_length (seqs [0]);

    // reverse compliment query
    vsw = tmap_vsw_init ((uint8_t*) tmap_seq_get_bases(seqs[1])->s, orig_query_len, softclip_end, softclip_start, opt->vsw_type, opt->vsw_fallback, vsw_opt);

    for (end = 0, i = 0; end < sams->n; ++end) // for each placement
    {
        tmap_map_sam_t* src_sam = sams->sams + end; // pointer to the receiving sam (last filled element in sams_tmp)
        tmap_map_sam_t* dest_sam = sams_tmp->sams + i;

        if (find_alignment_start (
                src_sam,        // source: raw (position-only) mapping
                dest_sam,       // destination: refined (aligned) mapping
                seq,
                seqs,           // array of size 4 that contains pre-computed inverse / complement combinations
                refseq,         // reference server
                softclip_start, // is 5' softclip allowed
                softclip_end,   // is 3' softclip allowed
                opt,            // tmap parameters
                target,         // reference data
                vsw,            // vectorized aligner object
                stat
                ))
            ++i;
    }

    // realloc
    tmap_map_sams_realloc (sams_tmp, i);

    // free memory
    tmap_map_sams_destroy (sams);
    tmap_vsw_destroy (vsw);
    tmap_vsw_opt_destroy (vsw_opt);

    // sort by max score, then min coordinate

    return sams_tmp;
}

void tmap_map_find_amplicons 
(
    uint32_t stage_ord,
    tmap_map_opt_t* stage_opt,
    tmap_sw_param_t* def_sw_par,
    tmap_refseq_t *refseq,      // reference server
    tmap_map_sams_t *sams       // initial rough mapping
)
{
    uint32_t i, ampl_start, ampl_end;
    tmap_map_locopt_t* locopt = NULL;
    tmap_map_endstat_p_t ends;
    for (i = 0; i < sams->n; ++i) // for each placement
    {
        tmap_map_sam_t* dest_sam = sams->sams + i;
        // only try to assign amplions for the not yet assigned ones 
        // (if parameters overriding in candidate evaluation is enabled, the amplicons are already assigned)
// #define DEBUG_AMPLFIND_COMPARE
#ifndef DEBUG_AMPLFIND_COMPARE
        if (dest_sam->ampl_start == 0 && dest_sam->ampl_end == 0)
#endif
        {
            locopt = NULL, ampl_start = 0, ampl_end = 0;
            ends.positions = NULL, ends.starts_count = 0, ends.ends_count = 0;

            if (tmap_map_get_amplicon (refseq, dest_sam->seqid, dest_sam->pos, dest_sam->pos + dest_sam->target_len, dest_sam->strand, &ampl_start, &ampl_end, &locopt, &ends))
            {
#ifdef DEBUG_AMPLFIND_COMPARE
                if (dest_sam->ampl_start != 0 || dest_sam->ampl_end != 0)
                {
                    if (dest_sam->ampl_start != ampl_start || dest_sam->ampl_end != ampl_end)
                        tmap_warning ("Old amplicon find gives result different than new: old: %d-%d, new %d-%d\n", ampl_start, ampl_end, dest_sam->ampl_start, dest_sam->ampl_end);
                }
#endif
                dest_sam->ampl_start = ampl_start;
                dest_sam->ampl_end = ampl_end;
                dest_sam->param_ovr = locopt;
                //
                // check if override sw parameters for stage are already cached; cache if not
                // 
                // !!! TS-17814: do not do this here - causes race condition. Protecting against that causes seralization bottleneck
                // do not uncomment the line that follows without #defining CONCURRENT_PARAMETERS_CACHE
                // 
                // cache_sw_overrides (locopt, stage_ord, stage_opt, def_sw_par);
                // 
                dest_sam->read_ends = ends;
            }
#if 0 // this is redundant now
            else
            {
                dest_sam->ampl_start = 0;
                dest_sam->ampl_end = 0;
                dest_sam->param_ovr = NULL;
                dest_sam->read_ends.positions = NULL, dest_sam->read_ends.starts_count = 0, dest_sam->read_ends.ends_count = 0;
            }
#endif 
        }
    }
}


void tmap_map_util_align
(
    tmap_refseq_t *refseq,      // reference server
    tmap_map_sams_t *sams,      // mappings to compute alignments for
    tmap_seq_t **seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    ref_buf_t* target,          // reference data
    int32_t stage_ord,          // stage number in tmap pipeline; needed for fetching proper sw parameters override
    tmap_sw_param_t* swpar,     // Smith-Waterman scoring parameters
    tmap_sw_path_t** path_buf,  // buffer for traceback path
    int32_t* path_buf_sz,       // used portion and allocated size of traceback path
    tmap_map_stats_t *stat      // statistics
)
{
    uint32_t i;
    for (i = 0; i < sams->n; ++i) // for each placement
    {
        tmap_map_sam_t* dest_sam = sams->sams + i;
        uint8_t alt = 0;
        // check if there is override for SW parameters:
        tmap_sw_param_t* local_sw_params;
        if ((local_sw_params = tmap_map_locopt_get_stage_sw_params (dest_sam->param_ovr, stage_ord))) //extra parenthesis to suppress GCC -Wparenthesis warning
        {
            swpar = local_sw_params;
        }

        // get the amplicon if any, use parmaeters for that amplicon if any.
        compute_alignment
        (
            dest_sam,           // destination: refined (aligned) mapping
            seqs,               // array of size 4 that contains pre-computed inverse / complement combinations
            refseq,             // reference server
            target,             // pointer to the reference cache
            path_buf,           // buffer for traceback path
            path_buf_sz,        // used portion and allocated size of traceback path.
            swpar,              // Smith-Waterman scoring parameters
            stat                // statistics
        );
    }
}

void tmap_map_util_salvage_edge_indels
(
    tmap_refseq_t* refseq,      // reference server
    tmap_map_sams_t* sams,      // mappings to compute alignments for
    tmap_seq_t** seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    ref_buf_t* target,          // reference data
    tmap_map_opt_t* opt,        // TMAP stage parmeters
    int32_t stage_ord,          // tmap processing stage index, needed for fetching stage SW parameter overrides
    tmap_sw_param_t* swpar,     // Smith-Waterman scoring parameters
    tmap_sw_path_t** path_buf,  // buffer for traceback path
    int32_t* path_buf_sz,       // used portion and allocated size of traceback path.
    tmap_map_stats_t* stat      // statistics
)
{
    uint32_t i;
    for (i = 0; i < sams->n; ++i) // for each placement
    {
        tmap_map_sam_t* dest_sam = sams->sams + i;
        // check if we need to alter parameters:
        int32_t pen_gapl = opt->pen_gapl;
        int32_t gapl_len = opt->gapl_len;
        if (dest_sam->param_ovr)
        {
            tmap_sw_param_t* local_sw_params;
            if ((local_sw_params = tmap_map_locopt_get_stage_sw_params (dest_sam->param_ovr, stage_ord))) //extra parenthesis to suppress GCC -Wparenthesis warning
            {
                swpar = local_sw_params;
            }
            if (dest_sam->param_ovr->pen_gapl.over)
                pen_gapl = dest_sam->param_ovr->pen_gapl.value;
            if (dest_sam->param_ovr->gapl_len.over)
                gapl_len = dest_sam->param_ovr->gapl_len.value;
        }
        if (pen_gapl < 0) // disabled globally (and not overriden to enabled) or overriden to disabled
            continue;
        salvage_long_indel_at_edges (
            dest_sam,      // destination: refined (aligned) mapping
            seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
            refseq,        // reference server
            target,        // reference data cache
            path_buf,      // buffer for traceback path
            path_buf_sz,   // used portion and allocated size of traceback path.
            pen_gapl, // the number of extra bases to add when searching for long indels
            gapl_len, // the extra bases to add before and after the target during Smith-Waterman
            swpar,           // Smith-Waterman scoring parameters
            stat           // statistics
            );
    }
}

void tmap_map_util_cure_softclips
(
    tmap_map_sams_t *sams,      // mappings to compute alignments for
    tmap_seq_t **seqs          // array of size 4 that contains pre-computed inverse / complement combinations
)
{
    uint32_t i;
    for (i = 0; i < sams->n; ++i) // for each placement
    {
        tmap_map_sam_t* dest_sam = sams->sams + i;
        extend_softclips_to_read_edges
        (
            dest_sam,    // mapping being adjusted
            seqs         // 4-element array containing fwd, rec, compl and rev/compl read sequence
        );
    }
}

void tmap_map_util_trim_key
(
    tmap_map_sams_t *sams,      // mappings to compute alignments for
    tmap_seq_t *seq,            // read
    tmap_seq_t **seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_refseq_t *refseq,      // reference server
    ref_buf_t* target,          // reference data cache
    tmap_map_stats_t *stat      // statistics

)
{
    uint32_t i;
    for (i = 0; i < sams->n; ++i) // for each placement
    {
        tmap_map_sam_t* dest_sam = sams->sams + i;
        key_trim_alignment (
            dest_sam,      // mapping being trimmed
            seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
            seq,           // read
            refseq,        // reference server
            target,        // reference data cache
            stat           // statistics
        );
    }
}

void tmap_map_util_end_repair_bulk
(
    tmap_refseq_t *refseq,      // reference server
    tmap_map_sams_t *sams,      // mappings to compute alignments for
    tmap_seq_t *seq,            // read
    tmap_seq_t **seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_map_opt_t *opt,        // tmap parameters
    ref_buf_t* target,          // reference data cache
    tmap_sw_path_t** path_buf,  // buffer for traceback path
    int32_t* path_buf_sz,       // used portion and allocated size of traceback path.
    tmap_map_stats_t *stat      // statistics
)
{
    uint32_t i;
    for (i = 0; i < sams->n; ++i) // for each placement
    {
        tmap_map_sam_t* dest_sam = sams->sams + i;
        end_repair (
                dest_sam,      // mapping being trimmed
                seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
                seq,           // read
                refseq,        // reference server
                opt,           // tmap options
                target,        // pointer to reference cache control structure
                path_buf,      // address of the pointer to buffer for traceback path,
                path_buf_sz,   // pointer to the integer holding allocated size of traceback path.
                stat           // statistics
        );
    }
}

// REPAiR (Read End Position AlIgnment Repair
void tmap_map_util_REPAiR_bulk
(
    tmap_refseq_t* refseq, 
    tmap_map_sams_t* sams, 
    tmap_seq_t* seq, 
    tmap_seq_t** seqs, 
    tmap_map_opt_t* opt, 
    int32_t stage_ord, 
    tmap_sw_param_t* swpar, 
    ref_buf_t* target, 
    tmap_sw_path_t** path_buf, 
    int32_t* path_buf_sz, 
    tmap_map_stats_t* stat
)
{
    uint32_t i;
    int32_t r = 0, rr = 0;
    for (i = 0; i < sams->n; ++i) // for each placement
    {
        tmap_map_sam_t* dest_sam = sams->sams + i;
        r = REPAiR (
                dest_sam,      // mapping being trimmed
                seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
                seq,           // read
                refseq,        // reference server
                stage_ord,     // processing stage index
                swpar,         // default alignment parameters
                opt,           // tmap options
                target,        // pointer to reference cache control structure
                path_buf,      // address of the pointer to buffer for traceback path,
                path_buf_sz,   // pointer to the integer holding allocated size of traceback path.
                stat           // statistics
        );
        if (r)
            rr = 1;
    }
    if (rr)
        stat->reads_REPAiRed++;
}

#if 0
tmap_map_sams_t *
tmap_map_util_sw_gen_cigar (tmap_refseq_t* refseq,
                 tmap_map_sams_t* sams,
                 tmap_seq_t* seq,
                 tmap_seq_t** seqs,
                 tmap_map_opt_t* opt,
                 tmap_map_stats_t *stat
)
{
    // int32_t start, end;
    int32_t i;    // index into results buffer;

    tmap_map_sams_t *sams_tmp = NULL; // buffer to receive the alignments

    ref_buf_t target;
    target_cache_init (&target);
    tmap_sw_path_t *path_buf = NULL; // buffer for traceback path
    int32_t path_buf_sz = 0; // used portion and allocated size of traceback path.
    tmap_sw_param_t sw_par;
    tmap_map_util_populate_sw_par (&sw_par, opt);

    // DK: seems nothing in this function depend on the mappings order
    // // sort by strand/chr/pos/score
    // tmap_sort_introsort (tmap_map_sam_sort_coord, sams->n, sams->sams);

    // Step 1: fing alignment start point
    sams_tmp = tmap_map_util_find_align_starts
    (
        refseq,             // reference server
        sams,               // initial rough mapping
        seq,                // read
        seqs,               // array of size 4 that contains pre-computed inverse / complement combinations
        opt,                // tmap parameters
        &target,            // target cache control structure
        stat                // statistics
    );

    // Step 2: compute alignment
    tmap_map_util_align
    (
        refseq,             // reference server
        sams_tmp,           // mappings to compute alignments for
        seqs,               // array of size 4 that contains pre-computed inverse / complement combinations
        &target,            // target cache control structure
        &path_buf,          // buffer for traceback path
        &path_buf_sz,       // used portion and allocated size of traceback path.
        &sw_par,            // Smith-Waterman scoring parameters
        stat,               // statistics
        opt->use_param_ovr  // use parameters override if present in BED
    );

    // Step 3: salvage long indels close to the read edges
    if (opt->pen_gapl)
        tmap_map_util_salvage_edge_indels
        (
            refseq,             // reference server
            sams_tmp,           // mappings to compute alignments for
            seqs,               // array of size 4 that contains pre-computed inverse / complement combinations
            opt,                // tmap parameters
            &sw_par,            // Smith-Waterman scoring parameters
            &target,            // target cache control structure
            &path_buf,          // buffer for traceback path
            &path_buf_sz,       // used portion and allocated size of traceback path.
            stat                // statistics
        );


    // Step 4: extend alignment to the read edges by adding soft clips
    tmap_map_util_cure_softclips
    (
        sams_tmp,           // mappings to compute alignments for
        seqs                // array of size 4 that contains pre-computed inverse / complement combinations
    );

    // Step 5: key trim the data
    if (opt->softclip_key)
        tmap_map_util_trim_key
        (
            sams_tmp,           // mappings to compute alignments for
            seq,                // read
            seqs,               // array of size 4 that contains pre-computed inverse / complement combinations
            refseq,             // reference server
            &target,            // target cache control structure
            stat                // statistics
        );

    // Step 6: end repair
    if (opt->end_repair)
        tmap_map_util_end_repair_bulk
        (
            refseq,             // reference server
            sams_tmp,           // mappings to compute alignments for
            seq,                // read
            seqs,               // array of size 4 that contains pre-computed inverse / complement combinations
            opt,                // tmap parameters
            &target,            // target cache control structure
            &path_buf,          // buffer for traceback path
            &path_buf_sz,       // used portion and allocated size of traceback path.
            stat                // statistics
        );

    // free memory
    // tmap_map_sams_destroy (sams);
    free (path_buf);
    target_cache_free (&target);

    // sort by max score, then min coordinate
    if(1 < sams_tmp->n)
        tmap_sort_introsort (tmap_map_sam_sort_score_coord, sams_tmp->n, sams_tmp->sams);

    return sams_tmp;
}
#endif

// if fws parameters are overriden, fills passed in alt_ap and returns it
// otherwise, returns defaul_ap
tmap_fsw_param_t* tmap_fsw_gen_override_ap
(
    tmap_map_locopt_t* locopt,
    tmap_fsw_param_t* default_ap,
    tmap_fsw_param_t* alt_ap,
    int32_t score_match,
    int32_t pen_mm,
    int32_t pen_gapo,
    int32_t pen_gape,
    int32_t fscore
)
{
    if (!locopt)
        return default_ap;

    if ((!locopt->score_match.over || locopt->score_match.value == score_match) &&
        (!locopt->pen_mm.over      || locopt->pen_mm.value      == pen_mm     ) &&
        (!locopt->pen_gapo.over    || locopt->pen_gapo.value    == pen_gapo   ) &&
        (!locopt->pen_gape.over    || locopt->pen_gape.value    == pen_gape   ) &&
        (!locopt->fscore.over      || locopt->fscore.value      == fscore     ))
        return default_ap;

    int32_t i;
    for (i = 0; i < TMAP_FSW_MATRIX_SIZE; ++i)
    {
        alt_ap->matrix [i] = -100 * (locopt->pen_mm.over ? locopt->pen_mm.value : pen_mm);
    }
    for (i = 0; i < 4; ++i)
    {
        alt_ap->matrix [i * 5 + i] = 100 * (locopt->score_match.over ? locopt->score_match.value : score_match);
    }
    alt_ap->gap_open = 100 * (locopt->pen_gapo.over ? locopt->pen_gapo.value : pen_gapo);
    alt_ap->gap_ext  = 100 * (locopt->pen_gape.over ? locopt->pen_gape.value : pen_gape);
    alt_ap->gap_end  = 100 * (locopt->pen_gape.over ? locopt->pen_gape.value : pen_gape);
    alt_ap->fscore   = 100 * (locopt->fscore.over ? locopt->fscore.value : fscore);
    alt_ap->row = 5;
    return alt_ap;
}

// TODO: make sure the "longest" read alignment is found
int32_t
tmap_map_util_fsw
(
    tmap_seq_t *seq,
    tmap_map_sams_t *sams,
    tmap_refseq_t *refseq,
    int32_t bw,
    int32_t softclip_type,
    int32_t score_thr,
    int32_t score_match,
    int32_t pen_mm,
    int32_t pen_gapo,
    int32_t pen_gape,
    int32_t fscore,
    int32_t use_flowgram,
    int32_t stage_fsw_use,
    int32_t use_param_ovr,
    tmap_map_stats_t* stat
)
{
    int32_t i, j, k, l;
    uint8_t *target = NULL;
    int32_t target_mem = 0, target_len = 0;
    int32_t was_int = 1;

    tmap_fsw_flowseq_t *fseq = NULL;
    tmap_fsw_path_t *path = NULL;
    int32_t path_mem = 0, path_len = 0;
    tmap_fsw_param_t param;
    tmap_fsw_param_t loc_param;
    tmap_fsw_param_t *actual_param = NULL;
    int32_t matrix [TMAP_FSW_MATRIX_SIZE];
    int32_t loc_matrix [TMAP_FSW_MATRIX_SIZE];
    param.matrix = matrix;
    loc_param.matrix = loc_matrix;
    uint8_t *flow_order = NULL;
    int32_t flow_order_len = 0;
    uint8_t *key_seq = NULL;
    int32_t key_seq_len = 0;

    if (0 == sams->n || NULL == seq->fo || NULL == seq->ks)
        return 0;

    // check if we need any fsw realignments at all
    uint32_t need_fsw = 0;
    for (i = 0; i < sams->n; ++i)
    {
        tmap_map_sam_t *s = &sams->sams [i];
        if (s->param_ovr && s->param_ovr->aln_flowspace.over && !s->param_ovr->aln_flowspace.value)
            continue;
        if (!s->param_ovr && !stage_fsw_use)
            continue;
        need_fsw = 1;
        break;
    }
    if (!need_fsw)
        return 1;

    // flow order
    flow_order_len = strlen (seq->fo);
    flow_order = tmap_malloc (sizeof (uint8_t) * (flow_order_len + 1), "flow_order");
    memcpy (flow_order, seq->fo, flow_order_len);
    tmap_to_int ((char*) flow_order, flow_order_len);

    // key sequence
    key_seq_len = strlen (seq->ks);
    key_seq = tmap_malloc (sizeof (uint8_t) * (key_seq_len+1), "key_seq");
    memcpy (key_seq, seq->ks, key_seq_len);
    tmap_to_int ((char*) key_seq, key_seq_len);

    // generate the alignment parameters
    param.matrix = matrix;
    param.band_width = 0;
    param.offset = TMAP_MAP_OPT_FSW_OFFSET; // this sets the hp difference
    __tmap_fsw_gen_ap1 (param, score_match, pen_mm, pen_gapo, pen_gape, fscore); // This is default parameters set. would be more efficient to calculate it not for every read, but once for all.

    was_int = tmap_seq_is_int (seq);
    if (0 == tmap_seq_is_int(seq))
        tmap_seq_to_int(seq);

    // get flow sequence
    fseq = tmap_fsw_flowseq_from_seq (NULL, seq, flow_order, flow_order_len, key_seq, key_seq_len, use_flowgram);

    if (!fseq)
    {
        free (flow_order);
        free (key_seq);
        return 0;
    }

    // go through each hit
    for (i = 0; i < sams->n; ++i)
    {
        tmap_map_sam_t *s = &sams->sams [i];
        if (!stage_fsw_use)
            if (!use_param_ovr || !(s->param_ovr && s->param_ovr->aln_flowspace.over && s->param_ovr->aln_flowspace.value))
                continue;

        uint32_t ref_start, ref_end;
        // get the reference end position
        // NB: soft-clipping at the start may cause ref_start to be moved
        param.band_width = 0;
        int32_t bw_r = 0, bw_l = 0;
        ref_start = ref_end = s->pos + 1;
        for (j = 0; j < s->n_cigar; ++j)
        {
            int32_t op, op_len;

            op = TMAP_SW_CIGAR_OP (s->cigar [j]);
            op_len = TMAP_SW_CIGAR_LENGTH (s->cigar [j]);

            switch(op)
            {
                case BAM_CMATCH:
                    ref_end += op_len;
                    break;
                case BAM_CDEL:
                    // if (param.band_width < op_len) // DK: potential error: 2 consecutive deletions would not fit
                    //     param.band_width += op_len;
                    bw_l += op_len;
                    ref_end += op_len;
                    break;
                case BAM_CINS:
                    bw_r += op_len;
                    // if (param.band_width < op_len)
                    //     param.band_width += op_len; // DK: potential error: 2 consecutive inserts would not fit
                    break;
                case BAM_CSOFT_CLIP:
                    if (0 == j)
                    {
                        if (ref_start <= op_len)
                            ref_start = 1;
                        else
                            ref_start -= op_len;
                    }
                    else
                        ref_end += op_len;
                    break;
                default:
                    // ignore
                    break;
            }
        }
        ref_end--; // NB: since we want this to be one-based
        param.band_width = (bw_l > bw_r)?bw_l:bw_r;
        // check bounds
        if (ref_start < 1)
            ref_start = 1;
        if (refseq->annos [s->seqid].len < ref_end)
            ref_end = refseq->annos [s->seqid].len;
        else if (ref_end < 1)
            ref_end = 1;

        // get the target sequence
        target_len = ref_end - ref_start + 1;
        if(target_mem < target_len)
        {
            target_mem = target_len;
            tmap_roundup32 (target_mem);
            target = tmap_realloc (target, sizeof (uint8_t) * target_mem, "target");
        }
        target_len = tmap_refseq_subseq (refseq, ref_start + refseq->annos [s->seqid].offset, target_len, target);
        /*
        // NB: IUPAC codes are turned into mismatches
        if(NULL == tmap_refseq_subseq2(refseq, sams->sams[end].seqid+1, start_pos, end_pos, target, 1, NULL)) {
            tmap_bug();
        }
        */

        if (1 == s->strand) // reverse compliment
            tmap_reverse_compliment_int(target, target_len);

        // add to the band width
        param.band_width += 2 * bw;

        // make sure we have enough memory for the path
        while (path_mem <= target_len + fseq->num_flows) // lengthen the path
        {
            path_mem = target_len + fseq->num_flows + 1;
            tmap_roundup32 (path_mem);
            path = tmap_realloc (path, sizeof (tmap_fsw_path_t) * path_mem, "path");
        }

        /*
        fprintf(stderr, "strand=%d\n", s->strand);
        fprintf(stderr, "ref_start=%d ref_end=%d\n", ref_start, ref_end);
        fprintf(stderr, "base_calls:\n");
        for(j=0;j<fseq->num_flows;j++) {
            for(k=0;k<fseq->base_calls[j];k++) {
                fputc("ACGTN"[fseq->flow_order[j % fseq->flow_order_len]], stderr);
            }
        }
        fputc('\n', stderr);
        fprintf(stderr, "target:\n");
        for(j=0;j<target_len;j++) {
            fputc("ACGTN"[target[j]], stderr);
        }
        fputc('\n', stderr);
        for(j=0;j<fseq->flow_order_len;j++) {
            fputc("ACGTN"[fseq->flow_order[j]], stderr);
        }
        fputc('\n', stderr);
        */

        // check if we need to use custom SW parameters
        actual_param = &param;
        if (use_param_ovr && s->param_ovr)
        {
            actual_param = tmap_fsw_gen_override_ap (
                                                    s->param_ovr,
                                                    &param,
                                                    &loc_param,
                                                    score_match,
                                                    pen_mm,
                                                    pen_gapo,
                                                    pen_gape,
                                                    fscore
                                                    );
        }
        // re-align
        s->ascore = s->score;
        path_len = path_mem;
        //fprintf(stderr, "old score=%d\n", s->score);
        switch (softclip_type)
        {
            case TMAP_MAP_OPT_SOFT_CLIP_ALL:
                s->score = tmap_fsw_clipping_core (target, target_len, fseq, actual_param,
                                                1, 1, s->strand, path, &path_len);
                break;
            case TMAP_MAP_OPT_SOFT_CLIP_LEFT:
                s->score = tmap_fsw_clipping_core (target, target_len, fseq, actual_param,
                                                1, 0, s->strand, path, &path_len);
                break;
            case TMAP_MAP_OPT_SOFT_CLIP_RIGHT:
                s->score = tmap_fsw_clipping_core (target, target_len, fseq, actual_param,
                                                0, 1, s->strand, path, &path_len);
                break;
            case TMAP_MAP_OPT_SOFT_CLIP_NONE:
                s->score = tmap_fsw_clipping_core (target, target_len, fseq, actual_param,
                                                0, 0, s->strand, path, &path_len);
                break;
            default:
                tmap_error("soft clipping type was not recognized", Exit, OutOfRange);
                break;
        }
        s->score_subo = INT32_MIN;
        //fprintf(stderr, "new score=%d path_len=%d\n", s->score, path_len);

        if (0 < path_len)
        { // update
            /*
            for(j=0;j<path_len;j++) {
                fprintf(stderr, "j=%d path[j].i=%d path[j].j=%d path[j].type=%d\n", j, path[j].i, path[j].j, path[j].ctype);
            }
            */

            // score
            s->score = (int32_t)((s->score + 99.99) / 100.0);

            // position
            s->pos = ref_start - 1; // zero-based
            // NB: must be careful of leading insertions and strandedness
            if (0 == s->strand)
            {
                if (0 <= path [path_len - 1].j)
                    s->pos += (path [path_len - 1].j);
            }
            else
            {
                if (path [0].j < target_len)
                    s->pos += target_len - path [0].j - 1;
            }

            if (refseq->len < s->pos)
                tmap_bug ();

            // new cigar
            free (s->cigar);
            s->cigar = tmap_fsw_path2cigar (path, path_len, &s->n_cigar, 1);

            // reverse the cigar
            if (1 == s->strand)
                for(i=0; i < (s->n_cigar >> 1); i++)
                {
                    uint32_t t = s->cigar [i];
                    s->cigar [i] = s->cigar [s->n_cigar - i - 1];
                    s->cigar [s->n_cigar - i - 1] = t;
                }

            int32_t skipped_start, skipped_end;
            skipped_start = skipped_end = 0;

            // soft-clipping
            if (0 < path [path_len - 1].i) // skipped beginning flows
                // get the number of bases to clip
                for (j = 0; j < path [path_len - 1].i; j++)
                    skipped_start += fseq->base_calls[j];

            if (path [0].i + 1 < fseq->num_flows) // skipped ending flows
                // get the number of bases to clip
                for (j = path [0].i + 1; j < fseq->num_flows; j++)
                    skipped_end += fseq->base_calls [j];

            if (1 == s->strand) // swap
            {
                k = skipped_start;
                skipped_start = skipped_end;
                skipped_end = k;
            }
            if (0 < skipped_start) // start soft clip
            {
                s->cigar = tmap_realloc (s->cigar, sizeof (uint32_t) * (1 + s->n_cigar), "s->cigar");
                for (l = s->n_cigar-1; 0 <= l; l--)
                    s->cigar [l + 1] = s->cigar [l];

                TMAP_SW_CIGAR_STORE (s->cigar[0], BAM_CSOFT_CLIP, skipped_start);
                s->n_cigar++;
            }
            if (0 < skipped_end) // end soft clip
            {
                s->cigar = tmap_realloc (s->cigar, sizeof (uint32_t) * (1 + s->n_cigar), "s->cigar");
                s->cigar [s->n_cigar] = (skipped_end << 4) | 4;
                s->n_cigar++;
            }
        }
    }
    // free
    if (fseq)
    {
        free (fseq->flow_order);
        free (fseq->base_calls);
        free (fseq->flowgram);
        free (fseq);
    }
    free (target);
    free (path);
    free (flow_order);
    free (key_seq);

    if (0 == was_int)
        tmap_seq_to_char(seq);

    return 1;
}


int32_t tmap_map_util_remove_5_prime_softclip
(
    tmap_refseq_t *refseq,      // reference server
    tmap_map_sam_t *dest_sam,   // mappings to fix
    tmap_seq_t *seq,            // read
    tmap_seq_t **seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    ref_buf_t* target,          // reference data cache
    tmap_sw_path_t** path_buf,  // buffer for traceback path
    int32_t* path_buf_sz,       // used portion and allocated size of traceback path.
    int32_t stage_ord,          // index of TMAP processing stage (needed for fetching SW parameters override for a stage)
    tmap_sw_param_t* swpar,     // Smith-Waterman scoring parameters
    tmap_map_opt_t *opt,        // tmap parameters
    tmap_map_stats_t *stat      // statistics
)
{
    // check if we need to proceed at all
    uint32_t allow_5prime_softclip = 0;
    allow_5prime_softclip = (opt->end_repair_5_prime_softclip || opt->softclip_type == 0 || opt->softclip_type == 1);
    if (dest_sam->param_ovr)
    {
        if (dest_sam->strand) // reverse => 5' on high coordinate (right) end
        {
            if (dest_sam->param_ovr->end_repair_5_prime_softclip_he.over)
                allow_5prime_softclip = dest_sam->param_ovr->end_repair_5_prime_softclip_he.value;
            else if (dest_sam->param_ovr->end_repair_5_prime_softclip.over)
                allow_5prime_softclip = dest_sam->param_ovr->end_repair_5_prime_softclip.value;
        }
        else // forward => 5' on low coordinate (left) end
        {
            if (dest_sam->param_ovr->end_repair_5_prime_softclip_le.over)
                allow_5prime_softclip = dest_sam->param_ovr->end_repair_5_prime_softclip_le.value;
            else if (dest_sam->param_ovr->end_repair_5_prime_softclip.over)
                allow_5prime_softclip = dest_sam->param_ovr->end_repair_5_prime_softclip.value;
        }
    }
    if (allow_5prime_softclip)
        return 0;

    if (dest_sam->strand == 0 && (dest_sam->n_cigar == 0 || TMAP_SW_CIGAR_OP (dest_sam->cigar [0]) != BAM_CSOFT_CLIP))
        return 0;
    if (dest_sam->strand == 1 && (dest_sam->n_cigar == 0 || TMAP_SW_CIGAR_OP (dest_sam->cigar [dest_sam->n_cigar - 1]) != BAM_CSOFT_CLIP))
        return 0;
    // if we're here, there is a soft clip on 5'

    // check if there is override for SW parameters:
    tmap_sw_param_t* local_sw_params;
    if ((local_sw_params = tmap_map_locopt_get_stage_sw_params (dest_sam->param_ovr, stage_ord))) //extra parenthesis to suppress GCC -Wparenthesis warning
    {
        swpar = local_sw_params;
    }

    // if (0 == strcmp (seq->data.sam->name->s, "K0BK3:03674:08750"))
    //    printf ("Here");
    uint8_t* orig_query = (uint8_t*) tmap_seq_get_bases (seqs [dest_sam->strand])->s; // forward genomic strand
    int32_t orig_query_len = tmap_seq_get_bases_length (seqs [0]);

    int32_t sc_len = TMAP_SW_CIGAR_LENGTH ((dest_sam->strand == 0) ? (dest_sam->cigar [0]) : (dest_sam->cigar [dest_sam->n_cigar - 1]));
    int32_t max_ins_equiv = (opt->pen_gape != 0) ? ((opt->score_match * sc_len - opt->pen_gapo) / opt->pen_gape) : (opt->score_match * sc_len - opt->pen_gapo);
    if (max_ins_equiv < 0)
        max_ins_equiv = 0;
    int32_t target_len = sc_len + max_ins_equiv;

    int32_t start_pos, end_pos;

    if (dest_sam->strand == 0)
    {
        // 5' is a beginning of the alignment, align forward ref to forward read in no-clipping mode
        // target
        start_pos = (dest_sam->pos > target_len) ? (dest_sam->pos - target_len + 1) : 1;
        end_pos = dest_sam->pos;
        if (start_pos >= end_pos)
        {
            // use Ins in place of SC
            TMAP_SW_CIGAR_STORE (dest_sam->cigar [0], BAM_CINS, sc_len);
            dest_sam->score -= opt->pen_gapo + opt->pen_gape*sc_len;
            dest_sam->result.query_start = 0;
        }
        else
        {
            cache_target (target, refseq, dest_sam->seqid + 1, start_pos, end_pos);
            // path memory
            if (*path_buf_sz <= target->data_len + sc_len)
            {   // lengthen the path
                *path_buf_sz = target->data_len + sc_len;
                tmap_roundup32 (*path_buf_sz);
                *path_buf = tmap_realloc (*path_buf, sizeof (tmap_sw_path_t) * *path_buf_sz, "path_buf");
            }
            // align softclipped part of the query
            int32_t path_len;
            int32_t add_score = tmap_sw_clipping_core2 (
                                                        target->data,
                                                        target->data_len,
                                                        orig_query,
                                                        sc_len,
                                                        swpar,
                                                        1, 0, // only allow the start of the target to be skipped
                                                        0, 0, // do not allow any softclips
                                                        *path_buf,
                                                        &path_len,
                                                        0);

            // build the cigar
            int32_t addition_len;
            uint32_t* addition = tmap_sw_path2cigar (*path_buf, path_len, &addition_len);
            if (addition_len == 0) // alignment failure - use Ins
            {
                // use Ins in place of SC
                TMAP_SW_CIGAR_STORE (dest_sam->cigar [0], BAM_CINS, sc_len);
                dest_sam->score -= opt->pen_gapo + opt->pen_gape*sc_len;
                dest_sam->result.query_start = 0;
                return 1;
            }
            assert (addition);
            assert (dest_sam->n_cigar > 1);
            // compute and apply pos, target_len and alignment box adjustment
            int32_t cp, addition_ref_len = 0;
            for (cp = 0; cp != addition_len; ++cp)
            {
                int32_t op = TMAP_SW_CIGAR_OP (addition [cp]);
                int32_t oplen = TMAP_SW_CIGAR_LENGTH (addition [cp]);
                if (op == BAM_CMATCH || op == BAM_CDEL)
                    addition_ref_len += oplen;
            }

            //int32_t pos_adj = 0;  // DK: I do not quite understand this. Modeled after position computing in 'raw' alignment.
            // DK: end-repair does not do this adjustment
            //if ((*path_buf) [0].ctype == TMAP_SW_FROM_I)
            //    pos_adj++; // TODO: is this correct?

            stat->num_5_softclips [dest_sam->strand] ++;
            stat->bases_5_softclips_qry [dest_sam->strand] += dest_sam->result.query_start;
            stat->bases_5_softclips_ref [dest_sam->strand] += addition_ref_len;
            stat->score_5_softclips_total [dest_sam->strand] += add_score;

            assert (addition_ref_len <= dest_sam->pos);
            dest_sam->pos -= addition_ref_len; // - pos_adj;
            dest_sam->target_len += addition_ref_len;
            dest_sam->result.target_end += addition_ref_len;
            dest_sam->result.query_start = 0;


            //tmap_log_record_begin ();
            //tmap_map_log_text_align ("  After 5' unclip :\n", addition, addition_len, (const char *) orig_query, orig_query_len, !dest_sam->strand, (const char*) (target->data + (target->data_len - addition_ref_len)), dest_sam->pos);
            //tmap_log_record_end ();

            // merge addition with the cigar
            // check if last op of addition matches first op of orig cigar; merge ops if yes
            int32_t old_shift = addition_len - 1;  // -1 because we are not copying softclip, it is always replaced with the addition
            int32_t addition_last_op = TMAP_SW_CIGAR_OP (addition [addition_len - 1]);
            int32_t old_first_nonsc_op = TMAP_SW_CIGAR_OP (dest_sam->cigar [1]);
            if (addition_last_op == old_first_nonsc_op)
            {
                int32_t merged_len = TMAP_SW_CIGAR_LENGTH (addition [addition_len - 1]) + TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [1]);
                TMAP_SW_CIGAR_STORE (addition [addition_len - 1], addition_last_op, merged_len);
                --old_shift;
                // DK 01/15/2020 seems like the following score adjustment was missing unintentionally 
                if (addition_last_op == BAM_CINS || addition_last_op == BAM_CDEL) // cannot be sofclip by the parameters to sw_clipping_core2
                    add_score -= opt->pen_gapo; // no additional gap opening, indel operation continues from addition into original alignment
            }
            int32_t new_cigar_len = dest_sam->n_cigar + old_shift;
            if (new_cigar_len > dest_sam->n_cigar)
                dest_sam->cigar = tmap_realloc (dest_sam->cigar, new_cigar_len * sizeof (*(dest_sam->cigar)), "cigar");
            if (old_shift > 0)
                memmove (dest_sam->cigar + old_shift, dest_sam->cigar, dest_sam->n_cigar * sizeof (*(dest_sam->cigar)));
            if (old_shift < 0)
                memmove (dest_sam->cigar, dest_sam->cigar - old_shift, (dest_sam->n_cigar + old_shift) * sizeof (*(dest_sam->cigar)));
            memcpy (dest_sam->cigar, addition, addition_len * sizeof (*(dest_sam->cigar)));
            dest_sam->n_cigar = new_cigar_len;
            dest_sam->score += add_score;
            free (addition);

        }
    }
    else
    {
        // 5' is an end of the alignment, align foward ref to reverse read in no-clipping mode
        start_pos = dest_sam->pos + dest_sam->target_len + 1;
        end_pos = start_pos + target_len - 1;
        if (end_pos > refseq->annos [dest_sam->seqid].len)
            end_pos = refseq->annos [dest_sam->seqid].len;
        if (start_pos >= end_pos)
        {
            // use Ins in place of SC
            TMAP_SW_CIGAR_STORE (dest_sam->cigar [dest_sam->n_cigar - 1], BAM_CINS, sc_len);
            dest_sam->score -= opt->pen_gapo + opt->pen_gape*sc_len;
            dest_sam->result.query_end = orig_query_len - 1;
        }
        else
        {
            assert (start_pos < refseq->annos [dest_sam->seqid].len);
            //if (start_pos >=  end_pos)
            //    printf ("here");
            assert (start_pos < end_pos);
            cache_target (target, refseq, dest_sam->seqid + 1, start_pos, end_pos);
            // path memory
            if (*path_buf_sz <= target->data_len + sc_len)
            {   // lengthen the path
                *path_buf_sz = target->data_len + sc_len;
                tmap_roundup32 (*path_buf_sz);
                *path_buf = tmap_realloc (*path_buf, sizeof (tmap_sw_path_t) * *path_buf_sz, "path_buf");
            }
            // align softclipped part of the query
            int32_t path_len;
            int32_t add_score = tmap_sw_clipping_core2 (
                                                        target->data,
                                                        target->data_len,
                                                        orig_query,
                                                        sc_len,
                                                        swpar,
                                                        0, 1, // only allow the end of the target to be skipped
                                                        0, 0, // do not allow any softclips
                                                        *path_buf,
                                                        &path_len,
                                                        0);

            // build the cigar
            int32_t addition_len;
            uint32_t* addition = tmap_sw_path2cigar (*path_buf, path_len, &addition_len);
            if (addition_len == 0) // alignment failure - use Ins
            {
                // use Ins in place of SC
                TMAP_SW_CIGAR_STORE (dest_sam->cigar [dest_sam->n_cigar - 1], BAM_CINS, sc_len);
                dest_sam->score -= opt->pen_gapo + opt->pen_gape*sc_len;
                dest_sam->result.query_end = orig_query_len - 1;
                return 1;
            }
            assert (addition);

            // compute and apply pos, target_len amd alignment box adjustment
            int32_t cp, addition_ref_len = 0;
            for (cp = 0; cp != addition_len; ++cp)
            {
                int32_t op = TMAP_SW_CIGAR_OP (addition [cp]);
                int32_t oplen = TMAP_SW_CIGAR_LENGTH (addition [cp]);
                if (op == BAM_CMATCH || op == BAM_CDEL)
                    addition_ref_len += oplen;
            }

            stat->num_5_softclips [dest_sam->strand] ++;
            stat->bases_5_softclips_qry [dest_sam->strand] += dest_sam->result.query_start;
            stat->bases_5_softclips_ref [dest_sam->strand] += addition_ref_len;
            stat->score_5_softclips_total [dest_sam->strand] += add_score;

            assert (dest_sam->pos + addition_ref_len <= refseq->annos [dest_sam->seqid].len);
            dest_sam->target_len += addition_ref_len;
            dest_sam->result.target_end += addition_ref_len;
            dest_sam->result.query_end = orig_query_len - 1;
            // merge addition with the cigar
            int32_t addition_pos = dest_sam->n_cigar - 1; // -1 to replace original softclip
            assert (addition_len);
            assert (dest_sam->n_cigar > 1);
            int32_t addition_first_op = TMAP_SW_CIGAR_OP (addition [0]);
            int32_t old_last_nonsc_op = TMAP_SW_CIGAR_OP (dest_sam->cigar [dest_sam->n_cigar - 2]);
            if (addition_first_op == old_last_nonsc_op)
            {
                int32_t merged_len = TMAP_SW_CIGAR_LENGTH (addition [0]) + TMAP_SW_CIGAR_LENGTH (dest_sam->cigar [dest_sam->n_cigar - 2]);
                TMAP_SW_CIGAR_STORE (addition [0], addition_first_op, merged_len);
                --addition_pos;
                if (addition_first_op == BAM_CINS || addition_first_op == BAM_CDEL)
                    add_score -= opt->pen_gapo;
            }
            int32_t new_cigar_len = addition_len + addition_pos;
            if (new_cigar_len > dest_sam->n_cigar)
                dest_sam->cigar = tmap_realloc (dest_sam->cigar, new_cigar_len * sizeof (*(dest_sam->cigar)), "cigar");
            memcpy (dest_sam->cigar + addition_pos, addition, addition_len * sizeof (*(dest_sam->cigar)));
            dest_sam->n_cigar = new_cigar_len;
            dest_sam->score += add_score;
            free (addition);

        }
    }

    return 1;
}

static uint8_t bin_cigar_to_str (uint32_t* cigar, int32_t n_cigar, char* dest, uint32_t dest_sz)
{
    uint32_t* cent = cigar + n_cigar;
    char* dp = dest;
    uint32_t incr;
    for (; cigar != cent; ++cigar, ++dest)
    {
        incr = snprintf (dp, dest_sz - (dp - dest), "%d", TMAP_SW_CIGAR_LENGTH (*cigar));
        if (dest_sz - (dp - dest) < incr + 1)
            return 0;
        dp += incr;
        if (dest_sz - (dp - dest) < 2)
            return 0;
        *(dp++) = BAM_CIGAR_STR [TMAP_SW_CIGAR_OP (*cigar)];
    }
    if (dp - dest >= dest_sz)
        return 0;
    *dp = 0;
    return 1;
}

// checks that alignment score is close enough to the recorded one
// checks if cigar consistent with read size and target_len
// checks if the target boundaries are right
// checks if alignment box consistent with cigar
void cigar_sanity_check
(
    tmap_refseq_t *refseq,      // reference server
    tmap_map_sam_t *sam,        // mapping to check
    tmap_seq_t *seq,            // read
    tmap_seq_t **seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    ref_buf_t* target,          // reference data cache
    tmap_map_opt_t *opt         // tmap options (for this stage)
)
{
    // get override value for cigar_sanity_check if given in BED.
    int32_t cigar_sanity_check = opt->cigar_sanity_check;
    if (opt->use_param_ovr && sam->param_ovr && sam->param_ovr->cigar_sanity_check.over)
        cigar_sanity_check = sam->param_ovr->cigar_sanity_check.value;
    if (!cigar_sanity_check)
        return; //no check for this location

    cache_target (target, refseq, sam->seqid + 1, sam->pos + 1, sam->pos + sam->target_len);
    uint8_t* ref = target->data;
    int32_t ref_len = target->data_len;
    tmap_string_t* seq_str = seqs [sam->strand]->data.sam->seq;
    char* qry = seq_str->s;
    int32_t qry_len = seq_str ->l;

    uint32_t cigar_str_sz = 4 * sam->n_cigar + 1;
    char* cigar_str = alloca (cigar_str_sz);
    if (!bin_cigar_to_str (sam->cigar, sam->n_cigar, cigar_str, cigar_str_sz))
        tmap_bug ();

    if (sam->result.target_start != 0)
        tmap_failure ("Alignment sanity check error on read %s:\n  alignment box start is %d, should be 0", seq->data.sam->name->s, sam->result.target_start);

    uint32_t qry_pos = 0, ref_pos = 0;
    int32_t al_score = 0, nogap_score = 0, score_chg;
    uint32_t *opptr, *sent;
    uint32_t op, prev_op = 0, oplen, p; // prev_op to 0 to alleviate gcc's jealousy. It thinks it may be uninitialized downstream (logic guarantees otherwise, but it can see only syntax)
    for (opptr = sam->cigar, sent = opptr + sam->n_cigar; opptr != sent; ++opptr)
    {
        op = TMAP_SW_CIGAR_OP (*opptr);
        oplen = TMAP_SW_CIGAR_LENGTH (*opptr);
        if (opptr == sam->cigar && op != BAM_CSOFT_CLIP && sam->result.query_start != 0)
            tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  query offset is %d while cigar (%s) does not start with softclip", seq->data.sam->name->s, sam->result.query_start, cigar_str);
        switch (op)
        {
            case BAM_CMATCH:
                for (p = 0; p != oplen; ++p)
                {
                    if (ref_pos >= ref_len)
                        tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar (%s) takes more reference positions than target length (%d)", seq->data.sam->name->s, cigar_str, ref_len);
                    if (qry_pos >= qry_len)
                        tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar (%s) takes more query positions than query length (%d)", seq->data.sam->name->s, cigar_str, qry_len);
                    score_chg = (ref [ref_pos] == qry [qry_pos])?(opt->score_match):(-opt->pen_mm);
                    al_score += score_chg;
                    nogap_score += score_chg;
                    ++qry_pos;
                    ++ref_pos;
                }
                break;
            case BAM_CINS:
                if (qry_pos + oplen > qry_len)
                    tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar (%s) takes more query positions than query length (%d)", seq->data.sam->name->s, cigar_str, qry_len);
                al_score -= (opt->pen_gapo + opt->pen_gape*oplen);
                qry_pos += oplen;
                break;
            case BAM_CDEL:
                if (ref_pos + oplen > ref_len)
                    tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar (%s) takes more reference positions than target length (%d)", seq->data.sam->name->s, cigar_str, ref_len);
                al_score -= (opt->pen_gapo + opt->pen_gape*oplen);
                ref_pos += oplen;
                break;
            case BAM_CSOFT_CLIP:
                if (qry_pos + oplen > qry_len)
                    tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar (%s) takes more query positions than query length (%d)", seq->data.sam->name->s, cigar_str, qry_len);
                qry_pos += oplen;
                if (opptr == sam->cigar) // first operation
                {
                    if (qry_pos != sam->result.query_start)
                        tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar (%s) has starting softclip of size %d, while query start is %d", seq->data.sam->name->s, cigar_str, sam->result.query_start, oplen);
                }
                else if (opptr + 1 == sent) // last operation
                {
                    if (sam->result.query_end + 1 + oplen != qry_len)
                        tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar (%s) has ending softclip of size %d, while query alignment box ends %d bases before end", seq->data.sam->name->s, cigar_str, oplen, qry_len - (sam->result.query_end + 1));
                }
                else // SOFTCLIP not at the edge
                    tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar (%s) has internal softclip at position %d", seq->data.sam->name->s, cigar_str, opptr - sam->cigar);
                break;
            default:
                tmap_bug ();
        }
        if (opptr != sam->cigar && prev_op == op)
            tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar (%s) have consecutive cigar operations of the same type at position %d", seq->data.sam->name->s, cigar_str, opptr - sam->cigar - 1);
        prev_op = op;
    }
    if (qry_pos != qry_len)
        tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar (%s) takes %d query bases while query len is %d", seq->data.sam->name->s, cigar_str, qry_pos, qry_len);
    if (sam->n_cigar != 0 && prev_op != BAM_CSOFT_CLIP && sam->result.query_end + 1 != qry_len)
        tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  alignment box's query end (%d) does not match (query_len-1)(%d)", seq->data.sam->name->s, cigar_str, sam->result.query_end, qry_len-1);
    if (ref_pos != sam->target_len)
        tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar (%s) takes %d reference bases while target len is %d", seq->data.sam->name->s, cigar_str, ref_pos, ref_len);
    if (ref_pos != sam->result.target_end + 1)
        tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT), "Alignment sanity check error on read %s:\n  cigar target size (%d) does not match (alignment box + 1)(%d)", seq->data.sam->name->s, ref_pos, sam->result.target_end + 1);

#define TMIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define TMAX(X, Y) (((Y) < (X)) ? (X) : (Y))

    // check that original alignment is consistent with post-processed one
    // check if alignments differ
    if (sam->n_cigar != sam->n_orig_cigar || sam->pos != sam->orig_pos || 0 != memcmp (sam->cigar, sam->orig_cigar, sam->n_cigar * sizeof (*(sam->cigar))))
    {
#define TOTAL_OVERLAP_CHECK
#ifdef TOTAL_OVERLAP_CHECK
        {
            #define MIN_OVERLAP_FACTOR 2
            // compute total overlap length and total match lengths
            // use mergewalk to avoid quadratic runtime
            int32_t r1_off = sam->orig_pos, q1_off = 0; // reference and query offset at current match in cigar1 (original)
            int32_t r2_off = sam->pos, q2_off = 0;      // reference and query offset at current match in cigar2 (post-processed)
            uint32_t *opptr1 = sam->orig_cigar, *sent1 = opptr1 + sam->n_orig_cigar; // current operation pointer and sentinel for interating over cigar1 (original)
            uint32_t *opptr2, *sent2 = sam->cigar + sam->n_cigar; // current operation pointer and sentinel for interating over cigar2 (post-processed)
            uint32_t *tail2 = sam->cigar; // index in cigar2 (new) from which to start looking for overlap with current match in cigar1 (orig)
            int32_t tail2_q_off = 0, tail2_r_off = r2_off; // match offsets at the beginning of tail2 match
            int32_t op1, oplen1, op2, oplen2; // current cigar operations and operation lengths
            int32_t match1_tot = 0, overlap_tot = 0;
            int32_t stop, tail2_recorded; // flags for inner loop
            for (; opptr1 != sent1; ++opptr1) // iterate over cigar1 (once)
            {
                op1 = TMAP_SW_CIGAR_OP (*opptr1), oplen1 = TMAP_SW_CIGAR_LENGTH (*opptr1);
                switch (op1)
                {
                    case BAM_CINS:
                        q1_off += oplen1;
                        break;
                    case BAM_CDEL:
                        r1_off += oplen1;
                        break;
                    case BAM_CSOFT_CLIP:
                        q1_off += oplen1;
                        break;
                    case BAM_CMATCH:
                        // advance along cigar 2 from last remembered 'tail2' position until overlap with op1 is possible (until beg2 < end1)
                        stop = 0, tail2_recorded = 0;
                        for (opptr2 = tail2, q2_off = tail2_q_off, r2_off = tail2_r_off; opptr2 != sent2 && !stop; ++opptr2)
                        {
                            op2 = TMAP_SW_CIGAR_OP (*opptr2), oplen2 = TMAP_SW_CIGAR_LENGTH (*opptr2);
                            switch (op2)
                            {
                                case BAM_CINS:
                                    q2_off += oplen2;
                                    break;
                                case BAM_CDEL:
                                    r2_off += oplen2;
                                    break;
                                case BAM_CSOFT_CLIP:
                                    q2_off += oplen2;
                                    break;
                                case BAM_CMATCH:
                                    // check if overlap still possible (if beg2 < end1), break if not
                                    if (r2_off >= r1_off + oplen1)
                                    {
                                        stop = 1;
                                        break;
                                    }
                                    // remember where end2 overruned end1 as 'tail2' - these reads can potentially overlap next cigar1 match
                                    if (!tail2_recorded && r2_off + oplen2 > r1_off + oplen1)
                                    {
                                        tail2 = opptr2;
                                        tail2_r_off = r2_off;
                                        tail2_q_off = q2_off;
                                        tail2_recorded = 1;
                                    }
                                    // if diagonal is same, compute overlap and add to total
                                    if (r2_off - q2_off == r1_off - q1_off)
                                    {
                                        int32_t largest_beg = TMAX (r2_off, r1_off);
                                        int32_t smallest_end = TMIN (r2_off + oplen2, r1_off + oplen1);
                                        int32_t overlap = smallest_end - largest_beg;
                                        if (overlap > 0)
                                            overlap_tot += overlap;
                                    }
                                    r2_off += oplen2;
                                    q2_off += oplen2;
                                    break;
                                default:
                                    tmap_bug ();
                            }
                        }
                        q1_off += oplen1;
                        r1_off += oplen1;
                        match1_tot += oplen1;
                        break;
                    default:
                        tmap_bug ();
                }
            }
            // now compute post-processed alignment total match length - it is not trivial to do this in above loop as each match in pp cigar may be seen several times
            int32_t match2_tot = 0;
            for (opptr2 = sam->cigar, sent2 = opptr2 + sam->n_cigar; opptr2 != sent2; ++opptr2)
                if (TMAP_SW_CIGAR_OP (*opptr2) == BAM_CMATCH)
                    match2_tot += TMAP_SW_CIGAR_LENGTH (*opptr2);
            {
                if (overlap_tot * MIN_OVERLAP_FACTOR < TMIN (match1_tot, match2_tot))
                {
                    uint32_t orig_cigar_str_sz = 4 * sam->n_orig_cigar + 1;
                    char* orig_cigar_str = alloca (orig_cigar_str_sz);
                    if (!bin_cigar_to_str (sam->orig_cigar, sam->n_orig_cigar, orig_cigar_str, orig_cigar_str_sz))
                        tmap_bug ();
                    tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT_ALIGN), "Alignment sanity check error on read %s:\n  post-processed alignment %s at position %d of %s (%s) has insufficient overlap with the original one %s at position %d: overlap=%d, orig_match_len=%d, pp_match_len=%d", seq->data.sam->name->s, cigar_str, sam->pos, refseq->annos [sam->seqid].name->s, (sam->strand?"rev":"fwd"), orig_cigar_str, sam->orig_pos, overlap_tot, match1_tot, match2_tot);
                }
#if DEBUG_ALIGNMENT_OVERLAP
                else
                {
                    uint32_t orig_cigar_str_sz = 4 * sam->n_orig_cigar + 1;
                    char* orig_cigar_str = alloca (orig_cigar_str_sz);
                    if (!bin_cigar_to_str (sam->orig_cigar, sam->n_orig_cigar, orig_cigar_str, orig_cigar_str_sz))
                        tmap_bug ();
                    tmap_conderr (0, "Alignment sanity check Ok on read %s:\n  post-processed alignment %s at position %d has sufficient overlap with the original one %s at position %d: overlap=%d, match_len=%d", seq->data.sam->name->s, cigar_str, sam->pos, orig_cigar_str, sam->orig_pos, overlap_tot, match1_tot);
                }
#endif
            }
        }
#endif
// #define LONG_MATCH_OVERLAP_CHECK
#ifdef LONG_MATCH_OVERLAP_CHECK
        {
            // at least one long match should overlap significantly
            #define MIN_LONG_LEN 30
            #define MIN_CHECKED_MATCH_LEN 16
            #define MIN_OVERLAP_FACTOR 2
            int32_t r1_off = sam->orig_pos, r2_off, q1_off = 0, q2_off;
            uint32_t *opptr1, *sent1, *opptr2, *sent2;
            int32_t op1, oplen1, op2, oplen2;
            int32_t has_overlap = 0, has_long = 0;
            for (opptr1 = sam->orig_cigar, sent1 = opptr1 + sam->n_orig_cigar; opptr1 != sent1 && !has_overlap; ++opptr1)
            {
                op1 = TMAP_SW_CIGAR_OP (*opptr1);
                oplen1 = TMAP_SW_CIGAR_LENGTH (*opptr1);
                switch (op1)
                {
                    case BAM_CINS:
                        q1_off += oplen1;
                        break;
                    case BAM_CDEL:
                        r1_off += oplen1;
                        break;
                    case BAM_CSOFT_CLIP:
                        q1_off += oplen1;
                        break;
                    case BAM_CMATCH:
                        {
                            if (oplen1 >= MIN_LONG_LEN)
                                has_long = 1;
                            if (oplen1 > MIN_CHECKED_MATCH_LEN)
                            {
                                r2_off = sam->pos, q2_off = 0, has_overlap = 0;
                                for (opptr2 = sam->cigar, sent2 = opptr2 + sam->n_cigar ; opptr2 != sent2 && !has_overlap; ++opptr2)
                                {
                                    op2 = TMAP_SW_CIGAR_OP (*opptr2);
                                    oplen2 = TMAP_SW_CIGAR_LENGTH (*opptr2);
                                    switch (op2)
                                    {
                                        case BAM_CINS:
                                            q2_off += oplen2;
                                            break;
                                        case BAM_CDEL:
                                            r2_off += oplen2;
                                            break;
                                        case BAM_CSOFT_CLIP:
                                            q2_off += oplen2;
                                            break;
                                        case BAM_CMATCH:
                                            // check if match long enough and overlaps
                                            if (r1_off - q1_off == r2_off - q2_off) // same diagonal  && r1_off < r2_off + oplen2 && r2_off < r1_off + oplen1) // ref intervals intersect
                                            {
                                                // find the overlap size
                                                int32_t largest_beg = TMAX (r1_off, r2_off);
                                                int32_t smallest_end = TMIN (r1_off + oplen1, r2_off + oplen2);
                                                int32_t overlap = smallest_end - largest_beg;
                                                int32_t to_cover = oplen1;
                                                if (oplen2 * MIN_OVERLAP_FACTOR >= MIN_CHECKED_MATCH_LEN && oplen2 < oplen1)
                                                    to_cover = oplen2;
                                                if (overlap * MIN_OVERLAP_FACTOR >= to_cover)
                                                    has_overlap = 1;
                                            }
                                            q2_off += oplen2;
                                            r2_off += oplen2;
                                            break;
                                        default:
                                            tmap_bug ();
                                    }
                                }
                            }
                        }
                        q1_off += oplen1;
                        r1_off += oplen1;
                        break;
                    default:
                        tmap_bug ();
                }
            }
            if (has_long && !has_overlap) // some of long original alignment batches found to have no good overlapping match in final alignment
            {
                uint32_t orig_cigar_str_sz = 4 * sam->n_orig_cigar + 1;
                char* orig_cigar_str = alloca (orig_cigar_str_sz);
                if (!bin_cigar_to_str (sam->orig_cigar, sam->n_orig_cigar, orig_cigar_str, orig_cigar_str_sz))
                    tmap_bug ();
                tmap_conderr ((cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT_ALIGN), "Alignment sanity check error on read %s:\n  post-processed alignment %s at position %d has insufficient overlap with the original one %s at position %d", seq->data.sam->name->s, cigar_str, sam->pos, orig_cigar_str, sam->orig_pos);
            }
        }
#endif
    }

    if (cigar_sanity_check == TMAP_MAP_SANITY_WARN_ALL ||
        cigar_sanity_check == TMAP_MAP_SANITY_ERR_CONTENT_WARN_ALIGN_SCORE ||
        cigar_sanity_check >= TMAP_MAP_SANITY_ERR_CONTENT_ALIGN_WARN_SCORE)
    {
        if (nogap_score < 0)
            tmap_conderr ((cigar_sanity_check == TMAP_MAP_SANITY_ERR_ALL), "Alignment sanity check error on read %s:\n  Normal alignment score less gaps (%d) is negative (original score %d, cigar %s): possible misalignment", seq->data.sam->name->s, nogap_score, sam->result.score_fwd, cigar_str);
        else if (nogap_score < (sam->result.score_fwd / 2))
            tmap_conderr ((cigar_sanity_check == TMAP_MAP_SANITY_ERR_ALL), "Alignment sanity check error on read %s:\n  Normal alignment score less gaps (%d) is below 1/2 of unadjusted alignment score (%d), cigar %s: possible misalignment", seq->data.sam->name->s, nogap_score, sam->result.score_fwd, cigar_str);
    }
}


// updates alignment box (result) from cigar, pos and target_len
// assumes cigar covers entire query, does not check with actual query
void tmap_map_update_alignment_box (tmap_map_sam_t* sam)
{
    sam->result.target_start = 0;
    sam->result.target_end = sam->target_len - 1;
    sam->result.query_start = 0;
    uint32_t *cp, *sent, op, oplen;
    for (cp = sam->cigar, sent = cp + sam->n_cigar; cp != sent; ++cp)
    {
        op = TMAP_SW_CIGAR_OP (*cp);
        oplen = TMAP_SW_CIGAR_LENGTH (*cp);
        switch (op)
        {
            case BAM_CMATCH:
            case BAM_CDEL:
                sam->result.query_end += oplen;
                break;
            case BAM_CSOFT_CLIP:
                if (cp == sam->cigar) // first SC
                    sam->result.query_start = sam->result.query_end = oplen;
            default:
                break;
        }
    }
}


void cigar_log
(
    const uint32_t* cigar,
    unsigned cigar_sz
)
{
    const uint32_t* sent;
    for (sent = cigar+cigar_sz; cigar != sent; ++cigar)
    {
        uint32_t curop = bam_cigar_op (*cigar);
        uint32_t count = bam_cigar_oplen (*cigar);
        char schar;
        switch (curop)
        {
            case BAM_CHARD_CLIP:
                schar = 'H';
                break;
            case BAM_CSOFT_CLIP: // skip
                schar = 'S';
                break;
            case BAM_CMATCH:
                schar = 'M';
                break;
            case BAM_CEQUAL:
                schar = '=';
                break;
            case BAM_CDIFF:
                schar = '#';
                break;
            case BAM_CINS:
                schar = 'I';
                break;
            case BAM_CDEL:
                schar = 'D';
                break;
            default:
                schar = '?';
        }
        tmap_log ("%c%d", schar, count);
    }
}

uint32_t
cigar_to_batches
(
    const uint32_t* cigar,
    uint32_t cigar_sz,
    uint32_t* x_clip,
    AlBatch* batches,
    uint32_t max_batches
)
{
    uint32_t xpos = 0, ypos = 0;  // x is query, y is reference
    AlBatch* curb = batches;
    const uint32_t* sent = cigar + cigar_sz;
    for (; cigar != sent && curb - batches != max_batches; ++cigar)
    {
        uint32_t curop = bam_cigar_op (*cigar);
        uint32_t count = bam_cigar_oplen (*cigar);
        switch (curop)
        {
            case BAM_CHARD_CLIP: // skip
            case BAM_CSOFT_CLIP: // skip
                if (x_clip) *x_clip += count;
                break;
            case BAM_CMATCH:
            case BAM_CEQUAL:
            case BAM_CDIFF:
                curb->xpos = xpos;
                curb->ypos = ypos;
                curb->len = count;
                xpos += count;
                ypos += count;
                ++curb;
                x_clip = NULL;
                break;
            case BAM_CINS:
                xpos += count;
                x_clip = NULL;
                break;
            case BAM_CDEL:
                ypos += count;
                x_clip = NULL;
                break;
            default:
                ;
        }
    }
    return curb - batches;
}

#define MAXSTRLEN 256
#define NUMSTRLEN 11

static const char* IB = "ACGTNBDHKMRSVWYN";

void log_batches
(
    const char* xseq,
    unsigned xlen,
    uint8_t xrev,
    const char* yseq,
    unsigned ylen,
    uint8_t yrev,
    const AlBatch *b_ptr,
    int b_cnt,
    unsigned xoff,
    unsigned yoff
)
{
    const unsigned margin = 0, width = 120, zero_based = 1;

    unsigned slen = 0;
    unsigned blen = 0;
    unsigned x = b_ptr->xpos;
    unsigned y = b_ptr->ypos;
    unsigned xstart = x;
    unsigned ystart = y;
    unsigned char xc, yc;
    char s[3][MAXSTRLEN];

    assert (width < MAXSTRLEN);


    while (b_cnt > 0)
    {
        xc = xseq [x];
        yc = yseq [y];

        // special handling for (x < b_ptr->xpos && y < b_ptr->ypos)
        // treating as batch with special match symbols

        if (x < b_ptr->xpos && y < b_ptr->ypos)
        {
            s[0][slen] = (xc>='A')?xc:IB[xc];
            s[2][slen] = (yc>='A')?yc:IB[yc];
            s[1][slen] = '#';
            x++, y++, slen++;
        }
        // X insert
        else if (x < b_ptr->xpos)
        {
            s[0][slen] = (xc>='A')?xc:IB[xc];
            s[1][slen] = ' ';
            s[2][slen] = '-';
            x++, slen++;
        }
        // Y insert
        else if (y < b_ptr->ypos)
        {
            s[0][slen] = '-';
            s[1][slen] = ' ';
            s[2][slen] = (yc>='A')?yc:IB[yc];
            y++, slen++;
        }
        // emit text batch
        else if (blen < b_ptr->len)
        {
            s[0][slen] = (xc>='A')?xc:IB[xc];
            s[2][slen] = (yc>='A')?yc:IB[yc];
            s[1][slen] = (toupper (xc) == toupper (yc) || toupper (xc) == 'N' || toupper (yc) == 'N') ? '*' : ' ';
            x++, y++, slen++, blen++;
        }
        else
            blen = 0, b_cnt--, b_ptr++;

        //print accumulated lines
        if ((slen + NUMSTRLEN > width) || b_cnt <= 0)
        {
            //null terminate all strings
            for (int i = 0; i < 3; i++)
                s[i][slen] = 0;

            unsigned xdisp = (xrev ? xlen - xstart - 1 : xstart) + xoff + (zero_based ? 0 : 1);
            unsigned ydisp = (yrev ? ylen - ystart - 1 : ystart) + yoff + (zero_based ? 0 : 1);

            tmap_log ("\n%*s%*d %s", margin, "", NUMSTRLEN, xdisp, s[0]);
            tmap_log ("\n%*s%*s %s", margin, "", NUMSTRLEN, "", s[1]);
            tmap_log ("\n%*s%*d %s\n", margin, "", NUMSTRLEN, ydisp, s[2]);

            xstart = x, ystart = y, slen = 0;
        }
    }
}


void tmap_map_log_text_align
(
    const char* preceed,
    uint32_t* cigar,
    uint32_t n_cigar,
    const char* query,
    uint32_t query_len,
    uint32_t forward,
    const char* ref,
    uint32_t ref_off
)
{
    const unsigned MAX_BATCHNO = 100;
    AlBatch batches [MAX_BATCHNO];
    uint32_t q_clip = 0;
    int bno =  cigar_to_batches (cigar, n_cigar, &q_clip, batches, MAX_BATCHNO);

    uint32_t q_len_cigar, r_len_cigar;
    seq_lens_from_bin_cigar (cigar, n_cigar, &q_len_cigar, &r_len_cigar);

    if (preceed)
        tmap_log (preceed);
    log_batches (query+q_clip, query_len - q_clip, !forward, (const char*) ref, r_len_cigar, 0, batches, bno, q_clip, ref_off);
}
