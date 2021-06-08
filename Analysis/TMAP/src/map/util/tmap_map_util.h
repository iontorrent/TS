/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TMAP_MAP_UTIL_H
#define TMAP_MAP_UTIL_H

#include <sys/types.h>
#include "../../util/tmap_rand.h"
#include "../../sw/tmap_sw.h"
#include "../../sw/tmap_fsw.h"
#include "../../sw/tmap_vsw.h"
#include "tmap_map_opt.h"
#include "tmap_map_stats.h"
#include "tmap_map_locopt.h"

#if defined (__cplusplus)
extern "C"
{
#endif

#define __map_util_gen_ap(par, opt) do { \
    int32_t i; \
    for(i=0;i<ACTGN_MATRIX_SIZE;++i) { \
        (par).matrix[i] = -(opt)->pen_mm; \
    } \
    for(i=0;i<4;++i) { \
        (par).matrix[i*5+i] = (opt)->score_match; \
    } \
    (par).gap_open = (opt)->pen_gapo; (par).gap_ext = (opt)->pen_gape; \
    (par).gap_end = (opt)->pen_gape; \
    (par).row = 5; \
    (par).band_width = (opt)->bw; \
} while(0)

#define __tmap_map_util_reverse_soft_clipping(_sc) \
  (((_sc) == TMAP_MAP_OPT_SOFT_CLIP_LEFT) ? \
   TMAP_MAP_OPT_SOFT_CLIP_RIGHT : \
   (((_sc) == TMAP_MAP_OPT_SOFT_CLIP_RIGHT) ? \
    TMAP_MAP_OPT_SOFT_CLIP_LEFT : (_sc)))

/*! 
  Auxiliary data for map1
  */
typedef struct {
    uint16_t n_mm;  /*!< the current number of mismatches  */
    uint16_t n_gapo;  /*!< the current number of gap opens */
    uint16_t n_gape;  /*!< the current number of gap extensions */
    uint16_t aln_ref;  /*!< the number of reference bases in the alignment */
    uint32_t num_all_sa;  /*!< the number of hits produced by map1 (though fewer may be reported due to -b) */
} tmap_map_map1_aux_t;

/*! 
  Auxiliary data for map2
  */
typedef struct {
    uint16_t XF:2;  /*!< support for the forward/reverse alignment (1-forward 2-reverse 3-both) */
    uint16_t flag:1; /*!< 0 for non-repetitive hit, 1 for repetitive hit */
    uint16_t XE:13;  /*!< the number of supporting seeds */
    int32_t XI;  /*!< the suffix interval size */
} tmap_map_map2_aux_t;

/*! 
  Auxiliary data for map3
  */
typedef struct {
    void *ptr; // NULL
} tmap_map_map3_aux_t;

/*! 
  Auxiliary data for map4
  */
typedef struct {
    void *ptr; // NULL
} tmap_map_map4_aux_t;

/*! 
  Auxiliary data for map3
  */
typedef struct {
    void *ptr; // NULL
} tmap_map_map_vsw_aux_t;

/*!
  General data structure for holding a mapping; for easy outputting to the SAM format
  */
typedef struct {
    uint32_t algo_id:16; /*!< the algorithm id used to obtain this hit */
    uint32_t algo_stage:16; /*!< the algorithm id used to obtain this hit */
    uint8_t strand; /*!< the strand */
    uint32_t seqid;  /*!< the sequence index (0-based) */
    uint32_t pos; /*!< the position (0-based) */
    int16_t mapq; /*!< the mapping quality */
    int32_t score; /*!< the alignment score */
    int32_t ascore;  /*!< the base alignment score (SFF only) */
    int32_t pscore;  /*!< the pairing base alignment score (pairing only) */
    uint8_t proper_pair:1;  /*!< 0 if not a proper pair, 1 otherwise */
    uint8_t repr_hit:1; /*!< 1 if a representative repetitive hit, 0 otherwise */
    double num_stds;  /*!< the number of standard deviations from the mean insert size */
    int16_t pmapq; /*!< the pairing mapping quality */
    int32_t score_subo; /*!< the alignment score of the sub-optimal hit */
    int32_t n_cigar; /*!< the number of cigar operators */
    uint32_t *cigar; /*!< the cigar operator array */
    int32_t n_orig_cigar;
    uint32_t *orig_cigar; 
    uint32_t orig_pos;
    uint16_t target_len; /*!< internal variable, the target length estimated by the seeding step */ 
    uint16_t n_seeds; /*!< the number seeds in this hit */
    uint16_t fivep_offset; /*!< number of additional ref bases aligned if 5' not softclipped */
    uint32_t mapper_pos;
    uint32_t mapper_tlen;
    uint32_t ampl_start; // start of covering ampicon 
    uint32_t ampl_end; // end of covering amplicon
    tmap_map_locopt_t* param_ovr; // pointer to override parameters; ptr to default for no override
    tmap_map_endstat_p_t read_ends; // structure holding data on read end positions statistics for the amplicon this read is mapped to.
    union {
        tmap_map_map1_aux_t *map1_aux; /*!< auxiliary data for map1 */
        tmap_map_map2_aux_t *map2_aux; /*!< auxiliary data for map2 */
        tmap_map_map3_aux_t *map3_aux; /*!< auxiliary data for map3 */
        tmap_map_map4_aux_t *map4_aux; /*!< auxiliary data for map4 */
        tmap_map_map_vsw_aux_t *map_vsw_aux; /*!< auxiliary data for map_vsw */
    } aux;
    // for bounding the alignment with vectorized SW
    tmap_vsw_result_t result; /*!< the VSW boundaries (query/target start/end and scores) */
} tmap_map_sam_t;

/*!
  Stores multiple mappings for a given read
  */
typedef struct {
    int32_t n; /*!< the number of hits */
    int32_t max; /*!< the number of hits before filtering */
    tmap_map_sam_t *sams; /*!< array of hits */
} tmap_map_sams_t;

/*!
  The multi-end record structure
  */
typedef struct {
    tmap_map_sams_t **sams; /*!< the sam records */
    int32_t n; /*!< the number of records (multi-end) */
} tmap_map_record_t;

/*!
  Stores multiple mappings for BAMs
  */
typedef struct {
    bam1_t **bams; /*!< the bam records */
    int32_t n; /*!< the number of hits */
} tmap_map_bam_t;

/*!
  The multi-end BAM record structure
  */
typedef struct {
    tmap_map_bam_t **bams;  /*!< the bam hits */
    int32_t n; /*!< the number of records (multi-end) */
} tmap_map_bams_t;

/*!
 Control structure for reference sequence buffer
 */
typedef struct {
    uint8_t* buf;        /*!< pointer to the address of the memory buffer for unpacked reference sequence */
    uint32_t buf_sz;     /*!< pointer to the variable contining presently allocated size of target_buf */
    uint8_t* data;       /*!< pointer into buffer where actual requested fragment starts */
    uint32_t data_len;   /*!< length of last requested data */
    uint32_t position;   /*!< coordinate of the data start in the reference fragment being cached (0-based!) */
    uint32_t seqid;      /*!< sequence Id for the fragment stored in data member, 1-based */
    uint32_t seq_start;  /*!< offset of the first base of fragment stored in buf, 1-based */
    uint32_t seq_end;    /*!< offset of the last base stored in buf, 1-based */
} ref_buf_t;


/*!
 * initializes ref_buf_t for proper memory management (on-heap pointers to NULLs)
 * */
void target_cache_init (ref_buf_t* target);
/*!
 * frees memory held by ref_buf_t
 * */
void target_cache_free (ref_buf_t* target);
/*!
  extracts specified portion of the reference data and populates the ref_buf_t structure
  @param dest (ref_buf_t*) the reference cache control structure
  @param refseq (tmap_refseq_t *) pointer to the the reference server control structure
  @param seqid (uint32_t) sequence id to cache, 1-based, 
  @param seq_start (uint32_t) position of first base in a fragment to cache, 1-based
  @param seq_end (uint32_t) poition of last base of a fragment to cache, 1-based
  */
void cache_target (ref_buf_t* target, tmap_refseq_t *refseq, uint32_t seqid, uint32_t seq_start, uint32_t seq_end);


/*!
  initializes
  @param  s  the mapping structure
  */
void
tmap_map_sam_init(tmap_map_sam_t *s);

/*!
  make a copy of src and store it in dest
  @param  dest  the destination record
  @param  src   the source record
  */
void
tmap_map_sam_copy(tmap_map_sam_t *dest, tmap_map_sam_t *src);

/*!
  allocates memory for the auxiliary data specific to the algorithm specified by algo_id
  @param  s        the mapping structurem
  */
void
tmap_map_sam_malloc_aux(tmap_map_sam_t *s);

/*!
  destroys auxiliary data for the given mapping structure
  @param  s  the mapping structure
  */
void
tmap_map_sam_destroy_aux(tmap_map_sam_t *s);

/*!
  destroys the given mapping structure, including auxiliary data
  @param  s  the mapping structure
  */ 
void
tmap_map_sam_destroy(tmap_map_sam_t *s);

/*!
  allocate memory for an empty mapping structure, with no auxiliary data
  @param   prev  copies over the max from prev
  @return        a pointer to the initialized memory
  */
tmap_map_sams_t *
tmap_map_sams_init(tmap_map_sams_t *prev);

/*!
  reallocate memory for mapping structures; does not allocate auxiliary data
  @param  s  the mapping structure
  @param  n  the new number of mappings
  */
void
tmap_map_sams_realloc(tmap_map_sams_t *s, int32_t n);

/*!
  destroys memory associated with these mappings
  @param  s  the mapping structure
  */
void
tmap_map_sams_destroy(tmap_map_sams_t *s);

/*!
  Initializes a new multi-end mapping structure
  @param  num_ends  the number of ends in this record
  @return  the new multi-end mapping structure
 */
tmap_map_record_t*
tmap_map_record_init(int32_t num_ends);

/*!
  Clones a new multi-end mapping structure
  @param  src  the multi-end mapping structure to clone
  @return  the new multi-end mapping structure
 */
tmap_map_record_t*
tmap_map_record_clone(tmap_map_record_t *src);

/*!
  Merges the mappings of two multi-end mappings 
  @param  src   the multi-end mapping structure destination
  @param  dest  the multi-end mapping structure to merge from
 */
void
tmap_map_record_merge(tmap_map_record_t *dest, tmap_map_record_t *src);

/*!
  Destroys a record structure
  @param  record  the mapping structure
 */
void 
tmap_map_record_destroy(tmap_map_record_t *record);

// TODO
tmap_map_bam_t*
tmap_map_bam_init(int32_t n);

// TODO
void
tmap_map_bam_destroy(tmap_map_bam_t *b);

// TODO
tmap_map_bams_t*
tmap_map_bams_init(int32_t n);

// TODO
void
tmap_map_bams_destroy(tmap_map_bams_t *b); 

/*!
  merges src into dest
  @param  dest  the destination mapping structure
  @param  src   the source mapping structure
  */
void
tmap_map_sams_merge(tmap_map_sams_t *dest, tmap_map_sams_t *src);

/*!
  clones src
  @param  src   the source mapping structure
  */
tmap_map_sams_t *
tmap_map_sams_clone(tmap_map_sams_t *src);

/*!
  copies the source into the destination, nullifying the source
  @param  dest  the destination mapping structure
  @param  src   the source mapping structure
  */
void
tmap_map_sam_copy_and_nullify(tmap_map_sam_t *dest, tmap_map_sam_t *src);

/*!
  converts the alignment record to SAM/BAM records
  @param  seq           the original read sequence
  @param  refseq        the reference sequence
  @param  sams          the mappings to print
  @param  end_num       0 if there is no mate, 1 if this is the first fragment, 2 if the this is the last fragment
  @param  mates         the mate's mappings, NULL if there is no mate
  @param  sam_flowspace_tags  1 if SFF specific SAM tags are to be outputted, 0 otherwise
  @param  bidirectional  1 if a bidirectional SAM tag is to be added, 0 otherwise
  @param  seq_eq        1 if the SEQ field is to use '=' symbols, 0 otherwise
  @param  min_al_len    skip alignments shorter then min_al_len; do not skip anythong if min_al_len is 0
  @param  min_al_cov    skip alignments covering less then min_al_cov fraction of read; do not filter if min_al_cov is 0
  @param  min_identity  skip alignments with identity less then min_identity; do not filter if min_identity is 0
  @param  match_score   match score to use in identity filtering
  @param  filtered      pointer to the 64-bit filtering events counter
  @return  the BAM records for this read
  */
tmap_map_bam_t*
tmap_map_sams_print(tmap_seq_t *seq, tmap_refseq_t *refseq, tmap_map_sams_t *sams, int32_t end_num,
                    tmap_map_sams_t *mates, int32_t sam_flowspace_tags, int32_t bidirectional, int32_t seq_eq,
                    int32_t min_al_len, double min_coverage, double min_identity, int32_t match_score, uint64_t* filtered);

/*!
  keep only the mappings with the given score 
  @param  sams     the mappings to keep
  @param  algo_id  the algorithm identifier
  @param  score    the score to keep
  */
void
tmap_map_util_keep_score(tmap_map_sams_t *sams, int32_t algo_id, int32_t score);

/*!
  filters mappings based on the output mode
  @param  sams             the mappings to filter
  @param  aln_output_mode  the output mode
  @param  rand             the random number generator
  */
void
tmap_map_sams_filter(tmap_map_sams_t *sams, int32_t aln_output_mode, tmap_rand_t *rand);

/*!
  filters mappings based on the output mode
  @param  sams             the mappings to filter
  @param  aln_output_mode  the output mode
  @param  algo_id          the algorithm identifier
  @param  rand             the random number generator
  only filters mappings based on the algorithm id (none process all)
  */
void
tmap_map_sams_filter1(tmap_map_sams_t *sams, int32_t aln_output_mode, int32_t algo_id, tmap_rand_t *rand);

/*!
  filters mappings that pass both the scoring and mapping quality thresholds
  @param  sams       the mappings to filter
  @param  score_thr  the score threshold
  @param  mapq_thr   the mapping quality threshold
  */
void
tmap_map_sams_filter2(tmap_map_sams_t *sams, int32_t score_thr, int32_t mapq_thr);

/*!
  removes duplicate alignments that fall within a given window
  @param  sams        the mappings to adjust 
  @param  dup_window  the window size to cluster mappings
  @param  rand        the random number generator
  */
void
tmap_map_util_remove_duplicates(tmap_map_sams_t *sams, int32_t dup_window, tmap_rand_t *rand);

/*!
 Computes the mapping quality score from a small set of summary statistics.
 @param  seq_len          the sequence length
 @param  n_best           the number of best scores
 @param  best_score       the best score
 @param  n_best_subo      the number of best suboptimal scores
 @param  best_subo_score  the best suboptimal score
 @param  opt              the program parameters
 @return                  the mapping quality
 */
int32_t
tmap_map_util_mapq_score (int32_t seq_len, int32_t n_best, int32_t best_score, int32_t n_best_subo, int32_t best_subo_score, tmap_map_opt_t *opt);

/*!
 Computes the mapping quality from the mappings of multiple algorithms
 @param  sams     the sams to update
 @param  seq_len  the sequence length
 @param  opt      the program parameters
 @return          0 upon success, non-zero otherwise
 */
int32_t
tmap_map_util_mapq (tmap_map_sams_t *sams, int32_t seq_len, tmap_map_opt_t *opt, tmap_refseq_t *refseq);

/*!
  perform local alignment
  @details              only fills in the score, start and end of the alignments
  @param  refseq        the reference sequence
  @param  seq           the original query record
  @param  sams          the seeded sams
  @param  seqs          the query sequence (forward, reverse compliment, reverse, and compliment)
  @param  rand          the random number generator
  @param  opt           the program parameters
  @param  num_after_grouping used to return the number seeds after grouping
  @return               the locally aligned sams
  */

tmap_map_sams_t*
tmap_map_util_sw_gen_score
(
    tmap_refseq_t *refseq,
    tmap_seq_t *seq,
    tmap_map_sams_t *sams,
    tmap_seq_t **seqs,
    tmap_rand_t *rand,
    tmap_map_opt_t *opt,
    int32_t *num_after_grouping
);

void tmap_map_util_populate_sw_par_iupac_direct 
(
    tmap_sw_param_t* par, 
    int32_t score_match, 
    int32_t pen_mm, 
    int32_t pen_gapo, 
    int32_t pen_gape, 
    int32_t bw
);

void tmap_map_util_populate_sw_par_iupac 
(
    tmap_sw_param_t* par, 
    tmap_map_opt_t* opt
);

void tmap_map_util_populate_stage_sw_par 
(
    tmap_sw_param_t* par, 
    tmap_map_opt_t* opt
);

/*!
  find alignment starts for raw mappings
  @details              generates the cigar after tmap_map_util_sw_gen_score has been called
  @param  refseq        the reference sequence
  @param  sams          the seeded sams
  @param  seq           the original query sequence 
  @param  seqs          the query sequence (forward, reverse compliment, reverse, and compliment)
  @param  opt           the program parameters
  @param  target        reference data
  @param  stat          statistics
  @return               the locally aligned sams
  */

// Find alignment box for all mappings of a read
tmap_map_sams_t*
tmap_map_util_find_align_starts 
(
    tmap_refseq_t *refseq,      // reference server
    tmap_map_sams_t *sams,      // initial rough mapping 
    tmap_seq_t *seq,            // read
    tmap_seq_t **seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_map_opt_t *opt,        // tmap parameters
    ref_buf_t* target,          // reference data
    tmap_map_stats_t *stat      // statistics
);

// find amplion for locations of each mapping of a read
void
tmap_map_find_amplicons 
(
    uint32_t stage,             // tmap stage index
    tmap_map_opt_t* stage_opt,  // tmap stage options
    tmap_sw_param_t* def_par,   // stage SW parameters - needed for override check
    tmap_refseq_t *refseq,      // reference server
    tmap_map_sams_t *sams       // initial rough mapping 
);

// align all mappings of the read
void 
tmap_map_util_align 
(
    tmap_refseq_t *refseq,      // reference server
    tmap_map_sams_t *sams,      // mappings to compute alignments for
    tmap_seq_t **seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    ref_buf_t* target,          // reference data cache
    int32_t stage_ord,          // tmap processing stage index, needed for fetching stage SW parameter overrides
    tmap_sw_param_t* swpar,     // Smith-Waterman scoring parameters
    tmap_sw_path_t** path_buf,  // buffer for traceback path
    int32_t* path_buf_sz,       // used portion and allocated size of traceback path. 
    tmap_map_stats_t *stat      // statistics
);

// long indel rescue for all mappings of the read
void 
tmap_map_util_salvage_edge_indels 
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
);

// adjust softclips according to specs for all mappings of a read
void 
tmap_map_util_cure_softclips 
(
    tmap_map_sams_t *sams,      // mappings to compute alignments for
    tmap_seq_t **seqs          // array of size 4 that contains pre-computed inverse / complement combinations
);

// trim key for all mappings for a read
void 
tmap_map_util_trim_key 
(
    tmap_map_sams_t *sams,      // mappings to compute alignments for
    tmap_seq_t *seq,            // read
    tmap_seq_t **seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    tmap_refseq_t *refseq,      // reference server
    ref_buf_t* target,          // reference data cache
    tmap_map_stats_t *stat      // statistics
);

// end repair all mappings for a read
void 
tmap_map_util_end_repair_bulk 
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
);

// REPAiR all mappings for a read
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
);

# if 0
/*!
  perform local alignment
  @details              generates the cigar after tmap_map_util_sw_gen_score has been called
  @param  refseq        the reference sequence
  @param  sams          the seeded sams
  @param  seq           the original query sequence 
  @param  seqs          the query sequence (forward, reverse compliment, reverse, and compliment)
  @param  opt           the program parameters
  @param  stat          statistics
  @return               the locally aligned sams
  */
tmap_map_sams_t *
tmap_map_util_sw_gen_cigar (
    tmap_refseq_t *refseq,
    tmap_map_sams_t *sams, 
    tmap_seq_t *seq,
    tmap_seq_t **seqs,
    tmap_map_opt_t *opt,
    tmap_map_stats_t *stat
);
#endif

/*!
  re-aligns mappings in flow space
  @param  seq            the seq read sequence
  @param  sams           the mappings to adjust 
  @param  refseq         the reference sequence
  @param  bw             the band width
  @param  softclip_type  the soft clip type
  @param  score_thr      the alignment score threshold
  @param  score_match    the match score
  @param  pen_mm         the mismatch penalty
  @param  pen_gapo       the gap open penalty
  @param  pen_gape       the gap extension penalty
  @param  fscore         the flow penalty
  @param  use_flowgram   1 to use the flowgram if available, 0 otherwise
  @param  stage_fsw_use  stage-wide flowspace alignemnt flag
  @param  use_param_ovr  parameters override enabled flag
  @param  stat           tmap statistics
  @return  1 if successful, 0 otherwise
  */

int32_t tmap_map_util_fsw 
(
  tmap_seq_t* seq,
  tmap_map_sams_t* sams,
  tmap_refseq_t* refseq,
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
);


// updates alignment box (result) from cigar, pos and target_len
void tmap_map_update_alignment_box (tmap_map_sam_t* sam);


/*!
  turns softclip on 5' into non-clipped alignment
  @param  refseq         the reference sequence
  @param  sams           the mappings to adjust 
  @param  seq            the read
  @param  seqs           the 'alternate forms' of query sequences (forward, reverse compliment, reverse, and compliment)
  @param  target         reference data 
  @param  path_buf       buffer for traceback path
  @param  path_buf_sz    used portion and allocated size of traceback path. 
  @param  par            Smith-Waterman scoring parameters
  @param  opt            the program parameters
  @param  stat           tmap statistics
  @return  1 if 5-prime was found and replaced, 0 if there was no 5' softclip
  */

int32_t
tmap_map_util_remove_5_prime_softclip 
(
    tmap_refseq_t *refseq,      // reference server
    tmap_map_sam_t *dest_sam,   // mapping to fix
    tmap_seq_t *seq,            // read
    tmap_seq_t **seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    ref_buf_t* target,          // reference data cache
    tmap_sw_path_t** path_buf,  // buffer for traceback path
    int32_t* path_buf_sz,       // used portion and allocated size of traceback path. 
    int32_t stage_ord,          // index of TMAP processing stage (needed for fetching SW parameters override for a stage)
    tmap_sw_param_t* swpar,     // stage-wide Smith-Waterman scoring parameters
    tmap_map_opt_t *opt,        // tmap options (for this stage)
    tmap_map_stats_t *stat      // statistics
);

void 
cigar_sanity_check 
(
    tmap_refseq_t *refseq,      // reference server
    tmap_map_sam_t *dest_sam,   // mapping to fix
    tmap_seq_t *seq,            // read
    tmap_seq_t **seqs,          // array of size 4 that contains pre-computed inverse / complement combinations
    ref_buf_t* target,          // reference data cache
    tmap_map_opt_t *opt         // tmap options (for this stage)
);

typedef struct
{
    unsigned xpos;
    unsigned ypos;
    unsigned len;
} 
AlBatch;

void cigar_log 
(
    const uint32_t* cigar,
    unsigned cigar_sz
);

uint32_t cigar_to_batches 
(
    const uint32_t* cigar,
    uint32_t cigar_sz,
    uint32_t* x_clip,
    AlBatch* batches,
    uint32_t max_batches
);

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
);

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
);

int
tmap_map_get_amplicon
(
    tmap_refseq_t *refseq,
    int32_t seqid,
    uint32_t start,
    uint32_t end,
    uint32_t *ampl_start,
    uint32_t *ampl_end,
    tmap_map_locopt_t** overrides,
    tmap_map_endstat_p_t* ends,
    uint32_t strand
);

uint8_t
cache_sw_overrides 
(
    tmap_map_locopt_t* locopt,
    int32_t stage_ord,
    tmap_map_opt_t* stage_opt,
    tmap_sw_param_t* def_sw_par
);

#if defined (__cplusplus)
}
#endif

#endif // TMAP_MAP_UTIL_H
