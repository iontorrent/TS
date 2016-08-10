/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TMAP_MAP_STATS_H
#define TMAP_MAP_STATS_H

/*!
  The mapping statistics structure.
 */
typedef struct {
    uint64_t num_reads; /*!< the number of reads with at least one mapping */
    uint64_t num_with_mapping; /*!< the number of reads with at least one mapping */
    uint64_t num_after_seeding; /*!< the number of hits after seeding */
    uint64_t num_after_grouping; /*!< the number of hits after grouping */
    uint64_t num_after_scoring; /*!< the number of hits after scoring */
    uint64_t num_after_rmdup; /*!< the number of hits after duplicate removal */
    uint64_t num_after_filter; /*!< the number of hits after filtering */
    // realigner statistics
    uint64_t num_realign_invocations; /*!< the number of alignments ran through realignment procedure*/
    uint64_t num_realign_already_perfect; /*!< the number of alignments that considered already perfect and thus were not re-processed*/
    uint64_t num_realign_not_clipped; /*!< the number of alignments could not be clipped (due to size / edge proximity / HP constraints*/ 
    uint64_t num_realign_sw_failures; /*!< the number of realigner Smith-Waterman procedure failures*/ 
    uint64_t num_realign_unclip_failures; /*!< the number of realigner un-clipping procedure failures*/ 
    uint64_t num_realign_changed; /*!< the number of alignments that were modified adjusted by realigner*/ 
    uint64_t num_realign_shifted; /*!< the number of alignments for which the position on the reference was altered by realigner*/ 
    // tail repeat clipping stats
    uint64_t num_seen_tailclipped;
    uint64_t bases_seen_tailclipped;
    uint64_t num_tailclipped;
    uint64_t bases_tailclipped;
    uint64_t num_fully_tailclipped;
    uint64_t bases_fully_tailclipped;
    // statistics for realignment with context-dependent gap cost
    uint64_t num_hpcost_invocations;
    uint64_t num_hpcost_skipped;
    uint64_t num_hpcost_modified;
    uint64_t num_hpcost_shifted;
    // number of filtered alignments 
    uint64_t num_filtered_als;
} tmap_map_stats_t;

/*!
  @return  a new stats structure
 */
tmap_map_stats_t*
tmap_map_stats_init();

/*!
  @param  s  the mapping driver stats to destroy
 */
void
tmap_map_stats_destroy(tmap_map_stats_t *s);

/*!
  Adds the src stats to the dest stats
  @param  dest  the destination
  @param  src   the source
 */
void
tmap_map_stats_add(tmap_map_stats_t *dest, tmap_map_stats_t *src);

/*!
  Zeroes all counters 
  @param  s  the pointer to stats holding object
 */
void 
tmap_map_stats_zero(tmap_map_stats_t *s);

#endif // TMAP_MAP_STATS_H
