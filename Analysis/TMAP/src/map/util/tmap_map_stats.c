/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdlib.h>
#include <stdio.h>
#include <config.h>
#include <unistd.h>
#include <memory.h>
#include <string.h>
#include "../../util/tmap_error.h"
#include "../../util/tmap_alloc.h"
#include "../../util/tmap_definitions.h"
#include "tmap_map_stats.h"

tmap_map_stats_t*
tmap_map_stats_init()
{
  tmap_map_stats_t* r = tmap_calloc (1, sizeof(tmap_map_stats_t), "stats");

  // DVK - guarantee initialization to zero of all members (actually calloc does this anyway)
  tmap_map_stats_zero (r);

  return r;
}

void
tmap_map_stats_destroy(tmap_map_stats_t *s)
{
  free (s);
}

void 
tmap_map_stats_zero(tmap_map_stats_t *s)
{
  memset (s, 0, sizeof (tmap_map_stats_t));
}


void
tmap_map_stats_add(tmap_map_stats_t *dest, tmap_map_stats_t *src)
{
  int i;
  dest->num_reads += src->num_reads;
  dest->num_with_mapping += src->num_with_mapping;
  dest->num_after_seeding += src->num_after_seeding;
  dest->num_after_grouping += src->num_after_grouping;
  dest->num_after_scoring += src->num_after_scoring;
  dest->num_after_rmdup += src->num_after_rmdup;
  dest->num_after_filter += src->num_after_filter;

  dest->num_hpcost_invocations += src->num_hpcost_invocations;
  dest->num_hpcost_skipped += src->num_hpcost_skipped;
  dest->num_hpcost_modified += src->num_hpcost_modified;
  dest->num_hpcost_shifted += src->num_hpcost_shifted;

  dest->num_realign_invocations += src->num_realign_invocations;
  dest->num_realign_already_perfect += src->num_realign_already_perfect;
  dest->num_realign_not_clipped += src->num_realign_not_clipped;
  dest->num_realign_sw_failures += src->num_realign_sw_failures;
  dest->num_realign_unclip_failures += src->num_realign_unclip_failures;
  dest->num_realign_changed += src->num_realign_changed;
  dest->num_realign_shifted += src->num_realign_shifted;

  dest->reads_salvaged += src->reads_salvaged;
  for (i = 0; i != 4; ++i) dest->num_salvaged [i] += src->num_salvaged [i];
  for (i = 0; i != 4; ++i) dest->bases_salvaged_qry [i] += src->bases_salvaged_qry [i];
  for (i = 0; i != 4; ++i) dest->bases_salvaged_ref [i] += src->bases_salvaged_ref [i];
  for (i = 0; i != 4; ++i) dest->score_salvaged_total [i] += src->score_salvaged_total [i];

  dest->reads_end_repair_clipped += src->reads_end_repair_clipped;
  for (i = 0; i != 4; ++i) dest->num_end_repair_clipped [i] += src->num_end_repair_clipped [i];
  for (i = 0; i != 4; ++i) dest->bases_end_repair_clipped [i] += src->bases_end_repair_clipped [i];
  dest->reads_end_repair_extended += src->reads_end_repair_extended;
  for (i = 0; i != 4; ++i) dest->num_end_repair_extended [i] += src->num_end_repair_extended [i];
  for (i = 0; i != 4; ++i) dest->bases_end_repair_extended [i] += src->bases_end_repair_extended [i];
  for (i = 0; i != 4; ++i) dest->total_end_repair_indel [i] += src->total_end_repair_indel [i];

  for (i = 0; i != 2; ++i) dest->num_5_softclips [i] += src->num_5_softclips [i];
  for (i = 0; i != 2; ++i) dest->bases_5_softclips_qry [i] += src->bases_5_softclips_qry [i];
  for (i = 0; i != 2; ++i) dest->bases_5_softclips_ref [i] += src->bases_5_softclips_ref [i];
  for (i = 0; i != 2; ++i) dest->score_5_softclips_total [i] += src->score_5_softclips_total [i];

  dest->num_seen_tailclipped += src->num_seen_tailclipped;
  dest->bases_seen_tailclipped += src->bases_seen_tailclipped;
  dest->num_tailclipped += src->num_tailclipped;
  dest->bases_tailclipped += src->bases_tailclipped;
  dest->num_fully_tailclipped += src->num_fully_tailclipped;
  dest->bases_fully_tailclipped += src->bases_fully_tailclipped;

  dest->num_filtered_als += src->num_filtered_als;
}

void
tmap_map_stats_print(tmap_map_stats_t *s)
{
  fprintf (stderr, "num_reads=%llu\n", (unsigned long long int) s->num_reads);
  fprintf (stderr, "num_with_mapping=%llu\n", (unsigned long long int) s->num_with_mapping);
  fprintf (stderr, "num_after_seeding=%llu\n", (unsigned long long int) s->num_after_seeding);
  fprintf (stderr, "num_after_grouping=%llu\n", (unsigned long long int) s->num_after_grouping);
  fprintf (stderr, "num_after_scoring=%llu\n", (unsigned long long int) s->num_after_scoring);
  fprintf (stderr, "num_after_rmdup=%llu\n", (unsigned long long int) s->num_after_rmdup);
  fprintf (stderr, "num_after_filter=%llu\n", (unsigned long long int) s->num_after_filter);
  
  fprintf (stderr, "num_hpcost_invocations=%llu\n", (unsigned long long int) s->num_hpcost_invocations);
  fprintf (stderr, "num_hpcost_skipped=%llu\n", (unsigned long long int) s->num_hpcost_skipped);
  fprintf (stderr, "num_hpcost_modified=%llu\n", (unsigned long long int) s->num_hpcost_modified);
  fprintf (stderr, "num_hpcost_shifted=%llu\n", (unsigned long long int) s->num_hpcost_shifted);

  fprintf (stderr, "num_realign_invocations=%llu\n", (unsigned long long int) s->num_realign_invocations);
  fprintf (stderr, "num_realign_already_perfect=%llu\n", (unsigned long long int) s->num_realign_already_perfect);
  fprintf (stderr, "num_realign_not_clipped=%llu\n", (unsigned long long int) s->num_realign_not_clipped);
  fprintf (stderr, "num_realign_sw_failures=%llu\n", (unsigned long long int) s->num_realign_sw_failures);
  fprintf (stderr, "num_realign_unclip_failures=%llu\n", (unsigned long long int) s->num_realign_unclip_failures);
  fprintf (stderr, "num_realign_changed=%llu\n", (unsigned long long int) s->num_realign_changed);
  fprintf (stderr, "num_realign_shifted=%llu\n", (unsigned long long int) s->num_realign_shifted);


  fprintf (stderr, "num_salvaged [reads, fwd5', fwd3', rev5', rev3']=[%llu, %llu, %llu, %llu, %llu]\n", 
            (unsigned long long int) s->reads_salvaged,
            (unsigned long long int) s->num_salvaged [F5P],
            (unsigned long long int) s->num_salvaged [F3P],
            (unsigned long long int) s->num_salvaged [R5P],
            (unsigned long long int) s->num_salvaged [R3P]);

  fprintf (stderr, "bases_salvaged_qry [fwd5', fwd3', rev5', rev3']=[%llu, %llu, %llu, %llu]\n", 
            (unsigned long long int) s->bases_salvaged_qry [F5P],
            (unsigned long long int) s->bases_salvaged_qry [F3P],
            (unsigned long long int) s->bases_salvaged_qry [R5P],
            (unsigned long long int) s->bases_salvaged_qry [R3P]);

  fprintf (stderr, "bases_salvaged_ref [fwd5', fwd3', rev5', rev3']=[%llu, %llu, %llu, %llu]\n", 
            (unsigned long long int) s->bases_salvaged_ref [F5P],
            (unsigned long long int) s->bases_salvaged_ref [F3P],
            (unsigned long long int) s->bases_salvaged_ref [R5P],
            (unsigned long long int) s->bases_salvaged_ref [R3P]);

  fprintf (stderr, "score_salvaged_total [fwd5', fwd3', rev5', rev3']=[%lld, %lld, %lld, %lld]\n", 
            (long long int) s->score_salvaged_total [F5P],
            (long long int) s->score_salvaged_total [F3P],
            (long long int) s->score_salvaged_total [R5P],
            (long long int) s->score_salvaged_total [R3P]);

  fprintf (stderr, "num_end_repair_clipped [reads, fwd5', fwd3', rev5', rev3']=[%llu, %llu, %llu, %llu, %llu]\n", 
            (unsigned long long int) s->reads_end_repair_clipped,
            (unsigned long long int) s->num_end_repair_clipped [F5P],
            (unsigned long long int) s->num_end_repair_clipped [F3P],
            (unsigned long long int) s->num_end_repair_clipped [R5P],
            (unsigned long long int) s->num_end_repair_clipped [R3P]);

  fprintf (stderr, "bases_end_repair_clipped [fwd5', fwd3', rev5', rev3']=[%llu, %llu, %llu, %llu]\n", 
            (unsigned long long int) s->bases_end_repair_clipped [F5P],
            (unsigned long long int) s->bases_end_repair_clipped [F3P],
            (unsigned long long int) s->bases_end_repair_clipped [R5P],
            (unsigned long long int) s->bases_end_repair_clipped [R3P]);

  fprintf (stderr, "num_end_repair_extended [reads, fwd5', fwd3', rev5', rev3']=[%llu, %llu, %llu, %llu, %llu]\n", 
            (unsigned long long int) s->reads_end_repair_extended,
            (unsigned long long int) s->num_end_repair_extended [F5P],
            (unsigned long long int) s->num_end_repair_extended [F3P],
            (unsigned long long int) s->num_end_repair_extended [R5P],
            (unsigned long long int) s->num_end_repair_extended [R3P]);

  fprintf (stderr, "bases_end_repair_extended [fwd5', fwd3', rev5', rev3']=[%llu, %llu, %llu, %llu]\n", 
            (unsigned long long int) s->bases_end_repair_extended [F5P],
            (unsigned long long int) s->bases_end_repair_extended [F3P],
            (unsigned long long int) s->bases_end_repair_extended [R5P],
            (unsigned long long int) s->bases_end_repair_extended [R3P]);

  fprintf (stderr, "total_end_repair_indel [fwd5', fwd3', rev5', rev3']=[%llu, %llu, %llu, %llu]\n", 
            (unsigned long long int) s->total_end_repair_indel [F5P],
            (unsigned long long int) s->total_end_repair_indel [F3P],
            (unsigned long long int) s->total_end_repair_indel [R5P],
            (unsigned long long int) s->total_end_repair_indel [R3P]);

  fprintf (stderr, "num_5_softclips [fwd, rev]=[%llu, %llu]\n", 
            (unsigned long long int) s->num_5_softclips [0],
            (unsigned long long int) s->num_5_softclips [1]);

  fprintf (stderr, "bases_5_softclips_qry [fwd, rev]=[%llu, %llu]\n", 
            (unsigned long long int) s->bases_5_softclips_qry [0],
            (unsigned long long int) s->bases_5_softclips_qry [1]);

  fprintf (stderr, "bases_5_softclips_ref [fwd, rev]=[%llu, %llu]\n", 
            (unsigned long long int) s->bases_5_softclips_ref [0],
            (unsigned long long int) s->bases_5_softclips_ref [1]);
  
  fprintf (stderr, "bases_5_softclips_ref [fwd, rev]=[%llu, %llu]\n", 
            (unsigned long long int) s->bases_5_softclips_ref [0],
            (unsigned long long int) s->bases_5_softclips_ref [1]);

  fprintf (stderr, "score_5_softclips_total [fwd, rev]=[%lld, %lld]\n", 
            (long long int) s->score_5_softclips_total [0],
            (long long int) s->score_5_softclips_total [1]);

  fprintf (stderr, "num_seen_tailclipped=%llu\n", (unsigned long long int)s->num_seen_tailclipped);
  fprintf (stderr, "bases_seen_tailclipped=%llu\n", (unsigned long long int)s->bases_seen_tailclipped);
  fprintf (stderr, "num_tailclipped=%llu\n", (unsigned long long int)s->num_tailclipped);
  fprintf (stderr, "bases_tailclipped=%llu\n", (unsigned long long int)s->bases_tailclipped);
  fprintf (stderr, "num_fully_tailclipped=%llu\n", (unsigned long long int)s->num_fully_tailclipped);
  fprintf (stderr, "bases_fully_tailclipped=%llu\n", (unsigned long long int)s->bases_fully_tailclipped);
  

  fprintf (stderr, "num_filtered_als=%llu\n", (unsigned long long int)s->num_filtered_als);

}
