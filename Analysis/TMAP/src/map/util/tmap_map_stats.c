/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdlib.h>
#include <stdio.h>
#include <config.h>
#include <unistd.h>
#include <memory.h>
#include "../../util/tmap_error.h"
#include "../../util/tmap_alloc.h"
#include "../../util/tmap_definitions.h"
#include "tmap_map_stats.h"

tmap_map_stats_t*
tmap_map_stats_init()
{
  tmap_map_stats_t* r = tmap_calloc(1, sizeof(tmap_map_stats_t), "return");

  // DVK - guarantee initialization to zero of all members (actually calloc does this anyway)
  tmap_map_stats_zero (r);

  return r;
}

void
tmap_map_stats_destroy(tmap_map_stats_t *s)
{
  free(s);
}

void 
tmap_map_stats_zero(tmap_map_stats_t *s)
{
  memset (s, 0, sizeof (tmap_map_stats_t));
}


void
tmap_map_stats_add(tmap_map_stats_t *dest, tmap_map_stats_t *src)
{
  dest->num_reads += src->num_reads;
  dest->num_with_mapping += src->num_with_mapping;
  dest->num_after_seeding += src->num_after_seeding;
  dest->num_after_grouping += src->num_after_grouping;
  dest->num_after_scoring += src->num_after_scoring;
  dest->num_after_rmdup += src->num_after_rmdup;
  dest->num_after_filter += src->num_after_filter;

  dest->num_realign_invocations += src->num_realign_invocations;
  dest->num_realign_already_perfect += src->num_realign_already_perfect;
  dest->num_realign_not_clipped += src->num_realign_not_clipped;
  dest->num_realign_sw_failures += src->num_realign_sw_failures;
  dest->num_realign_unclip_failures += src->num_realign_unclip_failures;
  dest->num_realign_changed += src->num_realign_changed;
  dest->num_realign_shifted += src->num_realign_shifted;

  dest->num_seen_tailclipped += src->num_seen_tailclipped;
  dest->bases_seen_tailclipped += src->bases_seen_tailclipped;
  dest->num_tailclipped += src->num_tailclipped;
  dest->bases_tailclipped += src->bases_tailclipped;
  dest->num_fully_tailclipped += src->num_fully_tailclipped;
  dest->bases_fully_tailclipped += src->bases_fully_tailclipped;

  dest->num_hpcost_invocations += src->num_hpcost_invocations;
  dest->num_hpcost_skipped += src->num_hpcost_skipped;
  dest->num_hpcost_modified += src->num_hpcost_modified;
  dest->num_hpcost_shifted += src->num_hpcost_shifted;

  dest->num_filtered_als += src->num_filtered_als;
}

void
tmap_map_stats_print(tmap_map_stats_t *s)
{
  fprintf(stderr, "num_reads=%llu\n", (unsigned long long int)s->num_reads);
  fprintf(stderr, "num_with_mapping=%llu\n", (unsigned long long int)s->num_with_mapping);
  fprintf(stderr, "num_after_seeding=%llu\n", (unsigned long long int)s->num_after_seeding);
  fprintf(stderr, "num_after_grouping=%llu\n", (unsigned long long int)s->num_after_grouping);
  fprintf(stderr, "num_after_scoring=%llu\n", (unsigned long long int)s->num_after_scoring);
  fprintf(stderr, "num_after_rmdup=%llu\n", (unsigned long long int)s->num_after_rmdup);
  fprintf(stderr, "num_after_filter=%llu\n", (unsigned long long int)s->num_after_filter);
  
  fprintf(stderr, "num_realign_invocations=%llu\n", (unsigned long long int)s->num_realign_invocations);
  fprintf(stderr, "num_realign_already_perfect=%llu\n", (unsigned long long int)s->num_realign_already_perfect);
  fprintf(stderr, "num_realign_not_clipped=%llu\n", (unsigned long long int)s->num_realign_not_clipped);
  fprintf(stderr, "num_realign_sw_failures=%llu\n", (unsigned long long int)s->num_realign_sw_failures);
  fprintf(stderr, "num_realign_unclip_failures=%llu\n", (unsigned long long int)s->num_realign_unclip_failures);
  fprintf(stderr, "num_realign_changed=%llu\n", (unsigned long long int)s->num_realign_changed);
  fprintf(stderr, "num_realign_shifted=%llu\n", (unsigned long long int)s->num_realign_shifted);

  fprintf(stderr, "num_seen_tailclipped=%llu\n", (unsigned long long int)s->num_seen_tailclipped);
  fprintf(stderr, "bases_seen_tailclipped=%llu\n", (unsigned long long int)s->bases_seen_tailclipped);
  fprintf(stderr, "num_tailclipped=%llu\n", (unsigned long long int)s->num_tailclipped);
  fprintf(stderr, "bases_tailclipped=%llu\n", (unsigned long long int)s->bases_tailclipped);
  fprintf(stderr, "num_fully_tailclipped=%llu\n", (unsigned long long int)s->num_fully_tailclipped);
  fprintf(stderr, "bases_fully_tailclipped=%llu\n", (unsigned long long int)s->bases_fully_tailclipped);
  
  fprintf(stderr, "num_hpcost_invocations=%llu\n", (unsigned long long int)s->num_hpcost_invocations);
  fprintf(stderr, "num_hpcost_skipped=%llu\n", (unsigned long long int)s->num_hpcost_skipped);
  fprintf(stderr, "num_hpcost_modified=%llu\n", (unsigned long long int)s->num_hpcost_modified);
  fprintf(stderr, "num_hpcost_shifted=%llu\n", (unsigned long long int)s->num_hpcost_shifted);

  fprintf(stderr, "num_filtered_als=%llu\n", (unsigned long long int)s->num_filtered_als);

}
