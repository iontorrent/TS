/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <config.h>

#include "../samtools/kstring.h"
#include "../samtools/sam.h"
#include "../samtools/bam.h"
#include "../samtools/sam.h"
#include "../samtools/sam_header.h"

#include "../util/tmap_alloc.h"
#include "../util/tmap_error.h"
#include "../util/tmap_definitions.h"
#include "../util/tmap_string.h"
#include "tmap_sam.h"

tmap_sam_t *
tmap_sam_init()
{
  return tmap_calloc(1, sizeof(tmap_sam_t), "sam");
}

void
tmap_sam_destroy(tmap_sam_t *sam)
{
  if(NULL == sam) return;
  if(NULL != sam->name) tmap_string_destroy(sam->name);
  if(NULL != sam->seq) tmap_string_destroy(sam->seq);
  if(NULL != sam->qual) tmap_string_destroy(sam->qual);
  if(NULL != sam->b) bam_destroy1(sam->b);
  free(sam);
}

inline tmap_sam_t*
tmap_sam_clone(tmap_sam_t *sam)
{
  tmap_sam_t *ret = tmap_calloc(1, sizeof(tmap_sam_t), "ret");

  ret->name = tmap_string_clone(sam->name);
  ret->seq = tmap_string_clone(sam->seq);
  ret->qual = tmap_string_clone(sam->qual);
  ret->is_int = sam->is_int;

  // do not clone flow space info

  return ret;
}

void
tmap_sam_reverse(tmap_sam_t *sam)
{
  tmap_string_reverse(sam->seq);
  tmap_string_reverse(sam->qual);
}

void
tmap_sam_reverse_compliment(tmap_sam_t *sam)
{
  tmap_string_reverse_compliment(sam->seq, sam->is_int);
  tmap_string_reverse(sam->qual);
}

void
tmap_sam_compliment(tmap_sam_t *sam)
{
  tmap_string_compliment(sam->seq, sam->is_int);
}

void
tmap_sam_to_int(tmap_sam_t *sam)
{
  int i;
  if(1 == sam->is_int) return;
  for(i=0;i<sam->seq->l;i++) {
      sam->seq->s[i] = tmap_nt_char_to_int[(int)sam->seq->s[i]];
  }
  sam->is_int = 1;
}

void
tmap_sam_to_char(tmap_sam_t *sam)
{
  int i;
  if(0 == sam->is_int) return;
  for(i=0;i<sam->seq->l;i++) {
      sam->seq->s[i] = "ACGTN"[(int)sam->seq->s[i]];
  }
  sam->is_int = 0;
}

inline tmap_string_t *
tmap_sam_get_bases(tmap_sam_t *sam)
{
    return sam->seq;
}

inline tmap_string_t *
tmap_sam_get_qualities(tmap_sam_t *sam)
{
    return sam->qual;
}

int32_t
tmap_sam_get_flowgram(tmap_sam_t *sam, uint16_t **flowgram)
{
  uint8_t *tag = NULL;
  int32_t len = -1;
  (*flowgram) = NULL;
  // FZ
  tag = bam_aux_get(sam->b, "FZ");
  if(NULL != tag) {
      (*flowgram) = bam_auxB2S(tag, &len);
  }
  return len;
}

char*
tmap_sam_get_rg_id(tmap_sam_t *sam)
{
  uint8_t *tag = NULL;
  char *value = NULL;
  // RG
  tag = bam_aux_get(sam->b, "RG");
  if(NULL == tag) return NULL;
  value = bam_aux2Z(tag);
  return value;
}

int32_t
tmap_sam_get_fo_start_idx(tmap_sam_t *sam)
{
  uint8_t *tag = NULL;
  // ZF
  tag = bam_aux_get(sam->b, "ZF");
  if(NULL != tag) return bam_aux2i(tag);
  else return -1;
}

int32_t
tmap_sam_get_zb(tmap_sam_t *sam)
{
  uint8_t *tag = NULL;
  // ZB
  if(NULL == sam->b) 
      tmap_bug();
  tag = bam_aux_get(sam->b, "ZB");
  if(NULL != tag) return bam_aux2i(tag);
  else return -1;
}

int32_t
tmap_sam_get_za(tmap_sam_t *sam)
{
  uint8_t *tag = NULL;
  // ZA
  if(NULL == sam->b) tmap_bug();
  tag = bam_aux_get(sam->b, "ZA");
  if(NULL != tag) return bam_aux2i(tag);
  else return -1;
}

