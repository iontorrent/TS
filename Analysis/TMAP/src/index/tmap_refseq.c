/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#define _POSIX_C_SOURCE 200112L // to make declaration of strtok_r explicit

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <config.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <getopt.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <errno.h>
#include <alloca.h>
#include <assert.h>
#include <limits.h>

#include "../util/tmap_error.h"
#include "../util/tmap_alloc.h"
#include "../util/tmap_string.h"
#include "../util/tmap_progress.h"
#include "../util/tmap_definitions.h"
#include "../seq/tmap_seq.h"
#include "../io/tmap_file.h"
#include "../io/tmap_seq_io.h"
#include "../map/util/tmap_map_locopt.h"
#include "tmap_refseq.h"
#include "../util/tmap_error.h"
#include <getopt.h>

static const uint32_t OVR_PAR_MEM_INIT_CHUNK = 64; // allocation chunk for per-amplicon parameters override structures
static const uint32_t READ_ENDS_MEM_INIT_CHUNK = 64; // allocation chunk for read ends statistics records

const char *
tmap_refseq_get_version_format(const char *v)
{
  static const int32_t tmap_index_versions_num = 6;
  static const char *tmap_index_versions[34] = {
      "0.0.1", "tmap-f1",
      "0.0.17", "tmap-f2",
      "0.3.0", "tmap-f3"
  };
  int32_t i, cmp;
  for(i=tmap_index_versions_num-2;0<=i;i-=2) {
      cmp = tmap_compare_versions(tmap_index_versions[i], v);
      if(cmp <= 0) {
          i++;
          break;
      }
  }
  if(i < 0) {
      tmap_bug();
  }

  return tmap_index_versions[i];
}

static inline int32_t
tmap_refseq_supported(tmap_refseq_t *refseq)
{
  int32_t i, j;
  char *refseq_v = refseq->package_version->s;
  char *tmap_v = PACKAGE_VERSION;

  // sanity check on version names
  for(i=j=0;i<(int32_t)strlen(refseq_v);i++) {
      if('.' == refseq_v[i]) j++;
  }
  if(2 != j) {
      tmap_error("did not find three version numbers", Exit, OutOfRange);
  }
  for(i=j=0;i<(int32_t)strlen(tmap_v);i++) {
      if('.' == tmap_v[i]) j++;
  }
  if(2 != j) {
      tmap_error("did not find three version numbers", Exit, OutOfRange);
  }

  if(tmap_compare_versions(tmap_v, refseq_v) < 0) {
      return 0;
  }

  // get the format ids
  if(0 == strcmp(tmap_refseq_get_version_format(refseq_v), tmap_refseq_get_version_format(tmap_v))) {
      return 1;
  }
  return 0;
}

static inline void
tmap_refseq_write_header(tmap_file_t *fp, tmap_refseq_t *refseq)
{
  // size_t ll = 5;
  if(1 != tmap_file_fwrite(&refseq->version_id, sizeof(uint64_t), 1, fp)
     || 1 != tmap_file_fwrite(&refseq->package_version->l, sizeof(size_t), 1, fp)
     || refseq->package_version->l+1 != tmap_file_fwrite(refseq->package_version->s, sizeof(char), refseq->package_version->l+1, fp)
     || 1 != tmap_file_fwrite(&refseq->num_annos, sizeof(uint32_t), 1, fp)
     || 1 != tmap_file_fwrite(&refseq->len, sizeof(uint64_t), 1, fp)) {
      tmap_error(NULL, Exit, WriteFileError);
  }
}

static inline void
tmap_refseq_print_header(tmap_file_t *fp, tmap_refseq_t *refseq)
{
  int32_t i;
  tmap_file_fprintf(fp, "version id:\t%llu\n", (unsigned long long int)refseq->version_id);
  tmap_file_fprintf(fp, "format:\t%s\n", tmap_refseq_get_version_format(refseq->package_version->s));
  tmap_file_fprintf(fp, "package version:\t%s\n", refseq->package_version->s);
  for(i=0;i<refseq->num_annos;i++) {
      tmap_file_fprintf(fp, "contig-%d:\t%s\t%u\n", i+1, refseq->annos[i].name->s, refseq->annos[i].len);
  }
  tmap_file_fprintf(fp, "length:\t%llu\n", (unsigned long long int)refseq->len);
  tmap_file_fprintf(fp, "supported:\t%s\n", (0 == tmap_refseq_supported(refseq)) ? "false" : "true");
}

static inline void
tmap_refseq_write_annos(tmap_file_t *fp, tmap_anno_t *anno)
{
  uint32_t len = anno->name->l+1; // include null terminator

  if(1 != tmap_file_fwrite(&len, sizeof(uint32_t), 1, fp)
     || len != tmap_file_fwrite(anno->name->s, sizeof(char), len, fp)
     || 1 != tmap_file_fwrite(&anno->len, sizeof(uint64_t), 1, fp)
     || 1 != tmap_file_fwrite(&anno->offset, sizeof(uint64_t), 1, fp)
     || 1 != tmap_file_fwrite(&anno->num_amb, sizeof(uint32_t), 1, fp)) {
      tmap_error(NULL, Exit, WriteFileError);
  }
  if(0 < anno->num_amb) {
      if(anno->num_amb != tmap_file_fwrite(anno->amb_positions_start, sizeof(uint32_t), anno->num_amb, fp)
         || anno->num_amb != tmap_file_fwrite(anno->amb_positions_end, sizeof(uint32_t), anno->num_amb, fp)
         || anno->num_amb != tmap_file_fwrite(anno->amb_bases, sizeof(uint8_t), anno->num_amb, fp)) {
          tmap_error(NULL, Exit, WriteFileError);
      }
  }
}

static inline void
tmap_refseq_write_anno(tmap_file_t *fp, tmap_refseq_t *refseq)
{
  int32_t i;

  // write annotation file
  tmap_refseq_write_header(fp, refseq); // write the header
  for(i=0;i<refseq->num_annos;i++) { // write the annotations
      tmap_refseq_write_annos(fp, &refseq->annos[i]);
  }
}

static inline void
tmap_refseq_anno_clone(tmap_anno_t *dest, tmap_anno_t *src, int32_t reverse)
{
  uint32_t i;
  dest->name = tmap_string_clone(src->name);
  dest->len = src->len;
  dest->offset = src->offset;
  dest->num_amb = src->num_amb;
  dest->amb_positions_start = tmap_malloc(sizeof(uint32_t) * dest->num_amb, "dest->amb_positions_start");
  dest->amb_positions_end = tmap_malloc(sizeof(uint32_t) * dest->num_amb, "dest->amb_positions_end");
  dest->amb_bases = tmap_malloc(sizeof(uint8_t) * dest->num_amb, "dest->amb_bases");
  // ZZ: the copy below does not make sense, for reverse, shall not only reverse the order, ...
  if(0 == reverse) {
      for(i=0;i<dest->num_amb;i++) {
          dest->amb_positions_start[i] = src->amb_positions_start[dest->num_amb-i-1];
          dest->amb_positions_end[i] = src->amb_positions_end[dest->num_amb-i-1];
          dest->amb_bases[i] = src->amb_bases[dest->num_amb-i-1];
      }
  }
  else {
      for(i=0;i<dest->num_amb;i++) {
          dest->amb_positions_start[i] = src->amb_positions_start[i];
          dest->amb_positions_end[i] = src->amb_positions_end[i];
          dest->amb_bases[i] = src->amb_bases[i];
      }
  }
}

uint64_t
tmap_refseq_fasta2pac(const char *fn_fasta, int32_t compression, int32_t fwd_only, int32_t old_v)
{
  tmap_file_t *fp_pac = NULL, *fp_anno = NULL;
  tmap_seq_io_t *seqio = NULL;
  tmap_seq_t *seq = NULL;
  tmap_refseq_t *refseq = NULL;
  char *fn_pac = NULL, *fn_anno = NULL;
  uint8_t buffer[TMAP_REFSEQ_BUFFER_SIZE];
  int64_t i, j, l, buffer_length;
  uint32_t num_IUPAC_found= 0, amb_bases_mem = 0;
  uint8_t x = 0;
  uint64_t k, ref_len;

  if(0 == fwd_only) {
      tmap_progress_print("packing the reference FASTA");
  }
  else {
      tmap_progress_print("packing the reference FASTA (forward only)");
  }

  refseq = tmap_calloc(1, sizeof(tmap_refseq_t), "refseq");

  refseq->version_id = TMAP_VERSION_ID;
  if (old_v == 0)
     refseq->package_version = tmap_string_clone2(PACKAGE_VERSION);
  else
     refseq->package_version = tmap_string_clone2("0.3.1");
  refseq->seq = buffer; // IMPORTANT: must nullify later
  refseq->annos = NULL;
  refseq->num_annos = 0;
  refseq->len = 0;
  refseq->is_shm = 0;
  refseq->bed_exist = 0;
  memset(buffer, 0, TMAP_REFSEQ_BUFFER_SIZE);
  buffer_length = 0;

  // input files
  seqio = tmap_seq_io_init(fn_fasta, TMAP_SEQ_TYPE_FQ, 0, compression);
  seq = tmap_seq_init(TMAP_SEQ_TYPE_FQ);

  // output files
  fn_pac = tmap_get_file_name(fn_fasta, TMAP_PAC_FILE);
  fp_pac = tmap_file_fopen(fn_pac, "wb", TMAP_PAC_COMPRESSION);

  // read in sequences
  while(0 <= (l = tmap_seq_io_read(seqio, seq))) {
      tmap_anno_t *anno = NULL;
      tmap_progress_print2("packing contig [%s:1-%d]", seq->data.fq->name->s, l);

      refseq->num_annos++;
      refseq->annos = tmap_realloc(refseq->annos, sizeof(tmap_anno_t)*refseq->num_annos, "refseq->annos");
      anno = &refseq->annos[refseq->num_annos-1];

      anno->name = tmap_string_clone(seq->data.fq->name);
      anno->len = l;
      anno->offset = (1 == refseq->num_annos) ? 0 : refseq->annos[refseq->num_annos-2].offset + refseq->annos[refseq->num_annos-2].len;
      anno->amb_positions_start = NULL;
      anno->amb_positions_end = NULL;
      anno->amb_bases = NULL;
      anno->num_amb = 0;
      amb_bases_mem = 0;

      // fill the buffer
      for(i=0;i<l;i++) {
          uint8_t c = tmap_nt_char_to_int[(int)seq->data.fq->seq->s[i]];
          // handle IUPAC codes
          if(4 <= c) {
              int32_t k;
              // warn users about IUPAC codes
              if(0 == num_IUPAC_found) {
                  tmap_error("IUPAC codes were found and will be converted to non-matching DNA bases", Warn, OutOfRange);
                  for(j=4;j<15;j++) {
                      c = tmap_iupac_char_to_bit_string[(int)tmap_iupac_int_to_char[j]];
                      // get the lexicographically smallest base not compatible with this code
                      for(k=0;k<4;k++) {
                          if(!(c & (0x1 << k))) {
                              break;
                          }
                      }
                      tmap_progress_print2("IUPAC code %c will be converted to %c", tmap_iupac_int_to_char[j], "ACGTN"[k & 3]);
                  }
              }
              num_IUPAC_found++;

              // change it to a mismatched base than the IUPAC code
              c = tmap_iupac_char_to_bit_string[(int)seq->data.fq->seq->s[i]];

              // store IUPAC bases
              if(amb_bases_mem <= anno->num_amb) { // allocate more memory if necessary
                  amb_bases_mem = anno->num_amb + 1;
                  tmap_roundup32(amb_bases_mem);
                  anno->amb_positions_start = tmap_realloc(anno->amb_positions_start, sizeof(uint32_t) * amb_bases_mem, "anno->amb_positions_start");
                  anno->amb_positions_end = tmap_realloc(anno->amb_positions_end, sizeof(uint32_t) * amb_bases_mem, "anno->amb_positions_end");
                  anno->amb_bases = tmap_realloc(anno->amb_bases, sizeof(uint8_t) * amb_bases_mem, "anno->amb_bases");
              }
              // encode stretches of the same base
              if(0 < anno->num_amb
                 && anno->amb_positions_end[anno->num_amb-1] == i
                 && anno->amb_bases[anno->num_amb-1] == tmap_iupac_char_to_int[(int)seq->data.fq->seq->s[i]]) {
                 anno->amb_positions_end[anno->num_amb-1]++; // expand the range
              }
              else {
                  // new ambiguous base and range
                  anno->num_amb++;
                  anno->amb_positions_start[anno->num_amb-1] = i+1; // one-based
                  anno->amb_positions_end[anno->num_amb-1] = i+1; // one-based
                  anno->amb_bases[anno->num_amb-1] = tmap_iupac_char_to_int[(int)seq->data.fq->seq->s[i]];
              }

              // get the lexicographically smallest base not compatible with
              // this code
              for(j=0;j<4;j++) {
                  if(!(c & (0x1 << j))) {
                      break;
                  }
              }
              c = j & 3; // Note: Ns will go to As
          }
          if(3 < c) {
              tmap_bug();
          }
          if(buffer_length == (TMAP_REFSEQ_BUFFER_SIZE << 2)) { // 2-bit
              if(tmap_refseq_seq_memory(buffer_length) != tmap_file_fwrite(buffer, sizeof(uint8_t), tmap_refseq_seq_memory(buffer_length), fp_pac)) {
                  tmap_error(fn_pac, Exit, WriteFileError);
              }
              memset(buffer, 0, TMAP_REFSEQ_BUFFER_SIZE);
              buffer_length = 0;
          }
          tmap_refseq_seq_store_i(refseq, buffer_length, c);
          buffer_length++;
      }
      refseq->len += l;
      // re-size the ambiguous bases
      if(anno->num_amb < amb_bases_mem) {
          amb_bases_mem = anno->num_amb;
          anno->amb_positions_start = tmap_realloc(anno->amb_positions_start, sizeof(uint32_t) * amb_bases_mem, "anno->amb_positions_start");
          anno->amb_positions_end = tmap_realloc(anno->amb_positions_end, sizeof(uint32_t) * amb_bases_mem, "anno->amb_positions_end");
          anno->amb_bases = tmap_realloc(anno->amb_bases, sizeof(uint8_t) * amb_bases_mem, "anno->amb_bases");
      }
  }
  if(0 == refseq->len) {
      tmap_error("no bases found", Exit, OutOfRange);
  }
  // write out the buffer
  if(tmap_refseq_seq_memory(buffer_length) != tmap_file_fwrite(buffer, sizeof(uint8_t), tmap_refseq_seq_memory(buffer_length), fp_pac)) {
      tmap_error(fn_pac, Exit, WriteFileError);
  }
  if(refseq->len % 4 == 0) { // add an extra byte if we completely filled all bits
      if(1 != tmap_file_fwrite(&x, sizeof(uint8_t), 1, fp_pac)) {
          tmap_error(fn_pac, Exit, WriteFileError);
      }
  }
  // store number of unused bits at the last byte
  x = refseq->len % 4;
  if(1 != tmap_file_fwrite(&x, sizeof(uint8_t), 1, fp_pac)) {
      tmap_error(fn_pac, Exit, WriteFileError);
  }
  refseq->seq = NULL; // IMPORTANT: nullify this
  ref_len = refseq->len; // save for return

  tmap_progress_print2("total genome length [%u]", refseq->len);
  if(0 < num_IUPAC_found) {
      if(1 == num_IUPAC_found) {
          tmap_progress_print("%u IUPAC base was found and converted to a DNA base", num_IUPAC_found);
      }
      else {
          tmap_progress_print("%u IUPAC bases were found and converted to DNA bases", num_IUPAC_found);
      }
  }

  // write annotation file
  fn_anno = tmap_get_file_name(fn_fasta, TMAP_ANNO_FILE);
  fp_anno = tmap_file_fopen(fn_anno, "wb", TMAP_ANNO_COMPRESSION);
  tmap_refseq_write_anno(fp_anno, refseq);

  // close files
  tmap_file_fclose(fp_pac);
  tmap_file_fclose(fp_anno);

  // check sequence name uniqueness
  l = refseq->num_annos;
  if(0 == fwd_only) l /= 2; // only check the fwd
  for(i=0;i<l;i++) {
      for(j=i+1;j<l;j++) {
          if(0 == strcmp(refseq->annos[i].name->s, refseq->annos[j].name->s)) {
              tmap_file_fprintf(tmap_file_stderr, "Contigs have the same name: #%d [%s] and #%d [%s]\n",
                                i+1, refseq->annos[i].name->s,
                                j+1, refseq->annos[j].name->s);
              tmap_error("Contig names must be unique", Exit, OutOfRange);
          }
      }
  }

  tmap_refseq_destroy(refseq);
  tmap_seq_io_destroy(seqio);
  tmap_seq_destroy(seq);
  free(fn_pac);
  free(fn_anno);

  // pack the reverse compliment
  if(0 == fwd_only) {
      int32_t num_annos;
      uint64_t len;
      uint64_t len_fwd, len_rev;

      tmap_progress_print2("packing the reverse compliment FASTA for BWT/SA creation");

      refseq = tmap_refseq_read(fn_fasta);

      // original length
      num_annos = refseq->num_annos;
      len = refseq->len;

      // more annotations
      refseq->num_annos *= 2;
      refseq->annos = tmap_realloc(refseq->annos, sizeof(tmap_anno_t)*refseq->num_annos, "refseq->annos");

      // allocate more memory for the sequence
      refseq->len *= 2;
      if(refseq->refseq_fp)
      {
          uint8_t *newSwq = (uint8_t *)tmap_malloc(sizeof(uint8_t) * tmap_refseq_seq_memory(refseq->len), "refseq->seq");
          memcpy(newSwq,refseq->seq,tmap_refseq_seq_memory(len));
          tmap_file_fclose((tmap_file_t *)refseq->refseq_fp);
          refseq->refseq_fp=NULL;
          refseq->seq = newSwq;
      }
      else
      {
          refseq->seq = tmap_realloc(refseq->seq, sizeof(uint8_t) * tmap_refseq_seq_memory(refseq->len), "refseq->seq");
      }

      memset(refseq->seq + tmap_refseq_seq_memory(len), 0,
             (tmap_refseq_seq_memory(refseq->len) - tmap_refseq_seq_memory(len)) * sizeof(uint8_t));

      for(i=0,j=num_annos-1,len_fwd=len-1,len_rev=len;i<num_annos;i++,j--) {
          tmap_anno_t *anno_fwd = NULL;
          tmap_anno_t *anno_rev = NULL;

          anno_fwd = &refseq->annos[j]; // source
          anno_rev = &refseq->annos[i+num_annos]; // destination

          // clone the annotations
          tmap_refseq_anno_clone(anno_rev, anno_fwd, 1); // ZZ: this part does not make sense, but since this is not used later, not cause problem yet.
          anno_rev->offset = refseq->annos[i+num_annos-1].offset + refseq->annos[i+num_annos-1].len;

          // fill the buffer
          for(k=0;k<anno_fwd->len;k++,len_fwd--,len_rev++) { // reverse
              uint8_t c = tmap_refseq_seq_i(refseq, len_fwd);
              if(3 < c) {
                  tmap_bug();
              }
              c = 3 - c; // compliment
              tmap_refseq_seq_store_i(refseq, len_rev, c);
          }
      }
      if(len_rev != refseq->len) {
          tmap_bug();
      }

      // write
      tmap_refseq_write(refseq, fn_fasta);

      // free memory
      tmap_refseq_destroy(refseq);
  }

  tmap_progress_print2("packed the reference FASTA");

  return ref_len;
}

void
tmap_refseq_write(tmap_refseq_t *refseq, const char *fn_fasta)
{
  tmap_file_t *fp_pac = NULL, *fp_anno = NULL;
  char *fn_pac = NULL, *fn_anno = NULL;
  uint8_t x = 0;

  // write annotation file
  fn_anno = tmap_get_file_name(fn_fasta, TMAP_ANNO_FILE);
  fp_anno = tmap_file_fopen(fn_anno, "wb", TMAP_ANNO_COMPRESSION);
  tmap_refseq_write_anno(fp_anno, refseq);
  tmap_file_fclose(fp_anno);
  free(fn_anno);

  // write the sequence
  fn_pac = tmap_get_file_name(fn_fasta, TMAP_PAC_FILE);
  fp_pac = tmap_file_fopen(fn_pac, "wb", TMAP_PAC_COMPRESSION);
  if(tmap_refseq_seq_memory(refseq->len) != tmap_file_fwrite(refseq->seq, sizeof(uint8_t), tmap_refseq_seq_memory(refseq->len), fp_pac)) {
      tmap_error(NULL, Exit, WriteFileError);
  }
  if(refseq->len % 4 == 0) { // add an extra byte if we completely filled all bits
      if(1 != tmap_file_fwrite(&x, sizeof(uint8_t), 1, fp_pac)) {
          tmap_error(fn_pac, Exit, WriteFileError);
      }
  }
  // store number of unused bits at the last byte
  x = refseq->len % 4;
  if(1 != tmap_file_fwrite(&x, sizeof(uint8_t), 1, fp_pac)) {
      tmap_error(fn_pac, Exit, WriteFileError);
  }
  tmap_file_fclose(fp_pac);
  free(fn_pac);
}

static inline void
tmap_refseq_read_header(tmap_file_t *fp, tmap_refseq_t *refseq, int32_t ignore_version)
{
  size_t package_version_l = 0;
  if(1 != tmap_file_fread(&refseq->version_id, sizeof(uint64_t), 1, fp)
     || 1 != tmap_file_fread(&package_version_l, sizeof(size_t), 1, fp)) {
      tmap_error(NULL, Exit, ReadFileError);
  }
  if(refseq->version_id != TMAP_VERSION_ID) {
      tmap_error("version id did not match", Exit, ReadFileError);
  }

  refseq->package_version = tmap_string_init(package_version_l+1); // add one for the null terminator
  refseq->package_version->l = package_version_l;
  if(refseq->package_version->l+1 != tmap_file_fread(refseq->package_version->s, sizeof(char), refseq->package_version->l+1, fp)) {
      tmap_error(NULL, Exit, ReadFileError);
  }
  if(0 == ignore_version && 0 == tmap_refseq_supported(refseq)) {
      fprintf(stderr, "reference version: %s\n", refseq->package_version->s);
      fprintf(stderr, "package version: %s\n", PACKAGE_VERSION);
      tmap_error("the reference index is not supported", Exit, ReadFileError);
  }

  if(1 != tmap_file_fread(&refseq->num_annos, sizeof(uint32_t), 1, fp)
     || 1 != tmap_file_fread(&refseq->len, sizeof(uint64_t), 1, fp)) {
      tmap_error(NULL, Exit, ReadFileError);
  }

}

static inline void
tmap_refseq_read_annos(tmap_file_t *fp, tmap_anno_t *anno)
{
  uint32_t len = 0; // includes the null-terminator

  if(1 != tmap_file_fread(&len, sizeof(uint32_t), 1, fp)) {
      tmap_error(NULL, Exit, ReadFileError);
  }

  anno->name = tmap_string_init(len);

  if(len != tmap_file_fread(anno->name->s, sizeof(char), len, fp)
     || 1 != tmap_file_fread(&anno->len, sizeof(uint64_t), 1, fp)
     || 1 != tmap_file_fread(&anno->offset, sizeof(uint64_t), 1, fp)
     || 1 != tmap_file_fread(&anno->num_amb, sizeof(uint32_t), 1, fp)) {
      tmap_error(NULL, Exit, ReadFileError);
  }
  if(0 < anno->num_amb) {
      anno->amb_positions_start = tmap_malloc(sizeof(uint32_t) * anno->num_amb, "anno->amb_positions_start");
      anno->amb_positions_end = tmap_malloc(sizeof(uint32_t) * anno->num_amb, "anno->amb_positions_end");
      anno->amb_bases = tmap_malloc(sizeof(uint8_t) * anno->num_amb, "anno->amb_bases");
      if(anno->num_amb != tmap_file_fread(anno->amb_positions_start, sizeof(uint32_t), anno->num_amb, fp)
         || anno->num_amb != tmap_file_fread(anno->amb_positions_end, sizeof(uint32_t), anno->num_amb, fp)
         || anno->num_amb != tmap_file_fread(anno->amb_bases, sizeof(uint8_t), anno->num_amb, fp)) {
          tmap_error(NULL, Exit, ReadFileError);
      }
  }
  else {
      anno->amb_positions_start = NULL;
      anno->amb_positions_end = NULL;
      anno->amb_bases = NULL;
  }
  // set name length
  anno->name->l = len-1;
}

static inline void
tmap_refseq_read_anno(tmap_file_t *fp, tmap_refseq_t *refseq, int32_t ignore_version)
{
  int32_t i;
  // read annotation file
  tmap_refseq_read_header(fp, refseq, ignore_version); // read the header
  refseq->annos = tmap_calloc(refseq->num_annos, sizeof(tmap_anno_t), "refseq->annos"); // allocate memory
  for(i=0;i<refseq->num_annos;i++) { // read the annotations
      tmap_refseq_read_annos(fp, &refseq->annos[i]);
  }
}

tmap_refseq_t *
tmap_refseq_read(const char *fn_fasta)
{
  tmap_file_t *fp_pac = NULL, *fp_anno = NULL;
  char *fn_pac = NULL, *fn_anno = NULL;
  tmap_refseq_t *refseq = NULL;

  // allocate some memory
  refseq = tmap_calloc(1, sizeof(tmap_refseq_t), "refseq");
  refseq->is_shm = 0;
  refseq->bed_exist = 0;

  // read annotation file
  fn_anno = tmap_get_file_name(fn_fasta, TMAP_ANNO_FILE);
  fp_anno = tmap_file_fopen(fn_anno, "rb", TMAP_ANNO_COMPRESSION);
  tmap_refseq_read_anno(fp_anno, refseq, 0);
  tmap_file_fclose(fp_anno);
  free(fn_anno);

  // read the sequence
  fn_pac = tmap_get_file_name(fn_fasta, TMAP_PAC_FILE);
  fp_pac = tmap_file_fopen(fn_pac, "rb", TMAP_PAC_COMPRESSION);
#ifndef TMAP_MMAP
  refseq->seq = tmap_malloc(sizeof(uint8_t)*tmap_refseq_seq_memory(refseq->len), "refseq->seq"); // allocate
  if(tmap_refseq_seq_memory(refseq->len)
     != tmap_file_fread(refseq->seq, sizeof(uint8_t), tmap_refseq_seq_memory(refseq->len), fp_pac)) {
      tmap_error(NULL, Exit, ReadFileError);
  }
  tmap_file_fclose(fp_pac);
#else
  size_t len=0;
  refseq->seq = tmap_file_mmap(fp_pac,&len);
  if(refseq->seq == NULL){
      tmap_error(NULL, Exit, ReadFileError);
  }
  refseq->refseq_fp = (void *)fp_pac;
#endif
  free(fn_pac);


  return refseq;
}

size_t
tmap_refseq_approx_num_bytes(uint64_t len)
{
  // returns the number of bytes to allocate for shared memory
  int32_t i;
  size_t n = 0;

  n += sizeof(uint64_t); // version_id
  n += sizeof(size_t); // package_version->l
  n += sizeof(uint32_t); // annos
  n += sizeof(uint64_t); // len
  n += sizeof(char)*(strlen(PACKAGE_VERSION)); // ~package_version->s
  n += sizeof(uint8_t)*tmap_refseq_seq_memory(len); // ~seq
  for(i=0;i<32;i++) { // ~refseq->num_anos;
      n += sizeof(uint64_t); // len
      n += sizeof(uint64_t); // offset
      n += sizeof(size_t); // annos[i].name->l
      n += sizeof(uint32_t); // annos[i].num_amb
      n += sizeof(char)*(32); // ~annos[i].name->s
      n += sizeof(uint32_t)*256; // ~amb_positions_start
      n += sizeof(uint32_t)*256; // ~amb_positions_end
      n += sizeof(uint8_t)*256; // amb_bases
  }

  return n;
}

size_t
tmap_refseq_shm_num_bytes(tmap_refseq_t *refseq)
{
  // returns the number of bytes to allocate for shared memory
  int32_t i;
  size_t n = 0;

  n += sizeof(uint64_t); // version_id
  n += sizeof(size_t); // package_version->l
  n += sizeof(uint32_t); // annos
  n += sizeof(uint64_t); // len
  n += sizeof(char)*(refseq->package_version->l+1); // package_version->s
  n += sizeof(uint8_t)*tmap_refseq_seq_memory(refseq->len); // seq
  for(i=0;i<refseq->num_annos;i++) {
      n += sizeof(uint64_t); // len
      n += sizeof(uint64_t); // offset
      n += sizeof(size_t); // annos[i].name->l
      n += sizeof(uint32_t); // annos[i].num_amb
      n += sizeof(char)*(refseq->annos[i].name->l+1); // annos[i].name->s
      n += sizeof(uint32_t)*refseq->annos[i].num_amb; // amb_positions_start
      n += sizeof(uint32_t)*refseq->annos[i].num_amb; // amb_positions_end
      n += sizeof(uint8_t)*refseq->annos[i].num_amb; // amb_bases
  }

  return n;
}

size_t
tmap_refseq_shm_read_num_bytes(const char *fn_fasta)
{
  size_t n = 0;
  tmap_file_t *fp_anno = NULL;
  char *fn_anno = NULL;
  tmap_refseq_t *refseq = NULL;

  // allocate some memory
  refseq = tmap_calloc(1, sizeof(tmap_refseq_t), "refseq");
  refseq->is_shm = 0;
  refseq->bed_exist = 0;

  // read the annotation file
  fn_anno = tmap_get_file_name(fn_fasta, TMAP_ANNO_FILE);
  fp_anno = tmap_file_fopen(fn_anno, "rb", TMAP_ANNO_COMPRESSION);
  tmap_refseq_read_anno(fp_anno, refseq, 1);
  tmap_file_fclose(fp_anno);
  free(fn_anno);

  // No need to read in the pac
  refseq->seq = NULL;

  // get the number of bytes
  n = tmap_refseq_shm_num_bytes(refseq);

  // destroy
  tmap_refseq_destroy(refseq);

  return n;
}

uint8_t *
tmap_refseq_shm_pack(tmap_refseq_t *refseq, uint8_t *buf)
{
  int32_t i;

  // fixed length data
  memcpy(buf, &refseq->version_id, sizeof(uint64_t)); buf += sizeof(uint64_t);
  memcpy(buf, &refseq->package_version->l, sizeof(size_t)); buf += sizeof(size_t);
  memcpy(buf, &refseq->num_annos, sizeof(uint32_t)); buf += sizeof(uint32_t);
  memcpy(buf, &refseq->len, sizeof(uint64_t)); buf += sizeof(uint64_t);
  // variable length data
  memcpy(buf, refseq->package_version->s, sizeof(char)*(refseq->package_version->l+1));
  buf += sizeof(char)*(refseq->package_version->l+1);
  memcpy(buf, refseq->seq, tmap_refseq_seq_memory(refseq->len)*sizeof(uint8_t));
  buf += tmap_refseq_seq_memory(refseq->len)*sizeof(uint8_t);

  for(i=0;i<refseq->num_annos;i++) {
      // fixed length data
      memcpy(buf, &refseq->annos[i].len, sizeof(uint64_t)); buf += sizeof(uint64_t);
      memcpy(buf, &refseq->annos[i].offset, sizeof(uint64_t)); buf += sizeof(uint64_t);
      memcpy(buf, &refseq->annos[i].name->l, sizeof(size_t)); buf += sizeof(size_t);
      memcpy(buf, &refseq->annos[i].num_amb, sizeof(uint32_t)); buf += sizeof(uint32_t);
      // variable length data
      memcpy(buf, refseq->annos[i].name->s, sizeof(char)*(refseq->annos[i].name->l+1));
      buf += sizeof(char)*(refseq->annos[i].name->l+1);
      if(0 < refseq->annos[i].num_amb) {
          memcpy(buf, refseq->annos[i].amb_positions_start, sizeof(uint32_t)*refseq->annos[i].num_amb);
          buf += sizeof(uint32_t)*refseq->annos[i].num_amb;
          memcpy(buf, refseq->annos[i].amb_positions_end, sizeof(uint32_t)*refseq->annos[i].num_amb);
          buf += sizeof(uint32_t)*refseq->annos[i].num_amb;
          memcpy(buf, refseq->annos[i].amb_bases, sizeof(uint8_t)*refseq->annos[i].num_amb);
          buf += sizeof(uint8_t)*refseq->annos[i].num_amb;
      }
  }

  return buf;
}

tmap_refseq_t *
tmap_refseq_shm_unpack(uint8_t *buf)
{
  int32_t i;
  tmap_refseq_t *refseq = NULL;

  if(NULL == buf) return NULL;

  refseq = tmap_calloc(1, sizeof(tmap_refseq_t), "refseq");
  refseq->bed_exist = 0;
  // fixed length data
  memcpy(&refseq->version_id, buf, sizeof(uint64_t)) ; buf += sizeof(uint64_t);
  if(refseq->version_id != TMAP_VERSION_ID) {
      tmap_error("version id did not match", Exit, ReadFileError);
  }

  refseq->package_version = tmap_string_init(0);
  memcpy(&refseq->package_version->l, buf, sizeof(size_t)); buf += sizeof(size_t);
  memcpy(&refseq->num_annos, buf, sizeof(uint32_t)) ; buf += sizeof(uint32_t);
  memcpy(&refseq->len, buf, sizeof(uint64_t)) ; buf += sizeof(uint64_t);

  // variable length data
  refseq->package_version->s = (char*)buf;
  refseq->package_version->m = refseq->package_version->l+1;
  buf += sizeof(char)*(refseq->package_version->l+1);
  if(0 == tmap_refseq_supported(refseq)) {
      tmap_error("the reference index is not supported", Exit, ReadFileError);
  }
  refseq->seq = (uint8_t*)buf;
  buf += tmap_refseq_seq_memory(refseq->len)*sizeof(uint8_t);
  refseq->annos = tmap_calloc(refseq->num_annos, sizeof(tmap_anno_t), "refseq->annos");
  for(i=0;i<refseq->num_annos;i++) {
      // fixed length data
      memcpy(&refseq->annos[i].len, buf, sizeof(uint64_t)); buf += sizeof(uint64_t);
      memcpy(&refseq->annos[i].offset, buf, sizeof(uint64_t)); buf += sizeof(uint64_t);
      refseq->annos[i].name = tmap_string_init(0);
      memcpy(&refseq->annos[i].name->l, buf, sizeof(size_t)); buf += sizeof(size_t);
      refseq->annos[i].name->m = refseq->annos[i].name->l+1;
      memcpy(&refseq->annos[i].num_amb, buf, sizeof(uint32_t)); buf += sizeof(uint32_t);
      // variable length data
      refseq->annos[i].name->s = (char*)buf;
      buf += sizeof(char)*refseq->annos[i].name->l+1;
      if(0 < refseq->annos[i].num_amb) {
          refseq->annos[i].amb_positions_start = (uint32_t*)buf;
          buf += sizeof(uint32_t)*refseq->annos[i].num_amb;
          refseq->annos[i].amb_positions_end = (uint32_t*)buf;
          buf += sizeof(uint32_t)*refseq->annos[i].num_amb;
          refseq->annos[i].amb_bases = (uint8_t*)buf;
          buf += sizeof(uint8_t)*refseq->annos[i].num_amb;
      }
      else {
          refseq->annos[i].amb_positions_start = NULL;
          refseq->annos[i].amb_positions_end = NULL;
          refseq->annos[i].amb_bases = NULL;
      }
  }

  refseq->is_shm = 1;

  return refseq;
}

void
tmap_refseq_destroy (tmap_refseq_t *refseq)
{
  int32_t i;

  if (1 == refseq->is_shm) 
  {
      free (refseq->package_version);
      for (i = 0; i < refseq->num_annos; ++i)
          free (refseq->annos [i].name);
      free (refseq->annos);
  }
  else 
  {
      tmap_string_destroy (refseq->package_version);
      for(i = 0; i < refseq->num_annos; ++i) 
      {
          tmap_string_destroy (refseq->annos [i].name);
          free (refseq->annos [i].amb_positions_start);
          free (refseq->annos [i].amb_positions_end);
          free (refseq->annos [i].amb_bases);
      }
      free (refseq->annos);

      if (refseq->refseq_fp)
          tmap_file_fclose ((tmap_file_t *)refseq->refseq_fp);
      else
          free (refseq->seq);
  }
  if (1 == refseq->bed_exist) 
  {
    int i;
    for (i = 0; i < refseq->beditem; ++i) 
    {
        if (refseq->bednum [i]> 0) 
        {
            free (refseq->bedstart [i]);
            free (refseq->bedend [i]);
        }
    }
    free (refseq->bednum);
    free (refseq->bedstart);
    free (refseq->bedend);
    if (refseq->parovr)
    {
        for (i = 0; i < refseq->num_annos; ++i) 
            free  (refseq->parovr [i]);
        free (refseq->parovr);
        for (i = 0; i < refseq->parmem_used; ++i)
            tmap_map_locopt_destroy (refseq->parmem + i);
        free (refseq->parmem);
    }
    if (refseq->read_ends)
    {
        for (i = 0; i < refseq->num_annos; ++i) 
            free  (refseq->read_ends [i]);
        free (refseq->read_ends);
        free (refseq->endposmem);
    }
  }
  free(refseq);
}

// zero-based
static inline int32_t
tmap_refseq_get_seqid1(const tmap_refseq_t *refseq, tmap_bwt_int_t pacpos)
{
  int32_t left, right, mid;

  if(refseq->len < pacpos) {
      tmap_error("Coordinate was larger than the reference", Exit, OutOfRange);
  }

  left = 0; mid = 0; right = refseq->num_annos;
  while (left < right) {
      mid = (left + right) >> 1;
      if(refseq->annos[mid].offset < pacpos) {
          if(mid == refseq->num_annos - 1) break;
          if(pacpos <= refseq->annos[mid+1].offset) break;
          left = mid + 1;
      } else right = mid;
  }

  if(refseq->num_annos < mid) {
      return refseq->num_annos;
  }

  return mid;
}

// zero-based
static inline int32_t
tmap_refseq_get_seqid(const tmap_refseq_t *refseq, tmap_bwt_int_t pacpos, uint32_t aln_len)
{
  int32_t seqid_left, seqid_right;

  seqid_left = tmap_refseq_get_seqid1(refseq, pacpos);
  if(1 == aln_len) return seqid_left;

  seqid_right = tmap_refseq_get_seqid1(refseq, pacpos+aln_len-1);
  if(seqid_left != seqid_right) return -1; // overlaps two chromosomes

  return seqid_left;
}

// zero-based
static inline uint32_t
tmap_refseq_get_pos(const tmap_refseq_t *refseq, tmap_bwt_int_t pacpos, uint32_t seqid)
{
  // note: offset is zero-based
  return pacpos - refseq->annos[seqid].offset;
}

inline tmap_bwt_int_t
tmap_refseq_pac2real(const tmap_refseq_t *refseq, tmap_bwt_int_t pacpos, uint32_t aln_length, uint32_t *seqid, uint32_t *pos, uint8_t *strand)
{
  if((refseq->len << 1) < pacpos) {
      tmap_error("Coordinate was larger than the reference", Exit, OutOfRange);
  }

  // strand
  if(refseq->len < pacpos) {
      (*strand) = 1;
      pacpos = pacpos - refseq->len; // between [1, refseq->len]
      pacpos = refseq->len - pacpos + 1; // reverse around
      // adjust based on the opposite strand
      if(pacpos < aln_length) {
          aln_length = pacpos;
          pacpos = 1;
      }
      else {
          pacpos -= aln_length-1;
      }
  }
  else {
      (*strand) = 0;
  }
  // seqid
  (*seqid) = tmap_refseq_get_seqid(refseq, pacpos, aln_length);
  if((*seqid) == (uint32_t)-1) {
      (*pos) = (uint32_t)-1;
      return 0;
  }
  // position
  (*pos) = tmap_refseq_get_pos(refseq, pacpos, (*seqid));

  return (*pos);
}

inline int32_t
tmap_refseq_subseq(const tmap_refseq_t *refseq, tmap_bwt_int_t pacpos, uint32_t length, uint8_t *target)
{
  tmap_bwt_int_t k, pacpos_upper;
  uint32_t l;
  if(0 == length) {
      return 0;
  }
  else if(pacpos + length - 1 < refseq->len) {
      pacpos_upper = pacpos + length - 1;
  }
  else {
      pacpos_upper = refseq->len;
  }
  for(k=pacpos,l=0;k<=pacpos_upper;k++,l++) {
      // k-1 since pacpos is one-based
      target[l] = tmap_refseq_seq_i(refseq, k-1);
  }
  return l;
}

inline uint8_t*
tmap_refseq_subseq2(const tmap_refseq_t *refseq, uint32_t seqid, uint32_t start, uint32_t end, uint8_t *target, int32_t to_n, int32_t *conv)
{
  uint32_t i, j;

  if(0 == seqid || (uint32_t)refseq->num_annos < seqid || end < start) {
      return NULL;
  }

  if(NULL == target) {
      target = tmap_malloc(sizeof(char) * (end - start + 1), "target");
  }
  if((end - start + 1) != (uint32_t)tmap_refseq_subseq(refseq, refseq->annos[seqid-1].offset + start, end - start + 1, target)) {
      free(target);
      return NULL;
  }

  // check if any IUPAC bases fall within the range
  // NB: this could be done more efficiently, since we we know start <= end
  if(NULL != conv) (*conv) = 0;
  if(0 < tmap_refseq_amb_bases(refseq, seqid, start, end)) {
      // modify them
      for(i=start;i<=end;i++) {
          j = tmap_refseq_amb_bases(refseq, seqid, i, i); // Note: j is one-based
          if(0 < j) {
              target[i-start] = (0 == to_n) ? refseq->annos[seqid-1].amb_bases[j-1] : 4;
              if(NULL != conv) (*conv)++;
          }
      }
  }

  return target;
}

inline int32_t
tmap_refseq_amb_bases(const tmap_refseq_t *refseq, uint32_t seqid, uint32_t start, uint32_t end)
{
  int64_t low, high, mid;
  int32_t c;
  tmap_anno_t *anno;

  anno = &refseq->annos[seqid-1];

  if(0 == anno->num_amb) {
      return 0;
  }
  else if(1 == anno->num_amb) {
      if(0 == tmap_interval_overlap(start, end, anno->amb_positions_start[0], anno->amb_positions_end[0])) {
          return 1;
      }
      else {
          return 0;
      }
  }

  low = 0;
  high = anno->num_amb - 1;
  while(low <= high) {
      mid = (low + high) / 2;
      c = tmap_interval_overlap(start, end, anno->amb_positions_start[mid], anno->amb_positions_end[mid]);
      if(0 == c) {
          return mid+1;
      }
      else if(0 < c) {
          low = mid + 1;
      }
      else {
          high = mid - 1;
      }
  }

  return 0;
}

int
tmap_refseq_fasta2pac_main(int argc, char *argv[])
{
  int c, help=0, fwd_only=0, old_v = 0;

  while((c = getopt(argc, argv, "fvhp")) >= 0) {
      switch(c) {
        case 'v': tmap_progress_set_verbosity(1); break;
        case 'f': fwd_only = 1; break;
        case 'h': help = 1; break;
    case 'p': old_v = 1; break;
        default: return 1;
      }
  }
  if(1 != argc - optind || 1 == help) {
      tmap_file_fprintf(tmap_file_stderr, "Usage: %s %s [-fvhp] <in.fasta>\n", PACKAGE, argv[0]);
      return 1;
  }

  tmap_refseq_fasta2pac(argv[optind], TMAP_FILE_NO_COMPRESSION, fwd_only, old_v);

  return 0;
}

int
tmap_refseq_refinfo_main(int argc, char *argv[])
{
  int c, help=0;
  tmap_refseq_t *refseq = NULL;
  tmap_file_t *fp_anno = NULL;
  char *fn_anno = NULL;
  char *fn_fasta = NULL;

  while((c = getopt(argc, argv, "vh")) >= 0) {
      switch(c) {
        case 'v': tmap_progress_set_verbosity(1); break;
        case 'h': help = 1; break;
        default: return 1;
      }
  }
  if(1 != argc - optind || 1 == help) {
      tmap_file_fprintf(tmap_file_stderr, "Usage: %s %s [-vh] <in.fasta>\n", PACKAGE, argv[0]);
      return 1;
  }
  fn_fasta = argv[optind];

  // Note: 'tmap_file_stdout' should not have been previously modified
  tmap_file_stdout = tmap_file_fdopen(fileno(stdout), "wb", TMAP_FILE_NO_COMPRESSION);

  // allocate some memory
  refseq = tmap_calloc(1, sizeof(tmap_refseq_t), "refseq");
  refseq->is_shm = 0;
  refseq->bed_exist = 0;

  // read the annotation file
  fn_anno = tmap_get_file_name(fn_fasta, TMAP_ANNO_FILE);
  fp_anno = tmap_file_fopen(fn_anno, "rb", TMAP_ANNO_COMPRESSION);
  tmap_refseq_read_anno(fp_anno, refseq, 1);
  tmap_file_fclose(fp_anno);
  free(fn_anno);

  // no need to read in the pac
  refseq->seq = NULL;

  // print the header
  tmap_refseq_print_header(tmap_file_stdout, refseq);

  // destroy
  tmap_refseq_destroy(refseq);

  // close the output
  tmap_file_fclose(tmap_file_stdout);

  return 0;
}

int
tmap_refseq_pac2fasta_main(int argc, char *argv[])
{
  int c, help=0, amb=0;
  int32_t i;
  uint32_t j, k;
  char *fn_fasta = NULL;
  tmap_refseq_t *refseq = NULL;

  while((c = getopt(argc, argv, "avh")) >= 0) {
      switch(c) {
        case 'a': amb = 1; break;
        case 'v': tmap_progress_set_verbosity(1); break;
        case 'h': help = 1; break;
        default: return 1;
      }
  }
  if(1 != argc - optind || 1 == help) {
      tmap_file_fprintf(tmap_file_stderr, "Usage: %s %s [-avh] <in.fasta>\n", PACKAGE, argv[0]);
      return 1;
  }

  fn_fasta = argv[optind];

  // Note: 'tmap_file_stdout' should not have been previously modified
  tmap_file_stdout = tmap_file_fdopen(fileno(stdout), "wb", TMAP_FILE_NO_COMPRESSION);

  // read in the reference sequence
  refseq = tmap_refseq_read(fn_fasta);

  for(i=0;i<refseq->num_annos;i++) {
      tmap_file_fprintf(tmap_file_stdout, ">%s", refseq->annos[i].name->s); // new line handled later
      for(j=k=0;j<refseq->annos[i].len;j++) {
          if(0 == (j % TMAP_REFSEQ_FASTA_LINE_LENGTH)) {
              tmap_file_fprintf(tmap_file_stdout, "\n");
          }
          if(1 == amb && 0 < refseq->annos[i].num_amb) {
              // move the next ambiguous region
              while(k < refseq->annos[i].num_amb && refseq->annos[i].amb_positions_end[k] < j+1) {
                  k++;
              }
              // check for the ambiguous region
              if(k < refseq->annos[i].num_amb
                 && 0 == tmap_interval_overlap(j+1, j+1, refseq->annos[i].amb_positions_start[k], refseq->annos[i].amb_positions_end[k])) {
                  tmap_file_fprintf(tmap_file_stdout, "%c", tmap_iupac_int_to_char[refseq->annos[i].amb_bases[k]]);
              }
              else {
                  tmap_file_fprintf(tmap_file_stdout, "%c", "ACGTN"[(int)tmap_refseq_seq_i(refseq, j + refseq->annos[i].offset)]);
              }
          }
          else {
              tmap_file_fprintf(tmap_file_stdout, "%c", "ACGTN"[(int)tmap_refseq_seq_i(refseq, j + refseq->annos[i].offset)]);
          }
      }
      tmap_file_fprintf(tmap_file_stdout, "\n");
  }

  // destroy
  tmap_refseq_destroy(refseq);

  // close the output
  tmap_file_fclose(tmap_file_stdout);

  return 0;
}

static int32_t
tmap_refseq_get_id(tmap_refseq_t *refseq, char *chr)
{
    int i;
    for (i = 0; i < refseq->num_annos; i++) {
    if (strcmp(refseq->annos[i].name->s, chr) == 0) return i;
    }
    return -1;
}

static char *strsave(char *s)
{
    char *t = malloc(sizeof(char)*(strlen(s)+1));
    strcpy(t, s);
    return t;
}

// DK:Parse tmap override parameters embedded in BED file
// returns 0 if no overrides found, 1 otherwise
enum ovr_opt_code
{
    // --no-bed-er
    OO_no_bed_er,
    // -A,--score-match
    OO_score_match = 127, //over any ASCII value, so short opts can be checked same way
    // -M,--pen-mismatch
    OO_pen_mismatch,
    // -O,--pen-gap-open
    OO_pen_gap_open,
    // -E,--pen-gap-extension
    OO_pen_gap_extension,
    // -G,--pen-gap-long
    OO_pen_gap_long,
    // -K,--gap-long-length
    OO_gap_long_length,
    // -w,--band-width
    OO_band_width,
    // -g,--softclip-type
    OO_softclip_type,
    // --do-realign
    OO_do_realign,
    // --r-mat
    OO_r_mat,
    // --r-mis
    OO_r_mis,
    // --r-gip
    OO_r_gip,
    // --r-gep
    OO_r_gep,
    // --r-bw
    OO_r_bw,
    // --r-clip
    OO_r_clip,
    // --do-repeat-clip
    OO_do_repeat_clip,
    // --repclip-cont
    OO_repclip_cont,
    // --context
    OO_context,
    // --gap-scale
    OO_gap_scale,
    // --c-mat
    OO_c_mat,
    // --c-mis
    OO_c_mis,
    // --c-gip
    OO_c_gip,
    // --c-gep
    OO_c_gep,
    // --c-bw
    OO_c_bw,

    // --end-repair
    OO_end_repair,
    // --max-one-large-indel-rescue
    OO_max_one_large_indel_rescue,
    // --min-anchor-large-indel-rescue
    OO_min_anchor_large_indel_rescue,
    // --max-amplicon-overrun-large-indel-rescue
    OO_max_amplicon_overrun_large_indel_rescue,
    // --max-adapter-bases-for-soft-clipping
    OO_max_adapter_bases_for_soft_clipping,
    // --er-no5clip
    OO_er_no5clip,
    // --repair
    OO_repair,
    // --repair-min-adapter
    OO_repair_min_adapter,
    // --repair-max-overhang
    OO_repair_max_overhang,
    // --repair-identity-drop-limits
    OO_repair_identity_drop_limit,
    // --repair_max_primer_zone_dist
    OO_repair_max_primer_zone_dist,
    // --repair_clip_ext
    OO_repair_clip_ext,

    // --end-repair-he
    OO_end_repair_he,
    // --max-one-large-indel-rescue-he
    OO_max_one_large_indel_rescue_he,
    // --min-anchor-large-indel-rescue-he
    OO_min_anchor_large_indel_rescue_he,
    // --max-amplicon-overrun-large-indel-rescue-he
    OO_max_amplicon_overrun_large_indel_rescue_he,
    // --max-adapter-bases-for-soft-clipping-he
    OO_max_adapter_bases_for_soft_clipping_he,
    // --er-no5clip-he
    OO_er_no5clip_he,
    // --repair-he
    OO_repair_he,
    // --repair-min-adapter-he
    OO_repair_min_adapter_he,
    // --repair-max-overhang-he
    OO_repair_max_overhang_he,
    // --repair-identity-drop-limits-he
    OO_repair_identity_drop_limit_he,
    // --repair_max_primer_zone_dist-he
    OO_repair_max_primer_zone_dist_he,
    // --repair_clip_ext_he
    OO_repair_clip_ext_he,

    // --end-repair-le
    OO_end_repair_le,
    // --max-one-large-indel-rescue-le
    OO_max_one_large_indel_rescue_le,
    // --min-anchor-large-indel-rescue-le
    OO_min_anchor_large_indel_rescue_le,
    // --max-amplicon-overrun-large-indel-rescue-le
    OO_max_amplicon_overrun_large_indel_rescue_le,
    // --max-adapter-bases-for-soft-clipping-le
    OO_max_adapter_bases_for_soft_clipping_le,
    // --er-no5clip-le
    OO_er_no5clip_le,
    // --repair-le
    OO_repair_le,
    // --repair-min-adapter-le
    OO_repair_min_adapter_le,
    // --repair-max-overhang-le
    OO_repair_max_overhang_le,
    // --repair-identity-drop-limits-le
    OO_repair_identity_drop_limit_le,
    // --repair_max_primer_zone_dist-le
    OO_repair_max_primer_zone_dist_le,
    // --repair_clip_ext_le
    OO_repair_clip_ext_le,


    // --log
    OO_log,
    // --debug-log
    OO_debug_log,
    // --pen_flow_error
    OO_pen_flow_error,
    // --softclip-key
    OO_softclip_key,
    // --ignore-flowgram
    OO_ignore_flowgram,
    // --final_flowspace
    OO_aln_flowspace 
};

typedef struct option sysopt_t;

static const sysopt_t overridable_opts [] =
{
    // --no-bed-er
    { "no-bed-er",                              optional_argument,  NULL, OO_no_bed_er },
    // -A,--score-match
    { "score-match",                            required_argument,  NULL, OO_score_match },
    // -M,--pen-mismatch
    { "pen-mismatch",                           required_argument,  NULL, OO_pen_mismatch },
    // -O,--pen-gap-open
    { "pen-gap-open",                           required_argument,  NULL, OO_pen_gap_open },
    // -E,--pen-gap-extension
    { "pen-gap-extension",                      required_argument,  NULL, OO_pen_gap_extension },
    // -G,--pen-gap-long
    { "pen-gap-long",                           required_argument,  NULL, OO_pen_gap_long },
    // -K,--gap-long-length
    { "gap-long-length",                        required_argument,  NULL, OO_gap_long_length },
    // -w,--band-width
    { "pen-band-width",                         required_argument,  NULL, OO_band_width },
    // -g,--softclip-type
    { "softclip-type",                          required_argument,  NULL, OO_softclip_type },
    // --do-realign
    { "do-realign",                             optional_argument,  NULL, OO_do_realign },
    // --r-mat
    { "r-mat",                                  required_argument,  NULL, OO_r_mat },
    // --r-mis
    { "r-mis",                                  required_argument,  NULL, OO_r_mis },
    // --r-gip
    { "r-gip",                                  required_argument,  NULL, OO_r_gip },
    // --r-gep
    { "r-gep",                                  required_argument,  NULL, OO_r_gep },
    // --r-bw
    { "r-bw",                                   required_argument,  NULL, OO_r_bw },
    // --r-clip
    { "r-clip",                                 required_argument,  NULL, OO_r_clip },
    // --do-repeat-clip
    { "do-repeat-clip",                         optional_argument,  NULL, OO_do_repeat_clip },
    // --repclip-cont
    { "repclip-cont",                           optional_argument,  NULL, OO_repclip_cont },
    // --context
    { "context",                                optional_argument,  NULL, OO_context },
    // --gap-scale
    { "gap-scale",                              required_argument,  NULL, OO_gap_scale },
    // --c-mat
    { "c-mat",                                  required_argument,  NULL, OO_c_mat },
    // --c-mis
    { "c-mis",                                  required_argument,  NULL, OO_c_mis },
    // --c-gip
    { "c-gip",                                  required_argument,  NULL, OO_c_gip },
    // --c-gep
    { "c-gep",                                  required_argument,  NULL, OO_c_gep },
    // --c-bw
    { "c-bw",                                   required_argument,  NULL, OO_c_bw },

    // Following options are recognised per se as well as with -le and -he suffixes:
    // --end-repair
    { "end-repair",                             required_argument,  NULL, OO_end_repair },
    // --max-one-large-indel-rescue
    { "max-one-large-indel-rescue",                     required_argument,  NULL, OO_max_one_large_indel_rescue },
    // --min-anchor-large-indel-rescue
    { "min-anchor-large-indel-rescue",                  required_argument,  NULL, OO_min_anchor_large_indel_rescue },
    // --max-er-5clip-large-indel-rescue
    { "max-amplicon-overrun-large-indel-rescue",        required_argument,  NULL, OO_max_amplicon_overrun_large_indel_rescue },
    // -J, --max-adapter-bases-for-soft-clipping
    { "max-adapter-bases-for-soft-clipping",            required_argument,  NULL, OO_max_adapter_bases_for_soft_clipping },
    // --er-no5clip
    { "er-no5clip",                                     optional_argument,  NULL, OO_er_no5clip },
    // --repair
    { "repair",                                         optional_argument,  NULL, OO_repair },
    // --repair-min-adapter
    { "repair-min-adapter",                             required_argument,  NULL, OO_repair_min_adapter },
    // --repair-max-overhang
    { "repair-max-overhang",                            required_argument,  NULL, OO_repair_max_overhang },
    // --repair-identity-drop-limit
    { "repair-identity-drop-limit",                     required_argument,  NULL, OO_repair_identity_drop_limit },
    // --repair-max-primer-zone-dist
    { "repair-max-primer-zone-dist",                    required_argument,  NULL, OO_repair_max_primer_zone_dist },
    // --repair-clip-ext
    { "repair-clip-ext",                                required_argument,  NULL, OO_repair_clip_ext },

    // lower end of amplicon
    // --end-repair-le
    { "end-repair-le",                                  required_argument,   NULL, OO_end_repair_le },
    // --max-one-large-indel-rescue-le
    { "max-one-large-indel-rescue-le",                  required_argument,  NULL, OO_max_one_large_indel_rescue_le },
    // --min-anchor-large-indel-rescue-le
    { "min-anchor-large-indel-rescue-le",               required_argument,  NULL, OO_min_anchor_large_indel_rescue_le },
    // --max-er-5clip-large-indel-rescue-le
    { "max-amplicon-overrun-large-indel-rescue-le",     required_argument,  NULL, OO_max_amplicon_overrun_large_indel_rescue_le },
    // --max-adapter-bases-for-soft-clipping-le
    { "max-adapter-bases-for-soft-clipping-le",         required_argument,  NULL, OO_max_adapter_bases_for_soft_clipping_le },
    // --er-no5clip-le
    { "er-no5clip-le",                                  optional_argument,  NULL, OO_er_no5clip_le },
    // --repair-le
    { "repair-le",                                      optional_argument,  NULL, OO_repair_le },
    // --repair-min-adapter-le
    { "repair-min-adapter-le",                          required_argument,  NULL, OO_repair_min_adapter_le },
    // --repair-max-overhang-le
    { "repair-max-overhang-le",                         required_argument,  NULL, OO_repair_max_overhang_le },
    // --repair-identity-drop-limit-le
    { "repair-identity-drop-limit-le",                  required_argument,  NULL, OO_repair_identity_drop_limit_le },
    // --repair-max-primer-zone-dist-le
    { "repair-max-primer-zone-dist-le",                 required_argument,  NULL, OO_repair_max_primer_zone_dist_le },
    // --repair-clip-ext-le
    { "repair-clip-ext-le",                             required_argument,  NULL, OO_repair_clip_ext_le },

    // higher end of amplicon
    // --end-repair-he
    { "end-repair-he",                                  required_argument,  NULL, OO_end_repair_he },
    // --max-one-large-indel-rescue-he
    { "max-one-large-indel-rescue-he",                  required_argument,  NULL, OO_max_one_large_indel_rescue_he },
    // --min-anchor-large-indel-rescue-he
    { "min-anchor-large-indel-rescue-he",               required_argument,  NULL, OO_min_anchor_large_indel_rescue_he },
    // --max-er-5clip-large-indel-rescue-he
    { "max-amplicon-overrun-large-indel-rescue-he",     required_argument,  NULL, OO_max_amplicon_overrun_large_indel_rescue_he },
    // --max-adapter-bases-for-soft-clipping-he
    { "max-adapter-bases-for-soft-clipping-he",         required_argument,  NULL, OO_max_adapter_bases_for_soft_clipping_he },
    // --er-no5clip-he
    { "er-no5clip-he",                                  optional_argument,  NULL, OO_er_no5clip_he },
    // --repair-he
    { "repair-he",                                      optional_argument,  NULL, OO_repair_he },
    // --repair-min-adapter-he
    { "repair-min-adapter-he",                          required_argument,  NULL, OO_repair_min_adapter_he },
    // --repair-max-overhang-he
    { "repair-max-overhang-he",                         required_argument,  NULL, OO_repair_max_overhang_he },
    // --repair-identity-drop-limit-he
    { "repair-identity-drop-limit-he",                  required_argument,  NULL, OO_repair_identity_drop_limit_he },
    // --repair-max-primer-zone-dist-he
    { "repair-max-primer-zone-dist-he",                 required_argument,  NULL, OO_repair_max_primer_zone_dist_he },
    // --repair-clip-ext-he
    { "repair-clip-ext-he",                             required_argument,  NULL, OO_repair_clip_ext_he },


    // --log
    { "log",                                            no_argument,        NULL, OO_log },
    // --debug-log
    { "debug-log",                                      no_argument,        NULL, OO_debug_log },
    // --pen_flow_error
    { "pen-flow-error",                                 required_argument,  NULL, OO_pen_flow_error },
    // --softclip-key
    { "softclip-key",                                   optional_argument,  NULL, OO_softclip_key },
    // --ignore-flowgram
    { "ignore-flowgram",                                optional_argument,  NULL, OO_ignore_flowgram },
    // --final_flowspace
    { "final-flowspace",                                optional_argument,  NULL, OO_aln_flowspace },
    { NULL,                                             0,                  NULL, 0 }
};

static const char* short_opts = "A:M:O:E:G:K:w:g:J:X:YySF";

int string_to_args (char* argstring, char*** argv_p) // this would return 0 or 2 and above: never should return 1 (if works properly)
{
    // scan line for space-separated tokens, count and save them
    static const char *delim = " ";
    int argc;
    char** argv = NULL;
    int argv_alloc = 0, argv_chunk = 8;
    static char firstarg [] = "BEDOPT"; // :)
    char *token, *initstr, *context;
    for (argc = 0, initstr = argstring; (token = strtok_r (initstr, delim, &context)); ++argc, initstr = NULL)
    {
        if (argc >= argv_alloc)
        {
            argv_alloc += argv_chunk;
            argv_chunk *= 2;
            if (!argv)
                argv = tmap_malloc (sizeof (char*) * argv_alloc, "override_argv_alloc");
            else
                argv = tmap_realloc (argv, sizeof (char*) * argv_alloc, "override_argv_realloc");
        }
        if (!argc)
            argv [argc ++] = firstarg;
        argv [argc] = token;
    }
    if (argv_alloc > argc)
        argv = tmap_realloc (argv, sizeof (char*) * argc, "override_argv_realloc_shrink");
    *argv_p = argv;
    return argc;
}

// neither atoi nor strtol properly check for errors, and sscanf is too expensive.
// this amends strtol with naive error checking
static uint8_t str2int (const char* str, int* value)
{
    char* endptr;
    int errono_save = errno;
    errno = 0;
    long val = strtol (str, &endptr, 10);
    if (*endptr) // there was some garbage down a string
        return 0;
    if (val == 0)
    {
        // check if there is simple evidence that 0 just indicates no valid conversion
        while (*str)
        {
            if (!(isspace (*str) || *str == '+' || *str == '-' || (*str >=  '0' && *str <= '9')))
                return 0;
            if (*str >= '1' && *str <= '9')
                return 0;
            ++ str;
        }
    }
    if (val > INT_MAX || val < INT_MIN)
        return 0;
    *value = (int) val;
    return 1;
}

static uint8_t str2double (const char* str, double* value)
{
    char* endptr;
    int errono_save = errno;
    errno = 0;
    double val = strtod (str, &endptr);
    if (*endptr) // there was some garbage down a string
        return 0;
    if (val == 0.0)
    {
        // check if there is simple evidence that 0 just indicates no valid conversion
        while (*str)
        {
            if (!(isspace (*str) || *str == '+' || *str == '-' || *str == '.' || *str == 'e' || *str == 'E' || (*str >=  '0' && *str <= '9')))
                return 0;
            ++ str;
        }
        // 'semi-blindly assume best: if no invalid chars here, and 0.0 returned, than exponent was too small...
    }
    *value = val;
    return 1;
}

// parses string to get override parameters
// following TMAP options are recognized:
// -A,--score-match
// -M,--pen-mismatch
// -O,--pen-gap-open
// -E,--pen-gap-extension
// -G,--pen-gap-long
// -K,--gap-long-length
// -w,--band-width
// -g,--softclip-type
// --do-realign
// --r-mat
// --r-mis
// --r-gip
// --r-gep
// --r-bw
// --r-clip
// --do-repeat-clip
// --repclip-cont
// --context
// --gap-scale
// --c-mat
// --c-mis
// --c-gip
// --c-gep
// --c-bw
// --debug-log
// -X, --pen-flow-error
// --softclip-key
// --ignore-flowgram
// -F, --final-flowspace
// Following options are recognised per se as well as with -le and -he suffixes:
// --end-repair
// --max-one-large-indel-rescue
// --min-anchor-large-indel-rescue
// --max-er-5clip-large-indel-rescue
// -J,--max-adapter-bases-for-soft-clipping
// --er-no5clip
// the following option has a special meaning:
// --log : if specified, cancels the post-processing logging for all amplicons but the ones for which it is specified.
// returns number of parameter overrides succesfully parsed

uint32_t parse_overrides (tmap_map_locopt_t* local_params, char* param_spec_str, int32_t* specific_log, char* bed_fname, int lineno)
{
    char** argv = NULL;
    int argc = string_to_args (param_spec_str, &argv);
    if (argc < 2)
        return 0;
    // now use getopt_long to extract parameters
    uint32_t ovr_count = 0;
    optind = 0; // reset global state for getopt_long. This is GLIBC way to reset internal getopt_long state. Non-portable. Should not use 1 as for original K&R implementation.
    while (1)
    {
        int option_index;
        int value;
        double dvalue;
        int c = getopt_long (argc, argv, short_opts, overridable_opts, &option_index);
        if (c == -1)
            break;

        switch (c)
        {
            case OO_no_bed_er:
                if (!optarg) // optional argument not given : treat as 1
                    local_params->use_bed_in_end_repair.value = 0,
                    local_params->use_bed_in_end_repair.over = 1,
                    ++ovr_count;
                else if (str2int (optarg, &value))
                {
                    if (value < 0 || value > 1)
                        tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --no-bed-er", value);
                    else
                        local_params->use_bed_in_end_repair.value = value?0:1,
                        local_params->use_bed_in_end_repair.over = 1,
                        ++ovr_count;
                }
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --no-bed-er");
                break;
            case 'A':
            case OO_score_match:
                if (str2int (optarg, &value))
                    local_params->score_match.value = value,
                    local_params->score_match.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --score-match (-A)");
                break;
            case 'M':
            case OO_pen_mismatch:
                if (str2int (optarg, &value))
                    local_params->pen_mm.value = value,
                    local_params->pen_mm.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --pen-mismatch (-M)");
                break;
            case 'O':
            case OO_pen_gap_open:
                if (str2int (optarg, &value))
                    local_params->pen_gapo.value = value,
                    local_params->pen_gapo.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --pen-gap-open (-O)");
                break;
            case 'E':
            case OO_pen_gap_extension:
                if (str2int (optarg, &value))
                    local_params->pen_gape.value = value,
                    local_params->pen_gape.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --pen-gap-extension (-E)");
                break;
            case 'G':
            case OO_pen_gap_long:
                if (str2int (optarg, &value))
                    local_params->pen_gapl.value = value,
                    local_params->pen_gapl.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --pen-gap-long (-G)");
                break;
            case 'K':
            case OO_gap_long_length:
                if (str2int (optarg, &value))
                    local_params->gapl_len.value = value,
                    local_params->gapl_len.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --gap-long-length (-K)");
                break;
            case 'w':
            case OO_band_width:
                if (str2int (optarg, &value))
                    local_params->bw.value = value,
                    local_params->bw.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --band_width (-w)");
                break;
            case 'g':
            case OO_softclip_type:
                if (str2int (optarg, &value))
                {
                    if (value < 0 || value > 3)
                        tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --softclip-type (-g)", value);
                    else
                        local_params->softclip_type.value = value,
                        local_params->softclip_type.over = 1,
                        ++ovr_count;
                }
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --softclip-type (-g)");
                break;
            case OO_do_realign:
                if (!optarg) // optional argument not given : treat as 1
                    local_params->do_realign.value = 1,
                    local_params->do_realign.over = 1,
                    ++ovr_count;
                else if (str2int (optarg, &value))
                {
                    if (value < 0 || value > 1)
                        tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --do-realign", value);
                    else
                        local_params->do_realign.value = value,
                        local_params->do_realign.over = 1,
                        ++ovr_count;
                }
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --do-realign");
                break;
            case OO_r_mat:
                if (str2int (optarg, &value))
                    local_params->realign_mat_score.value = value,
                    local_params->realign_mat_score.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --r-mat");
                break;
            case OO_r_mis:
                if (str2int (optarg, &value))
                    local_params->realign_mis_score.value = value,
                    local_params->realign_mis_score.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --r-mis");
                break;
            case OO_r_gip:
                if (str2int (optarg, &value))
                    local_params->realign_gip_score.value = value,
                    local_params->realign_gip_score.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --r-gip");
                break;
            case OO_r_gep:
                if (str2int (optarg, &value))
                    local_params->realign_gep_score.value = value,
                    local_params->realign_gep_score.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --r-gep");
                break;
            case OO_r_bw:
                if (str2int (optarg, &value))
                    local_params->realign_bandwidth.value = value,
                    local_params->realign_bandwidth.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --r-bw");
                break;
            case OO_r_clip:
                if (str2int (optarg, &value))
                {
                    if (value < 0 || value > 4)
                        tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --r-clip", value);
                    else
                        local_params->realign_cliptype.value = value,
                        local_params->realign_cliptype.over = 1,
                        ++ovr_count;
                }
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --r-clip");
                break;
            case OO_do_repeat_clip:
                if (!optarg) // optional argument not given : treat as 1
                    local_params->do_repeat_clip.value = 1,
                    local_params->do_repeat_clip.over = 1,
                    ++ovr_count;
                else if (str2int (optarg, &value))
                {
                    if (value < 0 || value > 1)
                        tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --do-repeat-clip", value);
                    else
                        local_params->do_repeat_clip.value = value,
                        local_params->do_repeat_clip.over = 1,
                        ++ovr_count;
                }
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --do-repeat-clip");
                break;
            case OO_repclip_cont:
                if (!optarg) // optional argument not given : treat as 1
                    local_params->repclip_continuation.value = 1,
                    local_params->repclip_continuation.over = 1,
                    ++ovr_count;
                else if (str2int (optarg, &value))
                {
                    if (value < 0 || value > 1)
                        tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --repclip-cont", value);
                    else
                        local_params->repclip_continuation.value = value,
                        local_params->repclip_continuation.over = 1,
                        ++ovr_count;
                }
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repclip-cont");
                break;
            case OO_context:
                if (!optarg) // optional argument not given : treat as 1
                    local_params->do_hp_weight.value = 1,
                    local_params->do_hp_weight.over = 1,
                    ++ovr_count;
                else if (str2int (optarg, &value))
                {
                    if (value < 0 || value > 1)
                        tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --context", value);
                    else
                        local_params->do_hp_weight.value = value,
                        local_params->do_hp_weight.over = 1,
                        ++ovr_count;
                }
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --context");
                break;
            case OO_c_mat:
                if (str2double (optarg, &dvalue))
                    local_params->context_mat_score.value = dvalue,
                    local_params->context_mat_score.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --c-mat");
                break;
            case OO_c_mis:
                if (str2double (optarg, &dvalue))
                    local_params->context_mis_score.value = dvalue,
                    local_params->context_mis_score.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --c-mis");
                break;
            case OO_c_gip:
                if (str2double (optarg, &dvalue))
                    local_params->context_gip_score.value = dvalue,
                    local_params->context_gip_score.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --c-gip");
                break;
            case OO_c_gep:
                if (str2double (optarg, &dvalue))
                    local_params->context_gep_score.value = dvalue,
                    local_params->context_gep_score.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --c-gep");
                break;
            case OO_c_bw:
                if (str2int (optarg, &value))
                    local_params->context_extra_bandwidth.value = value,
                    local_params->context_extra_bandwidth.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --c-bw");
                break;
            case OO_end_repair:
                if (str2int (optarg, &value))
                    local_params->end_repair.value = value,
                    local_params->end_repair.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --end-repair");
                break;
            case OO_end_repair_he:
                if (str2int (optarg, &value))
                    local_params->end_repair_he.value = value,
                    local_params->end_repair_he.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --end-repair-he");
                break;
            case OO_end_repair_le:
                if (str2int (optarg, &value))
                    local_params->end_repair_le.value = value,
                    local_params->end_repair_le.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --end-repair-le");
                break;
            case OO_max_one_large_indel_rescue:
                if (str2int (optarg, &value))
                    local_params->max_one_large_indel_rescue.value = value,
                    local_params->max_one_large_indel_rescue.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --max-one-large-indel-rescue");
                break;
            case OO_max_one_large_indel_rescue_he:
                if (str2int (optarg, &value))
                    local_params->max_one_large_indel_rescue_he.value = value,
                    local_params->max_one_large_indel_rescue_he.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --max-one-large-indel-rescue-he");
                break;
            case OO_max_one_large_indel_rescue_le:
                if (str2int (optarg, &value))
                    local_params->max_one_large_indel_rescue_le.value = value,
                    local_params->max_one_large_indel_rescue_le.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --max-one-large-indel-rescue-le");
                break;
            case OO_min_anchor_large_indel_rescue:
                if (str2int (optarg, &value))
                    local_params->min_anchor_large_indel_rescue.value = value,
                    local_params->min_anchor_large_indel_rescue.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --min-anchor-large-indel-rescue");
                break;
            case OO_min_anchor_large_indel_rescue_he:
                if (str2int (optarg, &value))
                    local_params->min_anchor_large_indel_rescue_he.value = value,
                    local_params->min_anchor_large_indel_rescue_he.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --min-anchor-large-indel-rescue-he");
                break;
            case OO_min_anchor_large_indel_rescue_le:
                if (str2int (optarg, &value))
                    local_params->min_anchor_large_indel_rescue_le.value = value,
                    local_params->min_anchor_large_indel_rescue_le.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --min-anchor-large-indel-rescue-le");
                break;
            case OO_max_amplicon_overrun_large_indel_rescue:
                if (str2int (optarg, &value))
                    local_params->max_amplicon_overrun_large_indel_rescue.value = value,
                    local_params->max_amplicon_overrun_large_indel_rescue.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --amplicon-overrun");
                break;
            case OO_max_amplicon_overrun_large_indel_rescue_he:
                if (str2int (optarg, &value))
                    local_params->max_amplicon_overrun_large_indel_rescue_he.value = value,
                    local_params->max_amplicon_overrun_large_indel_rescue_he.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --amplicon-overrun-he");
                break;
            case OO_max_amplicon_overrun_large_indel_rescue_le:
                if (str2int (optarg, &value))
                    local_params->max_amplicon_overrun_large_indel_rescue_le.value = value,
                    local_params->max_amplicon_overrun_large_indel_rescue_le.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --amplicon-overrun-le");
                break;
            case 'J':
            case OO_max_adapter_bases_for_soft_clipping:
                if (str2int (optarg, &value))
                    local_params->max_adapter_bases_for_soft_clipping.value = value,
                    local_params->max_adapter_bases_for_soft_clipping.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --max-adapter-bases-for-soft-clipping");
                break;
            case OO_max_adapter_bases_for_soft_clipping_he:
                if (str2int (optarg, &value))
                    local_params->max_adapter_bases_for_soft_clipping_he.value = value,
                    local_params->max_adapter_bases_for_soft_clipping_he.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --max-adapter-bases-for-soft-clipping-he");
                break;
            case OO_max_adapter_bases_for_soft_clipping_le:
                if (str2int (optarg, &value))
                    local_params->max_adapter_bases_for_soft_clipping_le.value = value,
                    local_params->max_adapter_bases_for_soft_clipping_le.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --max-adapter-bases-for-soft-clipping-le");
                break;
            case OO_er_no5clip:
                if (!optarg) // optional argument not given : treat as 1 => disable er_5'_softclip (store ZERO)
                    local_params->end_repair_5_prime_softclip.value = 0, // 0 is correct
                    local_params->end_repair_5_prime_softclip.over = 1,
                    ++ovr_count;
                else
                {
                    if (str2int (optarg, &value))
                    {
                        if (value < 0 || value > 1)
                            tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --er-no5clip", value);
                        local_params->end_repair_5_prime_softclip.value = !value,
                        local_params->end_repair_5_prime_softclip.over = 1,
                        ++ovr_count;
                    }
                    else
                        tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --er-no5clip");
                }
                break;
            case OO_er_no5clip_he:
                if (!optarg) // optional argument not given : treat as 1 => disable er_5'_softclip (store ZERO)
                    local_params->end_repair_5_prime_softclip_he.value = 0, // 0 is correct
                    local_params->end_repair_5_prime_softclip_he.over = 1,
                    ++ovr_count;
                else
                {
                    if (str2int (optarg, &value))
                    {
                        if (value < 0 || value > 1)
                            tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --er-no5clip-he", value);
                        local_params->end_repair_5_prime_softclip_he.value = !value,
                        local_params->end_repair_5_prime_softclip_he.over = 1,
                        ++ovr_count;
                    }
                    else
                        tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --er-no5clip-he");
                }
                break;
            case OO_er_no5clip_le:
                if (!optarg) // optional argument not given : treat as 1 => disable er_5'_softclip (store ZERO)
                    local_params->end_repair_5_prime_softclip_le.value = 0, // 0 is correct
                    local_params->end_repair_5_prime_softclip_le.over = 1,
                    ++ovr_count;
                else
                {
                    if (str2int (optarg, &value))
                    {
                        if (value < 0 || value > 1)
                            tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --er-no5clip-le", value);
                        local_params->end_repair_5_prime_softclip_le.value = !value,
                        local_params->end_repair_5_prime_softclip_le.over = 1,
                        ++ovr_count;
                    }
                    else
                        tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --er-no5clip-le");
                }
                break;

            case OO_repair_min_adapter:
                if (str2int (optarg, &value))
                    local_params->repair_min_adapter.value = value,
                    local_params->repair_min_adapter.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-min-adapter");
                break;
            case OO_repair_min_adapter_he:
                if (str2int (optarg, &value))
                    local_params->repair_min_adapter_he.value = value,
                    local_params->repair_min_adapter_he.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-min-adapter-he");
                break;
            case OO_repair_min_adapter_le:
                if (str2int (optarg, &value))
                    local_params->repair_min_adapter_le.value = value,
                    local_params->repair_min_adapter_le.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-min-adapter-le");
                break;
            case OO_repair_max_overhang:
                if (str2int (optarg, &value))
                    local_params->repair_max_overhang.value = value,
                    local_params->repair_max_overhang.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-max-overhang");
                break;
            case OO_repair_max_overhang_he:
                if (str2int (optarg, &value))
                    local_params->repair_max_overhang_he.value = value,
                    local_params->repair_max_overhang_he.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-max-overhang-he");
                break;
            case OO_repair_max_overhang_le:
                if (str2int (optarg, &value))
                    local_params->repair_max_overhang_le.value = value,
                    local_params->repair_max_overhang_le.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-max-overhang-le");
                break;
            case OO_repair_identity_drop_limit:
                if (str2double (optarg, &dvalue))
                    local_params->repair_identity_drop_limit.value = dvalue,
                    local_params->repair_identity_drop_limit.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-identity-drop-limit");
                break;
            case OO_repair_identity_drop_limit_he:
                if (str2double (optarg, &dvalue))
                    local_params->repair_identity_drop_limit_he.value = dvalue,
                    local_params->repair_identity_drop_limit_he.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-identity-drop-limit-he");
                break;
            case OO_repair_identity_drop_limit_le:
                if (str2double (optarg, &dvalue))
                    local_params->repair_identity_drop_limit_le.value = dvalue,
                    local_params->repair_identity_drop_limit_le.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-identity-drop-limit-le");
                break;
            case OO_repair_max_primer_zone_dist:
                if (str2int (optarg, &value))
                    local_params->repair_max_primer_zone_dist.value = value,
                    local_params->repair_max_primer_zone_dist.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-max-primer-zone-dist");
                break;
            case OO_repair_max_primer_zone_dist_he:
                if (str2int (optarg, &value))
                    local_params->repair_max_primer_zone_dist_he.value = value,
                    local_params->repair_max_primer_zone_dist_he.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-max-primer-zone-dist-he");
                break;
            case OO_repair_max_primer_zone_dist_le:
                if (str2int (optarg, &value))
                    local_params->repair_max_primer_zone_dist_le.value = value,
                    local_params->repair_max_primer_zone_dist_le.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-max-primer-zone-dist-le");
                break;

            case OO_repair_clip_ext:
                if (str2int (optarg, &value))
                    local_params->repair_clip_ext.value = value,
                    local_params->repair_clip_ext.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-clip-ext");
                break;
            case OO_repair_clip_ext_he:
                if (str2int (optarg, &value))
                    local_params->repair_clip_ext_he.value = value,
                    local_params->repair_clip_ext_he.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-clip-ext-he");
                break;
            case OO_repair_clip_ext_le:
                if (str2int (optarg, &value))
                    local_params->repair_clip_ext_le.value = value,
                    local_params->repair_clip_ext_le.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --repair-clip-ext-le");
                break;

            case OO_log:
                local_params->specific_log.value = 1,
                local_params->specific_log.over = 1,
                ++(*specific_log);
                ++ovr_count;
                break;
            case OO_debug_log:
                local_params->debug_log.value = 1,
                local_params->debug_log.over = 1,
                ++ovr_count;
                break;
            case 'X':
            case OO_pen_flow_error:
                if (str2int (optarg, &value))
                    local_params->fscore.value = value,
                    local_params->fscore.over = 1,
                    ++ovr_count;
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --pen_flow_error (-X)");
                break;
            case 'Y':
            case OO_softclip_key:
                if (!optarg) // optional argument not given : treat as 1
                    local_params->softclip_key.value = 1,
                    local_params->softclip_key.over = 1,
                    ++ovr_count;
                else if (str2int (optarg, &value))
                {
                    if (value < 0 || value > 1)
                        tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --softclip-key", value);
                    else
                        local_params->softclip_key.value = value,
                        local_params->softclip_key.over = 1,
                        ++ovr_count;
                }
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --softclip-key");
                break;
            case 'S':
            case OO_ignore_flowgram:
                if (!optarg) // optional argument not given : treat as 1
                    local_params->ignore_flowgram.value = 1,
                    local_params->ignore_flowgram.over = 1,
                    ++ovr_count;
                else if (str2int (optarg, &value))
                {
                    if (value < 0 || value > 1)
                        tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --ignore_flowgram", value);
                    else
                        local_params->ignore_flowgram.value = value,
                        local_params->ignore_flowgram.over = 1,
                        ++ovr_count;
                }
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --ignore_flowgram");
                break;
            case 'F':
            case OO_aln_flowspace:
                if (!optarg) // optional argument not given : treat as 1
                    local_params->aln_flowspace.value = 1,
                    local_params->aln_flowspace.over = 1,
                    ++ovr_count;
                else if (str2int (optarg, &value))
                {
                    if (value < 0 || value > 1)
                        tmap_user_fileproc_msg (bed_fname, lineno, "Invalid value (%d) for override for --final-flowspace", value);
                    else
                        local_params->aln_flowspace.value = value,
                        local_params->aln_flowspace.over = 1,
                        ++ovr_count;
                }
                else
                    tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters override: --final-flowspace");
                break;
        }
    }
    free (argv);
    return ovr_count;
}

// parse description (last) field from "BED Detail" file to obtain parameters.
// The parameters should appear same way as on command line. No escape or quotation parsing is performed, as none of the overridable parameters can carry complex string values.
// format is TMAP_OVERRIDE{ --tmap_option[ value] ....}
// returns numer of overrides parsed

uint32_t extract_overrides (tmap_map_locopt_t* local_params, const char* description, int32_t* local_logs, char* bed_fname, int lineno)
{
    // finds TMAP_OVERRIDE{...} block, cals parse_override on it's content
    static const char BLOCK_HEAD [] = "TMAP_OVERRIDE";
    static const char BLOCK_OPEN = '{';
    static const char BLOCK_CLOSE = '}';
    char* block_beg, *block_end;
    block_beg = strstr (description, BLOCK_HEAD);
    if (!block_beg)
        return 0;
    block_beg += sizeof (BLOCK_HEAD) - 1;
    // scan for opening brace
    while (isspace (*block_beg) || *block_beg == '=')
        ++block_beg;
    if (*block_beg != BLOCK_OPEN)
    {
        tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters overrides: No parameters block opening found");
        return 0;
    }
    ++block_beg; // point to first char inside param block
    // find block close
    block_end = block_beg;
    while (*block_end && *block_end != BLOCK_CLOSE)
        ++block_end;
    if (*block_end != BLOCK_CLOSE)
    {
        tmap_user_fileproc_msg (bed_fname, lineno, "Error parsing parameters overrides: Parameters block is not closed");
        return 0;
    }
    // end string at block end;
    // *block_end = 0;
    // pass parmeters block strng to parser
    // copy block to temp buffer as parse_overrides will modify it while parsing
    size_t block_len = block_end - block_beg;
    char* block = alloca (block_len + 1);
    memcpy (block, block_beg, block_len);
    // replace '@' symbols with '=' symbols: this is needed to keep BED description consistent with key = value; syntax
    char* b, *sent;
    for (b = block, sent = block + block_len; b != sent; b ++)
        if (*b == '@') *b = '=';
    block [block_len] = 0;
    return parse_overrides (local_params, block, local_logs, bed_fname, lineno);
}

// assumes passed in string is semicolon-separated list of tags
// extracts string value for a given tag; 
// returns pointer to where the value string starts, put value's length into variable pointed by len
// if no value for a tag can be found, returns NULL
char* extract_tag_value (const char* string, const char* tag, int* len)
{
    static const char tag_delim [] = ";\n";
    static const char ignored [] = "= ";

    char* entry = strstr (string, tag);
    if (!entry)
    {
        if (len) *len = 0;
        return NULL;
    }
    entry += strlen (tag);
    while (*entry && strchr (ignored, *entry))
        ++entry;
    if (!*entry)
    {
        if (len) *len = 0;
        return entry;
    }
    char* end = strpbrk (entry, tag_delim);
    if (!end)
        end = entry + strlen (entry);
    if (len) *len = end - entry;
    return entry;
}

static void endpos_init (tmap_map_endpos_t* endpos)
{
    endpos->coord = 0;
    endpos->count = 0;
    endpos->fraction = 0.0;
    endpos->flag = 0;
};

const unsigned MAX_SAFE_DIST = 50; // warn if read start/end is out of the amplicon boundaries by more then this number of bases

int32_t extract_and_store_read_ends_tag_values (const char* description, const char* tag, tmap_refseq_t* refseq, char* bed_fname, int lineno, const char* chr, int32_t ampl_beg, int32_t ampl_end)
{
    static const char* BLOCK_DELIM = ":";
    static const char* ELEM_DELIM = "|";
    int32_t valid_blocks = 0;
    int value_len;
    char* value = extract_tag_value (description, tag, &value_len);
    if (!value)
        return 0;
    if (!value_len)
    {
        tmap_user_fileproc_msg (bed_fname, lineno, "Warning: Empty value for %s", tag);
        return 0;
    }
    // make copy - no changes to passed in description is made
    char* vcopy = alloca (value_len + 1);
    memcpy (vcopy, value, value_len);
    vcopy  [value_len] = 0;
    // parse and store
    char *block, *block_context, *block_initstr;
    int blockno;
    for (block_initstr = block = vcopy, blockno = 0; block != NULL; ++blockno, block_initstr = NULL)
    {
        if ((block = strtok_r (block_initstr, BLOCK_DELIM, &block_context)))
        {
            char *tok, *tok_context, *tok_initstr;
            int tokno;
            tmap_map_endpos_t endpos;
            endpos_init (&endpos);
            int blocklen = strlen (block);
            // assert (blocklen <= value_len); // sanity
            char* bcopy = alloca (blocklen + 1);
            memcpy (bcopy, block, blocklen +1);
            //  printf ("Blocklen=%d, block = %s\n", blocklen, bcopy);
            uint8_t block_valid = 1;
            for (tok_initstr = tok = bcopy, tokno = 0; 
                (tok = strtok_r (tok_initstr, ELEM_DELIM, &tok_context)) != NULL && block_valid; 
                tok_initstr = NULL, ++tokno)
            {
                switch (tokno)
                {
                    case 0:
                        if (!str2int (tok, &endpos.coord))
                        {
                            tmap_user_fileproc_msg (bed_fname, lineno, "Warning: Can not parse coordinate (from '%s') from block %d of %s (%s). Block ignored.", tok, blockno, tag, block);
                            block_valid = 0;
                        }
                        else if (endpos.coord < 0)
                        {
                            tmap_user_fileproc_msg (bed_fname, lineno, "Warning: %s: invalid position (%d) in block# %d (amplicon %s:%d-%d). Block ignored.", tag, endpos.coord, blockno, chr, ampl_beg, ampl_end);
                            block_valid = 0;
                        }
                        // maybe add validation by the chromosome size? 
                        else if (endpos.coord + MAX_SAFE_DIST < ampl_beg || endpos.coord > ampl_end + MAX_SAFE_DIST)
                        {
                            tmap_user_fileproc_msg (bed_fname, lineno, "Warning: %s position %d in block# %d is too far out of the amplicon boundaries (%s:%d-%d). Block ignored.", tag, endpos.coord, blockno, chr, ampl_beg, ampl_end);
                            block_valid = 0;
                        }
                        break;
                    case 1:
                        if (!str2int (tok, &endpos.count))
                        {
                            tmap_user_fileproc_msg (bed_fname, lineno, "Warning: Can not parse count (from '%s') from block %d of %s (%s). Block ignored.", tok, blockno, tag, block);
                            block_valid = 0;
                        }
                        else if (endpos.count < 0)
                        {
                            tmap_user_fileproc_msg (bed_fname, lineno, "Warning: %s: invalid count (%d) in block# %d (amplicon %s:%d-%d). Block ignored.", tag, endpos.count, blockno, chr, ampl_beg, ampl_end);
                            block_valid = 0;
                        }
                        break;
                    case 2:
                        if (!str2double (tok, &endpos.fraction))
                        {
                            tmap_user_fileproc_msg (bed_fname, lineno, "Warning: Can not parse fraction (from '%s') from %d block %d of %s (%s). Block ignored.", tok, blockno, tag, block);
                            block_valid = 0;
                        }
                        else if (endpos.fraction < 0)
                        {
                            tmap_user_fileproc_msg (bed_fname, lineno, "Warning: %s: invalid fraction (%g) in block# %d (amplicon %s:%d-%d). Block ignored.", tag, endpos.fraction, blockno, chr, ampl_beg, ampl_end);
                            block_valid = 0;
                        }
                        break;
                    case 3:
                        // we may want some validation here. 
                        endpos.flag = *tok;
                        break;
                    default:
                        ;
                }
            }
            if (!block_valid)
                continue;
            if (tokno > 4)
                tmap_user_fileproc_msg (bed_fname, lineno, "Warning: Ignoring %d extra fields in %s block %d (%s).", tokno - 3, tag, blockno, block);
            if (tokno > 0 && tokno < 3)
                tmap_user_fileproc_msg (bed_fname, lineno, "Warning: Ignoring incomplete %s block %d (%s): %d fields found, 3 or more needed.", tag, blockno, block, tokno);
            if (tokno == 3)
                tmap_user_fileproc_msg (bed_fname, lineno, "Warning: %s block %d (%s) is missing Flag fied. Zero value is used.", tag, blockno, block);
            if (tokno >= 3)
            {
                // manage memory if necessary
                if (refseq->endposmem_used == refseq->endposmem_size)
                {
                    refseq->endposmem_size = refseq->endposmem_size ? (refseq->endposmem_size<<1) : READ_ENDS_MEM_INIT_CHUNK;
                    refseq->endposmem = tmap_realloc (refseq->endposmem, refseq->endposmem_size * sizeof (tmap_map_endpos_t), "refseq->endposmem"); // no need to disctiminate malloc ad realloc: C realloc handles null pointers safely
                }
                // store
                refseq->endposmem [refseq->endposmem_used ++] = endpos;
                ++valid_blocks;
            }
        }
    }
    return valid_blocks;
}

// extracts read_ends and put into storage (refseq->endposmem), reallocating as needed and updating control memebers endposmem_used and endposmem_size
// puts the addresses of saved data into tmap_map_endstat_t, passed in by pointer.
// returns number of ends stored, or -1 on (unrecoverable) error

int32_t extract_read_ends (char* description, tmap_map_endstat_t* read_ends, tmap_refseq_t* refseq, char* bed_fname, int lineno, const char* chr, int32_t ampl_beg, int32_t ampl_end)
{
    static const char* READ_STARTS_TAG = "READ_STARTS";
    static const char* READ_ENDS_TAG = "READ_ENDS";
    read_ends->index = UINT32_MAX, read_ends->starts_count = 0, read_ends->ends_count = 0;

    int32_t starts_count = extract_and_store_read_ends_tag_values (description, READ_STARTS_TAG, refseq, bed_fname, lineno, chr, ampl_beg, ampl_end);
    if (starts_count == -1)
        tmap_failure ("%s:%d : Unrecoverable error while parsing READ_STARTS",  bed_fname, lineno);
    int32_t ends_count = extract_and_store_read_ends_tag_values (description, READ_ENDS_TAG, refseq, bed_fname, lineno, chr, ampl_beg, ampl_end);
    if (ends_count == -1)
        tmap_failure ("%s:%d : Unrecoverable error while parsing READ_ENDS",  bed_fname, lineno);
    uint32_t stored = starts_count + ends_count;
    if (stored)
    {
        read_ends->index = refseq->endposmem_used - stored;
        read_ends->starts_count = starts_count;
        read_ends->ends_count = ends_count;
    }
    return stored;
}


// ZZ:The bed file need to be sorted by start positions.
static int
tmap_refseq_read_bed_core (tmap_refseq_t *refseq, char *bedfile, int flag, int32_t use_par_ovr, int32_t  use_read_ends, int* local_logs, char **chrs, int max_num_chr)
{
    if (bedfile == NULL)
    {
        refseq->bed_exist = 0;
        return 1;
    }
    if (flag == 0 && refseq->num_annos == 0)
    {
        refseq->bed_exist = 0;
        tmap_error ("Refseq does not have any contigs, cannot read bed file", Warn, OutOfRange);
        return 0;
    }
    FILE *fp = fopen(bedfile, "r");
    if (fp == NULL) 
        tmap_warning ("Cannot open bed file %s\n", bedfile);
    if (fp == NULL) 
    {
        refseq->bed_exist = 0;
        return 0; //this causes TMAP to terminate.
    }
    else
        refseq->bed_exist = 1;

    char line[10000], last_chr[100];
    int32_t seq_id = -1;
    uint32_t i, num = 0, *b = NULL, *e = NULL, memsize = 0;
    uint32_t last_parovr_mem_size = 0;
    uint32_t last_read_ends_mem_size = 0;
    uint32_t overrides_count = 0;
    uint32_t read_ends_count = 0;
    uint32_t n_anno = refseq->num_annos;
    if (flag) n_anno = max_num_chr;
    refseq->bednum = tmap_malloc (sizeof(uint32_t) * n_anno, "refseq->bednum");
    refseq->bedstart = tmap_malloc (sizeof(uint32_t *) * n_anno, "refseq->bedstart");
    refseq->bedend = tmap_malloc (sizeof(uint32_t *) * n_anno, "refseq->bedend");
    refseq->parovr = NULL; // lazy alloc if needed
    refseq->parmem = NULL;
    refseq->parmem_size = refseq->parmem_used = 0;
    memset (refseq->bednum, 0, sizeof (uint32_t) * n_anno);
    memset (refseq->bedstart, 0, sizeof (uint32_t *) * n_anno);
    memset (refseq->bedend, 0, sizeof (uint32_t *) * n_anno);
    refseq->beditem = n_anno;
    last_chr[0] = 0;
    char *token, *context, *initstr;
    const char *delim = "\t"; // use tab delimiting; if real BEDs can be space delimited, we should change the processing significantly (it'll be not easy to handle such case, as number of fields in bed detail can be anything from 4+1 to 12+2)
    uint32_t columns_no = -1; // for chechking consistency of columns count across different lines
    uint32_t lineno = 0;
    uint32_t recno = 0;
    tmap_map_locopt_t local_params;
    tmap_map_endstat_t read_ends;
    uint32_t tokno;
    char chr [100];
    uint32_t beg, end;
    uint8_t track_line_seen = 0;

    while (fgets (line, sizeof (line), fp))
    {
        ++lineno;
        if (line [0] == '#')
                continue;
        if (strncmp ("track", line, 5) == 0)
        {
            if (recno)
            {
                tmap_user_warning ("BED file %s contains trac record on line %d, after %d region records: Track record ignored", bedfile, lineno, recno);
                continue;
            }
            if (track_line_seen)
            {
                tmap_user_warning ("BED file %s contains multiple trac records: extra at line %s. All but first are ignored", bedfile, lineno);
                continue;
            }
            // DK check if track line contains 'type=bedDetail' clause
            // do just a silly check here, assume no funny concatenations should be cared about
            if (use_par_ovr && !strstr (line, "type=bedDetail"))
            {
                tmap_user_warning ("Per-amplicon parameter override is requested, but BED file %s is not of the bedDetail format. Amplicon specific parameters can not be used.", bedfile);
                use_par_ovr = 0; // disable override use
            }
            if (use_read_ends && !strstr (line, "type=bedDetail"))
            {
                tmap_user_warning ("Use of read ends statistics is requested, but BED file %s is not of the bedDetail format. Read ends statistics can not be used.", bedfile);
                use_read_ends = 0; // disable override use
            }
            track_line_seen = 1;
            continue;
        }
        sscanf(line, "%99s %u %u", chr, &beg, &end);
        ++recno;
        overrides_count = 0;
        read_ends_count = 0;
        if (use_par_ovr || use_read_ends)
        {
            // parse last column, extract parameters overrides if any
            char* last_tok = NULL;
            for (tokno = 0, initstr = token = line; token; ++tokno, initstr = NULL)
            {
                last_tok = token;
                token = strtok_r (initstr, delim, &context);
            }
            if (columns_no == -1)
            {
                if (tokno < 6)
                    tmap_user_warning ("Bed file format violation: file %s, line %d: bedDetail format is specified in 'track' line, but only %d columns (<6) present. Description field not extracted from this line.", bedfile, lineno, tokno);
                else
                    columns_no = tokno;
            }
            else
            {
                if (tokno < 6)
                    tmap_user_warning ("Bed file format violation: file %s, line %d: bedDetail format is specified in 'track' line, but only %d columns (<6) present. Description field not extracted from this line.", bedfile, lineno, tokno);
                if (tokno != columns_no)
                    tmap_user_warning ("Bed file format violation: file %s, line %d: Inconsistent number of fields: %d seen on this line, %d seen on earlier line(s).", bedfile, lineno, tokno, columns_no);
            }
            if (tokno >= 6) // description field present on this line
            {
                if (use_par_ovr)
                {
                    // find the TMAP_OVERRIDE block if any; get the overrides from the block
                    tmap_map_locopt_init (&local_params);
                    overrides_count = extract_overrides (&local_params, last_tok, local_logs, bedfile, lineno);
                }
                if (use_read_ends)
                {
                    // find the READ_STARTS and READ_ENDS blocks and parse / store them
                    read_ends_count = extract_read_ends (last_tok, &read_ends, refseq, bedfile, lineno, chr, beg, end);
                }
            }
        }

        if (strcmp (last_chr, chr) != 0) // next (or first) contig (assuming BED is sorted)
        {
            memsize = 1000;
            last_parovr_mem_size = 0;
            last_read_ends_mem_size = 0;
            if (num > 0)
            {
                refseq->bednum [seq_id] = num;
                num = 0;
                refseq->bedstart [seq_id] = b;
                refseq->bedend [seq_id] = e;
                if (flag) 
                    chrs [seq_id] = strsave(last_chr);
            }
            int32_t next_id = flag ? (seq_id + 1) : tmap_refseq_get_id (refseq, chr);
            if (next_id < 0 || next_id <= seq_id)
            {
                // fprintf(stderr, "ZZ warning %s %d\t%d\n", chr, seq_id, next_id);
                tmap_error("Bed file is not sorted by chromosome order", Warn,  OutOfRange);
                fclose (fp);
                return 0;
            }
            seq_id = next_id;
            if (flag && seq_id > n_anno)
                tmap_error("exceed the max number of chromosomes", Warn, OutOfRange);

            strcpy (last_chr, chr);
            b = tmap_malloc (sizeof (uint32_t) *memsize, "tmpb");
            e = tmap_malloc (sizeof (uint32_t) *memsize, "tmpe");
        }
        else
        {
            if (num >= memsize)
            {
                uint32_t prevsize = memsize;
                memsize *= 3;
                b = tmap_realloc (b, sizeof (uint32_t) *memsize, "realloc_b");
                e = tmap_realloc (e, sizeof (uint32_t) *memsize, "realloc_e");
                // TS-17849
                // if there are allready any parovr entries allocated for this contig, keep the size of contig's parovr array in sync with the size of amplicon beg/end arrays.
                if (refseq->parovr && refseq->parovr [seq_id]) // could be check for (last_parovr_mem_size != NULL)
                {
                    refseq->parovr [seq_id] = tmap_realloc (refseq->parovr [seq_id], sizeof (uint32_t) * memsize, "refseq->parovr[seq_id]");
                    uint32_t *par_idx, *par_idx_sent;
                    for (par_idx = refseq->parovr [seq_id] + last_parovr_mem_size, 
                            par_idx_sent = refseq->parovr [seq_id] + memsize; 
                            par_idx != par_idx_sent;
                            ++par_idx) 
                        *par_idx = UINT32_MAX;
                    last_parovr_mem_size = memsize;
                }
            }
            if (b [num-1] > beg) 
            {
                tmap_error ("Bed file is not sorted by begin", Warn, OutOfRange);
                fclose (fp);
                return 0;
            }
            else if (e [num-1] >= end)
            {
                // ZZ:current ampl is contained in the previous one
                continue;
                // DK: overrides for containing one preceed
            }
            else if (b [num-1] == beg)
            {
                // ZZ:current one has same beg, but larger end, replace previous one
                num--;
                // DK: overrides for longer one preceed
            }
        }
        b [num] = beg;
        e [num] = end;

        if (overrides_count) // this is possible only if use_par_ovr is not false
        {
            if (refseq->parovr == NULL) // encountered actual override. allocate storage for contig's overrides, set each contig's override pointer to NULL
            {
                refseq->parovr = tmap_malloc (sizeof (uint32_t *) * n_anno, "refseq->parovr");
                memset (refseq->parovr, 0, sizeof (uint32_t*) * n_anno); // NULL should always be all-bytes-zeroes, safe enough.
            }
            if (memsize > last_parovr_mem_size) // no need to check if the allocation is first: realloc handles NULL ptr reallocation safely. 
                                                // After TS-17849 fix, This actually is invoked at first allocation only; for the rest, size of provr array is kept in sync with the size of ampl beg/end arrays
                                                // (so the test can be replaced with if (!last_parovr_mem_size)
            {
                refseq->parovr [seq_id] = tmap_realloc (refseq->parovr [seq_id], sizeof (uint32_t) * memsize, "refseq->parovr[seq_id]");
                // memset (refseq->parovr [seq_id] + last_parovr_mem_size, 0xFF, (memsize - last_parovr_mem_size) * sizeof (tmap_map_locopt_t*));
                // the above is not so portable, explicit (below) is safe
                uint32_t *par_idx, *par_idx_sent;
                for (par_idx = refseq->parovr [seq_id] + last_parovr_mem_size, 
                     par_idx_sent = refseq->parovr [seq_id] + memsize; 
                     par_idx != par_idx_sent;
                     ++par_idx) 
                    *par_idx = UINT32_MAX;
                last_parovr_mem_size = memsize;
            }
            // make sure there is free slot for this set of overrides
            if (refseq->parmem_used == refseq->parmem_size) // no need to check if the allocation is first: realloc handles NULL ptr reallocation safely.
            {
                refseq->parmem_size = refseq->parmem_size ? (refseq->parmem_size * 2) : OVR_PAR_MEM_INIT_CHUNK;
                refseq->parmem = tmap_realloc (refseq->parmem, refseq->parmem_size * sizeof (tmap_map_locopt_t), "parameter overrides storage");
            }
            // store (plain copy)
            refseq->parmem [refseq->parmem_used] = local_params;
            // the pointer is (refseq->parmem + refseq->parmem_used); store it below
            refseq->parovr [seq_id][num] = refseq->parmem_used;
            ++refseq->parmem_used;
        }
        if (read_ends_count)
        {
            if (refseq->read_ends == NULL)
            {
                refseq->read_ends = tmap_malloc (sizeof (tmap_map_endstat_t*) * n_anno, "refseq->read_ends");
                memset (refseq->read_ends, 0, sizeof (tmap_map_endstat_t*) * n_anno); // NULL should always be all-bytes-zeroes, safe enough.
            }
            if (memsize > last_read_ends_mem_size) // no need to check if the allocation is first: realloc handles NULL ptr reallocation safely.
            {
                refseq->read_ends [seq_id] = tmap_realloc (refseq->read_ends [seq_id], sizeof (tmap_map_endstat_t) * memsize, "refseq->read_ends[seq_id]");
                tmap_map_endstat_t *read_ends_ptr, *read_ends_sent;
                for (read_ends_ptr = refseq->read_ends [seq_id] + last_read_ends_mem_size, 
                     read_ends_sent = refseq->read_ends [seq_id] + memsize; 
                     read_ends_ptr != read_ends_sent;
                     ++read_ends_ptr)
                    read_ends_ptr->index = UINT32_MAX, read_ends_ptr->starts_count = 0, read_ends_ptr->ends_count = 0;
                last_read_ends_mem_size = memsize;
            }
            refseq->read_ends [seq_id][num] = read_ends;
        }
        ++num;
    }

    // last ones
    if (seq_id >=0)  
    {
        refseq->bednum [seq_id] = num;
        refseq->bedstart [seq_id] = b;
        refseq->bedend [seq_id] = e;
        if (flag) 
            chrs [seq_id] = strsave (last_chr);
    }
    if (flag) 
        refseq->beditem = seq_id+1;
    fclose (fp);
    return 1;
}

int
tmap_refseq_read_bed (tmap_refseq_t *refseq, char *bedfile, int32_t use_par_ovr, int32_t use_read_ends, int32_t* local_logs)
{
    return tmap_refseq_read_bed_core (refseq, bedfile, 0, use_par_ovr, use_read_ends, local_logs, NULL, 0);
}

static void
find_next_bed(tmap_refseq_t *refseq, int cur_ind, int cur_pos, int *b, int *e, int *n)
{
    int i = *n;
    *b = -1;
    while (i < refseq->bednum[cur_ind] && refseq->bedend[cur_ind][i] <= cur_pos) i++;
    *n = i;
    if (i < refseq->bednum[cur_ind]) {
    *b = refseq->bedstart[cur_ind][i];
    *e = refseq->bedend[cur_ind][i];
    }
}

static int
find_chr_ind(char **chrs, int nchr, char *cur_chr)
{
    int i = 0;
    for (; i < nchr; i++) {
    if (strcmp(cur_chr, chrs[i]) == 0) return i;
    }
    return -1;
}

static void all_N(char *start)
{
    while (*start && *start != '\n') {
        *start = 'N';
        start++;
    }
}


int tmap_refseq_fasta2maskedfasta_main(int argc, char *argv[])
{
    // load bed file in some arrayint argc, char *argv[])
  int c, help=0;
  char *out_nomask = NULL;
  int max_num_chr = 1000;

  while((c = getopt(argc, argv, "o:m:vh")) >= 0) {
      switch(c) {
        case 'v': tmap_progress_set_verbosity(1); break;
        case 'h': help = 1; break;
    case 'o': out_nomask = optarg; break;
    case 'm': max_num_chr = atoi(optarg); break;
        default: return 1;
      }
  }
  if(2!= argc - optind || 1 == help) {
      tmap_file_fprintf(tmap_file_stderr, "Usage: %s %s [-o outNomas -m maxChr -vh] <in.bed> <in.fasta>\n", PACKAGE, argv[0]);
      return 1;
  }

  //alloc refseq
  tmap_refseq_t *refseq =  tmap_calloc(1, sizeof(tmap_refseq_t), "refseq");

  refseq->version_id = TMAP_VERSION_ID;
  refseq->annos = NULL;
  refseq->num_annos = 0;
  refseq->len = 0;


  char **chrs = tmap_malloc (sizeof(char *)*max_num_chr, "chrs");
  tmap_refseq_read_bed_core (refseq, argv[optind], 1, 0, 0, NULL, chrs, max_num_chr);


    // read in fasta file and put mask at bases

        FILE *fp = fopen(argv[optind+1], "r");
        if (fp == NULL) {
                tmap_file_fprintf(tmap_file_stderr, "Cannot open %s\n", argv[optind+1]);
                return 1;
        }
    FILE *fo = NULL;
    if (out_nomask) fo = fopen(out_nomask, "w");

        char line[100000];
        char cur_chr[100];
        int b = -1, e = -1, n;
        int cur_pos = 0;
    int chr_ind = -1;
        while (fgets(line, sizeof line, fp)) {
                if (line[0] == '>') {
                        sscanf(line+1, "%s", cur_chr);
                        cur_pos = n = 0;
            chr_ind = find_chr_ind(chrs, refseq->beditem, cur_chr);
            if (chr_ind >= 0) {
                find_next_bed(refseq, chr_ind, cur_pos, &b, &e, &n);
                if(fo) fprintf(fo, "%s", line);
                printf("%s", line);
            }
                        continue;
                }
        if (chr_ind == -1) continue;
        if(fo) fprintf(fo, "%s", line);
                if (b == -1) {
                    all_N(line);
                    printf("%s", line);
                    continue;
                }
                // now cur_chr == chr
                char *s = line;
                while (*s != '\n' &&  *s != 0) {
                    if(cur_pos >= e) {
                        find_next_bed(refseq, chr_ind, cur_pos, &b, &e, &n);
                        if (b == -1) {
                                all_N(s);
                                break;
                        }
                    }
                    if (cur_pos < b) {
                        *s = 'N';
                    }
                    s++; cur_pos++;
                }
                printf("%s", line);
        }


  return 0;
}

