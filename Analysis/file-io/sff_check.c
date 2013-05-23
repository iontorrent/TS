#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <unistd.h>

#include "main.h"
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "sff_header.h"
#include "sff_read_header.h"
#include "sff_read.h"
#include "sff_index.h"
#include "sff_file.h"
#include "sff_iter.h"
#include "sff.h"
#include "sff_check.h"
      
// NB: how do we round?
static int32_t
sff_check_cmp_flow_signal(int32_t base_call, int32_t flow_signal)
{
  int32_t r;

  r = flow_signal % 100;
  if(base_call == (int)((flow_signal + 50) / 100)) {
      return 1;
  }
  else if(49 == r && base_call == (int)((flow_signal + 51) / 100)) {
      return 1;
  }
  else if(50 == r && base_call == (int)((flow_signal + 49) / 100)) {
      return 1;
  }
  return 0;
}

void
sff_check_rheader(sff_t *sff, int32_t *n_err3, int32_t print){
  uint32_t nBases = sff->rheader->n_bases;
  uint16_t clipAdapterRight = sff->rheader->clip_adapter_right;
  uint16_t clipQualRight = sff->rheader->clip_qual_right;
  uint16_t clipAdapterLeft = sff->rheader->clip_adapter_left;
  uint16_t clipQualLeft = sff->rheader->clip_qual_left;
  *n_err3 = 0;

  if(clipQualLeft > nBases){
    (*n_err3)++;
    if(print)
        fprintf(stdout, "Type three: clip qual left(%d)  is larger than read length(%d)\n", clipQualLeft, nBases);
  }

  if(clipAdapterLeft > nBases){
    (*n_err3)++;
    if(print)
        fprintf(stdout, "Type three: clip adapter left(%d)  is larger than read length(%d)\n", clipAdapterLeft, nBases);
  }

  if(clipQualRight > nBases){
    (*n_err3)++;
    if(print)
        fprintf(stdout, "Type three: clip qual right(%d)  is larger than read length(%d)\n", clipQualRight, nBases);
  }

  if(clipAdapterRight > nBases){
    (*n_err3)++;
    if(print)
        fprintf(stdout, "Type three: clip adapter right(%d)  is larger than read length(%d)\n", clipAdapterRight, nBases);
  }

}

void
sff_check(sff_t *sff, int32_t *n_err1, int32_t *n_err2, int32_t print)
{
  int32_t i, j, l, fl;
  ion_string_t *fo, *bases;
  char prev_base = 0;
  uint16_t *fg;

  fo = sff->gheader->flow;
  fl = sff->gheader->flow_length; 
  bases = sff->read->bases;
  fg = sff->read->flowgram;

  (*n_err1) = (*n_err2) = 0;

  i = j = 0;
  while(i < bases->l) {
      // track the empty flows
      while(fo->s[j] != bases->s[i]) {
          if(prev_base == fo->s[j] && 0 == sff_check_cmp_flow_signal(0, fg[j])) {
              if(print) {
                  fprintf(stdout, "Type two: base index = %d flow index = %d prev base = %c flow base = %c flow signal = %d\n",
                          i, j,
                          prev_base, fo->s[j],
                          fg[j]);
              }
              // Type two error: we should incorporate the base as early as possible
              (*n_err2)++;
          }
          j++;
          if(fl <= j) j = 0; // NB: is this necessary?
      }
      // skip over current hp
      prev_base = bases->s[i];
      l = 0;
      while(i<bases->l && prev_base == bases->s[i]) {
          i++;
          l++;
      }
      if(0 == sff_check_cmp_flow_signal(l, fg[j])) {
          if(print) {
              fprintf(stdout, "Type one: base index = %d flow index = %d base call = %d flow signal = %d\n",
                      i, j,
                      l, fg[j]);
          }
          // Disagreement between the flowgram and base call
          (*n_err1)++;
      }
      j++;
      if(fl <= j) j = 0; // NB: is this necessary?
  }
  
  if(print && (0 < (*n_err1) || 0 < (*n_err2))) {
      fprintf(stdout, "type one errors = %d\n", (*n_err1));
      fprintf(stdout, "type two errors = %d\n", (*n_err2));
      sff_print(stdout, sff);
  }
}

static int
usage()
{
  fprintf(stderr, "Usage: %s sffcheck [options] <in.sff>\n", PACKAGE_NAME);
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "         -r STRING   the minimum/maximum row range (ex. 0-20)\n");
  fprintf(stderr, "         -c STRING   the minimum/maximum col range (ex. 0-20)\n");
  fprintf(stderr, "         -p          print incorrect reads\n");
  fprintf(stderr, "         -h          print this message\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "\n");
  return 1;
}

int
sff_check_main(int argc, char *argv[])
{
  int c;
  sff_file_t *sff_file_in=NULL;
  sff_iter_t *sff_iter = NULL;
  sff_t *sff = NULL;
  int32_t print = 0;
  int32_t min_row, max_row, min_col, max_col;
  int32_t n_reads = 0;
  int32_t n_err1, n_err2, n_err3;
  int32_t n_err1_total, n_err2_total, n_err3_total;
  int32_t n_err1_reads, n_err2_reads, n_err3_reads;

  min_row = max_row = min_col = max_col = -1;

  while((c = getopt(argc, argv, "r:c:ph")) >= 0) {
      switch(c) {
        case 'r':
          if(ion_parse_range(optarg, &min_row, &max_row) < 0) {
              ion_error(__func__, "-r : format not recognized", Exit, OutOfRange);
          }
          break;
        case 'c':
          if(ion_parse_range(optarg, &min_col, &max_col) < 0) {
              ion_error(__func__, "-c : format not recognized", Exit, OutOfRange);
          }
          break;
        case 'p':
          print = 1;
          break;
        case 'h':
        default:
          return usage();
      }
  }
  if(argc != 1+optind) {
      return usage();
  }
  else {
      if(-1 != min_row || -1 != max_row || -1 != min_col || -1 != max_col) {
          sff_file_in = sff_fopen(argv[optind], "rbi", NULL, NULL);
          sff_iter = sff_iter_query(sff_file_in, min_row, max_row, min_col, max_col);
      }
      else {
          sff_file_in = sff_fopen(argv[optind], "rb", NULL, NULL);
      }
      
      if(print) {
          sff_header_print(stdout, sff_file_in->header);
      }

      n_err1_total = n_err2_total = n_err3_total = 0;
      n_err1_reads = n_err2_reads = n_err3_reads = 0;
      while(1) {
          if(-1 != min_row || -1 != max_row || -1 != min_col || -1 != max_col) {
              if(NULL == (sff = sff_iter_read(sff_file_in, sff_iter))) {
                  break;
              }
          }
          else {
              if(NULL == (sff = sff_read(sff_file_in))) {
                  break;
              }
          }

          sff_check(sff, &n_err1, &n_err2, print);
          if(0 < n_err1) n_err1_reads++;
          if(0 < n_err2) n_err2_reads++;
          n_err1_total += n_err1;
          n_err2_total += n_err2;

          sff_check_rheader(sff, &n_err3, print);
          if(0 < n_err3) n_err3_reads++;
          n_err3_total += n_err3;

          sff_destroy(sff);
          n_reads++;
      }

      sff_fclose(sff_file_in);
      if(-1 != min_row || -1 != max_row || -1 != min_col || -1 != max_col) {
          sff_iter_destroy(sff_iter);
      }
          
      fprintf(stderr, "** Examined %d reads **\n", n_reads);
      fprintf(stderr, "** Found %d reads with type one errors **\n", n_err1_reads);
      fprintf(stderr, "** Found %d reads with type two errors **\n", n_err2_reads);
      fprintf(stderr, "** Found %d reads with read header errors **\n", n_err3_reads);
      fprintf(stderr, "** Found %d type one errors across all reads **\n", n_err1_total);
      fprintf(stderr, "** Found %d type two errors across all reads **\n", n_err2_total);
      fprintf(stderr, "** Found %d read header errors across all reads **\n", n_err3_total);
  }
  return 0;
}
