#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>

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

sff_t *
sff_read(sff_file_t *sff_file)
{
  return sff_read1(sff_file->fp, sff_file->header);
}

sff_t *
sff_read1(FILE *fp, sff_header_t *header)
{
  sff_t *sff;

  sff = sff_init();

  sff->gheader = header;
  sff->rheader = sff_read_header_read(fp);
  if(NULL == sff->rheader) { // EOF
      sff_destroy(sff);
      return NULL;
  }
  sff->read = sff_read_read(fp, sff->gheader, sff->rheader);
  if(NULL == sff->read) { // EOF
      sff_destroy(sff);
      return NULL;
  }

  return sff;
}

uint32_t
sff_write(sff_file_t *sff_file, const sff_t *sff)
{
  return sff_write1(sff_file->fp, sff);
}

uint32_t 
sff_write1(FILE *fp, const sff_t *sff)
{
  uint32_t n = 0;

  n += sff_read_header_write(fp, sff->rheader);
  n += sff_read_write(fp, sff->gheader, sff->rheader, sff->read);

  return n;
}

void
sff_print(FILE *fp, sff_t *sff)
{
  sff_read_header_print(fp, sff->rheader);
  sff_read_print(fp, sff->read, sff->gheader, sff->rheader); 
}

sff_t *
sff_init()
{
  sff_t *sff = NULL;

  sff = ion_calloc(1, sizeof(sff_t), __func__, "sff");
  sff->gheader = NULL;
  sff->rheader = NULL;
  sff->read = NULL;

  return sff;
}

sff_t *
sff_init1()
{
  sff_t *sff = NULL;

  sff = ion_calloc(1, sizeof(sff_t), __func__, "sff");
  sff->gheader = NULL;
  sff->rheader = sff_read_header_init();
  sff->read = sff_read_init();

  return sff;
}

void 
sff_destroy(sff_t *sff)
{
  if(NULL == sff) return;
  sff_read_header_destroy(sff->rheader);
  sff_read_destroy(sff->read);
  free(sff);
}

static sff_read_header_t *
sff_read_header_clone(sff_read_header_t *rh)
{
  sff_read_header_t *ret = NULL;

  ret = ion_calloc(1, sizeof(sff_read_header_t), __func__, "rh");

  (*ret) = (*rh);
  ret->name = ion_string_clone(rh->name);

  return ret;
}

static sff_read_t *
sff_read_clone(sff_read_t *r, sff_header_t *gh, sff_read_header_t *rh)
{
  sff_read_t *ret = NULL;
  int32_t i;

  ret = ion_calloc(1, sizeof(sff_read_t), __func__, "r");

  ret->flowgram = ion_malloc(sizeof(uint16_t)*gh->flow_length, __func__, "ret->flowgram");
  for(i=0;i<gh->flow_length;i++) {
      ret->flowgram[i] = r->flowgram[i];
  }

  ret->flow_index = ion_malloc(sizeof(uint8_t)*rh->n_bases, __func__, "ret->flow_index");
  for(i=0;i<rh->n_bases;i++) {
      ret->flow_index[i] = r->flow_index[i];
  }

  ret->bases = ion_string_clone(r->bases);
  ret->quality = ion_string_clone(r->quality);

  return ret;
}

sff_t *
sff_clone(sff_t *sff)
{
  sff_t *ret = NULL;

  ret = sff_init();

  ret->gheader = sff->gheader;
  ret->rheader = sff_read_header_clone(sff->rheader);
  ret->read = sff_read_clone(sff->read, sff->gheader, sff->rheader);

  return ret;

}

void
sff_reverse_compliment(sff_t *sff)
{
  int32_t i;

  // reverse flowgram
  for(i=0;i<(sff->gheader->flow_length>>1);i++) {
      uint16_t tmp = sff->read->flowgram[sff->gheader->flow_length-1-i];
      sff->read->flowgram[sff->gheader->flow_length-1-i] = sff->read->flowgram[i];
      sff->read->flowgram[i] = tmp;
  }
  // reverse flow index
  for(i=0;i<(sff->rheader->n_bases>>1);i++) {
      uint8_t tmp = sff->read->flow_index[sff->rheader->n_bases-1-i];
      sff->read->flow_index[sff->rheader->n_bases-1-i] = sff->read->flow_index[i];
      sff->read->flow_index[i] = tmp;
  }
  // reverse compliment the bases
  ion_string_reverse_compliment(sff->read->bases, sff->is_int);
  // reverse the qualities
  ion_string_reverse(sff->read->quality);
}

void
sff_to_int(sff_t *sff)
{
  int32_t i;
  if(1 == sff->is_int) return;
  for(i=0;i<sff->read->bases->l;i++) {
      sff->read->bases->s[i] = ion_nt_char_to_int[(int)sff->read->bases->s[i]];
  }
  sff->is_int = 1;
}

void
sff_to_char(sff_t *sff)
{
  int32_t i;
  if(0 == sff->is_int) return;
  for(i=0;i<sff->read->bases->l;i++) {
      sff->read->bases->s[i] = "ACGTN"[(int)sff->read->bases->s[i]];
  }
  sff->is_int = 0;
}

inline ion_string_t *
sff_get_bases(sff_t *sff)
{
  return sff->read->bases;
}

inline ion_string_t *
sff_get_qualities(sff_t *sff)
{
  return sff->read->quality;
}

inline void
sff_remove_key_sequence(sff_t *sff)
{
  int32_t i;
  // remove the key sequence
  for(i=0;i<sff->rheader->n_bases - sff->gheader->key_length;i++) { 
      sff->read->bases->s[i] = sff->read->bases->s[i+sff->gheader->key_length];
      sff->read->quality->s[i] = sff->read->quality->s[i+sff->gheader->key_length];
  }
  sff->read->bases->l -= sff->gheader->key_length;
  sff->read->quality->l -= sff->gheader->key_length;
}

uint16_t
sff_clipped_read_left(sff_t *sff)
{
  uint16_t clip_position = 1;
  if(sff->rheader->clip_qual_left > 0)
    clip_position = sff->rheader->clip_qual_left;
  if(sff->rheader->clip_adapter_left > sff->rheader->clip_qual_left)
    clip_position = sff->rheader->clip_adapter_left;
  return(clip_position);
}

uint16_t
sff_clipped_read_right(sff_t *sff)
{
  uint16_t clip_position = sff->rheader->n_bases;
  if(sff->rheader->clip_qual_right > 0)
    clip_position = sff->rheader->clip_qual_right;
  if(sff->rheader->clip_adapter_right > 0 && sff->rheader->clip_adapter_right < clip_position)
    clip_position = sff->rheader->clip_adapter_right;
  return(clip_position);
}

static int
sff_view_usage()
{
  fprintf(stderr, "Usage: %s sffview [options] <in.sff>\n", PACKAGE_NAME);
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "         -r STRING   the minimum/maximum row range (ex. 0-20)\n");
  fprintf(stderr, "         -c STRING   the minimum/maximum col range (ex. 0-20)\n");
  fprintf(stderr, "         -R FILE     a file with each line being a read name to print\n");
  fprintf(stderr, "         -F STRING   replace the flow order in the header with this\n");
  fprintf(stderr, "         -K STRING   replace the key sequence in the header with this\n");
  fprintf(stderr, "         -q          print in FASTQ format\n");
  fprintf(stderr, "         -b          print in SFF binary format\n");
  fprintf(stderr, "         -h          print this message\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "\n");
  return 1;

}

int
sff_view_main(int argc, char *argv[])
{
  int i, c;
  sff_file_t *sff_file_in=NULL, *sff_file_out=NULL;
  sff_iter_t *sff_iter = NULL;
  sff_t *sff = NULL;
  char *fn_names = NULL;
  char **names = NULL;
  char *flow = NULL;
  char *key = NULL;
  int32_t names_num = 0, names_mem = 0;
  int32_t out_mode, min_row, max_row, min_col, max_col;

  out_mode = 0;
  min_row = max_row = min_col = max_col = -1;

  while((c = getopt(argc, argv, "r:c:R:F:K:bqh")) >= 0) {
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
        case 'R':
          free(fn_names);
          fn_names = strdup(optarg); break;
        case 'q':
          out_mode |= 1;
          break;
        case 'b':
          out_mode |= 2;
          break;
        case 'F':
          flow = strdup(optarg);
          break;
        case 'K':
          key = strdup(optarg);
          break;
        case 'h': 
        default: 
          return sff_view_usage();
      }
  }
  if(argc != 1+optind) {
      return sff_view_usage();
  }
  else {
      sff_header_t *header = NULL;
      if(3 == out_mode) {
          ion_error(__func__, "options -b and -q cannot be used together", Exit, CommandLineArgument);
      }

      // open the input SFF
      if(-1 != min_row || -1 != max_row || -1 != min_col || -1 != max_col) {
          sff_file_in = sff_fopen(argv[optind], "rbi", NULL, NULL);
      }
      else {
          sff_file_in = sff_fopen(argv[optind], "rb", NULL, NULL);
      }

      header = sff_header_clone(sff_file_in->header); /* copy header, but update n_reads if using index or names */

      if(NULL != flow) { // replace the flow order
          ion_string_copy1(header->flow, flow);
      }
      if(NULL != key) { // replace the key
          ion_string_copy1(header->key, key);
      }

      // read in the names
      if(NULL != fn_names) {
          FILE *fp = NULL;
          char name[1024]="\0"; // lets hope we don't exceed this length
          names_num = names_mem = 0;
          names = NULL;
          if(!(fp = fopen(fn_names, "rb"))) {
              fprintf(stderr, "** Could not open %s for reading. **\n", fn_names);
              ion_error(__func__, fn_names, Exit, OpenFileError);
          }
          while(EOF != fscanf(fp, "%s", name)) {
              while(names_num == names_mem) {
                  if(0 == names_mem) names_mem = 4;
                  else names_mem *= 2;
                  names = ion_realloc(names, sizeof(char*) * names_mem, __func__, "names");
              }
              names[names_num] = strdup(name);
              if(NULL == names[names_num]) {
                  ion_error(__func__, name, Exit, MallocMemory);
              }
              names_num++;
          }
          names = ion_realloc(names, sizeof(char*) * names_num, __func__, "names");
          fclose(fp);
          header->n_reads = names_num;
      }
      else {
	// if using index, then iterate once through the index to count the entries
	// so we can set the count correctly in the header
	if (-1 != min_row || -1 != max_row || -1 != min_col || -1 != max_col) {
	  int entries = 0;
          sff_iter = sff_iter_query(sff_file_in, min_row, max_row, min_col, max_col);
	  while (NULL != (sff = sff_iter_read(sff_file_in, sff_iter)))
	    entries++;
	  header->n_reads = entries;
	  /* reset sff_iter */
	  sff_iter_destroy(sff_iter);
	  sff_iter = sff_iter_query(sff_file_in, min_row, max_row, min_col, max_col);
	}
      }

      // print the header
      switch(out_mode) {
        case 0:
          sff_header_print(stdout, header);
          break;
        case 1:
          // do nothing: FASTQ
          break;
        case 2:
          sff_file_out = sff_fdopen(fileno(stdout), "wb", header, NULL);
          break;
      }


      while(1) {
          int32_t to_print = 1;
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
          if(0 < names_mem) {
              to_print = 0;
              for(i=0;i<names_num;i++) {
                  if(0 == strcmp(names[i], sff_name(sff))) {
                      to_print = 1;
                      break;
                  }
              }
              // shift down
              if(1 == to_print) { // i < names_num
                  free(names[i]);
                  names[i] = NULL;
                  for(;i<names_num-1;i++) {
                      names[i] = names[i+1];
                      names[i+1] = NULL;
                  }
                  names_num--;
              }
          }
          if(1 == to_print) {
              switch(out_mode) {
                case 0:
                  sff_print(stdout, sff);
                  break;
                case 1:
                  if(fprintf(stdout, "@%s\n%s\n+\n",
                             sff->rheader->name->s,
                             sff->read->bases->s + sff->gheader->key_length) < 0) {
                      ion_error(__func__, "stdout", Exit, WriteFileError);
                  }
                  for(i=sff->gheader->key_length;i<sff->read->quality->l;i++) {
                      if(fputc(QUAL2CHAR(sff->read->quality->s[i]), stdout) < 0) {
                          ion_error(__func__, "stdout", Exit, WriteFileError);
                      }
                  }
                  if(fputc('\n', stdout) < 0) {
                      ion_error(__func__, "stdout", Exit, WriteFileError);
                  }
                  break;
                case 2:
                  sff_write(sff_file_out, sff);
                  break;
              }
          }
          sff_destroy(sff);
          if(0 < names_mem && 0 == names_num) {
              break;
          }
      }

      sff_fclose(sff_file_in);
      if(2 == out_mode) {
          sff_fclose(sff_file_out);
      }
      if(-1 != min_row || -1 != max_row || -1 != min_col || -1 != max_col) {
          sff_iter_destroy(sff_iter);
      }

      if(0 != names_num) {
          fprintf(stderr, "** Did not find all the reads with (-R). **\n");
          ion_error(__func__, fn_names, Exit, OutOfRange);
      }

      sff_header_destroy(header);

  }
  if(NULL != names && 0 < names_num) {
      free(names);
  }
  free(fn_names);
  free(flow);
  free(key);
  return 0;
}

static sff_header_t*
sff_cat_merge_headers(sff_file_t **fps, int32_t n)
{
  sff_header_t *header = NULL;
  int32_t i, n_reads = 0;

  if(n < 1) return NULL;
  header = sff_header_clone(fps[0]->header); /* clone the first header */
  if(n == 1) return header;


  // get the total number of reads and do some error checking
  n_reads = header->n_reads;
  for(i=1;i<n;i++) {
      sff_header_t *alt = NULL;
      alt = fps[i]->header;

      // check the other headers against this one
      if(alt->magic != header->magic) {
          ion_error(__func__, "Header value did not match: magic", Exit, OutOfRange);
      }
      if(alt->version != header->version) {
          ion_error(__func__, "Header value did not match: version", Exit, OutOfRange);
      }
      if(alt->index_offset != header->index_offset) {
          ion_error(__func__, "Header value did not match: index_offset", Exit, OutOfRange);
      }
      if(alt->index_length != header->index_length) {
          ion_error(__func__, "Header value did not match: index_length", Exit, OutOfRange);
      }
      if(alt->gheader_length != header->gheader_length) {
          ion_error(__func__, "Header value did not match: gheader_length", Exit, OutOfRange);
      }
      if(alt->key_length != header->key_length) {
          ion_error(__func__, "Header value did not match: key_length", Exit, OutOfRange);
      }
      if(alt->flow_length != header->flow_length) {
          ion_error(__func__, "Header value did not match: flow_length", Exit, OutOfRange);
      }
      if(alt->flowgram_format != header->flowgram_format) {
          ion_error(__func__, "Header value did not match: flowgram_format", Exit, OutOfRange);
      }
      if(0 != strcmp(alt->flow->s, header->flow->s)) {
          ion_error(__func__, "Header value did not match: flow", Exit, OutOfRange);
      }
      if(0 != strcmp(alt->key->s, header->key->s)) {
          ion_error(__func__, "Header value did not match: key", Exit, OutOfRange);
      }

      // update the number of reads
      n_reads += alt->n_reads; 
  }

  // update the number of reads
  header->n_reads = n_reads;

  return header;
}

static int
sff_cat_usage()
{
  fprintf(stderr, "Usage: %s sffcat [options] <in1.sff <in2.sff> [...]\n", PACKAGE_NAME);
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "         -h          print this message\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "\n");
  return 1;

}

int
sff_cat_main(int argc, char *argv[])
{
  int i, c;
  sff_file_t **sff_files_in=NULL, *sff_file_out=NULL;
  int32_t sff_files_in_num=0;
  sff_t *sff = NULL;

  while((c = getopt(argc, argv, "h")) >= 0) {
      switch(c) {
        case 'h': 
        default: 
          return sff_cat_usage();
      }
  }
  if(argc <= optind) {
      return sff_cat_usage();
  }
  else {
      sff_header_t *header = NULL;

      // open input files
      sff_files_in_num = argc - optind;
      sff_files_in = ion_calloc(sff_files_in_num, sizeof(sff_file_t*), __func__, "sff_file_in");
      for(i=0;i<sff_files_in_num;i++) {
          sff_files_in[i] = sff_fopen(argv[optind+i], "rb", NULL, NULL);
      }

      // merge the headers
      header = sff_cat_merge_headers(sff_files_in, sff_files_in_num);

      // open the output file
      sff_file_out = sff_fdopen(fileno(stdout), "wb", header, NULL);

      // concatenate the records
      for(i=0;i<sff_files_in_num;i++) {
          while(NULL != (sff = sff_read(sff_files_in[i]))) {
              sff_write(sff_file_out, sff);
              sff_destroy(sff);
          }
      }

      // close the output file
      sff_fclose(sff_file_out);

      // close input files
      for(i=0;i<sff_files_in_num;i++) {
          sff_fclose(sff_files_in[i]);
      }
      free(sff_files_in);

      // destroy the header
      sff_header_destroy(header);
  }
  return 0;
}
