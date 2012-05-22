#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <netinet/in.h>

#include "main.h"
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "sff.h"
#include "sff_header.h"
#include "sff_file.h"
#include "sff_index.h"

static inline void
sff_index_ntoh(sff_index_t *idx)
{
  // convert values from big-endian
  idx->index_magic_number = ntohl(idx->index_magic_number);
  idx->num_rows = ntohl(idx->num_rows);
  idx->num_cols = ntohl(idx->num_cols);
  idx->type = ntohl(idx->type);
}

static inline void
sff_index_hton(sff_index_t *idx)
{
  // convert values to big-endian
  idx->index_magic_number = htonl(idx->index_magic_number);
  idx->num_rows = htonl(idx->num_rows);
  idx->num_cols = htonl(idx->num_cols);
  idx->type = htonl(idx->type);
}

uint32_t
sff_index_length(sff_index_t *idx)
{
  uint32_t n = 0;
  uint64_t len = 0;

  // index header
  n += sizeof(uint32_t)*2 + sizeof(int32_t)*3;

  // offsets
  if(SFF_INDEX_ROW_ONLY == idx->type) {
      len = 1 + idx->num_rows;
  }
  else if(SFF_INDEX_ALL == idx->type) {
      len = 1 + (idx->num_rows * idx->num_cols);
  }
  else {
      ion_error(__func__, "could not understand index type", Exit, OutOfRange);
  }
  n += sizeof(uint64_t) * len;

  // padding
  n += ion_write_padding(NULL, n);

  return n;
}

static sff_index_t *
sff_index_init()
{
  sff_index_t *idx;

  idx = ion_calloc(1, sizeof(sff_index_t), __func__, "idx");

  idx->index_version = SFF_INDEX_VERSION;
  idx->index_magic_number = SFF_INDEX_MAGIC;

  return idx;
}

// TODO: should we change the header:
// - must trake index_length
// - assumes row-major order
sff_index_t*
sff_index_create(sff_file_t *fp_in, sff_header_t *fp_out_header, int32_t num_rows, int32_t num_cols, int32_t type)
{
  int64_t len = 0;
  int32_t i, prev_row, prev_col, row, col;
  sff_index_t *idx;
  sff_t *sff;
  uint64_t fp_in_start, prev_pos;

  idx = sff_index_init();

  idx->num_rows = num_rows;
  idx->num_cols = num_cols;
  idx->type = type;

  // alloc
  switch(type) {
    case SFF_INDEX_ROW_ONLY:
      len = 1 + idx->num_rows;
      idx->offset = ion_malloc(len * sizeof(uint64_t), __func__, "idx->offset");
      break;
    case SFF_INDEX_ALL:
      len = 1 + (idx->num_rows * idx->num_cols);
      idx->offset = ion_malloc(len * sizeof(uint64_t), __func__, "idx->offset");
      break;
    default:
      ion_error(__func__, "this index type is currently not supported", Exit, OutOfRange);
  }

  // save where the sff entries started
  prev_pos = fp_in_start = ftell(fp_in->fp);
  if(-1L == fp_in_start) {
      ion_error(__func__, "ftell", Exit, ReadFileError);
  }

  // go through the input file
  i = 0;
  prev_row = prev_col = 0;
  while(NULL != (sff = sff_read(fp_in))) {
      // out of range
      if(len-1 <= i) {
          ion_error(__func__, "bug encountered", Exit, OutOfRange);
      }

      // get the row/col co-ordinates
      if(0 == ion_readname_to_rowcol(sff->rheader->name->s, &row, &col)) {
          ion_error(__func__, "could not understand the read name", Exit, OutOfRange);
      }

      // assumes row-major order, skips over reads that are not present
      if(row < prev_row || (row == prev_row && col < prev_col)) {
          ion_error(__func__, "SFF file was not sorted in row-major order", Exit, OutOfRange);
      }
      while(row != prev_row || col != prev_col) {
          // add in empty entry
          switch(type) {
            case SFF_INDEX_ROW_ONLY:
              if(0 == prev_col) { // first column
                  idx->offset[i] = UINT64_MAX;
                  // do not increment i, since we only do this when moving to a new row
              }
              break;
            case SFF_INDEX_ALL:
              // all rows and columns
              idx->offset[i] = UINT64_MAX;
              i++;
              break;
            default:
              ion_error(__func__, "this index type is currently not supported", Exit, OutOfRange);
          }
          if(len-1 <= i) {
              ion_error(__func__, "x/y was out of range", Exit, OutOfRange);
          }

          prev_col++;
          if(prev_col == idx->num_cols) {
              // new row
              prev_col = 0;
              prev_row++;
              if(SFF_INDEX_ROW_ONLY == type) {
                  i++;
              }
          }
      }

      // add to the index
      switch(type) {
        case SFF_INDEX_ROW_ONLY:
          if(0 == col) { // first column
              idx->offset[i] = prev_pos;
          }
          else if(0 < col && UINT64_MAX == idx->offset[i]) {
              idx->offset[i] = prev_pos;
              // do not move onto the next
          }
          break;
        case SFF_INDEX_ALL:
          // all rows and columns
          idx->offset[i] = prev_pos;
          i++;
          break;
        default:
          ion_error(__func__, "this index type is currently not supported", Exit, OutOfRange);
      }
      prev_row = row;
      prev_col = col;

      // destroy
      sff_destroy(sff);

      // next
      prev_col++;
      if(prev_col == idx->num_cols) {
          // new row
          prev_col = 0;
          prev_row++;
          if(SFF_INDEX_ROW_ONLY == type) {
              i++;
          }
      }

      prev_pos = ftell(fp_in->fp);
      if(-1L == prev_pos) {
          ion_error(__func__, "ftell", Exit, ReadFileError);
      }
  }
  // get the last offset
  idx->offset[len-1] = prev_pos;

  // update the index offset in the header
  fp_out_header->index_offset = fp_in_start; // insert between the header and sff entries
  // update the index length in the header
  fp_out_header->index_length = sff_index_length(idx);
  // update the offsets based on the index length
  for(i=0;i<len;i++) {
      if(UINT64_MAX != idx->offset[i]) {
          idx->offset[i] += fp_out_header->index_length;
      }
  }

  return idx;
}

sff_index_t *
sff_index_read(FILE *fp)
{
  int32_t i;
  uint32_t n = 0;
  uint64_t len = 0;
  sff_index_t *idx;

  idx = sff_index_init();

  // index header
  if(1 != fread(&idx->index_magic_number, sizeof(uint32_t), 1, fp)
     || 1 != fread(&idx->index_version, sizeof(uint32_t), 1, fp)
     || 1 != fread(&idx->num_rows, sizeof(int32_t), 1, fp)
     || 1 != fread(&idx->num_cols, sizeof(int32_t), 1, fp)
     || 1 != fread(&idx->type, sizeof(int32_t), 1, fp)) {
      ion_error(__func__, "fread", Exit, WriteFileError);
  }
  n += sizeof(uint32_t)*2 + sizeof(int32_t)*3;

  // convert values from big-endian
  sff_index_ntoh(idx);

  // offsets
  if(SFF_INDEX_ROW_ONLY == idx->type) {
      len = 1 + idx->num_rows;
  }
  else if(SFF_INDEX_ALL == idx->type) {
      len = 1 + (idx->num_rows * idx->num_cols);
  }
  else {
      ion_error(__func__, "could not understand index type", Exit, OutOfRange);
  }

  // alloc
  idx->offset = ion_malloc(sizeof(uint64_t) * len, __func__, "idx->offset");
  // read
  if(len != fread(idx->offset, sizeof(uint64_t), len, fp)) {
      ion_error(__func__, "fread", Exit, WriteFileError);
  }
  // convert values from big-endian
  for(i=0;i<len;i++) {
      idx->offset[i] = ntohll(idx->offset[i]);
  }
  n += sizeof(uint64_t) * len;

  // padding
  n += ion_read_padding(fp, n);

  return idx;
}

uint32_t
sff_index_write(FILE *fp, sff_index_t *idx)
{
  int32_t i;
  uint32_t n = 0;
  uint64_t len = 0;

  if(NULL == idx) {
      return 0;
  }

  // convert values to big-endian
  sff_index_hton(idx);

  // index header
  if(1 != fwrite(&idx->index_magic_number, sizeof(uint32_t), 1, fp)
     || 1 != fwrite(&idx->index_version, sizeof(uint32_t), 1, fp)
     || 1 != fwrite(&idx->num_rows, sizeof(int32_t), 1, fp)
     || 1 != fwrite(&idx->num_cols, sizeof(int32_t), 1, fp)
     || 1 != fwrite(&idx->type, sizeof(int32_t), 1, fp)) {
      ion_error(__func__, "fwrite", Exit, WriteFileError);
  }
  n += sizeof(uint32_t)*2 + sizeof(int32_t)*3;

  // convert values from big-endian
  sff_index_ntoh(idx);

  // offsets
  if(SFF_INDEX_ROW_ONLY == idx->type) {
      len = 1 + idx->num_rows;
  }
  else if(SFF_INDEX_ALL == idx->type) {
      len = 1 + (idx->num_rows * idx->num_cols);
  }
  else {
      ion_error(__func__, "could not understand index type", Exit, OutOfRange);
  }

  // convert values to big-endian
  for(i=0;i<len;i++) {
      idx->offset[i] = htonll(idx->offset[i]);
  }
  // write
  if(len != fwrite(idx->offset, sizeof(uint64_t), len, fp)) {
      ion_error(__func__, "fwrite", Exit, WriteFileError);
  }
  n += sizeof(uint64_t) * len;
  // convert values from big-endian
  for(i=0;i<len;i++) {
      idx->offset[i] = ntohll(idx->offset[i]);
  }

  // padding
  n += ion_write_padding(fp, n);

  return n;
}

void
sff_index_destroy(sff_index_t *idx)
{
  if(NULL != idx) {
      free(idx->offset);
      free(idx);
  }
}

static int
usage()
{
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage: %s sffindex <in.sff> [...]\n", PACKAGE_NAME);
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "         -r INT      the number of rows\n");
  fprintf(stderr, "         -c INT      the number of columns\n");
  fprintf(stderr, "         -C INT      the chip type (sets -r/-c):\n");
  fprintf(stderr, "                        0: 314 (1152 x 1280)\n");
  fprintf(stderr, "                        1: 316 (2640 x 2736)\n");
  fprintf(stderr, "                        2: 318 (3792 x 3392)\n");
  fprintf(stderr, "         -R          index only the rows\n");
  fprintf(stderr, "         -h          print this message\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "\n");
  return 1;
}

int
sff_index_create_main(int argc, char *argv[])
{
  int c;
  sff_file_t *fp_in, *fp_out;
  int32_t num_rows, num_cols, type;
  sff_header_t *fp_out_header;
  sff_index_t* index;
  sff_t *sff;

  num_rows = num_cols = -1;
  type = SFF_INDEX_ALL;

  while((c = getopt(argc, argv, "r:c:C:Rh")) >= 0) {
      switch(c) {
        case 'r':
          num_rows = atoi(optarg);
          break;
        case 'c':
          num_cols = atoi(optarg);
          break;
        case 'C':
          switch(atoi(optarg)) {
            case 0:
              num_rows = 1152;
              num_cols = 1280;
              break;
            case 1:
              num_rows = 2640;
              num_cols = 2736;
              break;
            case 2:
              num_rows = 3792;
              num_cols = 3392;
              break;
            default:
              break;
          }
        case 'R':
          type = SFF_INDEX_ROW_ONLY;
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
      // check cmd line args
      if(num_rows < 0) {
          ion_error(__func__, "-r must be specified and greater than zero", Exit, CommandLineArgument);
      }
      if(num_cols < 0) {
          ion_error(__func__, "-c must be specified and greater than zero", Exit, CommandLineArgument);
      }
      switch(type) {
        case SFF_INDEX_ROW_ONLY:
        case SFF_INDEX_ALL:
          break;
        default:
          ion_error(__func__, "bug encountered", Exit, OutOfRange);
          break;
      }

      fp_in = sff_fopen(argv[optind], "rb", NULL, NULL);
      fp_out_header = sff_header_clone(fp_in->header);
      index = sff_index_create(fp_in, fp_out_header, num_rows, num_cols, type);

      fp_out = sff_fdopen(fileno(stdout), "wbi", fp_out_header, index);

      // seek the input file to the beginning of the the entries, which is the same
      // location as where the index begins in the output file.
      if(0 != fseek(fp_in->fp, fp_out_header->index_offset, SEEK_SET)) {
	ion_error(__func__, "fseek", Exit, ReadFileError);
      }

      // write the sff entries
      while(NULL != (sff = sff_read(fp_in))) {
	sff_write(fp_out, sff);
	sff_destroy(sff);
      }

      // destroy the header.  Don't destroy index, sff_fclose does that
      sff_header_destroy(fp_out_header);
      //      sff_index_destroy(index);

      sff_fclose(fp_in);
      sff_fclose(fp_out);
  }

  return 0;
}
