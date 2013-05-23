#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

#include "main.h"
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "ion_sort.h"
#include "sff.h"
#include "sff_file.h"
#include "sff_sort.h"

// temporary struct so we do not need to keep regenerating the row/col co-ordinates
typedef struct {
    int32_t row, col;
    sff_t *sff;
} sff_sort_t;

// initialize sort routines
#define __sff_sort_lt(a, b) ((a).row < (b).row || ((a).row == (b).row && (a).col < (b).col))
ION_SORT_INIT(sff_sort, sff_sort_t, __sff_sort_lt)

void
sff_sort(sff_file_t *fp_in, sff_file_t *fp_out)
{
  int32_t i, row, col;
  sff_t *sff;
  int32_t requires_sort = 0;
  sff_sort_t *sffs = NULL;
  int32_t sffs_mem = 0, sffs_len = 0;

  // initialize memory
  sffs_mem = 1024;
  sffs = ion_malloc(sizeof(sff_sort_t) * sffs_mem, __func__, "sffs");

  // go through the input file
  while(NULL != (sff = sff_read(fp_in))) {
      // get the row/col co-ordinates
      if(0 == ion_readname_to_rowcol(sff->rheader->name->s, &row, &col)) {
          ion_error(__func__, "could not understand the read name", Exit, OutOfRange);
      }
      // copy over
      while(sffs_mem <= sffs_len) {
          sffs_mem <<= 1; // double
          sffs = ion_realloc(sffs, sizeof(sff_sort_t) * sffs_mem, __func__, "sffs");
      }
      sffs[sffs_len].row = row;
      sffs[sffs_len].col = col;
      sffs[sffs_len].sff = sff;
      sff = NULL;

      // check if we need to sort, for later
      if(0 < sffs_len && __sff_sort_lt(sffs[sffs_len], sffs[sffs_len-1])) {
          requires_sort = 1;
      }

      sffs_len++;
  }

  // resize
  sffs_mem = sffs_len; 
  sffs = ion_realloc(sffs, sizeof(sff_sort_t) * sffs_mem, __func__, "sffs");

  if(1 == requires_sort) {
      // sort
      ion_sort_introsort(sff_sort, sffs_len, sffs);
  }

  // write
  for(i=0;i<sffs_len;i++) {
      if(0 == sff_write(fp_out, sffs[i].sff)) {
          ion_error(__func__, "sff_write", Exit, WriteFileError);
      }
  }

  // destroy
  for(i=0;i<sffs_len;i++) {
      sff_destroy(sffs[i].sff);
  }
  free(sffs);
}

static int
usage()
{
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage: %s sffsort <in.sff> [...]\n", PACKAGE_NAME);
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "         -h          print this message\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "\n");
  return 1;
}

int
sff_sort_main(int argc, char *argv[])
{
  int c;
  sff_file_t *fp_in, *fp_out;

  while((c = getopt(argc, argv, "h")) >= 0) {
      switch(c) {
        case 'h':
        default:
          return usage();
      }
  }

  if(argc != 1 + optind) {
      return usage();
  }
  else {
      fp_in = sff_fopen(argv[optind], "rbi", NULL, NULL);
      fp_out = sff_fdopen(fileno(stdout), "wbi", fp_in->header, fp_in->index);

      sff_sort(fp_in, fp_out);

      sff_fclose(fp_in);
      sff_fclose(fp_out);
  }

  return 0;
}
