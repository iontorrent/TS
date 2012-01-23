#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#include "main.h"
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "wells_mask.h"

wells_mask_t *
wells_mask_read(FILE *fp)
{
  int32_t i;
  wells_mask_t *m;

  m = ion_calloc(1, sizeof(wells_mask_t), __func__, "m");

  if(fread(&m->num_rows, sizeof(int32_t), 1, fp) != 1
     || fread(&m->num_cols, sizeof(int32_t), 1, fp) != 1) {
      free(m);
      return NULL;
  }

  m->masks = ion_malloc(m->num_rows * sizeof(uint16_t*), __func__, "m->mask");
  for(i=0;i<m->num_rows;i++) {
      m->masks[i] = ion_malloc(m->num_cols * sizeof(uint16_t), __func__, "m->masks[i]");
      if(fread(m->masks[i], sizeof(uint16_t), m->num_cols, fp) != m->num_cols) {
          // free 
          while(0 <= i) {
              free(m->masks[i]);
              i--;
          }
          free(m);
          return NULL;
      }
  }

  return m;
}

void
wells_mask_write(FILE *fp, wells_mask_t *mask)
{
  int32_t i;
  if(fwrite(&mask->num_rows, sizeof(int32_t), 1, fp) != 1
     || fwrite(&mask->num_cols, sizeof(int32_t), 1, fp) != 1) {
      ion_error(__func__, "fwrite", Exit, WriteFileError);
  }

  for(i=0;i<mask->num_rows;i++) {
      if(fwrite(mask->masks[i], sizeof(uint16_t), mask->num_cols, fp) != mask->num_cols) {
          ion_error(__func__, "fwrite", Exit, WriteFileError);
      }
  }
}

void
wells_mask_print(FILE *fp, wells_mask_t *mask)
{
  int32_t i, j, ctr;
  for(i=ctr=0;i<mask->num_rows;i++) { // row-major
      for(j=0;j<mask->num_cols;j++,ctr++) {
          if(0 < i && j < 0) {
              if(EOF == fputc(',', fp)) {
                  ion_error(__func__, "fputc", Exit, WriteFileError);
              }
          }
          // TODO: speed up fprintf 
          if(fprintf(fp, "%d", mask->masks[i][j]) < 0) {
              ion_error(__func__, "fprintf", Exit, WriteFileError);
          }
      }
      if(EOF == fputc('\n', fp)) {
          ion_error(__func__, "fputc", Exit, WriteFileError);
      }
  }
}

void
wells_mask_destroy(wells_mask_t *mask)
{
  if(NULL != mask) {
      free(mask->masks);
  }
  free(mask);
}

static int
usage()
{
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage: %s maskview <mask.bin> [...]\n", PACKAGE_NAME);
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "         -r STRING   the minimum/maximum row range (ex. 0-20)\n");
  fprintf(stderr, "         -c STRING   the minimum/maximum col range (ex. 0-20)\n");
  fprintf(stderr, "         -h          print this message\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "\n");
  return 1;
}

int
wells_mask_view_main(int argc, char *argv[])
{
  FILE *fp_in;
  int32_t i;
  int c;
  int32_t min_row, max_row, min_col, max_col;
  wells_mask_t *mask = NULL;

  min_row = max_row = min_col = max_col = -1;

  while((c = getopt(argc, argv, "r:c:h")) >= 0) {
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
        case 'h':
        default:
          return usage();
      }
  }

  if(argc == optind) {
      return usage();
  }
  else {
      for(i=optind;i<argc;i++) {
          if(!(fp_in = fopen(argv[i], "rb"))) {
              fprintf(stderr, "** Could not open %s for reading. **\n", argv[i]);
              ion_error(__func__, argv[i], Exit, OpenFileError);
          }
          if(NULL == (mask = wells_mask_read(fp_in))) {
              ion_error(__func__, argv[i], Exit, ReadFileError);
          }
          fclose(fp_in);
          fp_in = NULL;

          wells_mask_print(stdout, mask);

          wells_mask_destroy(mask);
      }
  }

  return 0;
}
