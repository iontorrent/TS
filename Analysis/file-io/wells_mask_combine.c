#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#include "main.h"
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "wells_mask.h"

static int
usage()
{
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage: %s maskcombine <mask.bin> [...]\n", PACKAGE_NAME);
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "         -O          OR the masks (default: AND)\n");
  fprintf(stderr, "         -h          print this message\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "\n");
  return 1;
}

int
wells_mask_combine_main(int argc, char *argv[])
{
  FILE **fps_in = NULL;
  int32_t i, j, k;
  int c;
  int32_t n;
  wells_mask_t **masks = NULL;
  int32_t and = 1;

  while((c = getopt(argc, argv, "Oh")) >= 0) {
      switch(c) {
        case 'O':
          and = 0;
          break;
        case 'h':
        default:
          return usage();
      }
  }

  if(argc - optind < 2) {
      return usage();
  }
  else {
      n = argc - optind;

      masks = ion_calloc(n, sizeof(wells_mask_t*), __func__, "masks");
      
      // open the files
      fps_in = ion_calloc(n, sizeof(FILE*), __func__, "fps_in");
      for(i=optind;i<argc;i++) {
          if(!(fps_in[i-optind] = fopen(argv[i], "rb"))) {
              fprintf(stderr, "** Could not open %s for reading. **\n", argv[i]);
              ion_error(__func__, argv[i], Exit, OpenFileError);
          }
      }

      // read in the data
      for(i=0;i<n;i++) {
          if(NULL == (masks[i] = wells_mask_read(fps_in[i]))) {
              ion_error(__func__, argv[i+optind], Exit, ReadFileError);
          }
      }

      // check that we can combine
      for(i=1;i<n;i++) {
          if(masks[i-1]->num_rows != masks[i]->num_rows) {
              ion_error(__func__, "# of rows did not match", Exit, OpenFileError);
          }
          else if(masks[i-1]->num_cols != masks[i]->num_cols) {
              ion_error(__func__, "# of columns did not match", Exit, OpenFileError);
          }
      }

      // combine
      // NB: uses the first mask
      if(1 == and) {
          for(j=0;j<masks[0]->num_rows;j++) {
              for(k=0;k<masks[0]->num_cols;k++) {
                  for(i=1;i<n;i++) {
                      masks[0]->masks[i][j] &= masks[i]->masks[i][j];
                  }
              }
          }
      }
      else {
          for(j=0;j<masks[0]->num_rows;j++) {
              for(k=0;k<masks[0]->num_cols;k++) {
                  for(i=1;i<n;i++) {
                      masks[0]->masks[i][j] |= masks[i]->masks[i][j];
                  }
              }
          }
      }
      
      // write
      wells_mask_write(stdout, masks[0]);

      // destroy the masks and close the files
      for(i=0;i<n;i++) {
          wells_mask_destroy(masks[i]);
          fclose(fps_in[i]);
      }

      // free
      free(fps_in);
      free(masks);
  }

  return 0;
}
