#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "main.h"
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "wells_header.h"
#include "wells_data.h"
#include "wells_chip.h"

static int
usage()
{
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage: %s wellscombine <in.wells> [...]\n", PACKAGE_NAME); 
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "         -r STRING   the minimum/maximum row range (ex. 0-20)\n");
  fprintf(stderr, "         -c STRING   the minimum/maximum col range (ex. 0-20)\n");
  fprintf(stderr, "         -z          print only non-zero flows\n");
  fprintf(stderr, "         -h          print this message\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "\n");
  return 1;
}

int
wells_combine_main(int argc, char *argv[])
{
  FILE **fps_in = NULL;
  wells_chip_t **chips = NULL;
  int c;
  int32_t i, j, k, l, n;
  int32_t min_row, max_row, min_col, max_col;

  min_row = max_row = min_col = max_col = -1;

  while((c = getopt(argc, argv, "r:c:zh")) >= 0) {
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
        case 'z':
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

      // open the files
      fps_in = ion_calloc(n, sizeof(FILE*), __func__, "fps_in");
      for(i=optind;i<argc;i++) {
          if(!(fps_in[i-optind] = fopen(argv[i], "rb"))) {
              fprintf(stderr, "** Could not open %s for reading. **\n", argv[i]);
              ion_error(__func__, argv[i], Exit, OpenFileError);
          }
      }

      // read in the headers
      chips = ion_calloc(n, sizeof(wells_chip_t*), __func__, "chips");
      for(i=0;i<n;i++) {
          // NB: reads in both wells files
          if(NULL == (chips[i] = wells_chip_read1(fps_in[i]))) {
              ion_error(__func__, argv[i+optind], Exit, ReadFileError);
          }
      }

      // check we can combine the data
      for(i=1;i<n;i++) {
          if(chips[i-1]->header->num_wells != chips[i]->header->num_wells) {
              ion_error(__func__, "# of wells did not match", Exit, OutOfRange);
          }
          else if(chips[i-1]->header->num_flows != chips[i]->header->num_flows) {
              ion_error(__func__, "# of flows did not match", Exit, OutOfRange);
          }
          else if(0 != strcmp(chips[i-1]->header->flow_order, chips[i]->header->flow_order)) {
              ion_error(__func__, "flow order did not match", Exit, OutOfRange);
          }
          else if(chips[i-1]->num_rows != chips[i]->num_rows) {
              ion_error(__func__, "# of rows did not match", Exit, OutOfRange);
          }
          else if(chips[i-1]->num_cols != chips[i]->num_cols) {
              ion_error(__func__, "# of columns did not match", Exit, OutOfRange);
          }
      }

      // write the header
      if(0 == wells_header_write(stdout, chips[0]->header)) {
          ion_error(__func__, "Could not write to the output file", Exit, WriteFileError);
      }

      // combine the data
      // NB: modifies the wells for the first chip
      for(k=0;k<chips[0]->num_cols;k++) { // cols
          for(j=0;j<chips[0]->num_rows;j++) { // rows
              wells_data_t data_out;
              // make room for the new flow values
              data_out.flow_values = ion_calloc(chips[0]->header->num_flows, sizeof(float), __func__, "data_out->flow_values");
              for(i=0;i<n;i++) { // chip
                  wells_data_t *data_in = NULL;
                  // read data in
                  if(NULL == (data_in = wells_data_read(fps_in[i], chips[i]->header))) {
                      ion_error(__func__, argv[i+optind], Exit, ReadFileError);
                  }
                  // copy over relevant data
                  if(0 == i) {
                      data_out.x = data_in->x;
                      data_out.y = data_in->y;
                      data_out.rank = data_in->rank;
                  }
                  for(l=0;l<chips[i]->header->num_flows;l++) { // flows
                      // update sum
                      data_out.flow_values[l] += data_in->flow_values[l];
                  }
                  // destroy
                  wells_data_destroy(data_in);
              }
              // update
              for(l=0;l<chips[0]->header->num_flows;l++) { // flows
                  data_out.flow_values[l] /= n;
              }
              // write
              if(0 == wells_data_write(stdout, chips[0]->header, &data_out)) {
                  ion_error(__func__, "Could not write to the output file", Exit, WriteFileError);
              }
              // destroy
              free(data_out.flow_values);
          }
      }

      // close the files
      for(i=0;i<n;i++) {
          fclose(fps_in[i]);
      }

      // free
      free(fps_in);
      free(chips);
  }

  return 0;
}
