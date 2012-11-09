#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "main.h"
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "wells_header.h"
#include "wells_data.h"
#include "wells_chip.h"

wells_chip_t *
wells_chip_read1(FILE *fp)
{
  wells_chip_t *chip = NULL;
  int32_t i;
  wells_data_t tmp_wells_data;
  fpos_t fpos;

  chip = ion_calloc(1, sizeof(wells_chip_t), __func__, "chip");

  // read the header, which unfortunately does not have the # of rows/cols
  if(NULL == (chip->header = wells_header_read(fp))) {
      free(chip);
      return NULL;
  }

  // TRICKY: get the # of rows/cols, of course assuming all the data is there
  if(-1L == fgetpos(fp, &fpos)) { // save file position
      ion_error(__func__, "ftell", Exit, ReadFileError);
  }
  // assumes the input is column major
  tmp_wells_data.flow_values = ion_malloc(sizeof(float)*chip->header->num_flows, __func__, "tmp_wells_data.flow_values");
  chip->num_rows = -1;
  for(i=0;i<chip->header->num_wells;i++) {
      if(NULL == wells_data_read1(fp, chip->header, &tmp_wells_data)) {
          wells_header_destroy(chip->header);
          free(chip);
          return NULL;
      }
      if(0 < tmp_wells_data.y) {
          chip->num_rows = i; // 1-based
          break;
      }
  }
  free(tmp_wells_data.flow_values);
  if(-1 == chip->num_rows) { // sanity check
      ion_error(__func__, "-1 == chip->num_rows", Exit, OutOfRange);
  }
  if(0 != (chip->header->num_wells % chip->num_rows)) { // sanity check
      ion_error(__func__, "0 != (chip->header->num_wells % chip->num_cols)", Exit, OutOfRange);
  }
  chip->num_cols = chip->header->num_wells / chip->num_rows; // set the # of rows
  if(0 != fsetpos(fp, &fpos)) { // reset position
      ion_error(__func__, "ftell", Exit, ReadFileError);
  }

  return chip;
}

// TODO: we can avoid reading the whole .wells file if we know the byte offsets
// when any min_row/max_row/min_col/max_col are given
wells_chip_t *
wells_chip_read(FILE *fp, int32_t min_row, int32_t max_row, int32_t min_col, int32_t max_col)
{
  wells_chip_t *chip = NULL;
  int32_t i, j;

  if(NULL == (chip = wells_chip_read1(fp))) {
      return NULL;
  }


  if(-1 == min_row && -1 == max_row && -1 == min_col && -1 == max_col) {
      // malloc
      chip->data = ion_calloc(chip->num_rows, sizeof(wells_data_t*), __func__, "chip->data");
      for(i=0;i<chip->num_rows;i++) {
          chip->data[i] = ion_calloc(chip->num_cols, sizeof(wells_data_t), __func__, "chip->data[i]");
          for(j=0;j<chip->num_cols;j++) {
              chip->data[i][j].flow_values = ion_malloc(sizeof(float)*chip->header->num_flows, __func__, "chip->data[i][j].flow_values");
          }
      }
      // read in
      for(j=0;j<chip->num_cols;j++) {
          for(i=0;i<chip->num_rows;i++) {
              if(NULL == wells_data_read1(fp, chip->header, &chip->data[i][j])) {
                  wells_chip_destroy(chip);
                  return NULL;
              }
          }
      }
  }
  else {
      // bound values
      ion_bound_values(&min_row, &max_row, chip->num_rows);
      ion_bound_values(&min_col, &max_col, chip->num_cols);
      
      // allocate
      chip->data = ion_calloc((max_row-min_row+1), sizeof(wells_data_t*), __func__, "chip->data");
      for(i=0;i<(max_row-min_row+1);i++) {
          chip->data[i] = ion_calloc((max_col-min_col+1), sizeof(wells_data_t), __func__, "chip->data[i]");
          for(j=0;j<(max_col-min_col+1);j++) {
              chip->data[i][j].flow_values = ion_malloc(sizeof(float)*chip->header->num_flows, __func__, "chip->data[i][j].flow_values");
          }
      }

      // read
      if(0 < min_col) { // SEEK over unused cols
          if(0 != fseek(fp, min_col * chip->num_rows * wells_data_size(chip->header), SEEK_CUR)) {
              ion_error(__func__, "fseek", Exit, ReadFileError);
          } 
      }
      for(j=0;j<(max_col-min_col+1);j++) { // column-major
          if(0 < min_row) { // SEEK over unused rows
              if(0 != fseek(fp, min_row * wells_data_size(chip->header), SEEK_CUR)) {
                  ion_error(__func__, "fseek", Exit, ReadFileError);
              } 
          }
          // read in the rest of the column
          for(i=0;i<(max_row-min_row+1);i++) {
              if(NULL == wells_data_read1(fp, chip->header, &chip->data[i][j])) {
                  chip->num_rows = (max_row-min_row+1);
                  chip->num_cols = (max_col-min_col+1);
                  wells_chip_destroy(chip);
                  return NULL;
              }
          }
          if(max_col < chip->num_rows - 1) { // SEEK over unused columnns
              if(0 != fseek(fp, (chip->num_rows - (max_col + 1)) * wells_data_size(chip->header), SEEK_CUR)) {
                  ion_error(__func__, "fseek", Exit, ReadFileError);
              } 
          }
      }
      
      // adjust size
      chip->num_rows = (max_row-min_row+1);
      chip->num_cols = (max_col-min_col+1);
      chip->header->num_wells = chip->num_rows * chip->num_cols;
  }

  return chip;
}
    
int32_t
wells_chip_write(FILE *fp, wells_chip_t *chip)
{
  int32_t i, j;

  // write the header
  if(0 == wells_header_write(fp, chip->header)) {
      return 0;
  }

  // write the data
  for(j=0;j<chip->num_cols;j++) {
      for(i=0;i<chip->num_rows;i++) {
          if(0 == wells_data_write(fp, chip->header, &chip->data[i][j])) {
              return 0;
          }
      }
  }

  return 1;
}

void
wells_chip_print(FILE *fp, wells_chip_t *chip, int32_t nonzero)
{
  int32_t i, j;
  wells_header_print(fp, chip->header);
  for(i=0;i<chip->num_rows;i++) {
      for(j=0;j<chip->num_cols;j++) {
          wells_data_print(fp, chip->header, &chip->data[i][j], nonzero);
      }
  }
}

void
wells_chip_destroy(wells_chip_t *chip)
{
  int32_t i, j;

  wells_header_destroy(chip->header);
  for(i=0;i<chip->num_rows;i++) {
      for(j=0;j<chip->num_cols;j++) {
          free(chip->data[i][j].flow_values);
      }
      free(chip->data[i]);
  }
  free(chip->data);
  free(chip);
}

static int
usage()
{
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage: %s wellsview <in.wells> [...]\n", PACKAGE_NAME); 
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
wells_chip_view_main(int argc, char *argv[])
{
  FILE *fp_in;
  wells_chip_t *chip;
  int c;
  int32_t i;
  int32_t nonzero, min_row, max_row, min_col, max_col;

  min_row = max_row = min_col = max_col = -1;
  nonzero=0; 

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
          nonzero = 1;
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
          // Note: reading the entire structure, then writing is inefficient,
          // but this is for demonstration purposes...

          if(!(fp_in = fopen(argv[i], "rb"))) {
              fprintf(stderr, "** Could not open %s for reading. **\n", argv[i]);
              ion_error(__func__, argv[i], Exit, OpenFileError);
          }

          if(NULL == (chip = wells_chip_read(fp_in, min_row, max_row, min_col, max_col))) {
              ion_error(__func__, argv[i], Exit, ReadFileError);
          }
          fclose(fp_in);
          fp_in=NULL;

          wells_chip_print(stdout, chip, nonzero);

          wells_chip_destroy(chip);

          chip=NULL;
          fp_in=NULL;
      }
  }

  return 0;
}
