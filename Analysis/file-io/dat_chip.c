#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <netinet/in.h>
#include "main.h"
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "dat_io.h"
#include "dat_flow.h"
#include "dat_chip.h"

dat_chip_t *
dat_chip_read(FILE *fp)
{
  uint32_t i, j;
  dat_chip_t *chip = NULL;

  chip = ion_malloc(sizeof(dat_chip_t), __func__, "chip");

  if(1 != fread_big_endian_uint32_t(fp, &chip->num_flows)) {
      free(chip);
      return NULL;
  }

  chip->byte_offsets = ion_malloc(sizeof(uint32_t)*chip->num_flows, __func__, "chip->num_flows");

  if(fread(chip->byte_offsets, sizeof(uint32_t), chip->num_flows, fp) != chip->num_flows) {
      free(chip->byte_offsets);
      free(chip);
      return NULL;
  }
  for(i=0;i<chip->num_flows;i++) {
      chip->byte_offsets[i] = ntohl(chip->byte_offsets[i]);
  }

  chip->flows = ion_malloc(sizeof(dat_flow_t*)*chip->num_flows, __func__, "chip->num_flows");
  for(i=0;i<chip->num_flows;i++) {
      chip->flows[i] = dat_flow_read(fp);
      if(NULL == chip->flows[i]) {
          for(j=0;j<i;j++) {
              dat_flow_destroy(chip->flows[i]);
          }
          free(chip->byte_offsets);
          free(chip);
          return NULL;
      }
  }

  return chip;
}

dat_chip_t *
dat_chip_read1(FILE *fp, int32_t min_flow, int32_t max_flow, int32_t min_frame, int32_t max_frame, int32_t min_row, int32_t max_row, int32_t min_col, int32_t max_col)
{
  dat_chip_t *chip = NULL;
  uint32_t i, j;

  chip = ion_malloc(sizeof(dat_chip_t), __func__, "chip");

  if(1 != fread_big_endian_uint32_t(fp, &chip->num_flows)) {
      free(chip);
      return NULL;
  }

  if(ion_bound_values(&min_flow, &max_flow, chip->num_flows) < 0) {
      free(chip);
      return NULL;
  }

  chip->byte_offsets = ion_malloc(sizeof(uint32_t)*chip->num_flows, __func__, "chip->num_flows");

  if(fread(chip->byte_offsets, sizeof(uint32_t), chip->num_flows, fp) != chip->num_flows) {
      free(chip->byte_offsets);
      free(chip);
      return NULL;
  }
  for(i=0;i<chip->num_flows;i++) {
      chip->byte_offsets[i] = ntohl(chip->byte_offsets[i]);
  }

  // Seek to the right spot
  if(0 != fseek(fp, chip->byte_offsets[min_flow], SEEK_CUR)) {
      free(chip->byte_offsets);
      free(chip);
      return NULL;
  }

  // Read in flows
  chip->num_flows = max_flow - min_flow + 1;
  chip->flows = ion_malloc(sizeof(dat_flow_t*)*chip->num_flows, __func__, "chip->num_flows");
  for(i=0;i<chip->num_flows;i++) {
      chip->flows[i] = dat_flow_read1(fp, min_frame, max_frame, min_row, max_row, min_col, max_col);

      if(NULL == chip->flows[i]) {
          for(j=0;j<i;j++) {
              dat_flow_destroy(chip->flows[i]);
          }
          free(chip->byte_offsets);
          free(chip);
          return NULL;
      }
  }

  return chip;
}

void
dat_chip_import(FILE *fp, int32_t out_text, int32_t num_dats, char *dat_fns[])
{
  uint32_t i;
  FILE *fp_in = NULL;
  FILE *fp_out = NULL;

  if(0 == out_text) {
      dat_chip_t *chip = NULL;

      if(0 == (fp_out = fdopen(fileno(stdout), "wb"))) {
          ion_error(__func__, "stdout", Exit, OpenFileError);
      }

      chip = ion_malloc(sizeof(dat_chip_t), __func__, "chip");
      chip->num_flows = num_dats;

      if(1 != fwrite_big_endian_uint32_t(fp, &chip->num_flows)) {
          ion_error(__func__, "fwrite_big_endian_uint32_t", Exit, WriteFileError);
      }

      chip->byte_offsets = ion_malloc(sizeof(uint32_t)*chip->num_flows, __func__, "chip->byte_offsets");
      for(i=0;i<chip->num_flows;i++) {
          chip->byte_offsets[i] = htonl(0);
      }
      if(fwrite(chip->byte_offsets, sizeof(uint32_t), chip->num_flows, fp) != chip->num_flows) {
          ion_error(__func__, "fwrite", Exit, WriteFileError);
      }
      for(i=0;i<chip->num_flows;i++) {
          chip->byte_offsets[i] = ntohl(chip->byte_offsets[i]);
      }

      for(i=0;i<chip->num_flows;i++) {
          long offset = ftell(fp);
          if(offset < 0) {
              ion_error(__func__, "ftell", Exit, OutOfRange);
          }
          chip->byte_offsets[i] = offset;

          // Read in
          dat_flow_t *dat_flow = NULL;
          if(!(fp_in = fopen(dat_fns[i], "rb"))) {
              fprintf(stderr, "** Could not open %s for reading. **\n", dat_fns[i]);
              ion_error(__func__, dat_fns[i], Exit, OpenFileError);
          }

          dat_flow = dat_flow_read(fp_in);

          if(NULL == dat_flow) {
              ion_error(__func__, "Error reading file", Exit, OutOfRange);
          }

          fclose(fp_in);
          fp_in = NULL;

          if(1 != dat_flow_write(fp_out, dat_flow)) {
              ion_error(__func__, "dat_flow_write", Exit, WriteFileError);
          }
          dat_flow_destroy(dat_flow);
          dat_flow = NULL;
      }

      // Seek back and write byte offsets
      if(0 != fseek(fp, sizeof(uint32_t), SEEK_SET)) {
          free(chip->byte_offsets);
          free(chip);
          ion_error(__func__, "fseek", Exit, OutOfRange);
      }
      for(i=0;i<chip->num_flows;i++) {
          chip->byte_offsets[i] = htonl(chip->byte_offsets[i]);
      }
      if(fwrite(chip->byte_offsets, sizeof(uint32_t), chip->num_flows, fp) != chip->num_flows) {
          ion_error(__func__, "fwrite", Exit, OutOfRange);
      }

      free(chip->byte_offsets);
      free(chip);
      chip = NULL;

      fclose(fp_out);
  }
  else {
      if(fprintf(stdout, "@HD\tnum_flows=%d\n", num_dats) < 0) {
          ion_error(__func__, "fprintf", Exit, WriteFileError);
      }
      for(i=0;i<num_dats;i++) {
          dat_flow_t *dat_flow = NULL;
          if(!(fp_in = fopen(dat_fns[i], "rb"))) {
              fprintf(stderr, "** Could not open %s for reading. **\n", dat_fns[i]);
              ion_error(__func__, dat_fns[i], Exit, OpenFileError);
          }

          dat_flow = dat_flow_read(fp_in);

          if(NULL == dat_flow) {
              ion_error(__func__, "Error reading file", Exit, OutOfRange);
          }

          fclose(fp_in);
          fp_in = NULL;

          dat_flow_print(stdout, dat_flow, i);
          dat_flow_destroy(dat_flow);
          dat_flow = NULL;
      }
  }
}

int
dat_chip_write(FILE *fp, dat_chip_t *chip)
{
  uint32_t i;

  if(1 != fwrite_big_endian_uint32_t(fp, &chip->num_flows)) {
      return EOF;
  }

  for(i=0;i<chip->num_flows;i++) {
      chip->byte_offsets[i] = htonl(chip->byte_offsets[i]);
  }
  if(fwrite(chip->byte_offsets, sizeof(uint32_t), chip->num_flows, fp) != chip->num_flows) {
      return EOF;
  }
  for(i=0;i<chip->num_flows;i++) {
      chip->byte_offsets[i] = ntohl(chip->byte_offsets[i]);
  }

  for(i=0;i<chip->num_flows;i++) {
      if(1 != dat_flow_write(fp, chip->flows[i])) {
          return EOF;
      }
  }

  return 1;
}

void
dat_chip_print(FILE *fp, dat_chip_t *chip)
{
  uint32_t i;

  if(fprintf(fp, "@HD\tnum_flows=%d\n", chip->num_flows) < 0) {
      ion_error(__func__, "fprintf", Exit, WriteFileError);
  }

  for(i=0;i<chip->num_flows;i++) {
      dat_flow_print(fp, chip->flows[i], i);
  }
}

void
dat_chip_append_flow(dat_chip_t *chip, dat_flow_t *flow)
{
  chip->num_flows++;
  chip->flows = ion_realloc(chip->flows, sizeof(dat_flow_t)*chip->num_flows, __func__, "chip->flows");
  chip->flows[chip->num_flows-1] = flow;
}

void
dat_chip_destroy(dat_chip_t *chip)
{
  uint32_t i;
  for(i=0;i<chip->num_flows;i++) {
      dat_flow_destroy(chip->flows[i]);
  }
  free(chip->byte_offsets);
  free(chip);
}

static int 
usage()
{
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage: %s chipview <in.chip> [...]\n", PACKAGE_NAME);
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "         -F STRING   the minimum/maximum flow range (ex. 0-20)\n");
  fprintf(stderr, "         -f STRING   the minimum/maximum frame range (ex. 0-20)\n");
  fprintf(stderr, "         -r STRING   the minimum/maximum row range (ex. 0-20)\n");
  fprintf(stderr, "         -c STRING   the minimum/maximum col range (ex. 0-20)\n");
  fprintf(stderr, "         -i          import DAT files into a chip file\n");
  fprintf(stderr, "         -d          output a binary chip file\n");
  fprintf(stderr, "         -h          print this message\n");
  fprintf(stderr, "\n");
  return 1;
}

int
dat_chip_view_main(int argc, char *argv[])
{
  FILE *fp_in=NULL, *fp_out=NULL;
  int32_t i;
  dat_chip_t *dat_chip=NULL;
  int c;
  int32_t out_text, import;
  int32_t min_flow, min_frame, min_row, min_col;
  int32_t max_flow, max_frame, max_row, max_col;

  min_flow = min_frame = min_row = min_col = -1;
  max_flow = max_frame = max_row = max_col = -1;
  out_text = 1;
  import = 0;

  while((c = getopt(argc, argv, "f:r:c:F:diuh")) >= 0) {
      switch(c) {
        case 'f':
          if(ion_parse_range(optarg, &min_frame, &max_frame) < 0) {
              ion_error(__func__, "-f : format not recognized", Exit, OutOfRange);
          }
          break;
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
        case 'F':
          if(ion_parse_range(optarg, &min_flow, &max_flow) < 0) {
              ion_error(__func__, "-F : format not recognized", Exit, OutOfRange);
          }
          break;
        case 'd':
          out_text = 0; break;
        case 'i':
          import = 1; break;
        case 'h':
        default:
          return usage();
      }
  }

  if(argc == optind) {
      return usage();
  }
  else {
      if(0 == import) {
          // Note: reading the entire structure, then writing is inefficient,
          // but this is for demonstration purposes...
          
          // Notes:
          // - we could add to the header the sub-range being outputted
          for(i=optind;i<argc;i++) {
              if(!(fp_in = fopen(argv[i], "rb"))) {
                  fprintf(stderr, "** Could not open %s for reading. **\n", argv[i]);
                  ion_error(__func__, argv[i], Exit, OpenFileError);
              }
              if(NULL == (dat_chip = dat_chip_read1(fp_in, min_flow, max_flow, min_frame, max_frame, min_row, max_row, min_col, max_col))) {
                  ion_error(__func__, "Error reading file", Exit, OutOfRange);
              }
              fclose(fp_in);
              fp_in = NULL;

              if(1 == out_text) {
                  dat_chip_print(stdout, dat_chip);
              }
              else {
                  if(0 == (fp_out = fdopen(fileno(stdout), "wb"))) {
                      ion_error(__func__, "stdout", Exit, OpenFileError);
                  }
                  dat_chip_write(fp_out, dat_chip);
                  fclose(fp_out);
                  fp_out = NULL;
              }

              dat_chip_destroy(dat_chip);
              dat_chip = NULL;
          }
      }
      else {
          if(1 == out_text) {
              dat_chip_import(stdout, 1, argc-optind, argv+optind);
          }
          else {
              if(0 == (fp_out = fdopen(fileno(stdout), "wb"))) {
                  ion_error(__func__, "stdout", Exit, OpenFileError);
              }
              dat_chip_import(fp_out, 0, argc-optind, argv+optind);
              fclose(fp_out);
              fp_out = NULL;
          }
      }
  }
  return 0;
}
