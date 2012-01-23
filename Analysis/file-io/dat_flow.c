#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <unistd.h>
#include "main.h"
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "dat_header.h"
#include "dat_frame.h"
#include "dat_flow.h"

dat_flow_t 
*dat_flow_read(FILE *fp)
{
  dat_flow_t *d = NULL;
  int32_t i;

  // initialize memory
  d = ion_malloc(sizeof(dat_flow_t), __func__, "d");

  // read the header
  d->header = dat_header_read(fp);
  if(NULL == d->header) {
      free(d);
      return NULL;
  }

  // read the frames
  d->frames = ion_malloc(sizeof(dat_frame_t*)*(d->header->frames_in_file), __func__, "d->frames");
  for(i=0;i<d->header->frames_in_file;i++) {
      d->frames[i] = dat_frame_read(fp, (0 == i) ? NULL : d->frames[i-1], d->header);
      if(NULL == d->frames[i]) {
          ion_error(__func__, "dat_read_frame", Exit, EndOfFile);
      }
  }

  return d;
}	

int 
dat_flow_write(FILE *fp, dat_flow_t *d)
{
  int32_t i;

  // write the header
  if(1 != dat_header_write(fp, d->header)) {
      return EOF;
  }

  // write the frames
  for(i=0;i<d->header->frames_in_file;i++) {
      if(1 != dat_frame_write(fp, d->frames[i], (0 == i) ? NULL : d->frames[i-1], d->header)) {
          return EOF;
      }
  }

  return 1;
}	

void 
dat_flow_destroy(dat_flow_t *d)
{
  int32_t i;

  if(NULL != d->frames) { // free the frames
      for(i=0;i<d->header->frames_in_file;i++) {
          dat_frame_destroy(d->frames[i]);
      }
      free(d->frames);
      d->frames = NULL;
  }
  // free the header
  dat_header_destroy(d->header);
  // free your mind
  free(d);
}

void 
dat_flow_print(FILE *fp, dat_flow_t *d, uint32_t flow_index)
{
  uint32_t i, j, ctr, k;

  // print the header
  if(dat_header_print(fp, d->header) < 0) {
      ion_error(__func__, "dat_header_print", Exit, WriteFileError);
  }
  // print the data for each well
  // Note: this assumes the data is row major
  for(i=ctr=0;i<d->header->rows;i++) { // for each row
      for(j=0;j<d->header->cols;j++,ctr++) {  // for each column
          if(fprintf(fp, "%d,%d,%d", flow_index, i, j) < 0) {
              ion_error(__func__, "fprintf", Exit, WriteFileError);
          }
          for(k=0;k<d->header->frames_in_file;k++) { // for each frame 
              if(fprintf(fp, ",%d", d->frames[k]->data[ctr]) < 0) {
                  ion_error(__func__, "fprintf", Exit, WriteFileError);
              }
          }
          if(fprintf(fp, "\n") < 0) {
              ion_error(__func__, "fprintf", Exit, WriteFileError);
          }
      }
  }
}

// Note: Assumes an uninterlaced file
static inline int 
dat_flow_skip_to_frame(FILE *fp, int next_frame, dat_header_t *header)
{
  if(next_frame < 0 
     || (0 != next_frame 
         && 0 != fseek(fp, __dat_frame_size(next_frame, header), SEEK_CUR))) {
      return EOF;
  }
  else {
      return 1;
  }
}	

static void 
dat_flow_filter_row_col(dat_frame_t *frame, dat_header_t *header, int32_t min_row, int32_t max_row, int32_t min_col, int32_t max_col)
{
  int32_t row, col, ctr1, ctr2;
  if(0 < min_row 
     || max_row < header->rows-1 
     || 0 < min_col 
     || max_col < header->cols-1) {
      for(row=min_row,ctr1=0;row<=max_row;row++) {
          ctr2 = (row*header->cols) + min_col; // source
          for(col=min_col;col<=max_col;col++) {
              assert(ctr1 < header->rows*header->cols);
              assert(ctr2 < header->rows*header->cols);
              frame->data[++ctr1] = frame->data[++ctr2];
          }
      }
      frame->data = ion_realloc(frame->data, (max_row-min_row+1)*(max_col-min_col+1)*sizeof(uint16_t), __func__, "d->frames[i]->data");
  }
}

// TODO
// - update data size 
dat_flow_t *dat_flow_read1(FILE *fp, int32_t min_frame, int32_t max_frame, int32_t min_row, int32_t max_row, int32_t min_col, int32_t max_col)
{
  dat_flow_t *d = NULL;
  int32_t i, frames_in_file;

  d = ion_malloc(sizeof(dat_flow_t), __func__, "d");

  d->header = dat_header_read(fp);
  if(NULL == d->header) {
      free(d);
      return NULL;
  }

  // Update frame bounds
  // Update row bounds
  // Update col bounds
  if(ion_bound_values(&min_frame, &max_frame, d->header->frames_in_file) < 0 
     || ion_bound_values(&min_row, &max_row, d->header->rows) < 0 
     || ion_bound_values(&min_col, &max_col, d->header->cols) < 0) {
      free(d);
      return NULL;
  }

  if(DAT_HEADER_UNINTERLACED == d->header->interlace_type) {
      // Go through DAT frames
      frames_in_file = max_frame - min_frame + 1;

      // Skip to frame # 'min_frame' 
      if(1 != dat_flow_skip_to_frame(fp, min_frame, d->header)) {
          free(d->header);
          free(d);
          return NULL;
      }

      d->frames = ion_malloc(sizeof(dat_frame_t*)*frames_in_file, __func__, "d->frames");

      // Read in frames
      for(i=0;i<frames_in_file;i++) {
          d->frames[i] = dat_frame_read(fp, (0 == i) ? NULL : d->frames[i-1], d->header);
          if(NULL == d->frames[i]) {
              ion_error(__func__, "dat_read_frame", Exit, EndOfFile);
          }
          // Filter row/col
          dat_flow_filter_row_col(d->frames[i], d->header, min_row, max_row, min_col, max_col);
      }
      // Update the # of frames in the file
      d->header->frames_in_file = frames_in_file;
  }
  else {
      frames_in_file = max_frame + 1;

      d->frames = ion_malloc(sizeof(dat_frame_t*)*frames_in_file, __func__, "d->frames");
      for(i=0;i<=max_frame;i++) {
          d->frames[i] = dat_frame_read(fp, (0 == i) ? NULL : d->frames[i-1], d->header);
          if(NULL == d->frames[i]) {
              ion_error(__func__, "dat_read_frame", Exit, EndOfFile);
          }
          // free frames that are no longer needed
          if(0 <= i-1 && i-1 < min_frame) {
              dat_frame_destroy(d->frames[i-1]);
              d->frames[i-1] = NULL;
          }
      }

      // Filter row/col
      for(i=min_frame;i<=max_frame;i++) {
          dat_flow_filter_row_col(d->frames[i], d->header, min_row, max_row, min_col, max_col);
      }
      if(0 < min_frame) {
          // Shift down the frames
          for(i=min_frame;i<=max_frame;i++) {
              assert(NULL == d->frames[i-min_frame]);
              //dat_frame_destroy(d->frames[i-min_frame]);
              d->frames[i-min_frame] = d->frames[i];
              d->frames[i] = NULL;
          }
          frames_in_file = max_frame - min_frame + 1;
      }
      if(frames_in_file < d->header->frames_in_file) {
          d->header->frames_in_file = frames_in_file;
          d->frames = ion_realloc(d->frames, d->header->frames_in_file*sizeof(dat_frame_t), __func__, "d->frames");
      }
  }

  // Update the # of rows and columns
  // Note: we could do this while reading in each frame as to minimize memory
  if(0 < min_row 
     || max_row < d->header->rows-1 
     || 0 < min_col 
     || max_col < d->header->cols-1) {
      d->header->rows = (max_row - min_row + 1);
      d->header->cols = (max_col - min_col + 1);
  }

  return d;
}

// TODO
// - take advantage of when the data is uninterlaced
// - update data size 
dat_flow_t *dat_flow_read2(FILE *fp, int32_t min_frame, int32_t max_frame, int32_t *rows, int32_t nrows, int32_t *cols, int32_t ncols)
{
  dat_flow_t *d = NULL;
  int32_t i, j, k, ctr1, ctr2;

  // Check rows and columns exist
  if(nrows <= 0 || ncols <= 0) {
      return NULL;
  }

  // Read in entire DAT frame
  d = dat_flow_read(fp);

  // Update frame bounds
  if(ion_bound_values(&min_frame, &max_frame, d->header->frames_in_file) < 0) {
      dat_flow_destroy(d);
      return NULL;
  }
  // Check rows
  for(i=0;i<nrows;i++) {
      if(rows[i] < 0 
         || d->header->rows <= rows[i] 
         || (0 < i && rows[i+1] <= rows[i])) {
          dat_flow_destroy(d);
          return NULL;
      }
  }
  // Check cols
  for(i=0;i<ncols;i++) {
      if(cols[i] < 0 
         || d->header->cols <= cols[i] 
         || (0 < i && cols[i+1] <= cols[i])) {
          dat_flow_destroy(d);
          return NULL;
      }
  }

  // Filter frames
  if(0 < min_frame) {
      for(i=min_frame;i<=max_frame;i++) {
          dat_frame_destroy(d->frames[i-min_frame]);
          d->frames[i-min_frame] = d->frames[i];
          d->frames[i] = NULL;
      }
  }
  for(i=max_frame+1;i<d->header->frames_in_file;i++) {
      dat_frame_destroy(d->frames[i]);
      d->frames[i] = NULL;
  }
  d->header->frames_in_file = max_frame - min_frame + 1;
  d->frames = ion_realloc(d->frames, d->header->frames_in_file*sizeof(dat_frame_t), __func__, "d->frames");

  // Filter row/col
  for(i=0;i<d->header->frames_in_file;i++) {
      for(j=ctr1=0;j<nrows;j++) {
          ctr2=(rows[j]*d->header->cols);
          for(k=0;k<ncols;k++) {
              d->frames[i]->data[++ctr1] = d->frames[i]->data[ctr2+cols[k]];
          }
      }
      assert(ctr1 == nrows*ncols);
      d->frames[i]->data = ion_realloc(d->frames[i]->data, nrows*ncols*sizeof(uint16_t), __func__, "d->frames[i]->data");
  }
  d->header->rows = nrows;
  d->header->cols = ncols;

  return d;
}

static int 
usage()
{
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage: %s datview <in.dat> [...]\n", PACKAGE_NAME);
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "         -f STRING   the minimum/maximum frame range (ex. 0-20)\n");
  fprintf(stderr, "         -r STRING   the minimum/maximum row range (ex. 0-20)\n");
  fprintf(stderr, "         -c STRING   the minimum/maximum col range (ex. 0-20)\n");
  fprintf(stderr, "         -u          output an uninterlaced DAT file\n");
  fprintf(stderr, "         -d          output a binary DAT file\n");
  fprintf(stderr, "         -h          print this message\n");
  fprintf(stderr, "\n");
  return 1;
}

int 
dat_flow_view_main(int argc, char *argv[])
{
  FILE *fp_in=NULL, *fp_out=NULL;
  int32_t i;
  dat_flow_t *dat_flow=NULL;
  int c;
  int32_t min_frame, max_frame, min_row, max_row, min_col, max_col, uninterlaced, out_text;

  min_frame = max_frame = min_row = max_row = min_col = max_col = -1;
  uninterlaced = 0;
  out_text = 1;

  while((c = getopt(argc, argv, "f:r:c:duh")) >= 0) {
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
        case 'u':
          uninterlaced = 1; break;
        case 'd':
          out_text = 0; break;
        case 'h':
        default:
          return usage();
      }
  }

  if(argc == optind) {
      return usage();
  }
  else {
      // Note: reading the entire structure, then writing is inefficient,
      // but this is for demonstration purposes...

      if(1 == out_text && 1 == uninterlaced) {
          ion_error(__func__, "-u is ignored when -d is used", Warn, CommandLineArgument);
      }
      // Notes:
      // - we could read in and output by frame instead of reading in the entire flow.
      // - we could add to the header the sub-range being outputted
      for(i=optind;i<argc;i++) {
          if(!(fp_in = fopen(argv[i], "rb"))) {
              fprintf(stderr, "** Could not open %s for reading. **\n", argv[i]);
              ion_error(__func__, argv[i], Exit, OpenFileError);
          }


          if(-1 == min_frame && -1 == max_frame && -1 == min_row && -1 == max_row && -1 == min_col && -1 == max_col) {
              if(NULL == (dat_flow = dat_flow_read(fp_in))) {
                  ion_error(__func__, "Error reading file", Exit, OutOfRange);
              }
          }
          else {
              if(NULL == (dat_flow = dat_flow_read1(fp_in, min_frame, max_frame, min_row, max_row, min_col, max_col))) {
                  ion_error(__func__, "Error reading file", Exit, OutOfRange);
              }
          }
          fclose(fp_in);
          fp_in = NULL;

          if(1 == out_text) {
              dat_flow_print(stdout, dat_flow, i-optind);
          }
          else {
              if(1 == uninterlaced) {
                  dat_flow->header->interlace_type = DAT_HEADER_UNINTERLACED;
              }
              if(0 == (fp_out = fdopen(fileno(stdout), "wb"))) {
                  ion_error(__func__, "stdout", Exit, OpenFileError);
              }
              dat_flow_write(fp_out, dat_flow);
              fclose(fp_out);
              fp_out = NULL;
          }

          dat_flow_destroy(dat_flow);
          dat_flow = NULL;
      }
  }
  return 0;
}
