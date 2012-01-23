#include <stdlib.h>
#include <stdio.h>

#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "sff_definitions.h"
#include "sff.h"
#include "sff_iter.h"

sff_iter_t *
sff_iter_query(sff_file_t *sff_file,
               int32_t min_row, int32_t max_row,
               int32_t min_col, int32_t max_col)
{
  sff_iter_t *iter;
  sff_index_t *idx;

  idx = sff_file->index;
  if(NULL == idx) { 
      ion_error(__func__, "the input file has no index", Warn, OutOfRange);
      return NULL;
  }

  // bound values
  if(ion_bound_values(&min_row, &max_row, idx->num_rows) < 0
     || ion_bound_values(&min_col, &max_col, idx->num_cols) < 0) {
      return NULL;
  }

  iter = ion_calloc(1, sizeof(sff_iter_t), __func__, "iter");

  iter->min_row = min_row;
  iter->max_row = max_row;
  iter->min_col = min_col;
  iter->max_col = max_col;

  iter->next_row = min_row;
  iter->next_col = min_col;
  iter->new_row = 1;

  return iter;
}

sff_t *
sff_iter_read(sff_file_t *sff_file, sff_iter_t *iter)
{
  sff_index_t *idx;
  sff_t *sff = NULL;
  int32_t row, col, read_next, should_seek, found;
  uint64_t seek_pos = 0;

  /*
     fprintf(stdout, "BEG next[%d,%d] BOUNDS[%d-%d,%d-%d]\n",
     iter->next_row, iter->next_col,
     iter->min_row, iter->max_row,
     iter->min_col, iter->max_col);
     fprintf(stdout, "%s:A: %ld\n", __func__, ftell(sff_file->fp));
     */

  idx = sff_file->index;
  if(NULL == idx) { 
      ion_error(__func__, "the input file has no index", Warn, OutOfRange);
      return NULL;
  }

  // check if there are more entries
  if(iter->max_row < iter->next_row ||
     (iter->max_row == iter->next_row && iter->max_col < iter->next_col)) {
      //fprintf(stdout, "HERE RET 1\n");
      // none left within range
      return NULL;
  }


  /*
     fprintf(stdout, "BOUNDS[%d,%d] NEXT[%d,%d] %d seek_pos=%ld\n", 
     idx->num_rows, idx->num_cols,
     iter->next_row, iter->next_col,
     iter->next_col + (iter->next_row * idx->num_cols),
     (long int)seek_pos);
     */

  // find the read
  if(SFF_INDEX_ROW_ONLY == idx->type) {
      // skip over empty in this column
      found = 0;
      should_seek = iter->new_row;
      while(0 == found) { 
          if(iter->max_row < iter->next_row ||
             (iter->max_row == iter->next_row && iter->max_col < iter->next_col)) {
              // none left within range
              return NULL;
          }
          // get the start of this row
          seek_pos = idx->offset[iter->next_row];
          if(UINT64_MAX == seek_pos) { // empty row
              should_seek = 1;
              iter->next_row++;
              iter->next_col = iter->min_col;
              continue; // find a non-empty row
          }
          // seek, if necessary
          if(1 == should_seek || 1 == iter->new_row) {
              if(0 != fseek(sff_file->fp, seek_pos, SEEK_SET)) {
                  ion_error(__func__, "fseek", Exit, ReadFileError);
              }
          }
          // found a non-empty row, but are there any entries within range in this row?
          // may need to skip over entries at the begginning of the column
          read_next = 1;
          while(1) {
              // check if there are more entries
              if(iter->max_row < iter->next_row ||
                 (iter->max_row == iter->next_row && iter->max_col < iter->next_col)) {
                  // none left within range
                  return NULL;
              }
              if(1 == read_next) { // read in an sff
                  sff_destroy(sff); // destroy before reading
                  if(NULL == (sff = sff_read(sff_file))) {
                      return NULL;
                  }
                  // get the x/y co-ordinates
                  if(0 == ion_readname_to_rowcol(sff->rheader->name->s, &row, &col)) {
                      ion_error(__func__, "could not understand the read name", Exit, OutOfRange);
                  }
              }
              // check ranges
              if(iter->max_row < row ||
                 (iter->max_row == row && iter->max_col < col)) {
                  return NULL;
              }
              else if(row < iter->next_row) { // bug
                  ion_error(__func__, "bug encountered", Exit, OutOfRange);
              }
              else if(iter->next_row == row) { // same row
                  if(col < iter->next_col) {
                      // before the start column, skip
                      read_next = 1;
                  }
                  else if(col <= iter->max_col) {
                      // within range
                      iter->next_col = col;
                      found = 1; // found!
                      break; // while(1)
                  }
                  else {
                      // after the max column, go to a new row, skip
                      break; // while(1)
                  }
              }
              else { // new row
                  // go to a new row, do not skip, do not read in new sff
                  iter->next_row++;
                  iter->next_col = iter->min_col;
                  read_next = 0; // do not read in a new sff
              }
          }
      }
  }
  else if(SFF_INDEX_ALL  == idx->type) {
      // row-major
      should_seek = iter->new_row;
      seek_pos = idx->offset[iter->next_col 
        + (iter->next_row * idx->num_cols)];
      while(UINT64_MAX == seek_pos) { // empty row/col
          should_seek = 1;
          iter->next_col++; // next column
          if(iter->max_col < iter->next_col) {
              iter->next_col = iter->min_col;
              iter->next_row++;
          }
          if(iter->max_row < iter->next_row
             || (iter->max_row == iter->next_row 
                 && iter->max_col < iter->next_col)) { // out of range
              //fprintf(stdout, "HERE RET 2B\n");
              return NULL;
          }
          seek_pos = idx->offset[iter->next_col 
            + (iter->next_row * idx->num_cols)];
      }
      // seek, if necessary
      if(1 == should_seek || 1 == iter->new_row) {
          //fprintf(stdout, "seek_pos=%ld\n", (long int)seek_pos);
          if(0 != fseek(sff_file->fp, seek_pos, SEEK_SET)) {
              ion_error(__func__, "fseek", Exit, ReadFileError);
          }
      }
      // read
      sff = sff_read(sff_file);
  }
  else {
      ion_error(__func__, "index type not supported", Exit, OutOfRange);
  }

  /*
     fprintf(stdout, "HERE 4\n");
     if(NULL == sff) {
     fprintf(stdout, "RET NULL\n");
     }
     else {
     fprintf(stdout, "RET SFF\n");
     }
     */

  // next row/col
  iter->next_col++;
  if(iter->max_col < iter->next_col) {
      iter->next_col = iter->min_col;
      iter->next_row++;
      iter->new_row = 1;
  }
  else {
      iter->new_row = 0;
  }
  /*
     fprintf(stdout, "END next[%d,%d] BOUNDS[%d-%d,%d-%d]\n",
     iter->next_row, iter->next_col,
     iter->min_row, iter->max_row,
     iter->min_col, iter->max_col);
     */
  return sff;
}

void
sff_iter_destroy(sff_iter_t *iter)
{
  free(iter);
}
