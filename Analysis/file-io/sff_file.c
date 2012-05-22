#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <unistd.h>

#include "main.h"
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "sff_header.h"
#include "sff_read_header.h"
#include "sff_read.h"
#include "sff.h"
#include "sff_index.h"
#include "sff_file.h"

sff_header_t *
sff_header_clone(sff_header_t *h)
{
  sff_header_t *ret = NULL;

  ret = ion_calloc(1, sizeof(sff_header_t), __func__, "rh");

  (*ret) = (*h);
  ret->flow = ion_string_clone(h->flow);
  ret->key = ion_string_clone(h->key);

  return ret;
}

static sff_file_t *
sff_fileopen(const char *filename, int filedes, const char *mode, sff_header_t *header, sff_index_t *index)
{
  sff_file_t *sff_file;

  sff_file = ion_calloc(1, sizeof(sff_file_t), __func__, "sff_file");

  if(NULL == filename) {
      sff_file->fp = fdopen(filedes, mode);
  }
  else {
      if(0 == strcmp(filename, "-")) {
          if(NULL != strstr(mode, "r")) {
              sff_file->fp = fdopen(fileno(stdin), mode);
          }
          else {
              sff_file->fp = fdopen(fileno(stdout), mode);
          }
      }
      else {
          sff_file->fp = fopen(filename, mode);
      }
  }
  if(NULL == sff_file->fp) {
      if(NULL != strstr(mode, "r")) {
          fprintf(stderr, "** Could not open %s for reading. **\n", filename);
      }
      else {
          fprintf(stderr, "** Could not open %s for writing. **\n", filename);
      }
      ion_error(__func__, filename, Exit, OpenFileError);
  }

  if(NULL != strstr(mode, "r")) {
      // read in the header
      if(NULL == (sff_file->header = sff_header_read(sff_file->fp))) {
          ion_error(__func__, filename, Exit, ReadFileError);
      }
      sff_file->mode |= 0x1; // reading 
      if(0 < sff_file->header->index_length) {
          if(NULL != strstr(mode, "i")) { 
              // read in the index
              sff_file->index = sff_index_read(sff_file->fp);
              sff_file->mode |= 0x4; // index
          }
          else {
              // skip over
              if(0 != fseek(sff_file->fp, sff_file->header->index_offset + sff_file->header->index_length, 0)) {
                  ion_error(__func__, "fseek", Exit, ReadFileError);
              }
          }
      }
  }
  else if(NULL != strstr(mode, "w")) {
      sff_file->header = sff_header_clone(header);

      if(NULL==strstr(mode,"i") || NULL == index) 
	sff_file->header->index_length = sff_file->header->index_offset = 0; /* strip out index */

      // write the header
      if(0 == sff_header_write(sff_file->fp, sff_file->header)) {
          ion_error(__func__, filename, Exit, WriteFileError);
      }
      if(NULL != strstr(mode, "i")) {
		  if(NULL != index) { // do not write the index
			  sff_file->index = index;
			  sff_file->mode |= 0x4; // index
			  sff_index_write(sff_file->fp, sff_file->index);
		  }
      }
      sff_file->mode |= 0x1; // writing
  }
  else {
      ion_error(__func__, "either 'b' or 'w' must be specified", Exit, OutOfRange);
  }

  return sff_file;
}

sff_file_t *
sff_fopen(const char *filename, const char *mode, sff_header_t *header, sff_index_t *index)
{
  return sff_fileopen(filename, -1, mode, header, index);
}

sff_file_t *
sff_fdopen(int filedes, const char *mode, sff_header_t *header, sff_index_t *index)
{
  return sff_fileopen(NULL, filedes, mode, header, index);
}

void
sff_fclose(sff_file_t *sff_file)
{
  if(sff_file->mode & 0x4) { // the sff contains an index
      if(0 < sff_file->header->index_length) {
          sff_index_destroy(sff_file->index);
      }
  }

  sff_header_destroy(sff_file->header);
  fclose(sff_file->fp);
  free(sff_file);
}
