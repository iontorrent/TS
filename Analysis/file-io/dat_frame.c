#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <netinet/in.h>
#include "dat_io.h"
#include "ion_alloc.h"
#include "ion_error.h"
#include "dat_header.h"
#include "dat_frame.h"

#define DAT_FRAME_KEY_0     0x44
#define DAT_FRAME_KEY_8_1   0x99
#define DAT_FRAME_KEY_16_1  0xBB
#define DAT_FRAME_DATA_MASK 0x3fff

// TODO
// - is raw data big-endian?  It doesn't seem like it is when reading.

dat_frame_t *
dat_frame_read(FILE *fp, dat_frame_t *prev, dat_header_t *header) 
{
  dat_frame_t *cur = NULL;

  // malloc
  cur = ion_malloc(sizeof(dat_frame_t), __func__, "cur");
  cur->data = ion_malloc(sizeof(uint16_t)*header->rows*header->cols, __func__, "cur->data");

  // read in the data
  if(NULL == dat_frame_read1(fp, cur, prev, header)) {
      free(cur->data);
      free(cur);
  }

  return cur;
}

dat_frame_t *
dat_frame_read1(FILE *fp, dat_frame_t *cur, dat_frame_t *prev, dat_header_t *header) 
{
  int32_t i, mode;
  uint32_t ctr, compressed;
  int8_t *tmp_data8=NULL; 
  //FILE *fp_debug = stdout; // for debuggin

  // read the timestamp and compression flag for this frame
  if(fread_big_endian_uint32_t(fp, &cur->timestamp) != 1 
     || fread_big_endian_uint32_t(fp, &compressed) != 1) {
      return NULL;
  }

  // the first frame is always uncompressed...
  if(NULL == prev || DAT_HEADER_UNINTERLACED == header->interlace_type) { 
      if(0 != compressed) {
          ion_error(__func__, "compressed", Exit, OutOfRange);
      }

      // read in the data
      if(fread(cur->data, sizeof(uint16_t), header->rows*header->cols, fp) != header->rows*header->cols) { // image data
          return NULL;
      }
      // manually convert from big-endian
      for(i=0;i<header->rows*header->cols;i++) { 
          cur->data[i] = ntohs(cur->data[i]) & DAT_FRAME_DATA_MASK; // unmask data
      }
  }
  else { 
      uint32_t len, transitions, total, sentinel;
      uint32_t observed_transitions = 0;

      // read in the length of the frame, the # of transitions, the 
      // total sum of hte pixel values, and hte sentinel
      if(fread_big_endian_uint32_t(fp, &len) != 1 
         || fread_big_endian_uint32_t(fp, &transitions) != 1 
         || fread_big_endian_uint32_t(fp, &total) != 1
         || fread_big_endian_uint32_t(fp, &sentinel) != 1) {
          return NULL;
      }

      // check the sentinel
      if(sentinel != DAT_HEADER_SIGNATURE) {
          ion_error(__func__, "cur->sentinel", Exit, OutOfRange);
      }

      // subtract len, transitions, total, sentinel
      len -= sizeof(uint32_t)*4; 
      assert(0 < len);

      // initialize memory
      cur->data = ion_calloc(header->rows*header->cols, sizeof(uint16_t), __func__, "cur->data");
      tmp_data8 = ion_malloc(len*sizeof(int8_t), __func__, "tmp_data8");

      // read in the whole frame
      // NOTE: could read in byte-by-byte and process to reduce memory overhead
      if(len != fread(tmp_data8, sizeof(int8_t), len, fp)) {
          free(tmp_data8);
          return NULL;
      }

      // de-interlace the data
      i=mode=ctr=0;
      while(ctr < header->rows*header->cols) {
          if(len <= i) { // not enough bytes read
              ion_error(__func__, "len <= i", Exit, OutOfRange);
          }

          // switch to 8-bit mode, or 16-bit mode where appropriate
          if(i < len-1 && DAT_FRAME_KEY_0 == (uint8_t)tmp_data8[i]) {
              if(DAT_FRAME_KEY_8_1 == (uint8_t)tmp_data8[i+1]) {
                  // 16-bit to 8-bit 
                  observed_transitions++;
                  /*
                     fprintf(stderr, "[%d-%d] from %d-bit mode to 8-bit mode #%d/%d ctr=%d\n", i, i+1, mode, 
                     observed_transitions, transitions, ctr);
                     */
                  mode = 8;
                  i+=2;
              }
              else if(DAT_FRAME_KEY_16_1 == (uint8_t)tmp_data8[i+1]) {
                  // 8-bit to 16-bit
                  observed_transitions++;
                  /*
                     fprintf(stderr, "[%d-%d] from %d-bit mode to 16-bit mode #%d/%d ctr=%d\n", i, i+1, mode,
                     observed_transitions, transitions, ctr);
                     */
                  mode = 16;
                  i+=2;
              }
          }
          // Note: assumes we must have data read between mode switches
          // read in data
          switch(mode) {
            case 8:
              // 8-bit mode
              cur->data[ctr] = tmp_data8[i] + prev->data[ctr];
              ctr++;
              i++;
              break;
            case 16:
              // 16-bit mode
              cur->data[ctr] = (ntohs((int16_t)((tmp_data8[i] << 8) | tmp_data8[i+1])) & DAT_FRAME_DATA_MASK) + prev->data[ctr];
              ctr++;
              i+=2;
              break;
            default:
              // mode?
              ion_error(__func__, "mode", Exit, OutOfRange);
              break;
          }
      }
      if(((i+3) & ~0x3) != len) { // check that the data was quad-word aligned
          ion_error(__func__, "quad word alignment", Exit, OutOfRange);
      }

      // free tmp_data8
      free(tmp_data8);

      // check that the observed # of transitions equals the state # of
      // transitions
      if(transitions != observed_transitions) {
          ion_error(__func__, "transitions != observed_transitions", Exit, OutOfRange);
      }
  }
  return cur;
}

int 
dat_frame_write(FILE *fp, dat_frame_t *cur, dat_frame_t *prev, dat_header_t *header) 
{
  int32_t i;
  uint32_t compressed=0;

  // first frame or uninterlaced
  if(NULL == prev || DAT_HEADER_UNINTERLACED == header->interlace_type) {
      // timestamp and comprssion
      compressed = 0;
      if(fwrite_big_endian_uint32_t(fp, &cur->timestamp) != 1 
         || fwrite_big_endian_uint32_t(fp, &compressed) != 1) {
          return EOF;
      }
      // manually convert to big-endian
      for(i=0;i<header->rows*header->cols;i++) { 
          cur->data[i] = htons(cur->data[i]);
      }
      // write the data
      if(fwrite(cur->data, sizeof(uint16_t), header->rows*header->cols, fp) != header->rows*header->cols) { // image data
          return EOF;
      }
      // manually convert from big-endian
      for(i=0;i<header->rows*header->cols;i++) { 
          cur->data[i] = ntohs(cur->data[i]);
      }
  }
  else {
      // Currently not supported
      // TODO
      ion_error(__func__, "Writing interlaced DAT files currently not supported", Warn, OutOfRange);
      return EOF;
  }
  return 1;
}

void 
dat_frame_destroy(dat_frame_t *f)
{
  if(NULL != f) { // free the data
      free(f->data);
      f->data=NULL;
  }
  free(f);
}
