#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "ion_error.h" 
#include "ion_alloc.h"
#include "dat_io.h"
#include "dat_header.h"

dat_header_t *
dat_header_read(FILE *fp)
{
  dat_header_t *h=NULL;

  h = ion_malloc(sizeof(dat_header_t), __func__, "h");

  if(fread_big_endian_uint32_t(fp, &h->signature) != 1
     || fread_big_endian_uint32_t(fp, &h->version) != 1
     || fread_big_endian_uint32_t(fp, &h->header_size) != 1
     || fread_big_endian_uint32_t(fp, &h->data_size) != 1
     || fread_big_endian_uint32_t(fp, &h->wall_time) != 1
     || fread_big_endian_uint16_t(fp, &h->rows) != 1
     || fread_big_endian_uint16_t(fp, &h->cols) != 1
     || fread_big_endian_uint16_t(fp, &h->channels) != 1
     || fread_big_endian_uint16_t(fp, &h->interlace_type) != 1
     || fread_big_endian_uint16_t(fp, &h->frames_in_file) != 1
     || fread_big_endian_uint16_t(fp, &h->reserved) != 1
     || fread_big_endian_uint32_t(fp, &h->sample_rate) != 1
     || fread_big_endian_uint16_t(fp, &h->full_scale_voltage[0]) != 1
     || fread_big_endian_uint16_t(fp, &h->full_scale_voltage[1]) != 1
     || fread_big_endian_uint16_t(fp, &h->full_scale_voltage[2]) != 1
     || fread_big_endian_uint16_t(fp, &h->full_scale_voltage[3]) != 1
     || fread_big_endian_uint16_t(fp, &h->channel_offsets[0]) != 1
     || fread_big_endian_uint16_t(fp, &h->channel_offsets[1]) != 1
     || fread_big_endian_uint16_t(fp, &h->channel_offsets[2]) != 1
     || fread_big_endian_uint16_t(fp, &h->channel_offsets[3]) != 1
     || fread_big_endian_uint16_t(fp, &h->ref_electrode_offset) != 1
     || fread_big_endian_uint16_t(fp, &h->frame_interval) != 1) {
      free(h);
      return NULL;
  }

  if(3 != h->version) { // from Image.cpp
      ion_error(__func__, "h->version", Exit, OutOfRange);
  }

  // Check signature
  if(h->signature != DAT_HEADER_SIGNATURE) {
      ion_error(__func__, "h->signature", Exit, OutOfRange);
  }

  return h;
}	

int 
dat_header_write(FILE *fp, dat_header_t *h)
{
  if(fwrite_big_endian_uint32_t(fp, &h->signature) != 1
     || fwrite_big_endian_uint32_t(fp, &h->version) != 1
     || fwrite_big_endian_uint32_t(fp, &h->header_size) != 1
     || fwrite_big_endian_uint32_t(fp, &h->data_size) != 1
     || fwrite_big_endian_uint32_t(fp, &h->wall_time) != 1
     || fwrite_big_endian_uint16_t(fp, &h->rows) != 1
     || fwrite_big_endian_uint16_t(fp, &h->cols) != 1
     || fwrite_big_endian_uint16_t(fp, &h->channels) != 1
     || fwrite_big_endian_uint16_t(fp, &h->interlace_type) != 1
     || fwrite_big_endian_uint16_t(fp, &h->frames_in_file) != 1
     || fwrite_big_endian_uint16_t(fp, &h->reserved) != 1
     || fwrite_big_endian_uint32_t(fp, &h->sample_rate) != 1
     || fread_big_endian_uint16_t(fp, &h->full_scale_voltage[0]) != 1
     || fread_big_endian_uint16_t(fp, &h->full_scale_voltage[1]) != 1
     || fread_big_endian_uint16_t(fp, &h->full_scale_voltage[2]) != 1
     || fread_big_endian_uint16_t(fp, &h->full_scale_voltage[3]) != 1
     || fread_big_endian_uint16_t(fp, &h->channel_offsets[0]) != 1
     || fread_big_endian_uint16_t(fp, &h->channel_offsets[1]) != 1
     || fread_big_endian_uint16_t(fp, &h->channel_offsets[2]) != 1
     || fread_big_endian_uint16_t(fp, &h->channel_offsets[3]) != 1
     || fwrite_big_endian_uint16_t(fp, &h->ref_electrode_offset) != 1
     || fwrite_big_endian_uint16_t(fp, &h->frame_interval) != 1) {
      return EOF;
  }

  return 1;
}	

// TODO
// - could standardize the names for each of these, similar to the SAM format
int 
dat_header_print(FILE *fp, dat_header_t *h)
{
  if(fprintf(fp, "@HD,signature=%u", h->signature) < 0
     || fprintf(fp, ",version=%u", h->version) < 0
     || fprintf(fp, ",header_size=%u", h->header_size) < 0
     || fprintf(fp, ",data_size=%u", h->data_size) < 0
     || fprintf(fp, ",wall_time=%u", h->wall_time) < 0
     || fprintf(fp, ",rows=%hu", h->rows) < 0
     || fprintf(fp, ",cols=%hu", h->cols) < 0
     || fprintf(fp, ",channels=%hu", h->channels) < 0
     || fprintf(fp, ",interlace_type=%hu", h->interlace_type) < 0
     || fprintf(fp, ",frames_in_file=%hu", h->frames_in_file) < 0
     || fprintf(fp, ",reserved=%hu", h->reserved) < 0
     || fprintf(fp, ",sample_rate=%u", h->sample_rate) < 0
     || fprintf(fp, ",full_scale_voltage={%d,%d,%d,%d}", 
                h->full_scale_voltage[0], h->full_scale_voltage[1], 
                h->full_scale_voltage[2], h->full_scale_voltage[3]) < 0
     || fprintf(fp, ",channel_offsets={%d,%d,%d,%d}", 
                h->channel_offsets[0], h->channel_offsets[1], 
                h->channel_offsets[2], h->channel_offsets[3]) < 0
     || fprintf(fp, ",ref_electrode_offset=%hu", h->ref_electrode_offset) < 0
     || fprintf(fp, ",frame_interval=%hu", h->frame_interval) < 0 
     || fprintf(fp, "\n") < 0) {
      return EOF;
  }
  return 1;
}

void 
dat_header_destroy(dat_header_t *h)
{
  free(h);
}
