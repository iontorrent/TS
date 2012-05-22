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

sff_read_t*
sff_read_init()
{
  sff_read_t *r = NULL;
  r = ion_calloc(1, sizeof(sff_read_t), __func__, "r");
  return r;
}

sff_read_t *
sff_read_read(FILE *fp, sff_header_t *gh, sff_read_header_t *rh)
{
  sff_read_t *r = NULL;
  uint32_t i, n = 0;

  r = sff_read_init();

  r->flowgram = ion_malloc(sizeof(uint16_t)*gh->flow_length, __func__, "r->flowgram");
  r->flow_index = ion_malloc(sizeof(uint8_t)*rh->n_bases, __func__, "r->flow_index");

  r->bases = ion_string_init(rh->n_bases+1);
  r->quality = ion_string_init(rh->n_bases+1);

  if(gh->flow_length != fread(r->flowgram, sizeof(uint16_t), gh->flow_length, fp)
     || rh->n_bases != fread(r->flow_index, sizeof(uint8_t), rh->n_bases, fp)
     || rh->n_bases != fread(r->bases->s, sizeof(char), rh->n_bases, fp)
     || rh->n_bases != fread(r->quality->s, sizeof(char), rh->n_bases, fp)) {
      // truncated file, error
      ion_error(__func__, "fread", Exit, ReadFileError);
  }
  n += sizeof(uint16_t)*gh->flow_length + 3*sizeof(uint8_t)*rh->n_bases;

  // set length and null-terminators
  r->bases->l = rh->n_bases;
  r->quality->l = rh->n_bases;
  r->bases->s[r->bases->l]='\0';
  r->quality->s[r->quality->l]='\0';

  // convert flowgram to host order
  for(i=0;i<gh->flow_length;i++) {
      r->flowgram[i] = ntohs(r->flowgram[i]);
  }

  n += ion_read_padding(fp, n);

#ifdef ION_SFF_DEBUG
  sff_read_print(stderr, r, gh, rh);
#endif

  return r;
}

uint32_t
sff_read_write(FILE *fp, sff_header_t *gh, sff_read_header_t *rh, sff_read_t *r)
{
  uint32_t i, n = 0;
  
  // convert flowgram to network order
  for(i=0;i<gh->flow_length;i++) {
      r->flowgram[i] = htons(r->flowgram[i]);
  }
  
  if(gh->flow_length != fwrite(r->flowgram, sizeof(uint16_t), gh->flow_length, fp)
     || rh->n_bases != fwrite(r->flow_index, sizeof(uint8_t), rh->n_bases, fp)
     || rh->n_bases != fwrite(r->bases->s, sizeof(char), rh->n_bases, fp)
     || rh->n_bases != fwrite(r->quality->s, sizeof(char), rh->n_bases, fp)) {
      ion_error(__func__, "fread", Exit, ReadFileError);
  }
  n += sizeof(uint16_t)*gh->flow_length + 3*sizeof(uint8_t)*rh->n_bases;
  
  // convert flowgram to host order
  for(i=0;i<gh->flow_length;i++) {
      r->flowgram[i] = ntohs(r->flowgram[i]);
  }

  n += ion_write_padding(fp, n);

  return n;
}

void
sff_read_print(FILE *fp, sff_read_t *r, sff_header_t *gh, sff_read_header_t *rh)
{
  uint32_t i;

  // flowgram
  for(i=0;i<gh->flow_length;i++) {
      if(0 < i) {
          if(-1L == fputc(',', fp)) {
            ion_error(__func__, "fputc", Exit, WriteFileError);
          }
      }
      if(fprintf(fp, "%u", r->flowgram[i]) < 0) {
          ion_error(__func__, "fprintf", Exit, WriteFileError);
      }
  }
  if(-1L == fputc('\n', fp)) {
      ion_error(__func__, "fputc", Exit, WriteFileError);
  }

  // flow index
  for(i=0;i<rh->n_bases;i++) {
      if(0 < i) {
          if(-1L == fputc(',', fp)) { 
            ion_error(__func__, "fputc", Exit, WriteFileError);
          }
      }
      if(fprintf(fp, "%u", r->flow_index[i]) < 0) { 
        ion_error(__func__, "fprintf", Exit, WriteFileError);
      }
  }
  if(-1L == fputc('\n', fp)) {
      ion_error(__func__, "fputc", Exit, WriteFileError);
  }

  // bases
  if(fprintf(fp, "%s\n", r->bases->s) < 0) { 
    ion_error(__func__, "fprintf", Exit, WriteFileError);
  }

  // quality
  for(i=0;i<r->quality->l;i++) {
    if(EOF == fputc(QUAL2CHAR(r->quality->s[i]), fp)) {
        ion_error(__func__, "fputc", Exit, WriteFileError);
    }
  }
  fputc('\n', fp);
}

void
sff_read_destroy(sff_read_t *r)
{
  if(NULL == r) return;
  free(r->flowgram);
  free(r->flow_index);
  ion_string_destroy(r->bases);
  ion_string_destroy(r->quality);
  free(r);

}

ion_string_t*
sff_read_get_read_bases(sff_read_t* rd, int left_clip, int right_clip) {
	ion_string_t *bases;

    // account for NULL termination
    int bases_length = (right_clip - left_clip) + 1;

	bases = ion_string_init(bases_length);//MJ?+1
    
    // copy the relative substring
    int start = left_clip;
    int stop  = right_clip;
    int i, j = 0;

    for (i = start; i < stop; i++) {
        *(bases->s + j) = *(rd->bases->s + i);
        j++;
    }

    return bases;
}

ion_string_t*
sff_read_get_read_quality_values(sff_read_t* rd, int left_clip, int right_clip) {
	ion_string_t *quality;

    // account for NULL termination
    int quality_length = (right_clip - left_clip) + 1;

    // inititalize the quality array
    quality = ion_string_init(quality_length+1);
    if (!quality) {
        fprintf(stderr, "Out of memory! For read quality array!\n");
        exit(1);
    }
 
    // copy the relative substring
    int start = left_clip;
    int stop  = right_clip;
    int i, j = 0;

    for (i = start; i < stop; i++) {
        quality->s[j] = rd->quality->s[i];
        j++;
    }

    return quality;
}
