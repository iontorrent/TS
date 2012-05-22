#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <unistd.h>

#include "main.h"
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "sff_read_header.h"

#define min(a,b) ( (a) < (b) ? (a) : (b) )
#define max(a,b) ( (a) > (b) ? (a) : (b) )

static inline void
sff_read_header_ntoh(sff_read_header_t *rh)
{
  // convert values from big-endian
  rh->rheader_length = ntohs(rh->rheader_length);
  rh->name_length = ntohs(rh->name_length);
  rh->n_bases = ntohl(rh->n_bases);
  rh->clip_qual_left = ntohs(rh->clip_qual_left);
  rh->clip_qual_right = ntohs(rh->clip_qual_right);
  rh->clip_adapter_left = ntohs(rh->clip_adapter_left);
  rh->clip_adapter_right = ntohs(rh->clip_adapter_right);
}

static inline void
sff_read_header_hton(sff_read_header_t *rh)
{
  // convert values to big-endian
  rh->rheader_length = htons(rh->rheader_length);
  rh->name_length = htons(rh->name_length);
  rh->n_bases = htonl(rh->n_bases);
  rh->clip_qual_left = htons(rh->clip_qual_left);
  rh->clip_qual_right = htons(rh->clip_qual_right);
  rh->clip_adapter_left = htons(rh->clip_adapter_left);
  rh->clip_adapter_right = htons(rh->clip_adapter_right);
}

static inline void
sff_read_header_calc_bytes(sff_read_header_t *rh)
{
  rh->rheader_length = 0;
  rh->rheader_length += 6 * sizeof(uint16_t);
  rh->rheader_length += 1 * sizeof(uint32_t);
  rh->rheader_length += rh->name_length * sizeof(char);
  if(0 != (rh->rheader_length & 7)) {
      rh->rheader_length += 8 - (rh->rheader_length & 7);
  }
}

sff_read_header_t *
sff_read_header_init()
{
  sff_read_header_t *rh = NULL;
  rh = ion_calloc(1, sizeof(sff_read_header_t), __func__, "rh");
  return rh;
}

sff_read_header_t *
sff_read_header_read(FILE *fp)
{
  sff_read_header_t *rh = NULL;
  uint32_t n = 0;

  rh = sff_read_header_init();

  if(1 != fread(&rh->rheader_length, sizeof(uint16_t), 1, fp)
     || 1 != fread(&rh->name_length, sizeof(uint16_t), 1, fp)
     || 1 != fread(&rh->n_bases, sizeof(uint32_t), 1, fp)
     || 1 != fread(&rh->clip_qual_left, sizeof(uint16_t), 1, fp)
     || 1 != fread(&rh->clip_qual_right, sizeof(uint16_t), 1, fp)
     || 1 != fread(&rh->clip_adapter_left, sizeof(uint16_t), 1, fp)
     || 1 != fread(&rh->clip_adapter_right, sizeof(uint16_t), 1, fp)) {
      free(rh);
      return NULL;
  }
  n += sizeof(uint32_t) + 6*sizeof(uint16_t);

  // convert values from big-endian
  sff_read_header_ntoh(rh);

  rh->name = ion_string_init(rh->name_length+1);

  if(rh->name_length != fread(rh->name->s, sizeof(char), rh->name_length, fp)) {
      // truncated file, error
      ion_error(__func__, "fread", Exit, ReadFileError);
  }
  n += sizeof(char)*rh->name_length;

  // set read name length and null-terminator
  rh->name->l = rh->name_length;
  rh->name->s[rh->name->l]='\0';

  n += ion_read_padding(fp, n);

#ifdef ION_SFF_DEBUG
  sff_read_header_print(stderr, rh);
#endif

  if(rh->rheader_length != n) {
      ion_error(__func__, "SFF read header length did not match", Exit, ReadFileError);
  }

  return rh;
}

uint32_t
sff_read_header_write(FILE *fp, sff_read_header_t *rh)
{
  uint32_t n = 0;

  sff_read_header_calc_bytes(rh);

  // convert values to big-endian
  sff_read_header_hton(rh);

  if(1 != fwrite(&rh->rheader_length, sizeof(uint16_t), 1, fp)
     || 1 != fwrite(&rh->name_length, sizeof(uint16_t), 1, fp)
     || 1 != fwrite(&rh->n_bases, sizeof(uint32_t), 1, fp)
     || 1 != fwrite(&rh->clip_qual_left, sizeof(uint16_t), 1, fp)
     || 1 != fwrite(&rh->clip_qual_right, sizeof(uint16_t), 1, fp)
     || 1 != fwrite(&rh->clip_adapter_left, sizeof(uint16_t), 1, fp)
     || 1 != fwrite(&rh->clip_adapter_right, sizeof(uint16_t), 1, fp)) {
      ion_error(__func__, "fwrite", Exit, WriteFileError);
  }
  n += sizeof(uint32_t) + 6*sizeof(uint16_t);
 
  // convert values from big-endian
  sff_read_header_ntoh(rh);

  if(rh->name_length != fwrite(rh->name->s, sizeof(char), rh->name_length, fp)) {
      ion_error(__func__, "fwrite", Exit, ReadFileError);
  }
  n += sizeof(char)*rh->name_length;

  n += ion_write_padding(fp, n);

  return n;
}

void
sff_read_header_print(FILE *fp, sff_read_header_t *rh)
{
  if(fprintf(fp, "rheader_length=%u", rh->rheader_length) < 0
     || fprintf(fp, ",name_length=%u", rh->name_length) < 0
     || fprintf(fp, ",n_bases=%u", rh->n_bases) < 0
     || fprintf(fp, ",clip_qual_left=%u", rh->clip_qual_left) < 0
     || fprintf(fp, ",clip_qual_right=%u", rh->clip_qual_right) < 0
     || fprintf(fp, ",clip_adapter_left=%u", rh->clip_adapter_left) < 0
     || fprintf(fp, ",clip_adapter_right=%u", rh->clip_adapter_right) < 0
     || fprintf(fp, ",name=%s\n", rh->name->s) < 0) {
  }
} 

void
sff_read_header_destroy(sff_read_header_t *rh)
{
  if(NULL == rh) return;
  ion_string_destroy(rh->name);
  free(rh);
}

void
sff_read_header_get_clip_values(sff_read_header_t* rh,
								int trim_flag,
								int *left_clip,
								int *right_clip) {
    if (trim_flag) {
        (*left_clip)  =
            (int) max(1, max(rh->clip_qual_left, rh->clip_adapter_left));

        // account for the 1-based index value
        *left_clip = *left_clip - 1;

        (*right_clip) = (int) min(
              (rh->clip_qual_right    == 0 ? rh->n_bases : rh->clip_qual_right   ),
              (rh->clip_adapter_right == 0 ? rh->n_bases : rh->clip_adapter_right)
        );
    }
    else {
        (*left_clip)  = 0;
        (*right_clip) = (int) rh->n_bases;
    }
}