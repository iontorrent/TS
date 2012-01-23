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

static inline void
sff_header_ntoh(sff_header_t *h)
{
  // convert values from big-endian
  h->magic = ntohl(h->magic);
  h->version = ntohl(h->version);
  h->index_offset = ntohll(h->index_offset);
  h->index_length = ntohl(h->index_length);
  h->n_reads = ntohl(h->n_reads);
  h->gheader_length = ntohs(h->gheader_length);
  h->key_length = ntohs(h->key_length);
  h->flow_length = ntohs(h->flow_length);
}

static inline void
sff_header_hton(sff_header_t *h)
{
  h->magic = htonl(h->magic);
  h->version = htonl(h->version);
  h->index_offset = htonll(h->index_offset);
  h->index_length = htonl(h->index_length);
  h->n_reads = htonl(h->n_reads);
  h->gheader_length = htons(h->gheader_length);
  h->key_length = htons(h->key_length);
  h->flow_length = htons(h->flow_length);
}

sff_header_t *
sff_header_init()
{
  return ion_calloc(1, sizeof(sff_header_t), __func__, "return");
}

static void
sff_header_calc_bytes(sff_header_t *h)
{
  h->gheader_length = 0;
  h->gheader_length += 1 * sizeof(uint8_t);
  h->gheader_length += 3 * sizeof(uint16_t);
  h->gheader_length += 4 * sizeof(uint32_t);
  h->gheader_length += 1 * sizeof(uint64_t);
  h->gheader_length += (h->flow_length + h->key_length) * sizeof(char);
  if(0 != (h->gheader_length & 7)) {
      h->gheader_length += 8 - (h->gheader_length & 7); // 8 - (n % 8) -> add padding
  }
}
    
sff_header_t *
sff_header_init1(uint32_t n_reads, uint16_t flow_length, const char *flow_order, const char *key)
{
  sff_header_t *h;
  int32_t i, cycle_length, key_length;

  h = sff_header_init();

  h->magic = SFF_MAGIC;
  h->version = SFF_VERSION;
  h->n_reads = n_reads;
  h->key_length = strlen(key);
  h->flow_length = flow_length;
  h->flowgram_format = 1;
  h->flow = ion_string_init(flow_length+1);
  cycle_length = strlen(flow_order);
  for(i=0;i<flow_length;i++) {
      h->flow->s[i] = flow_order[i % cycle_length];
  }
  h->flow->l = flow_length;
  h->flow->s[h->flow->l] = '\0';
  key_length = strlen(key);
  h->key = ion_string_init(key_length+1);
  for(i=0;i<key_length;i++) {
    h->key->s[i] = key[i];
  }
  h->key->l = key_length;
  h->key->s[h->key->l] = '\0';

  // reset gheader_length
  sff_header_calc_bytes(h);

  return h;
}

sff_header_t *
sff_header_read(FILE *fp)
{
  sff_header_t *h = NULL;
  uint32_t n = 0;

  h = sff_header_init();

  if(1 != fread(&h->magic, sizeof(uint32_t), 1, fp)
     || 1 != fread(&h->version, sizeof(uint32_t), 1, fp)
     || 1 != fread(&h->index_offset, sizeof(uint64_t), 1, fp)
     || 1 != fread(&h->index_length, sizeof(uint32_t), 1, fp)
     || 1 != fread(&h->n_reads, sizeof(uint32_t), 1, fp)
     || 1 != fread(&h->gheader_length, sizeof(uint16_t), 1, fp)
     || 1 != fread(&h->key_length, sizeof(uint16_t), 1, fp)
     || 1 != fread(&h->flow_length, sizeof(uint16_t), 1, fp)
     || 1 != fread(&h->flowgram_format, sizeof(uint8_t), 1, fp)) {
      ion_error(__func__, "fread", Exit, ReadFileError);
  }
  n += 4*sizeof(uint32_t) + sizeof(uint64_t) + 3*sizeof(uint16_t) + sizeof(uint8_t);

  // convert values from big-endian
  sff_header_ntoh(h);

  if(SFF_MAGIC != h->magic) {
      ion_error(__func__, "SFF magic number did not match", Exit, ReadFileError);
  }
  if(h->version != SFF_VERSION) {
      ion_error(__func__, "SFF version number did not match", Exit, ReadFileError);
  }

  h->flow = ion_string_init(h->flow_length+1);
  h->key = ion_string_init(h->key_length+1);

  if(h->flow_length != fread(h->flow->s, sizeof(char), h->flow_length, fp)
     || h->key_length != fread(h->key->s, sizeof(char), h->key_length, fp)) {
      ion_error(__func__, "fread", Exit, ReadFileError);
  }
  n += sizeof(char)*(h->flow_length + h->key_length);

  // set the length and null-terminator
  h->flow->l = h->flow_length;
  h->key->l = h->key_length;
  h->flow->s[h->flow->l]='\0';
  h->key->s[h->key->l]='\0';

  n += ion_read_padding(fp, n);

#ifdef ION_SFF_DEBUG
  sff_header_print(stderr, h);
#endif

  if(h->gheader_length != n) {
      ion_error(__func__, "SFF global header length did not match", Exit, ReadFileError);
  }

  return h;
}

void
sff_header_print(FILE *fp, sff_header_t *h)
{
  if(fprintf(fp, "@HD") < 0
     || fprintf(fp, ",magic=%u", h->magic) < 0
     || fprintf(fp, ",version=%u", h->version) < 0
     || fprintf(fp, ",index_offset=%llu", (long long unsigned int)h->index_offset) < 0
     || fprintf(fp, ",index_length=%u", h->index_length) < 0
     || fprintf(fp, ",n_reads=%u", h->n_reads) < 0
     || fprintf(fp, ",gheader_length=%u", h->gheader_length) < 0
     || fprintf(fp, ",key_length=%u", h->key_length) < 0
     || fprintf(fp, ",flow_length=%u", h->flow_length) < 0
     || fprintf(fp, ",flowgram_format=%u", h->flowgram_format) < 0
     || fprintf(fp, ",flow=%s", h->flow->s) < 0
     || fprintf(fp, ",key=%s\n", h->key->s) < 0) {
      ion_error(__func__, "fprintf", Exit, WriteFileError);
  }
}

uint32_t 
sff_header_write(FILE *fp, sff_header_t *h)
{
  uint32_t n = 0;
  
  // reset gheader_length
  sff_header_calc_bytes(h); // just in case
  
  // convert values to big-endian
  sff_header_hton(h);

  if(1 != fwrite(&h->magic, sizeof(uint32_t), 1, fp)
     || 1 != fwrite(&h->version, sizeof(uint32_t), 1, fp)
     || 1 != fwrite(&h->index_offset, sizeof(uint64_t), 1, fp)
     || 1 != fwrite(&h->index_length, sizeof(uint32_t), 1, fp)
     || 1 != fwrite(&h->n_reads, sizeof(uint32_t), 1, fp)
     || 1 != fwrite(&h->gheader_length, sizeof(uint16_t), 1, fp)
     || 1 != fwrite(&h->key_length, sizeof(uint16_t), 1, fp)
     || 1 != fwrite(&h->flow_length, sizeof(uint16_t), 1, fp)
     || 1 != fwrite(&h->flowgram_format, sizeof(uint8_t), 1, fp)) {
      ion_error(__func__, "fwrite", Exit, WriteFileError);
  }
  n += 4*sizeof(uint32_t) + sizeof(uint64_t) + 3*sizeof(uint16_t) + sizeof(uint8_t);
  
  // convert values from big-endian
  sff_header_ntoh(h);

  if(h->flow_length != fwrite(h->flow->s, sizeof(char), h->flow_length, fp)
     || h->key_length != fwrite(h->key->s, sizeof(char), h->key_length, fp)) {
      ion_error(__func__, "fwrite", Exit, WriteFileError);
  }
  n += sizeof(char)*(h->flow_length + h->key_length);
  
  n += ion_write_padding(fp, n);

  return n;
}

void
sff_header_destroy(sff_header_t *h)
{
  if(NULL == h) return;
  ion_string_destroy(h->flow);
  ion_string_destroy(h->key);
  free(h);
}
