#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <netinet/in.h>
#include "dat_io.h"

int 
fread_big_endian_uint32_t(FILE *fp, uint32_t *data)
{
  if(fread(data, sizeof(uint32_t), 1, fp) != 1) {
      return EOF;
  }
  (*data) = ntohl((*data));
  return 1;
}

int 
fread_big_endian_uint16_t(FILE *fp, uint16_t *data)
{
  if(fread(data, sizeof(uint16_t), 1, fp) != 1) {
      return EOF;
  }
  (*data) = ntohs((*data));
  return 1;
}

int 
fwrite_big_endian_uint32_t(FILE *fp, uint32_t *data)
{
  uint32_t tmp_data = htonl((*data));
  if(fwrite(&tmp_data, sizeof(uint32_t), 1, fp) != 1) {
      return EOF;
  }
  return 1;
}

int 
fwrite_big_endian_uint16_t(FILE *fp, uint16_t *data)
{
  uint16_t tmp_data = htons((*data));
  if(fwrite(&tmp_data, sizeof(uint16_t), 1, fp) != 1) {
      return EOF;
  }
  return 1;
}
