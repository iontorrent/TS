#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "ion_error.h"
#include "ion_alloc.h"
#include "wells_header.h"

wells_header_t *
wells_header_read(FILE *fp)
{
  wells_header_t *h;

  h = ion_calloc(1, sizeof(wells_header_t), __func__, "h");

  if(fread(&h->num_wells, sizeof(uint32_t), 1, fp) != 1
     || fread(&h->num_flows, sizeof(uint16_t), 1, fp) != 1) {
      free(h);
      return NULL;
  } 

  h->flow_order = ion_malloc(sizeof(char)*(h->num_flows+1), __func__, "h");
  if(fread(h->flow_order, sizeof(char), h->num_flows, fp) != h->num_flows) {
      free(h);
      return NULL;
  }
  h->flow_order[h->num_flows]='\0';

  return h;
}
    
int32_t
wells_header_write(FILE *fp, wells_header_t *header)
{
  if(fwrite(&header->num_wells, sizeof(uint32_t), 1, fp) != 1
     || fwrite(&header->num_flows, sizeof(uint16_t), 1, fp) != 1
     || fwrite(header->flow_order, sizeof(char), header->num_flows, fp) != header->num_flows) {
      return 0;
  }
  return 1;
}

void
wells_header_print(FILE *fp, wells_header_t *h)
{
  if(fprintf(fp, "@HD,num_wells=%d,num_flows=%d,flow_order=%s\n",
             h->num_wells, 
             h->num_flows, 
             h->flow_order) < 0) {
      ion_error(__func__, "fprintf", Exit, WriteFileError);
  }
}

void
wells_header_destroy(wells_header_t *h)
{
  free(h->flow_order);
  free(h);
}
