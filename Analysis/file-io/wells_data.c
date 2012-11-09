#include <stdlib.h>
#include <stdio.h>

#include "ion_alloc.h"
#include "ion_error.h"
#include "wells_data.h"

wells_data_t *
wells_data_read(FILE *fp, wells_header_t *header)
{
  wells_data_t *data;

  data = ion_malloc(sizeof(wells_data_t), __func__, "data");
  data->flow_values = ion_malloc(sizeof(float)*header->num_flows, __func__, "data->flow_values");

  if(NULL == wells_data_read1(fp, header, data)) {
      wells_data_destroy(data);
      return NULL;
  }
  return data;
}

wells_data_t *
wells_data_read1(FILE *fp, wells_header_t *header, wells_data_t *data)
{
  if(fread(&data->rank, sizeof(uint32_t), 1, fp) != 1
     || fread(&data->x, sizeof(uint16_t), 1, fp) != 1
     || fread(&data->y, sizeof(uint16_t), 1, fp) != 1
     || fread(data->flow_values, sizeof(float), header->num_flows, fp) != header->num_flows) {
      return NULL;
  }
  return data;
}

/*
wells_data_t *
wells_data_read_xy(FILE *fp, wells_header_t *header, wells_data_t *data, int32_t x, int32_t y)
{
  int64_t offset;

  // skip over the header
  offset = wells_header_size(header);
  // skip over data
  // TODO: need # of rows and columns...
}
*/
    
int32_t
wells_data_write(FILE *fp, wells_header_t *header, wells_data_t *data)
{
  if(fwrite(&data->rank, sizeof(uint32_t), 1, fp) != 1
     || fwrite(&data->x, sizeof(uint16_t), 1, fp) != 1
     || fwrite(&data->y, sizeof(uint16_t), 1, fp) != 1
     || fwrite(data->flow_values, sizeof(float), header->num_flows, fp) != header->num_flows) {
      return 0;
  }
  return 1;
}

void
wells_data_print(FILE *fp, wells_header_t *header, wells_data_t *data, int32_t nonzero)
{
  int32_t i, output;

  output = 1;
  if(1 == nonzero) {
      output = 0;
      for(i=0;i<header->num_flows;i++) {
          if(0 < data->flow_values[i]) {
              output = 1;
              break;
          }
      }
  }
  if(1 == output) {
      if(fprintf(fp, "%d,%d,%d", data->rank, data->x, data->y) < 0) {
          ion_error(__func__, "fprintf", Exit, WriteFileError);
      }
      for(i=0;i<header->num_flows;i++) {
          if(fprintf(fp, ",%f", data->flow_values[i]) < 0) {
              ion_error(__func__, "fprintf", Exit, WriteFileError);
          }
      }
      if(fprintf(fp, "\n") < 0) {
          ion_error(__func__, "fprintf", Exit, WriteFileError);
      }
  }
}

void
wells_data_destroy(wells_data_t *data)
{
  free(data->flow_values);
  free(data);
}
