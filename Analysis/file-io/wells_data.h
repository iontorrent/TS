/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef WELLS_DATA_H
#define WELLS_DATA_H

#include <stdint.h>
#include "wells_header.h"

#define wells_data_size(_header) (sizeof(int32_t) + (2 * sizeof(uint16_t)) + (_header->num_flows * sizeof(float)))

/*! 
  Structure for holding data for a single well
  */
typedef struct {
    int32_t rank;  /*!< unknown field */
    uint16_t x;  /*!< the x-coordinate of the well (0-based) */
    uint16_t y;  /*!< the y-coordinate of the well (0-based) */
    float *flow_values;  /*!< the values for each flow for this well */
} wells_data_t;

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      read in a wells-worth of data
      @param  fp      the file pointer
      @param  header  the header previously read in
      @return         a pointer to a wells-worth of data, NULL if unsuccessful
      */
    wells_data_t *
      wells_data_read(FILE *fp, wells_header_t *header);

    /*! 
      read in a wells-worth of data
      @param  fp      the file pointer
      @param  header  the header previously read in
      @param  data    a pointer where to store the data
      @return         the data pointer, NULL if unsuccessful
      */
    wells_data_t *
      wells_data_read1(FILE *fp, wells_header_t *header, wells_data_t *data);
    
    /*! 
      write out a wells-worth of data
      @param  fp      the file pointer
      @param  header  the header 
      @param  data    the data to write
      @return         1 if successful, 0 otherwise
      */
    int32_t
      wells_data_write(FILE *fp, wells_header_t *header, wells_data_t *data);

    /*! 
      print a wells-worth of data
      @param  fp       the file pointer
      @param  header   the WELLS header 
      @param  data     the data to write
      @param  nonzero  1 will print only flows that have at least one nonzero flow, 0 otherwise
      */
    void
      wells_data_print(FILE *fp, wells_header_t *header, wells_data_t *data, int32_t nonzero);

    /*! 
      @param  data  a pointer to the data to destroy
      */
    void
      wells_data_destroy(wells_data_t *data);

#ifdef __cplusplus
}
#endif

#endif // WELLS_DATA_H
