/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef WELLS_HEADER_H
#define WELLS_HEADER_H

#include <stdint.h>

#define wells_header_size(h) (sizeof(uint32_t) + sizeof(uint16_t) + ((h)->num_flows * sizeof(char))) 

/*! 
  Structure describing the given WELLS data
  */
typedef struct {
    uint32_t num_wells;  /*!< the number of wells */
    uint16_t num_flows;  /*!< the number of flows per well */
    char *flow_order;  /*!< the flow base for each flow */
} wells_header_t;

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      read in the WELLS file header
      @param  fp  the file pointer
      @return     a pointer to the initialized memory
      */
    wells_header_t *
      wells_header_read(FILE *fp);

    /*! 
      write the WELLS file header
      @param  fp  the file pointer
      @return     1 if successful, 0 otherwise
      */
    int32_t
      wells_header_write(FILE *fp, wells_header_t *header);

    /*! 
      print the WELLS file header
      @param  fp  the file pointer
      @param  h   pointer to the header to print
      */
    void
      wells_header_print(FILE *fp, wells_header_t *h);

    /*! 
      destroys the data associated with this header
      @param  h  pointer to the header
      */
    void
      wells_header_destroy(wells_header_t *h);

#ifdef __cplusplus
}
#endif

#endif // WELLS_HEADER_H
