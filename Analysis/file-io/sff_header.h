/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFF_HEADER_H
#define SFF_HEADER_H

#include <stdint.h>
#include "ion_string.h"
#include "sff_definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*!
      @return             a pointer to the empty sff 
      */
    sff_header_t *
      sff_header_init();
    
    /*!
      @param  n_reads     the number of reads in the file
      @param  flow_length  the number of nucleotide flows used in this experiment
      @param  flow_order   the flow order (one cycle only)
      @param  key         the key sequence
      @return             a pointer to the initialized sff 
      */
    sff_header_t *
      sff_header_init1(uint32_t n_reads, uint16_t flow_length, const char *flow_order, const char *key);

    /*! 
      @param  fp  the file pointer from which to read
      @return     a pointer to the sff header to read in
      */
    sff_header_t *
      sff_header_read(FILE *fp);

    /*! 
      @param  fp  the file pointer to which to write
      @param  h   a pointer to the sff header to write
      @return     the number of bytes written, including padding
      */
    uint32_t
      sff_header_write(FILE *fp, sff_header_t *h);

    /*!
      @param  fp  the file pointer to which to print
      @param  h   a pointer to the sff header to print
      */
    void
      sff_header_print(FILE *fp, sff_header_t *h);

    /*! 
      @param  h  a pointer to the sff header to destroy
      */
    void
      sff_header_destroy(sff_header_t *h);

#ifdef __cplusplus
}
#endif

#endif // SFF_HEADER_H
