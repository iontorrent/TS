/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFF_READ_HEADER_H
#define SFF_READ_HEADER_H

#include <stdint.h>
#include "ion_string.h"
#include "sff_definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

    sff_read_header_t *
      sff_read_header_init();

    /*! 
      @param  fp  the file pointer from which to read
      @return     a pointer to the sff read header read in
      */
    sff_read_header_t *
      sff_read_header_read(FILE *fp);    
 
    void 
    sff_read_header_get_clip_values(sff_read_header_t* rh,
									int trim_flag,
									int *left_clip,
									int *right_clip);
    
    /*! 
      @param  fp  the file pointer to which to write
      @param  rh  a pointer to the sff read header to write 
      @return     the number of bytes written, including the padding
      */
    uint32_t
      sff_read_header_write(FILE *fp, sff_read_header_t *rh);

    /*!
      @param  fp  the file pointer to which to print
      @param  rh  a pointer to the sff read header to print
      */
    void
      sff_read_header_print(FILE *fp, sff_read_header_t *rh);

    /*! 
      @param  rh  a pointer to the sff read header to destroy
      */
    void
      sff_read_header_destroy(sff_read_header_t *rh);

#ifdef __cplusplus
}
#endif

#endif // SFF_READ_HEADER_H
