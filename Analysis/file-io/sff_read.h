/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFF_READ_H
#define SFF_READ_H

#include "sff_definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

    sff_read_t*
      sff_read_init();

    /*! 
      @param  fp  the file pointer from which to read
      @param  gh  the sff global header
      @param  rh  the sff read header
      @return     a pointer to the sff read to read in
      */
    sff_read_t *
      sff_read_read(FILE *fp, sff_header_t *gh, sff_read_header_t *rh);
    
    ion_string_t* 
	  sff_read_get_read_bases(sff_read_t* rd, int left_clip, int right_clip);
	ion_string_t* 
	  sff_read_get_read_quality_values(sff_read_t* rd, int left_clip, int right_clip);    
    
    /*! 
      @param  fp  the file pointer to which to write
      @param  gh  the sff global header
      @param  rh  the sff read header
      @param  r   the sff read
      @return     the number of bytes written, including padding
      */
    uint32_t 
      sff_read_write(FILE *fp, sff_header_t *gh, sff_read_header_t *rh, sff_read_t *r);

    /*!
      @param  fp  the file pointer to which to print
      @param  r   a pointer to the sff read to print
      @param  gh  the sff global header
      @param  rh  the sff read header
      */
    void
      sff_read_print(FILE *fp, sff_read_t *r, sff_header_t *gh, sff_read_header_t *rh);

    /*! 
      @param  r  a pointer to the sff read to destroy
      */
    void
      sff_read_destroy(sff_read_t *r);

#ifdef __cplusplus
}
#endif

#endif // SFF_READ_H
