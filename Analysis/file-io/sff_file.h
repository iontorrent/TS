/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFF_FILE_H
#define SFF_FILE_H

#include <stdint.h>
#include "ion_string.h"
#include "sff_definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*!
      @param  filename  the name of the file to open
      @param  mode     the file access mode /[rw](b?)(i?): 'r' for reading, 
      'w' for writing, 'i' for reading/writing an SFF index 
      @param  header   the SFF header if the file is to opened for writing, otherwise ignored
      @param  index    the SFF index if the file is to opened for writing, otherwise ignored. 
      The index is shallow-copied (pointer only), so do not destroy the index before calling sff_close.
      @return          the opened SFF file, with the header read in or written 
      */
    sff_file_t *
      sff_fopen(const char *filename, const char *mode, sff_header_t *header, sff_index_t *index);
    
    /*!
      @param  fildes    the file descriptor of the file to open
      @param  mode     the file access mode /[rw](b?)(i?): 'r' for reading, 
      'w' for writing, 'i' for reading/writing an SFF index 
      @param  header   the SFF header if the file is to opened for writing, otherwise ignored
      @param  index    the SFF index if the file is to opened for writing, otherwise ignored
      The index is shallow-copied (pointer only), so do not destroy the index before calling sff_close.
      @return          the opened SFF file, with the header read in or written 
      */
    sff_file_t *
      sff_fdopen(int filedes, const char *mode, sff_header_t *header, sff_index_t *index);

    /*!
      @param  sff_file  a file pointer to the SFF
      */
    void
      sff_fclose(sff_file_t *sff_file);

    /*!
      @param  header  pointer to global header to copy
	  @return         copy of header
      */
    sff_header_t *
      sff_header_clone(sff_header_t *header);

#ifdef __cplusplus
}
#endif

#endif // SFF_FILE_H
