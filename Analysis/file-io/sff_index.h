/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFF_INDEX_H
#define SFF_INDEX_H

#include "sff_definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*!
      @param  fp_in     the file poitner from which to read
      @param  fp_out_header  the header to be written (already dup'd from fp_in)
      @param  num_rows  the number of rows
      @param  num_cols  the number of columns
      @param  type      the type of SFF index
      *
      *@return the index to be inserted in the output file. Side effect: fp_out_header is also appropriately modified.
      */
    sff_index_t*
      sff_index_create(sff_file_t *fp_in, sff_header_t *fp_out_header, int32_t num_rows, int32_t num_cols, int32_t type);

    /*!
      @param  fp  the file pointer 
      @return     pointer to the initialized pointer
      */
    sff_index_t *
      sff_index_read(FILE *fp);

    /*!
      @param  fp   the file pointer 
      @param  idx  the index to write
      @return      the number of bytes written, including padding
      */
    uint32_t
      sff_index_write(FILE *fp, sff_index_t *idx);

    /*!
      @param  idx  the index to destroy
      */
    void
      sff_index_destroy(sff_index_t *idx);

    /*! 
      main function for indexing a SFF file
      @param  argc  the number of command line arguments
      @param  argv  the command line arguments
      @return       0 if successful
      */
    int
      sff_index_create_main(int argc, char *argv[]);


#ifdef __cplusplus
}
#endif

#endif // SFF_INDEX_H
