/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFF_SORT_H
#define SFF_SORT_H

#include "sff_definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*!
      @param  fp_in   the file poitner from which to read
      @param  fp_out  the file pointer to which to write
      @details        sorts by x/y co-ordinate in row-major (x-major) order
      */
    void
      sff_sort(sff_file_t *fp_in, sff_file_t *fp_out);

    /*! 
      main function for sorting a SFF file
      @param  argc  the number of command line arguments
      @param  argv  the command line arguments
      @return       0 if successful
      */
    int
      sff_sort_main(int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif // SFF_SORT_H
