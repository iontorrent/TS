/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFF_CHECK_H
#define SFF_CHECK_H

#include "sff_definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      @param  sff     a pointer to the sff to check
      @param  n_err1  the number of type one errors
      @param  n_err2  the number of type two errors
      @param  print   print out error information to stdout
      */
    void
      sff_check(sff_t *sff, int32_t *n_err1, int32_t *n_err2, int32_t print);

#ifdef __cplusplus
}
#endif

#endif // SFF_CHECK_H
