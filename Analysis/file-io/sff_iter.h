/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFF_ITER_H
#define SFF_ITER_H

typedef struct {
    int32_t min_row;
    int32_t max_row;
    int32_t min_col;
    int32_t max_col;
    int32_t next_row;
    int32_t next_col;
    int64_t new_row;
} sff_iter_t;

#ifdef __cplusplus
extern "C" {
#endif

    /*!
      @param  sff_file  the SFF file to query
      @param  min_row  the 0-based minimum row to read in, -1 will use the lowest row 
      @param  max_row  the 0-based maximum row to read in, -1 will use the highest row 
      @param  min_col  the 0-based minimum col to read in, -1 will use the lowest col 
      @param  max_col  the 0-based maximum col to read in, -1 will use the highest col 
      @return          the initialized SFF file iterator 
      */
    sff_iter_t *
      sff_iter_query(sff_file_t *sff_file,
                     int32_t min_row, int32_t max_row,
                     int32_t min_col, int32_t max_col);

    /*!
      @param  sff_file  the SFF file from which to read
      @param  iter     the SFF file iterator
      @return          the SFF read in, or NULL if unsuccessful
      */
    sff_t *
      sff_iter_read(sff_file_t *sff_file, sff_iter_t *iter);

    /*!
      @param  iter  the SFF file iterator
      */
    void
      sff_iter_destroy(sff_iter_t *iter);

#ifdef __cplusplus
}
#endif

#endif // SFF_ITER_H
