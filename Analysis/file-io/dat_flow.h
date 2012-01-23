/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DAT_FLOW_H
#define DAT_FLOW_H

#include "dat_header.h"
#include "dat_frame.h"

/*! 
  Structure for the DAT flow
  */
typedef struct {
    dat_header_t *header; /*!< the DAT flow header */
    dat_frame_t **frames; /*!< the DAT frames, in row-major order */
} dat_flow_t;

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      read in a DAT flow from the file pointer
      @param  fp  the file pointer
      @return     a pointer to the DAT flow structure, NULL if unsuccessful.
      */
    dat_flow_t *
      dat_flow_read(FILE *fp);

    /*! 
      write out a DAT flow from the file pointer
      @param  fp  the file pointer
      @param  d   a pointer to the DAT flow structure
      @return     1 if successful, EOF otherwise.
      */
    int 
      dat_flow_write(FILE *fp, dat_flow_t *d);

    /*! 
      free memory associated with the DAT flow.
      @param  d  the DAT flow pointer
      */
    void 
      dat_flow_destroy(dat_flow_t *d);

    /*! 
      print a comma-separated (CSV) list representing the flow.
      @param  fp  the file pointer to which to write
      @param  d   the DAT flow structure
      @param  i   the flow index (0-based)
      */
    void 
      dat_flow_print(FILE *fp, dat_flow_t *d, uint32_t i);

    /*! 
      read in a DAT flow from the file pointer
      @param  fp         the file pointer
      @param  min_frame  the 0-based minimum frame to read in, -1 will use the lowest frame 
      @param  max_frame  the 0-based maximum frame to read in, -1 will use the highest frame 
      @param  min_row    the 0-based minimum row to read in, -1 will use the lowest row 
      @param  max_row    the 0-based maximum row to read in, -1 will use the highest row 
      @param  min_col    the 0-based minimum col to read in, -1 will use the lowest col 
      @param  max_col    the 0-based maximum col to read in, -1 will use the highest col 
      @return            a pointer to the DAT flow structure, NULL if unsuccessful.
      */
    dat_flow_t *
      dat_flow_read1(FILE *fp, int32_t min_frame, int32_t max_frame, int32_t min_row, int32_t max_row, int32_t min_col, int32_t max_col); 

    /*! 
      read in a DAT flow from the file pointer
      @param  fp         the file pointer
      @param  min_frame  the 0-based minimum frame to read in, -1 will use the lowest frame 
      @param  max_frame  the 0-based maximum frame to read in, -1 will use the highest frame 
      @param  rows       an strictly increasing list of 0-based rows to read in
      @param  nrows      the length of the row list
      @param  cols       an strictly increasing list of 0-based cols to read in
      @param  ncols      the length of the col list
      @return            a pointer to the DAT flow structure, NULL if unsuccessful.
      */
    dat_flow_t *
      dat_flow_read2(FILE *fp, int32_t min_frame, int32_t max_frame, int32_t *rows, int32_t nrows, int32_t *cols, int32_t ncols);

    /*! 
      main function for viewing a DAT 
      @param  argc  the number of command line arguments
      @param  argv  the command line arguments
      @return       0 if successful
      */
    int 
      dat_flow_view_main(int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif // DAT_FLOW_H
