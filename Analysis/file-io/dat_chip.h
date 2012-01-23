/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DAT_CHIP_H
#define DAT_CHIP_H

/*! 
  Structure storing DAT flows from one run.  
  This structure is useful if all DATs from one run wish to be aggregated into one file
  */
typedef struct {
    dat_flow_t **flows;  /*!< pointers to the flows  */
    uint32_t *byte_offsets;  /*!< the byte offsets of each flow in the DATs file */
    uint32_t num_flows;  /*!< the number of DAT flows */
} dat_chip_t; 

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      read in a DAT chip structure from a file
      @param  fp  file pointer
      @return     a pointer to the DAT chip if successful, NULL otherwise
      */
    dat_chip_t *
      dat_chip_read(FILE *fp);

    /*!
      read in a DAT chip structure from a file containing only the given data
      @param  fp         file pointer
      @param  min_flow    the 0-based minimum flow to read in, -1 will use the lowest flow 
      @param  max_flow    the 0-based maximum flow to read in, -1 will use the highest flow 
      @param  min_frame  the 0-based minimum frame to read in, -1 will use the lowest frame 
      @param  max_frame  the 0-based maximum frame to read in, -1 will use the highest frame 
      @param  min_row    the 0-based minimum row to read in, -1 will use the lowest row 
      @param  max_row    the 0-based maximum row to read in, -1 will use the highest row 
      @param  min_col    the 0-based minimum col to read in, -1 will use the lowest col 
      @param  max_col    the 0-based maximum col to read in, -1 will use the highest col 
      @return            a pointer to the DAT chip if successful, NULL otherwise
      */
    dat_chip_t *
      dat_chip_read1(FILE *fp, int32_t min_flow, int32_t max_flow, int32_t min_frame, int32_t max_frame, int32_t min_row, int32_t max_row, int32_t min_col, int32_t max_col);

    /*! 
      imports DATs to a chip DAT structure
      @param  fp        the file pointer to write
      @param  out_text  1 if the data is to be written as text, 0 otherwise
      @param  num_dats  the number of DATs to be read in
      @param  dat_fns   the DAT file names
      */
    void
      dat_chip_import(FILE *fp, int32_t out_text, int32_t num_dats, char *dat_fns[]);

    /*! 
      write out a DAT chip structure from a file
      @param  fp    file pointer
      @param  chip  pointer to the DAT chip structure
      @return       1 if successful, EOF otherwise
      */
    int
      dat_chip_write(FILE *fp, dat_chip_t *chip);

    /*!
      prints the chip data
      @param  fp    a file pointer
      @param  chip  pointer to the chip to print
      */
    void
      dat_chip_print(FILE *fp, dat_chip_t *chip);

    /*! 
      add the DAT flow structure to the DAT chip structure 
      @param  chip  pointer to the DAT chip structure
      @param  flow   pointer to the DAT flow structure to append
      @details the pointer to the DAT flow structure is copied, not the DAT flow data itself
      */
    void 
      dat_chip_append_flow(dat_chip_t *chip, dat_flow_t *flow);

    /*! 
      free all memory associated with this DAT chip structure
      @param  chip  pointer to the DAT chip structure
      */
    void
      dat_chip_destroy(dat_chip_t *chip);

    /*! 
      main function for viewing a DAT chip 
      @param  argc  the number of command line arguments
      @param  argv  the command line arguments
      @return       0 if successful
      */
    int 
      dat_chip_view_main(int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif // DAT_CHIP_H
