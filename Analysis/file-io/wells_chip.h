/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef WELLS_CHIP_H
#define WELLS_CHIP_H

#include "wells_header.h"
#include "wells_data.h"

/*! 
  Structure for holding well data
  */
typedef struct {
    wells_header_t *header;  /*!< the WELLS header */
    wells_data_t **data;  /*!< the WELLS data */
    int32_t num_rows;  /*!< the number of rows */
    int32_t num_cols;  /*!< the number of cols */
} wells_chip_t;

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      @param  fp        the file pointer
      @return           an initialized WELLS structure with the header only read-in data, NULL if unsuccessful 
      @details          assumes all wells are present, which is used to calculate the dimensions
      */
    wells_chip_t *
      wells_chip_read1(FILE *fp);

    /*! 
      @param  fp        the file pointer
      @param  min_row   the 0-based minimum row to read in, -1 will use the lowest row 
      @param  max_row   the 0-based maximum row to read in, -1 will use the highest row 
      @param  min_col   the 0-based minimum col to read in, -1 will use the lowest col 
      @param  max_col   the 0-based maximum col to read in, -1 will use the highest col 
      @return           an initialized WELLS structure with the read-in data, NULL if unsuccessful 
      @details          assumes all wells are present, which is used to calculate the dimensions
      */
    wells_chip_t *
      wells_chip_read(FILE *fp, int32_t min_row, int32_t max_row, int32_t min_col, int32_t max_col);

    /*! 
      @param  fp        the file pointer
      @return           1 if successful, 0 otherwise
      */
    int32_t
      wells_chip_write(FILE *fp, wells_chip_t *chip);

    /*! 
      @param  fp       the file pointer
      @param  chip     the WELLS chip to print
      @param  nonzero  1 will print only flows that have at least one nonzero flow, 0 otherwise
      */
    void
      wells_chip_print(FILE *fp, wells_chip_t *chip, int32_t nonzero);

    /*! 

      @param  chip  pointer to the chip to destroy
      */
    void
      wells_chip_destroy(wells_chip_t *chip);

    /*! 
      main function for viewing a WELLS chip 
      @param  argc  the number of command line arguments
      @param  argv  the command line arguments
      @return       0 if successful
      */
    int
      wells_chip_view_main(int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif // WELLS_CHIP_H
