/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef WELLS_MASK_H
#define WELLS_MASK_H

/*!
  All wells start out with WellsMaskEmpty.
  WellsMaskEmpty, WellsMaskPinned, WellsMaskBead are mutually exclusive
  WellsMaskLive, WellsMaskDud, WellsMaskAmbiguous are mutually exclusive
  WellsMaskTF, WellsMaskLib are mututally exclusive
  */
enum {
    WellsMaskNone      = 0,
    WellsMaskEmpty     = (1<<0),
    WellsMaskBead      = (1<<1),
    WellsMaskLive      = (1<<2),
    WellsMaskDud       = (1<<3),
    WellsMaskAmbiguous = (1<<4),
    WellsMaskTF        = (1<<5),
    WellsMaskLib       = (1<<6),
    WellsMaskPinned    = (1<<7),
    WellsMaskIgnore    = (1<<8),
    WellsMaskWashout   = (1<<9),
    WellsMaskExclude   = (1<<10),
    WellsMaskKeypass   = (1<<11),
    WellsMaskAll       = 0xffff,
};

/*!
  Structure for storing the well masks, in row-major format
  */
typedef struct {
    int32_t num_rows; /*!< the number of rows */
    int32_t num_cols; /*!< the number of columns */
    uint16_t **masks; /*!< the mask for each well */
} wells_mask_t;

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      @param  fp  a file pointer 
      @return     pointer to the initialized memory
      */
    wells_mask_t *
      wells_mask_read(FILE *fp);

    /*! 
      @param  fp    a file pointer 
      @param  mask  pointer to the mask to write
      @details      this will write in binary format
      */
    void
      wells_mask_write(FILE *fp, wells_mask_t *mask);

    /*! 
      @param  fp    a file pointer 
      @param  mask  pointer to the mask to write
      @details      this will write a CSV file in column-major order
      */
    void
      wells_mask_print(FILE *fp, wells_mask_t *mask);

    /*!
      @param  mask  pointer to hte mask to destroy
      */
    void
      wells_mask_destroy(wells_mask_t *mask);

#ifdef __cplusplus
}
#endif

#endif // WELLS_MASK_H
