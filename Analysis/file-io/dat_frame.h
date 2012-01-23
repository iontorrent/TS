/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DAT_FRAME_H
#define DAT_FRAME_H

/*! 
  calculate the size of an uninterlaced frame
  @param  _n       the number of frames to skip 
  @param  _header  the DAT header 
  @return          the number of bytes to skip
  @details         this includes the size of the eight reference pixels
  */
#define __dat_frame_size(_n, _header) \
  (_n*(sizeof(uint32_t)*2 + sizeof(uint16_t)*(8 + _header->rows*_header->cols)))

/*! 
  Structure for the DAT frame
  */
typedef struct {
    uint32_t timestamp;  /*!< relative time from the start of acquisition for this frame */
    uint16_t *data;  /*!< the frame data for all wells */
} dat_frame_t;

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      read in the DAT frame
      @param  fp      file pointer
      @param  prev    pointer to the previous frame data. NULL if this is the first frame.  This argument will be ignored if the DAT is uninterlaced.
      @param  header  pointer to the DAT header
      @return         pointer to the current frame data if read successfully, NULL otherwise
      */
    dat_frame_t *
      dat_frame_read(FILE *fp, dat_frame_t *prev, dat_header_t *header);

    /*! 
      read in the DAT frame
      @param  fp      file pointer
      @param  cur     pointer to the current dat frame
      @param  prev    pointer to the previous frame data. NULL if this is the first frame.  This argument will be ignored if the DAT is uninterlaced.
      @param  header  pointer to the DAT header
      @return         pointer to the current frame data if read successfully, NULL otherwise
      */
    dat_frame_t *
      dat_frame_read1(FILE *fp, dat_frame_t *cur, dat_frame_t *prev, dat_header_t *header);

    /*! 
      write the DAT frame
      @param  fp      file pointer
      @param  cur     pointer to the previous frame data. NULL if this is the first 
      @param  prev    pointer to the previous frame data. NULL if this is the first 
      frame.  
      @param  header  pointer to the DAT header
      @return         1 if successful, EOF otherwise
      */
    int
      dat_frame_write(FILE *fp, dat_frame_t *cur, dat_frame_t *prev, dat_header_t *header);

    /*! 
      free the DAT frame
      @param  f  pointer to the frame data 
      */
    void
      dat_frame_destroy(dat_frame_t *f);

#ifdef __cplusplus
}
#endif

#endif // DAT_FRAME_H
