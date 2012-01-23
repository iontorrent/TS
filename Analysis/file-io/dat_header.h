/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DAT_HEADER_H
#define DAT_HEADER_H

/*! the sentinel value for each interlaced frame */
#define DAT_HEADER_SIGNATURE 0xDEADBEEF
/*! the DAT file is uninterlaced */
#define DAT_HEADER_UNINTERLACED 0
/*! the DAT file is interlaced */
#define DAT_HEADER_INTERLACED 4

#define DAT_HEADER_DEVICE_CHANNELS 4

/*! 
  Structure for the DAT flow header
  */
typedef struct {
    uint32_t signature;  /*!< signature */
    uint32_t version;  /*!< the version number */
    uint32_t header_size;  /*!< size in bytes of the header */
    uint32_t data_size;  /*!< size in bytes of the data portion */
    uint32_t wall_time;  /*!< time of acquisition */
    uint16_t rows;  /*!< number of rows in the following images (includes pixel border) */
    uint16_t cols;  /*!< number of columns in the following images (includes pixel border) */
    uint16_t channels;  /*!< number of channels in the imaged chip */
    uint16_t interlace_type;  /*!< 0=uniterlaced, 4=compressed */
    uint16_t frames_in_file;  /*!< number of frames to follow this header */
    uint16_t reserved;  /*!< a reserved field */
    uint32_t sample_rate;  /*!< acquisition speed at which the image was taken    */
    uint16_t full_scale_voltage[DAT_HEADER_DEVICE_CHANNELS];  /*! max voltage for the channel A/D's */
    uint16_t channel_offsets[DAT_HEADER_DEVICE_CHANNELS];  /*! current voltage for the channel A/D's */
    uint16_t ref_electrode_offset;  /*!< voltage of the fluid flowing over the chip */
    uint16_t frame_interval;  /*!< time interval between frames */
} dat_header_t;

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      read in the DAT header
      @param  fp  file pointer
      @return     pointer to the header data if successfully read, NULL otherwise
      */
    dat_header_t *
      dat_header_read(FILE *fp);

    /*! 
      write out the DAT header
      @param  fp  file pointer
      @param  h   the DAT header
      @return     1 if successful, EOF otherwise
      */
    int
      dat_header_write(FILE *fp, dat_header_t *h);

    /*! 
      print the DAT header
      @param  fp  file pointer
      @param  h   pointer to the header data
      @return     1 if successful, EOF otherwise
      */
    int 
      dat_header_print(FILE *fp, dat_header_t *h);

    /*!  
      free memory associated with the DAT header structure
      @param  h  the DAT header pointer
      */
    void 
      dat_header_destroy(dat_header_t *h);

#ifdef __cplusplus
}
#endif

#endif // DAT_HEADER_H
