/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DAT_IO_H
#define DAT_IO_H

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      read in a big-endian unsigned 32-bit integer 
      @param  fp    file pointer
      @param  data  pointer where the data is to be stored
      @return       1 if successful, EOF otherwise
      */
    int
      fread_big_endian_uint32_t(FILE *fp, uint32_t *data);

    /*! 
      read in a big-endian unsigned 16-bit integer
      @param  fp    file pointer
      @param  data  pointer where the data is to be stored
      @return       1 if successful, EOF otherwise
      */
    int
      fread_big_endian_uint16_t(FILE *fp, uint16_t *data);

    /*! 
      write out a big-endian unsigned 32-bit integer 
      @param  fp    file pointer
      @param  data  pointer where the data is stored
      @return       1 if successful, EOF otherwise
      */
    int
      fwrite_big_endian_uint32_t(FILE *fp, uint32_t *data);

    /*! 
      write out a big-endian unsigned 16-bit integer
      @param  fp    file pointer
      @param  data  pointer where the data is stored
      @return       1 if successful, EOF otherwise
      */
    int
      fwrite_big_endian_uint16_t(FILE *fp, uint16_t *data);

#ifdef __cplusplus
}
#endif

#endif // DAT_IO_H
