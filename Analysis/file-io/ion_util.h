/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ION_UTIL_H
#define ION_UTIL_H

#include <stdint.h>

#ifndef ion_roundup32
/*! 
  rounds up to the nearest power of two integer
  @param  x  the integer to round up
  @return    the smallest integer greater than x that is a power of two 
  */
#define ion_roundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))
#endif

#ifndef htonll
/*! 
  converts a 64-bit value to network order
  @param  x  the 64-bit value to convert
  @return    the converted 64-bit value
  */
#define htonll(x) ((((uint64_t)htonl(x)) << 32) + htonl(x >> 32))
#endif

#ifndef ntohll
/*! 
  converts a 64-bit value to host order
  @param  x  the 64-bit value to convert
  @return    the converted 64-bit value
  */
#define ntohll(x) ((((uint64_t)ntohl(x)) << 32) + ntohl(x >> 32))
#endif

/*! 
  @param  c  the quality value in ASCII format
  @return    the quality value in integer format
  @details   no bounds are enforced
  */
#define CHAR2QUAL(c) ((unsigned char)c-33)

/*! 
  @param  q  the quality value in integer format
  @return    the quality value in ASCII format
  @details   no bounds are enforced
  */
#define QUAL2CHAR(q) (char)(q+33)

extern uint8_t ion_nt_char_to_rc_char[256];
extern uint8_t ion_nt_char_to_int[256];

#ifdef __cplusplus
extern "C" {
#endif

    /*!
      @param  id    the character array where to store the id
      @param  run   the run name to hash
      @param  size  the length of the run name
      */
    void
      ion_run_to_readname(char *id, char *run, int size);
    
    /*!
      @param  id    the character array where to store the id
      @param  x     the x co-ordinate (column)
      @param  y     the y co-ordinate (row)
      */
    void 
      ion_xy_to_readname(char *id, int x, int y);
    
    /*!
      returns the x/y co-ordinates stored in a SFF read name
      @param  readname  the read name
      @param  row       pointer to the row co-ordinate to return (y)
      @param  col       pointer to the column co-ordinate to return (x)
      @return           1 if successful, 0 otherwise
      */
    int32_t
      ion_readname_to_rowcol(const char *readname, int32_t *row, int32_t *col);

    /*!
      returns the x/y co-ordinates stored in a SFF read name
      @param  readname  the read name
      @param  x         pointer to the x co-ordinate to return (column)
      @param  y         pointer to the y co-ordinate to return (row)
      @return           1 if successful, 0 otherwise
      */
    int32_t
      ion_readname_to_xy(const char *readname, int32_t *x, int32_t *y);

    /*!
      returns the x/y co-ordinates from a legacy SFF read name
      @param  id  the hash id
      @param  x         pointer to the x co-ordinate to return (column)
      @param  y         pointer to the y co-ordinate to return (row)
      */
    void 
      ion_id_to_xy(const char *id, int *x, int *y);

    /*!
      @param  readname  the read name
      @return           1 if the name is in a legacy format, 0 otherwise
      */
    int32_t 
      ion_readname_legacy(const char *readname);

    /*!
      reads in byte badding
      @param  fp  file pointer from which to read
      @param  n   the number of bytes read so far
      @return     the number of bytes read after reading the padding
      @details    this assumes that zero padding is read until the number of bytes read is divisible by eight 
      */
    uint32_t
      ion_read_padding(FILE *fp, uint32_t n);

    /*!
      reads in byte badding
      @param  fp  file pointer from which to read
      @param  n   the number of bytes written so far
      @return     the number of bytes written after padding
      @details    this assumes that zero padding is written until the number of bytes read is divisible by eight 
      */
    uint32_t 
      ion_write_padding(FILE *fp, uint32_t n);

    /*! 
      bound a minimum and maximum value (range)
      @param  min  the minimum value to be bounded
      @param  max  the maximum value to be bounded
      @param  num  the upper bound
      @return      1 if succesfully bounded
      @details     the lower bound is assumed to be zero
      */
    int
      ion_bound_values(int32_t *min, int32_t *max, int32_t num);

    /*! 
      parse a range from a string
      @param  str    the string containing the range
      @param  start  pointer to the start of the range to be returned
      @param  end    pointer to the end of the range to be returned
      @return        1 if successful, -1 otherwise
      @details       the range is of the form "-?[0-9]+-?-?[0-9]*", where the first number must be less than or equal to the second.  Examples include "1", "1-2", "-1--1", etc.
      */ 
    int
      ion_parse_range(const char *str, int32_t *start, int32_t *end);

#ifdef __cplusplus
}
#endif

#endif // ION_UTIL_H
