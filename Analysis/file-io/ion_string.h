/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ION_STRING_H
#define ION_STRING_H

#include <stdint.h>

/*! 
  A Generic String Library
  */

extern uint8_t nt_char_to_rc_char[256];

/*! 
*/
typedef struct {
    size_t l;  /*!< the length of the string */
    size_t m;  /*!< the memory allocated for this string */
    char *s;  /*!< the pointer to the string */
} ion_string_t;

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      @param  mem  the initial memory to allocate for the string
      @return      a pointer to the initialized memory
      */
    ion_string_t *
      ion_string_init(int32_t mem);

    /*! 
      @param  str  a pointer to the string to destroy
      */
    void
      ion_string_destroy(ion_string_t *str);

    /*! 
      analagous to strcpy
      @param  dest  pointer to the destination string
      @param  src   pointer to the source string
      */
    void
      ion_string_copy(ion_string_t *dest, ion_string_t *src);

    /*! 
      analagous to strcpy
      @param  dest  pointer to the destination string
      @param  src   pointer to the source string
      */
    void
      ion_string_copy1(ion_string_t *dest, char *src);

    /*! 
      @param  str  a pointer to the string to clone
      @return      a pointer to the cloned string
      */
    ion_string_t *
      ion_string_clone(ion_string_t *str);

    /*! 
      @param  dest    pointer to the destination string
      @param  l       the number of leading characters to skip
      @param  format  the format for the string
      @param  ...     the arguments to fill in the format
      @details        the first l characters will not be modified
      */
    void
      ion_string_lsprintf(ion_string_t *dest, int32_t l, const char *format, ...);

    /*! 
      reverse the characters in the string
      @param  str  pointer to the string
      */
    void
      ion_string_reverse(ion_string_t *str);

    /*! 
      reverse compliments the string
      @param  str       pointer to the string
      @param  is_int    1 if the sequence is in integer format, 0 otherwise
      */
    void
      ion_string_reverse_compliment(ion_string_t *str, int32_t is_int);

#ifdef __cplusplus
}
#endif

#endif // ION_STRING_H
