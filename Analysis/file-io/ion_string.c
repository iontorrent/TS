#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include "ion_error.h"
#include "ion_alloc.h"
#include "ion_util.h"
#include "ion_string.h"

ion_string_t *
ion_string_init(int32_t mem)
{
  ion_string_t *str = NULL;

  str = ion_calloc(1, sizeof(ion_string_t), __func__, "str");
  if(0 < mem) {
      str->m = mem;
      str->s = ion_calloc(str->m, sizeof(char), __func__, "str->s");
      str->l = 0;
      str->s[str->l] = '\0';
  }

  return str;
}

void
ion_string_destroy(ion_string_t *str)
{
  if(NULL == str) return;
  free(str->s);
  free(str);
}

void
ion_string_copy(ion_string_t *dest, ion_string_t *src)
{
  int32_t i;
  if(dest->m < src->m) {
      dest->m = src->m;
      dest->s = ion_realloc(dest->s, sizeof(char)*dest->m, __func__, "dest->s");
  }
  for(i=0;i<src->m;i++) { // copy over all memory
      dest->s[i] = src->s[i];
  }
  dest->l = src->l;
}

void
ion_string_copy1(ion_string_t *dest, char *src)
{
  int32_t i, m;
  m = strlen(src)+1;
  if(dest->m < m) {
      dest->m = m;
      dest->s = ion_realloc(dest->s, sizeof(char)*dest->m, __func__, "dest->s");
  }
  for(i=0;i<m;i++) { // copy over all memory
      dest->s[i] = src[i];
  }
  dest->l = m-1;
}

ion_string_t *
ion_string_clone(ion_string_t *str)
{
  int32_t i;
  ion_string_t *ret = NULL;
  
  ret = ion_string_init(str->m);
  for(i=0;i<str->m;i++) { // copy over all memory
      ret->s[i] = str->s[i];
  }
  ret->l = str->l;

  return ret;
}

void
ion_string_lsprintf(ion_string_t *dest, int32_t l, const char *format, ...) 
{
  va_list ap;
  int32_t length;
  if(l < 0) ion_error(__func__, NULL, Exit, OutOfRange);
  va_start(ap, format);
  length = vsnprintf(dest->s + l, dest->m - l, format, ap);
  if(length < 0) ion_error(__func__, NULL, Exit, OutOfRange);
  va_end(ap);
  if(dest->m - l - 1 < length) {
      dest->m = length + l + 2;
      ion_roundup32(dest->m);
      dest->s = ion_realloc(dest->s, sizeof(char)*dest->m, __func__, "dest->s");
      va_start(ap, format);
      length = vsnprintf(dest->s + l, dest->m - l, format, ap);
      va_end(ap);
      if(length < 0) ion_error(__func__, NULL, Exit, OutOfRange);
  }
  dest->l += length;
}

void
ion_string_reverse(ion_string_t *str)
{
  int i;
  for(i = 0; i < (str->l >> 1); ++i) {
      char tmp = str->s[str->l-1-i];
      str->s[str->l-1-i] = str->s[i]; str->s[i] = tmp;
  }
}

void
ion_string_reverse_compliment(ion_string_t *str, int32_t is_int)
{
  int i;

  if(1 == is_int) { // bases are integer values
      for(i = 0; i < (str->l >> 1); ++i) {
          char tmp = str->s[str->l-1-i];
          str->s[str->l-1-i] = (4 <= str->s[i]) ? str->s[i] : 3 - str->s[i];
          str->s[i] = (4 <= tmp) ? tmp : 3 - tmp;
      }
      if(1 == (str->l & 1)) { // mod 2
          str->s[i] = (4 <= str->s[i]) ? str->s[i] : 3 - str->s[i];
      }
  }
  else { // bases are ASCII values
      for(i = 0; i < (str->l >> 1); ++i) {
          char tmp = str->s[str->l-1-i];
          str->s[str->l-1-i] = ion_nt_char_to_rc_char[(int)str->s[i]]; 
          str->s[i] = ion_nt_char_to_rc_char[(int)tmp];
      }
      if(1 == (str->l & 1)) { // mod 2
          str->s[i] = ion_nt_char_to_rc_char[(int)str->s[i]];
      }
  }
}
