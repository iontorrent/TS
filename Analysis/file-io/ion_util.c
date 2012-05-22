#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "ion_error.h"
#include "ion_util.h"

// Input: ASCII character
// Output: 2-bit DNA value
uint8_t ion_nt_char_to_int[256] = {
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};

// Input: ASCII character
// Output: ASCII reverse complimented DNA value
uint8_t ion_nt_char_to_rc_char[256] = {
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'T', 'N', 'G',  'N', 'N', 'N', 'C',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'A', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'T', 'N', 'G',  'N', 'N', 'N', 'C',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'A', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N'
};

/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

static int 
base36to10(char *str)
{
  int num = 0;
  int val;
  int i = (int)strlen(str);
  while(*str) {
      if (*str >= '0' && *str <= '9')
        val = 26 + *str - '0';
      else
        val = *str - 'a';
      num += val * (int)pow(36.0, i-1);
      i--;
      str++;
  }

  return num;
}

static void
base10to36(char *str, unsigned int num)
{
  // MGD - note: Obviously this is backwards, a '1' should 
  // encode as a '1', even in base-36 mode, yet it encodes 
  // as a 'B', but I keep this broken code to conform to 
  // the original sff format
  int i = 0;
  int digit;
  for(i=0;i<5;i++) {
      digit = num%36;
      if (digit < 26)
        str[4-i] = 'A' + digit;
      else
        str[4-i] = '0' + digit - 26;
      num /= 36;
  }
  str[i] = 0;
}

void 
ion_id_to_xy(const char *id, int *x, int *y)
{
  if (!x || !y)
    return;

  // only want to deal with lower-case string
  char str[32];
  int i = 0;
  while (id[i] && i < 31) {
      str[i] = tolower(id[i]);
      i++;
  }
  str[i] = 0; // we need a null terminator to make this a C string

  // base-36 string to base-10
  int num = base36to10(str);

  // number to XY
  *x = num/4096;
  *y = num%4096;
}

/*
 *	Hash function
 *	From http://www.partow.net/programming/hashfunctions
 */
static unsigned int 
DEKHash(char* str, unsigned int len)
{
  unsigned int hash = len;
  unsigned int i    = 0;

  for(i = 0; i < len; str++, i++)
    {
      hash = ((hash << 5) ^ (hash >> 27)) ^ (*str);
    }
  return hash;
}
/* End Of DEK Hash Function */

//
// Create 5 char unique string representing input char string (runname)
//
void 
ion_run_to_readname(char *id, char *run, int size)
{
  base10to36(id, DEKHash(run, size));
}

void 
ion_xy_to_readname(char *id, int x, int y)
{
  int num = x*4096 + y;
  base10to36(id, num);
}

int32_t
ion_readname_to_rowcol(const char *readname, int32_t *row, int32_t *col)
{
  return ion_readname_to_xy(readname, col, row);
}

int32_t 
ion_readname_legacy(const char *readname)
{
  return ((NULL != strstr(readname, "IONPGM_")) ? 1 : 0);
}

int32_t
ion_readname_to_xy(const char *readname, int32_t *x, int32_t *y)
{
  int32_t i, val, state, len;

  if(1 == ion_readname_legacy(readname)) {
      // Legacy read name format "IONPGM_XXXXX_YYYYY" where
      // YYYYY is ion_id_to_xy encoding of xy position
      len = strlen(readname);
      if(10 <= len) {
          ion_id_to_xy(readname + len - 5, x, y);
          return 1;
      }
      else {
          return 0;
      }
  }
  else {

      /* states:
         0 - skipping over read name (before first colon)
         1 - reading in x value (before second colon)
         2 - reading in y value (after second colon)
         */

      for(i=val=state=0;'\0' != readname[i];i++) {
          if(':' == readname[i]) {
              if(1 == state) {
                  (*y) = val;
              }
              state++;
              val = 0;
          }
          else if('0' <= readname[i] && readname[i] <= '9') {
              val *= 10;
              val += (int32_t)(readname[i] - '0');
          }
      }
      if(2 == state) {
          (*x) = val;
          return 1;
      }
      else {
          return 0;
      }
  }
}

uint32_t
ion_read_padding(FILE *fp, uint32_t n)
{
  char padding[8]="\0";
  n = (n & 7); // (n % 8)
  if(0 != n) {
      n = 8 - n; // number of bytes of padding
      if(NULL != fp) {
          if(n != fread(padding, sizeof(char), n, fp)) {
              ion_error(__func__, "fread", Exit, ReadFileError);
          }
      }
  }
  return n;
}

uint32_t
ion_write_padding(FILE *fp, uint32_t n)
{
  char padding[8]="\0\0\0\0\0\0\0\0";
  n = (n & 7); // (n % 8)
  if(0 != n) {
      n = 8 - n; // number of bytes of padding
      if(NULL != fp) {
          if(n != fwrite(padding, sizeof(char), n, fp)) {
              ion_error(__func__, "fwrite", Exit, WriteFileError);
          }
      }
  }
  return n;
}

int 
ion_bound_values(int32_t *min, int32_t *max, int32_t num)
{
  if(num <= (*min) || num <= (*max)) {
      ion_error(__func__, "Input range was out of bounds", Warn, OutOfRange);
      return -1;
  }
  if((*min) < 0) (*min) = 0;
  if((*max) < 0) (*max) = num-1;
  return 1;
}

int 
ion_parse_range(const char *str, int32_t *start, int32_t *end)
{
  if(2 == sscanf(str, "%d-%d", start, end)) {
      return 1;
  }
  else if(1 == sscanf(str, "%d", start)) {
      (*end) = (*start);
      return 1;
  }
  return -1;
}

int
rn_check_main(int argc, char *argv[])
{
  int32_t i;
  for(i=1;i<argc;i++) {
      char id[32]="\0";
      ion_run_to_readname(id, argv[i], strlen(argv[i]));
      fprintf(stderr, "%s -> %s\n", argv[i], id);
  }
  return 0;
}
