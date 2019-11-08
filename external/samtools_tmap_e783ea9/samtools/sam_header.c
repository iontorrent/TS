#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdarg.h>
#include "khash.h"
#include "ksort.h"
#include "bam.h"

#include "sam_header.h"

typedef struct {
    char tag[2];
} sam_header_tag_t;
#define __tageq(_tag_a, _tag_b) ((_tag_a).tag[0] == (_tag_b).tag[0] && (_tag_a).tag[1] == (_tag_b).tag[1])
#define __tag_hash_func(_tag) ((khint_t)(_tag).tag[1] << 8 | (khint_t)(_tag).tag[0])

KHASH_INIT(str, sam_header_tag_t, char *, 1, __tag_hash_func, __tageq);
KHASH_INIT(records, sam_header_tag_t, sam_header_records_t*, 1, __tag_hash_func, __tageq);

const char *SAM_HEADER_TYPE_TAGS[] = {"HD", "SQ", "RG", "PG", "CO", NULL};
const int32_t SAM_HEADER_TYPE_TAGS_MAX[] = {1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, -1};

const char *SAM_HEADER_HD_TAGS_REQ[] = {"VN", NULL};
const char *SAM_HEADER_HD_TAGS_OPT[] = {"SO", "GO", NULL};
const char *SAM_HEADER_HD_TAGS_UNQ[] = {NULL};

const char *SAM_HEADER_SQ_TAGS_REQ[] = {"SN", "LN", NULL};
const char *SAM_HEADER_SQ_TAGS_OPT[] = {"AS", "M5", "SP", "UR", NULL};
const char *SAM_HEADER_SQ_TAGS_UNQ[] = {"SN", NULL};

const char *SAM_HEADER_RG_TAGS_REQ[] = {"ID", NULL};
const char *SAM_HEADER_RG_TAGS_OPT[] = {"CN", "DS", "DT", "FO", "KS", "LB", "PG", "PI", "PL", "PU", "SM", "sk", "zt", "yt",  NULL};
const char *SAM_HEADER_RG_TAGS_UNQ[] = {"ID", NULL};

const char *SAM_HEADER_PG_TAGS_REQ[] = {"ID", NULL};
const char *SAM_HEADER_PG_TAGS_OPT[] = {"PN", "CL", "PP", "VN", NULL};
const char *SAM_HEADER_PG_TAGS_UNQ[] = {"ID", NULL};

const char *SAM_HEADER_CO_TAGS_REQ[] = {"  ", NULL};
const char *SAM_HEADER_CO_TAGS_OPT[] = {NULL};
const char *SAM_HEADER_CO_TAGS_UNQ[] = {NULL};


const char **SAM_HEADER_TAGS_REQ[] = {SAM_HEADER_HD_TAGS_REQ, SAM_HEADER_SQ_TAGS_REQ, SAM_HEADER_RG_TAGS_REQ, SAM_HEADER_PG_TAGS_REQ, SAM_HEADER_CO_TAGS_REQ, NULL}; 
const char **SAM_HEADER_TAGS_OPT[] = {SAM_HEADER_HD_TAGS_OPT, SAM_HEADER_SQ_TAGS_OPT, SAM_HEADER_RG_TAGS_OPT, SAM_HEADER_PG_TAGS_OPT, SAM_HEADER_CO_TAGS_OPT, NULL}; 
const char **SAM_HEADER_TAGS_UNQ[] = {SAM_HEADER_HD_TAGS_UNQ, SAM_HEADER_SQ_TAGS_UNQ, SAM_HEADER_RG_TAGS_UNQ, SAM_HEADER_PG_TAGS_UNQ, SAM_HEADER_CO_TAGS_UNQ, NULL}; 

static inline int32_t
tagcmp(const char tag_a[2], const char tag_b[2])
{
  if(tag_a[0] < tag_b[0] || (tag_a[0] == tag_b[0] && tag_a[1] < tag_b[1])) return -1;
  else if(tag_a[0] == tag_b[0] && tag_a[1] == tag_b[1]) return 0;
  else return 1;
}

static int32_t
sam_header_tag2int(const char *tag, const char **tags)
{
  int32_t i = 0;
  while(NULL != (*tags)) {
      if(0 == tagcmp((*tags), tag)) return i;
      tags++;
      i++;
  }
  return -1;
}

static void 
debug(const char *format, ...)
{
  va_list ap;
  va_start(ap, format);
  vfprintf(stderr, format, ap);
  va_end(ap);
}

int
sam_header_record_lt(sam_header_record_t *r1, sam_header_record_t *r2)
{
  char **tags_unq = NULL;
  if(r1->type != r2->type || 0 != tagcmp(r1->tag, r2->tag)) {
      debug("[%s] found inconsistency in the sam header [%c%c,%c%c]\n", __func__, 
            r1->tag[0], r1->tag[1],
            r2->tag[0], r2->tag[1]);
      exit(1);
  }
  tags_unq = (char**)SAM_HEADER_TAGS_UNQ[r1->type];
  while(NULL != tags_unq && NULL != (*tags_unq)) {
      char *v1 = sam_header_record_get(r1, (*tags_unq));
      char *v2 = sam_header_record_get(r2, (*tags_unq));
      if(NULL != v1 && NULL != v2) {
          int c = strcmp(v1, v2);
          if(c < 0) return 1;
          else if(0 < c) return 0;
      }
      else if(NULL != v1 && NULL == v2) {
          return 1;
      }
      else if(NULL == v1 && NULL != v2) {
          return 0;
      }

      tags_unq++;

  }
  return 0;
}

#define __sam_header_record_lt(r1, r2) (1 == sam_header_record_lt(&r1, &r2))

KSORT_INIT(sam_header_record, sam_header_record_t, __sam_header_record_lt)


// Mimics the behaviour of getline, except it returns pointer to the next chunk of the text
//  or NULL if everything has been read. The lineptr should be freed by the caller. The
//  newline character is stripped.
static const char *
nextline(char **lineptr, size_t *n, const char *text)
{
  int len;
  const char *to = text;

  if(0 == (*to)) return NULL;

  while(0 != (*to) && '\n' != (*to) && '\r' != (*to)) {
      to++;
  }
  len = to - text + 1;

  if ( *to )
    {
      // Advance the pointer for the next call
      if ( *to=='\n' ) to++;
      else if ( *to=='\r' && *(to+1)=='\n' ) to+=2;
    }
  if ( !len )
    return to;

  if ( !*lineptr )
    {
      *lineptr = malloc(len);
      *n = len;
    }
  else if ( *n<len )
    {
      *lineptr = realloc(*lineptr, len);
      *n = len;
    }
  if ( !*lineptr ) {
      debug("[%s] Insufficient memory!\n", __func__);
      return 0;
  }

  memcpy(*lineptr,text,len);
  (*lineptr)[len-1] = 0;

  return to;
}

sam_header_record_t*
sam_header_record_init(const char tag[2])
{
  sam_header_record_t *r = NULL;

  r = calloc(1, sizeof(sam_header_record_t));
  if(NULL == r) {
      return NULL;
  }

  r->type = sam_header_tag2int(tag, SAM_HEADER_TYPE_TAGS);
  r->tag[0] = tag[0]; r->tag[1] = tag[1];
  r->hash = (void*)kh_init(str);

  return r;
}

void
sam_header_record_destroy(sam_header_record_t *r)
{
  khash_t(str) *hash = (khash_t(str)*)r->hash;
  khiter_t k;
  for(k = kh_begin(hash); k != kh_end(hash); ++k) {
      if (kh_exist(hash, k)) {
          free(kh_value(hash, k));
          kh_value(hash, k) = NULL;
      }
  }
  kh_destroy(str, hash);
  free(r);
}

int32_t
sam_header_record_add(sam_header_record_t *r, const char *key, const char *value)
{
  khash_t(str) *hash = (khash_t(str)*)r->hash;
  khiter_t k;
  char *v = NULL;
  int ret;
  sam_header_tag_t key_str; key_str.tag[0] = key[0]; key_str.tag[1] = key[1];
  k = kh_get(str, hash, key_str);
  if(k != kh_end(hash)) {
      debug("[%s] The key %c%c is not unique.\n", __func__, key[0], key[1]);
      return 0;
  }
  k = kh_put(str, hash, key_str, &ret);
  v = malloc(sizeof(char) * (1 + strlen(value)));
  if(NULL == v) return 0;
  strcpy(v, value);
  kh_value(hash, k) = v;
  return 1;
}

int32_t
sam_header_record_set(sam_header_record_t *r, const char *tag, const char *value)
{
  khash_t(str) *hash = (khash_t(str)*)r->hash;
  khiter_t k;
  char *v = NULL;
  int ret;
  sam_header_tag_t key; key.tag[0] = tag[0]; key.tag[1] = tag[1];
  k = kh_put(str, hash, key, &ret);
  v = malloc(sizeof(char) * (1 + strlen(value)));
  if(NULL == v) return 0;
  strcpy(v, value);
  kh_value(hash, k) = v;
  return 1;
}

char*
sam_header_record_get(const sam_header_record_t *r, const char *tag)
{
  khash_t(str) *hash = (khash_t(str)*)r->hash;
  khiter_t k;
  sam_header_tag_t key; key.tag[0] = tag[0]; key.tag[1] = tag[1];
  k = kh_get(str, hash, key);
  return k == kh_end(hash) ? NULL : kh_val(hash, k);
}

int32_t
sam_header_record_remove(const sam_header_record_t *r, const char *tag)
{
  khash_t(str) *hash = (khash_t(str)*)r->hash;
  khiter_t k;
  sam_header_tag_t key; key.tag[0] = tag[0]; key.tag[1] = tag[1];
  k = kh_get(str, hash, key);
  if(kh_end(hash) == k) {
      return 0;
  }
  else {
      // destroy
      free(kh_value(hash, k));
      kh_value(hash, k) = NULL;
      // delete the key from the hash
      kh_del(str, hash, k);
      return 1;
  }
}

int32_t
sam_header_record_check(const sam_header_record_t *record)
{
  char **tags_req = NULL;

  if(NULL == record) return 0;
  if(-1 == record->type) return 1;
  tags_req = (char**)SAM_HEADER_TAGS_REQ[record->type];
  if (NULL == tags_req) return 1;
  while(NULL != (*tags_req)) { // go through the required tags
      if(NULL == sam_header_record_get(record, (*tags_req))) {
          debug("[%s] required tag [%s] missing from record type [%c%c]\n", __func__, (*tags_req), record->tag[0], record->tag[1]);
          return 0;
      }
      tags_req++;
  }
  return 1;
}

sam_header_record_t*
sam_header_record_parse(const char *buf)
{
  sam_header_record_t *r = NULL;
  const char *from = NULL, *to = NULL;
  char tag[3]="\0";
  char *value = NULL;
  size_t value_mem = 0;

  if(NULL == buf || 0 == (*buf)) return NULL;

  from = buf;

  if('@' != (*from)) { 
      debug("[%s] expected '@', got [%s]\n", __func__, buf);
  }
  to = ++from;

  // skip over first tab
  while(0 != (*to) && '\t' != (*to)) {
      to++;
  }
  if(2 != (to - from)) {
      debug("[%s] expected '@XY', got [%s]\nHint: The header tags must be tab-separated.\n", __func__, buf);
      return 0;
  }

  tag[0] = from[0]; tag[1] = from[1]; tag[2] = '\0';
  r = sam_header_record_init(tag);
  if(NULL == r) {
      //debug("[%s] non-standard header tag[%s]\n", tag);
      return 0;
  }

  // skip over the current tab(s)
  from = to;
  while(0 != (*to) && '\t' == (*to)) {
      to++;
  }
  if(1 != (to - from)) {
      debug("[%s] multiple tabs on line [%s] (%d)\n", __func__, buf, (int)(to-from));
      return 0;
  }
  from = to;
  while(0 != (*from)) {
      size_t len = 0;
      // skip to the next tab
      while(0 != (*to) && '\t' != (*to)) {
          to++;
      }

      if(SAM_HEADER_TYPE_CO == r->type) { // CO (comment tag)
          // CO is a special case, it can contain anything, including tabs
          if(0 != *to) { // more characters left
              to++; 
              continue;
          }
          tag[0] = tag[1] = ' '; tag[2] = '\0';
          len = (to - from);
          if(value_mem < len+1) {
              value = realloc(value, sizeof(char) * (len+1));
              value_mem = len + 1;
          }
          strncpy(value, from, len);
      }
      else {
          tag[0] = from[0]; tag[1] = from[1]; tag[2] = '\0';
          len = (to - from - 3);
          if(value_mem < len+1) {
              value = realloc(value, sizeof(char) * (len+1));
              value_mem = len + 1;
          }
          strncpy(value, from+3, len);
      }
      value[len] = '\0';

      // try to add
      if(0 == sam_header_record_add(r, tag, value)) {
          debug("[%s] The tag '%s' present (at least) twice on line [%s]\n", __func__, tag, buf);
      }
      //debug("[%s] added tag '%s' from '%s' record type with value '%s'\n", __func__, tag, r->tag, value);

      // skip over the current tab(s)
      from = to;
      while(0 != (*to) && '\t' == (*to)) {
          to++;
      }
      if(0 != (*to) && 1 != (to - from)) {
          debug("[%s] multiple tabs on line [%s] (%d)\n", __func__, buf, (int)(to-from));
          sam_header_record_destroy(r);
          r = NULL;
          return 0;
      }
      from = to;
  }
  free(value);

  // check required values etc.
  if(0 == sam_header_record_check(r)) {
      sam_header_record_destroy(r);
      r = NULL;
      return 0;
  }

  return r;
}

sam_header_record_t*
sam_header_record_clone(const sam_header_record_t *src)
{
  sam_header_record_t *dst = NULL;
  khash_t(str) *hash = NULL;
  khiter_t k;

  if(NULL == src) return NULL;
  hash = (khash_t(str)*)src->hash;
  dst = sam_header_record_init(src->tag);
  for(k = kh_begin(hash); k != kh_end(hash); ++k) {
      if (kh_exist(hash, k)) {
          sam_header_record_add(dst, kh_key(hash, k).tag, kh_value(hash, k));
      }
  }

  return dst;
}

sam_header_records_t*
sam_header_records_init(const char tag[2])
{
  sam_header_records_t *records = NULL;
  records = calloc(1, sizeof(sam_header_records_t));
  records->type = sam_header_tag2int(tag, SAM_HEADER_TYPE_TAGS);
  records->tag[0] = tag[0]; records->tag[1] = tag[1];
  return records;
}

void
sam_header_records_destroy(sam_header_records_t *records)
{
  int32_t i;
  if(NULL == records) return;
  for(i=0;i<records->n;i++) {
      sam_header_record_destroy(records->records[i]);
  }
  free(records->records);
  free(records);
}

// NB: shallow copy
int32_t
sam_header_records_add(sam_header_records_t *records, sam_header_record_t *record)
{
  if(records->type != record->type) {
      return 0;
  }
  records->n++;
  records->records = realloc(records->records, sizeof(sam_header_record_t*) * records->n);
  records->records[records->n-1] = record;
  return 1;
}

int32_t
sam_header_records_check(sam_header_records_t *records)
{
  int32_t i;
  char **tags_unq = NULL;
  sam_header_record_t *list = NULL;

  if(SAM_HEADER_TYPE_NONE == records->type) return 1;

  // check record
  if(SAM_HEADER_TYPE_TAGS_MAX[records->type] < records->n) { // too many
      debug("[%s] found too many lines for tag [%c%c] (%d < %d)\n", 
            __func__, records->tag[0], records->tag[1],
            SAM_HEADER_TYPE_TAGS_MAX[records->type], records->n); 
      return 0;
  }

  // store a list of records
  list = calloc(records->n, sizeof(sam_header_record_t));

  // check unique tags
  for(i=0;i<records->n;i++) {
      sam_header_record_t *r1 = records->records[i];

      // check individual records
      if(0 == sam_header_record_check(r1)) {
          return 0;
      }

      // save for the list
      list[i] = (*r1);
  }

  // sort the list...
  ks_introsort(sam_header_record, records->n, list);

  // check for inconsistencies
  for(i=0;i<records->n-1;i++) {
      sam_header_record_t *r1 = records->records[i];
      sam_header_record_t *r2 = records->records[i+1];
      if(r1->type != r2->type || 0 != tagcmp(r1->tag, r2->tag)) {
          debug("[%s] found inconsistency in the sam header [%d,%d,%c%c,%c%c]\n", __func__, i, i+1, 
                r1->tag[0], r1->tag[1],
                r2->tag[0], r2->tag[1]);
          return 0;
      }
      tags_unq = (char**)SAM_HEADER_TAGS_UNQ[records->type];
      while(NULL != tags_unq && NULL != (*tags_unq)) {
          char *v1 = sam_header_record_get(r1, (*tags_unq));
          char *v2 = sam_header_record_get(r2, (*tags_unq));
          if(NULL != v1 && NULL != v2 && 0 == strcmp(v1, v2)) {
              debug("[%s] value for %c%c.%s was not unique\n", __func__, records->tag[0], records->tag[1], (*tags_unq));
              return 0;
          }

          tags_unq++;
      }
  }

  free(list);
  list = NULL;

  return 1;
}

sam_header_t*
sam_header_init()
{
  sam_header_t *h = NULL;
  char **tags = (char**)SAM_HEADER_TYPE_TAGS;
  khiter_t k;
  int ret;

  khash_t(records) *hash = NULL;
  h = calloc(1, sizeof(sam_header_t));
  hash = kh_init(records);
  h->hash = (void*)hash;

  while(NULL != (*tags)) {
      sam_header_records_t *r = NULL;
      sam_header_tag_t key; key.tag[0] = (*tags)[0]; key.tag[1] = (*tags)[1];
      k = kh_put(records, hash, key, &ret);
      r = sam_header_records_init((*tags));
      kh_value(hash, k) = r;
      tags++;
  }

  return h;
}

void
sam_header_destroy(sam_header_t *h)
{
  khash_t(records) *hash;
  khiter_t k;
  if(NULL == h) return;
  hash = (khash_t(records)*)h->hash;
  if(NULL == hash) return;
  for(k = kh_begin(hash); k != kh_end(hash); ++k) {
      if (kh_exist(hash, k)) {
          sam_header_records_destroy(kh_value(hash, k));
          kh_value(hash, k) = NULL;
      }
  }
  kh_destroy(records, hash);
  free(h);
}

sam_header_records_t*
sam_header_get_records(const sam_header_t *h, const char type_tag[2])
{
  khash_t(records) *hash = NULL;
  if (NULL == h) return NULL;
  hash = (khash_t(records)*)h->hash;
  khiter_t k;
  sam_header_tag_t key; key.tag[0] = type_tag[0]; key.tag[1] = type_tag[1];
  k = kh_get(records, hash, key);
  return k == kh_end(hash) ? NULL : kh_val(hash, k);
}

// ZZ: This function differs from destroy_records in that it takes care to maintain the correct hash structure so the
// structure sam_header_t *h can still work. 
void
sam_header_remove_records(const sam_header_t *h, const char type_tag[2])
{
  khash_t(records) *hash = NULL;
  if (NULL == h) return;
  hash = (khash_t(records)*)h->hash;
  khiter_t k;
  sam_header_tag_t key; key.tag[0] = type_tag[0]; key.tag[1] = type_tag[1];
  k = kh_get(records, hash, key);
  if (k == kh_end(hash)) return;
  sam_header_records_destroy(kh_val(hash, k));
  kh_val(hash, k) = NULL;
  kh_del(records, hash, k);
}
  

sam_header_record_t**
sam_header_get_record(const sam_header_t *h, char type_tag[2], char tag[2], char *value, int32_t *n)
{
  int32_t i;
  sam_header_records_t *records = NULL;
  sam_header_record_t **list = NULL;
  
  if (NULL == h) return NULL;

  records = sam_header_get_records(h, type_tag);
  if(NULL == records) return NULL;

  for(i=0;i<records->n;i++) { // go through each record
      char *v = NULL;
      v = sam_header_record_get(records->records[i], tag);
      if(NULL == value) continue;
      if(0 == strcmp(value, v)) { // found
          (*n)++;
          list = realloc(list, sizeof(sam_header_record_t*) * (*n));
          list[(*n)-1] = records->records[i];
      }
  }
  return list;
}

// NB: shallow copy
int32_t
sam_header_add_record(sam_header_t *h, sam_header_record_t *record)
{
  if (NULL == h) return 0;
  khash_t(records) *hash = (khash_t(records)*)h->hash;
  khiter_t k;
  sam_header_records_t *records = NULL;
  int ret;
  sam_header_tag_t key; key.tag[0] = record->tag[0]; key.tag[1] = record->tag[1];
  k = kh_get(records, hash, key);
  if(k != kh_end(hash)) { // already exits
      records = kh_val(hash, k); 
      // check if we should multiple are allowed etc...
      if(SAM_HEADER_TYPE_NONE != records->type) {
          if(SAM_HEADER_TYPE_TAGS_MAX[records->type] <= records->n) {
              debug("[%s] too many lines for tag [%c%c] (%d <= %d)\n", 
                    __func__, records->tag[0], records->tag[1], SAM_HEADER_TYPE_TAGS_MAX[records->type], records->n); 
              return 0;
          }
      }
  }
  else { // new
      records = sam_header_records_init(record->tag); 
      k = kh_put(records, hash, key, &ret);
      kh_value(hash, k) = records;
  }
  sam_header_records_add(records, record);
  return 1;
}

sam_header_t*
sam_header_clone(const sam_header_t *h)
{
  sam_header_t *out = NULL;
  khash_t(records) *hash = NULL;
  khiter_t k;
  int32_t i;
  if(NULL == h) return NULL;

  out = sam_header_init();
  hash = (khash_t(records)*)h->hash;
  for(k = kh_begin(hash); k != kh_end(hash); ++k) {
      if (kh_exist(hash, k)) {
          sam_header_records_t *records = kh_value(hash, k);
          for(i=0;i<records->n;i++) {
              sam_header_record_t *record = records->records[i];
              if(0 == sam_header_add_record(out, sam_header_record_clone(record))) {
                  debug("[%s] error adding a record, trying to continue...\n", __func__);
              }
          }
      }
  }

  return out;
}

// NB: shallow copy
char **
sam_header_list(const sam_header_t *h, const char type_tag[2], const char key_tag[2], int *n)
{
  char **list = NULL;
  int32_t i;
  sam_header_records_t *records = NULL;

  (*n) = 0;

  records = sam_header_get_records(h, type_tag);
  if(NULL == records) return NULL; 

  for(i=0;i<records->n;i++) {
      sam_header_record_t *record = records->records[i];
      char *value = sam_header_record_get(record, key_tag);
      if(NULL != value) {
          (*n)++;
          list = realloc(list, sizeof(char*) * (*n));
          list[(*n)-1] = value; 
      }
  }

  return list;
}

// NB: shallow copy
void*
sam_header_table(const sam_header_t *h, char type_tag[2], char key_tag[2], char value_tag[2])
{
  khash_t(str) *tbl = kh_init(str);
  khiter_t k;
  int32_t i;
  sam_header_records_t *records = NULL;
  int ret;

  records = sam_header_get_records(h, type_tag);
  if(NULL == records) return tbl; 

  for(i=0;i<records->n;i++) {
      sam_header_record_t *record = records->records[i];
      const char *key = sam_header_record_get(record, key_tag);
      char *value = sam_header_record_get(record, value_tag);
      if(NULL != key && NULL != value) {
          sam_header_tag_t key_str; key_str.tag[0] = key[0]; key_str.tag[1] = key[1];

          k = kh_get(str, tbl, key_str);
          if(k != kh_end(tbl)) {
              debug("[%s] Found multiple values for [%s] -> [%s]!\n", __func__, key, kh_value(tbl, k));
          }
          else {
              k = kh_put(str, tbl, key_str, &ret);
              kh_value(tbl, k) = value;
          }
      }
  }

  return tbl;
}

const char *
sam_tbl_get(void *h, const char *key)
{
  khash_t(str) *tbl = (khash_t(str)*)h;
  khint_t k;
  sam_header_tag_t key_str; key_str.tag[0] = key[0]; key_str.tag[1] = key[1];
  k = kh_get(str, tbl, key_str);
  return k == kh_end(tbl)? 0 : kh_val(tbl, k);
}

int 
sam_tbl_size(void *h)
{
  khash_t(str) *tbl = (khash_t(str)*)h;
  return h? kh_size(tbl) : 0;
}

void 
sam_tbl_destroy(void *h)
{
  khash_t(str) *tbl = (khash_t(str)*)h;
  kh_destroy(str, tbl);
}

// NB: deep copy
// NB: not implemented
int32_t
sam_header_merge_into(sam_header_t *dst, const sam_header_t *src)
{
  khash_t(records) *hash_dst, *hash_src;
  khiter_t i, j;
  if(NULL == src) return 0;

  hash_dst = (khash_t(records)*)dst->hash;
  hash_src = (khash_t(records)*)src->hash;

  for(i = kh_begin(hash_dst); i != kh_end(hash_dst); i++) { 
      //sam_header_records_t *records_dst = kh_value(hash_dst, i);
      for(j = kh_begin(hash_src); j != kh_end(hash_src); j++) {
          //sam_header_records_t *records_src = kh_value(hash_src, j);
          // TODO
      }
  }
  return 1;
}

// NB: not implemented
sam_header_t*
sam_header_merge(int n, const sam_header_t **headers)
{
  int32_t i;
  sam_header_t *out = NULL;

  if(n < 2) return NULL;

  // clone the first one
  out = sam_header_clone(headers[0]);

  // merge the rest
  for(i=0;i<n;i++) {
      const sam_header_t *cur = headers[i];

      sam_header_merge_into(out, cur);
  }

  return NULL;
}

int32_t
sam_header_check(sam_header_t *h)
{
  khash_t(records) *hash = NULL;
  khiter_t k;
  if(NULL == h) return 0;

  hash = (khash_t(records)*)h->hash;
  for(k = kh_begin(hash); k != kh_end(hash); ++k) { // go through the record types
      if (kh_exist(hash, k)) { // exist
          sam_header_records_t *records = kh_value(hash, k);
          // check records
          if(0 == sam_header_records_check(records)) {
              return 0;
          }
      }
  }

  return 1;
}

sam_header_t*
sam_header_parse2(const char *text)
{
  sam_header_t *h = NULL;
  const char *p;
  char *buf = NULL;
  size_t nbuf = 0;

  h = sam_header_init();

  if(NULL == text) return h;

  p = text;
  while(NULL != (p = nextline(&buf, &nbuf, p))) {
      sam_header_record_t *record = NULL;
      record = sam_header_record_parse(buf);
      if(NULL != record) {
          // TODO: test with a large # of header records...
          if(0 == sam_header_add_record(h, record)) {
              debug("[%s] error adding a record, trying to continue...\n", __func__);
          }
      }
  }
  free(buf);
  buf = NULL;

  // check the consistency of records
  if(0 == sam_header_check(h)) {
      sam_header_destroy(h);
      return NULL;
  }

  return h;
}

static void
sam_header_write_add(char **text, size_t *text_len, size_t *text_mem, const char *value, int32_t value_len)
{
  if(value_len <= 0) return;
  // more memory
  while((*text_mem) < value_len + (*text_len)) {
      (*text_mem) = ((*text_mem) < 32) ? 32 : ((*text_mem) << 1);
      (*text) = realloc((*text), sizeof(char) * (*text_mem));
  }
  // copy over
  memcpy((*text) + (*text_len), value, value_len);
  (*text_len) += value_len;
}

static void
sam_header_record_write(const sam_header_record_t *record, char **text, size_t *text_len, size_t *text_mem)
{
  khash_t(str) *hash = NULL;
  khiter_t k;
  char **tags = NULL;

  sam_header_write_add(text, text_len, text_mem, "@", 1);
  sam_header_write_add(text, text_len, text_mem, SAM_HEADER_TYPE_TAGS[record->type], 2);

  if(SAM_HEADER_TYPE_NONE != record->type) { // standard tag
      // required tags
      tags = (char**)SAM_HEADER_TAGS_REQ[record->type];
      while(tags != NULL && NULL != (*tags)) {
          const char *value = sam_header_record_get(record, (*tags));
          if(NULL != value) {
              sam_header_write_add(text, text_len, text_mem, "\t", 1);
              if (**tags!=' ') {
		sam_header_write_add(text, text_len, text_mem, (*tags), 2);
              	sam_header_write_add(text, text_len, text_mem, ":", 1);
	      }
              sam_header_write_add(text, text_len, text_mem, value, strlen(value));
          }
          tags++;
      }
      tags = NULL;

      // optional tags
      tags = (char**)SAM_HEADER_TAGS_OPT[record->type];
      while(NULL != tags && NULL != (*tags)) {
          const char *value = sam_header_record_get(record, (*tags));
          if(NULL != value) {
              sam_header_write_add(text, text_len, text_mem, "\t", 1);
              sam_header_write_add(text, text_len, text_mem, (*tags), 2);
              sam_header_write_add(text, text_len, text_mem, ":", 1);
              sam_header_write_add(text, text_len, text_mem, value, strlen(value));
          }
          tags++;
      }
      tags = NULL;
  }
  else { // non-standard tag
      // individual tags, in any order
      hash = (khash_t(str)*)record->hash;
      for(k = kh_begin(hash); k != kh_end(hash); ++k) {
          if (kh_exist(hash, k)) {
              const char *key = NULL, *value = NULL;
              key = kh_key(hash, k).tag;
              value = kh_value(hash, k);
              sam_header_write_add(text, text_len, text_mem, "\t", 1);
              sam_header_write_add(text, text_len, text_mem, key, 2);
              sam_header_write_add(text, text_len, text_mem, ":", 1);
              sam_header_write_add(text, text_len, text_mem, value, strlen(value));
          }
      }
  }

  sam_header_write_add(text, text_len, text_mem, "\n", 1);
  (*text)[(*text_len)]='\0';
}

static void
sam_header_records_write(const sam_header_records_t *records, char **text, size_t *text_len, size_t *text_mem)
{
  int32_t i;

  // write in the same order as added
  for(i=0;i<records->n;i++) {
      sam_header_record_write(records->records[i], text, text_len, text_mem);
  }
}

char*
sam_header_write(const sam_header_t *h)
{
  char *text = NULL;
  size_t len = 0, mem = 0;
  khash_t(records) *hash = NULL;
  khiter_t k;
  char **tags = NULL;

  // standard tags, in order defined by the SAM spec.
  tags = (char**)SAM_HEADER_TYPE_TAGS;
  while(NULL != (*tags)) {
      sam_header_records_t *records = NULL;
      records = sam_header_get_records(h, (*tags));
      if(NULL != records) sam_header_records_write(records, &text, &len, &mem);
      tags++;
  }
  tags = NULL;

  // non-standard tags, in any order
  hash = (khash_t(records)*)h->hash;
  for(k = kh_begin(hash); k != kh_end(hash); ++k) {
      if (kh_exist(hash, k)) {
          sam_header_records_t *records = NULL;
          records = kh_value(hash, k);
          if(NULL == records) continue;
          if(SAM_HEADER_TYPE_NONE != records->type) continue; // ignore standard tags
          sam_header_records_write(records, &text, &len, &mem);
      }
  }

  // resize
  if(len+1 < mem) {
      text = realloc(text, sizeof(char) * (len + 1));
      mem = len + 1;
  }

  // EOL
  text[len] = '\0';

  return text;
}

extern void bam_init_header_hash(bam_header_t *header);

bam_header_t*
sam_header_to_bam_header(bam_header_t *bh)
{
  sam_header_t *sh = NULL;

  // Grab the SAM Header
  sh = bh->header;
  if(NULL == sh) {
      debug("[%s] error converting SAM header structure to the BAM header structure.\n", __func__);
      return bh;
  }

  // Destroy the BAM Header
  bh->header = NULL;
  bam_header_destroy(bh); // destroy the header
  bh = bam_header_init();

  // Copy previous SAM Header
  bh->header = sh; 

  // SAM Header to text
  bh->text = sam_header_write(sh);
  bh->l_text = strlen(bh->text);

  // Fill in the BAM Header based on the text
  sam_header_parse(bh);

  return bh;
}
