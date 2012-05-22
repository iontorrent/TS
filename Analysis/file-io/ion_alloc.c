#include <stdlib.h>

#include "ion_error.h"
#include "ion_alloc.h"

void *
ion_malloc(size_t size, const char *fn_name, const char *variable_name)
{
  void *ptr = malloc(size);
  if(NULL == ptr) {
      ion_error(fn_name, variable_name, Exit, MallocMemory);
  }
  return ptr;
}

void *
ion_realloc(void *ptr, size_t size, const char *fn_name, const char *variable_name)
{
  ptr = realloc(ptr, size);
  if(NULL == ptr) {
      ion_error(fn_name, variable_name, Exit, ReallocMemory);
  }
  return ptr;
}

void *
ion_calloc(size_t num, size_t size, const char *fn_name, const char *variable_name)
{
  void *ptr = calloc(num, size);
  if(NULL == ptr) {
      ion_error(fn_name, variable_name, Exit, MallocMemory);
  }
  return ptr;
}
