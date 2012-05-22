/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ION_ALLOC_H
#define ION_ALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      wrapper function for malloc
      @param  size           the size of the memory block, in bytes
      @param  fn_name        the calling function name 
      @param  variable_name  the variable name to be assigned this memory in the calling function
      @return                upon success, a pointer to the memory block allocated by the function; a null pointer otherwise.
      */
  void *
      ion_malloc(size_t size, const char *fn_name, const char *variable_name);

    /*! 
      wrapper function for realloc
      @param  ptr            the pointer to a memory block previously allocated
      @param  size           the size of the memory block, in bytes
      @param  fn_name        the calling function name 
      @param  variable_name  the variable name to be assigned this memory in the calling function
      @return 		 upon success, a pointer to the memory block allocated by the function; a null pointer otherwise.
      @details      	 the ptr must be a memory block previously allocated with malloc, calloc, or realloc to be reallocated; if the ptr is NULL, a new block of memory will be allocated. 
      */
    void *
      ion_realloc(void *ptr, size_t size, const char *fn_name, const char *variable_name);

    /*! 
      wrapper function for calloc
      @param  num            the number of elements to be allocated
      @param  size           the size of the memory block, in bytes
      @param  fn_name        the calling function name 
      @param  variable_name  the variable name to be assigned this memory in the calling function
      @return                upon success, a pointer to the memory block allocated by the function; a null pointer otherwise.
      */
    void *
      ion_calloc(size_t num, size_t size, const char *fn_name, const char *variable_name);

#ifdef __cplusplus
}
#endif

#endif // ION_ALLOC_H
