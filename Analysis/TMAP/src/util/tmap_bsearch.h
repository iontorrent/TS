/* Copyright (C) 2020 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef __tmap_bsearch_h__
#define __tmap_bsearch_h__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" 
{
#endif

// binary search for sorted arrays. Handles omitted keys
// passed in 'lt' function should compare the key pointed to by the first argument to the data element pointed to by the second argument, in this order
// returns the pointer to the first key that is not lower than the passed one.
// for multiple occurences returns pointer to the lowest occurence
// performance is always ln(N), no worst/best case difference (each search on the array of given size takes same number of steps)
const void* tmap_binary_search (const void* key, const void* data, size_t num_elem, size_t elem_size, int (*lt) (const void* key, const void* elem));

// 'keyless' variant of the binary search
// the passed in 'above' function should return 1 when the data element pointed by the argument is above the desired position, 0 otherwise
const void* tmap_binary_search_nokey (const void* data, size_t num_elem, size_t elem_size, int (*above) (const void* elem));

// 32-bit integer comparison
// 32-bit integer 'less then' comparison function, returns 1 if k<e, 0 otherwise
int lt_int (const void* k, const void* e);
int lt_uint (const void* k, const void* e);
int lt_int32 (const void* k, const void* e);
int lt_uint32 (const void* k, const void* e);
int lt_int64 (const void* k, const void* e);
int lt_uint64 (const void* k, const void* e);
int lt_double (const void* k, const void* e);

#ifdef __cplusplus
}
#endif

#endif
