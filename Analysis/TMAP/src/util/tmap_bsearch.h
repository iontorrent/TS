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
// passed in 'lt' function should be symmetric with respect to key and data pointers: (the bsearch would call it as lt (*key, *data) as well as lt (*data, *key))
// returns the pointer to the first data element that is not lower than the passed key (could be equal to the key).
// if there is no such entries in the passed array, returns a pointer to the upper array boudary (one position above the last element in the array, == (data + num_elem))
// for multiple occurences returns pointer to the lowest occurence
// performance is always ln(N), no worst/best case difference (each search on the array of given size takes same number of steps)
const void* tmap_binary_search_fixed (const void* key, const void* data, size_t num_elem, size_t elem_size, int (*lt) (const void* key, const void* elem));

#define tmap_binary_search tmap_binary_search_fixed

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
