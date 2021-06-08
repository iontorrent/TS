/* Copyright (C) 2020 Ion Torrent Systems, Inc. All Rights Reserved */
#include "tmap_bsearch.h"

// binary search for sorted arrays. Handles omitted keys
// passed in 'lt' function should compare the key pointed to by the first argument to the data element pointed to by the second argument, in this order
// returns the pointer to the first key that is not lower than the passed one.
// for multiple occurences returns pointer to the lowest occurencelowest occurence 
// performance is always ln(N), no worst/best case difference (each search on the array of given size takes same number of steps)
// can be improved to handle 'lucky strikes' if check for the equality in some form is available, or if no restriction on the comparison order is given


// (DK) WARNING! The implementation below is flowed! the search does not end (enters infinite loop) when searching for the value that is equal to 0 element in the array!
//

const void* tmap_binary_search (const void* key, const void* data, size_t num_elem, size_t elem_size, int (*lt) (const void* key, const void* elem))
{
    size_t lower_idx = 0; // lower boundary of the narrowed zone
    size_t upper_idx = num_elem; // upper boundary of the narrowed zone 
    size_t mid_idx;

    while (lower_idx < upper_idx) 
    {

        mid_idx = (lower_idx + upper_idx) >> 1;
        if (lt (key, data + (mid_idx * elem_size))) // key lower then mid
            upper_idx = mid_idx;
        else // key greater or equal to mid
            lower_idx = mid_idx;
    }
    return data + (lower_idx * elem_size);
}


const void* tmap_binary_search_nokey (const void* data, size_t num_elem, size_t elem_size, int (*above) (const void* elem))
{
    size_t lower_idx = 0; // lower boundary of the narrowed zone
    size_t upper_idx = num_elem; // upper boundary of the narrowed zone 
    size_t mid_idx;

    while (lower_idx < upper_idx) 
    {

        mid_idx = (lower_idx + upper_idx) >> 1;
        if (above (data + (mid_idx * elem_size))) // key lower then mid
            upper_idx = mid_idx;
        else // key greater or equal to mid
            lower_idx = mid_idx;
    }
    return data + (lower_idx * elem_size);
}

// the above functions can be improved to handle 'lucky strikes' if check for the equality in some form is available, or if no restriction on the comparison order is given

int lt_uint (const void* k, const void* e)
{
    return (*(unsigned *) k) < (*(unsigned *) e);
}

int lt_int (const void* k, const void* e)
{
    return (*(int *) k) < (*(int *) e);
}


int lt_uint32 (const void* k, const void* e)
{
    return (*(uint32_t *) k) < (*(uint32_t *) e);
}

int lt_int32 (const void* k, const void* e)
{
    return (*(int32_t *) k) < (*(int32_t *) e);
}

int lt_uint64 (const void* k, const void* e)
{
    return (*(uint64_t *) k) < (*(uint64_t *) e);
}

int lt_int64 (const void* k, const void* e)
{
    return (*(int64_t *) k) < (*(int64_t *) e);
}

int lt_double (const void* k, const void* e)
{
    return (*(double *) k) < (*(double *) e);
}
