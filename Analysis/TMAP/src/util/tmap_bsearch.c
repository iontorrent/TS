/* Copyright (C) 2020 Ion Torrent Systems, Inc. All Rights Reserved */
#include "tmap_bsearch.h"
#include "tmap_error.h"

// binary search for sorted arrays. Handles omitted keys
// passed in 'lt' function should compare the key pointed to by the first argument to the data element pointed to by the second argument, in this order
// returns the pointer to the first key that is not lower than the passed one.
// for multiple occurences returns pointer to the lowest occurencelowest occurence 
// performance is always ln(N), no worst/best case difference (each search on the array of given size takes same number of steps)
// can be improved to handle 'lucky strikes' if check for the equality in some form is available, or if no restriction on the comparison order is given


// (DK) WARNING! The implementation below is flawed! the search does not end (enters infinite loop) when searching for the value that is equal to 0 element in the array!
// (DK) A key greater then first and less then second element also makes a problem. 

const void* tmap_binary_search_bad (const void* key, const void* data, size_t num_elem, size_t elem_size, int (*lt) (const void* key, const void* elem))
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

// DK this should fix the problem in the above code
// DK 12/3/2020 Tested and debugged the fix by creating comprehensive (all-cases) TestSuite case. Ok now.

const void* tmap_binary_search_fixed (const void* key, const void* data, size_t num_elem, size_t elem_size, int (*lt) (const void* key, const void* elem))
{
    size_t lower_idx = 0; // lower boundary of the narrowed zone
    size_t upper_idx = num_elem; // upper boundary of the narrowed zone 
    size_t mid_idx;

    while (lower_idx < upper_idx) 
    {

        mid_idx = (lower_idx + upper_idx) >> 1;
        if (lt (key, data + (mid_idx * elem_size))) // key lower then mid
            upper_idx = mid_idx;
        else // key greater or equal to mid. 
        {
            if (lower_idx == mid_idx) // DK: in this case, greater and equal should be treated differently: equal - return lower, greater - return lower + 1 (upper in this case)
            {
                if (lt (data + (lower_idx * elem_size), key)) // mid == low is lower then key. Position of the next element is to be returned
                    ++lower_idx;
                break;
            }
            else
                lower_idx = mid_idx;
        }
    }
    return data + (lower_idx * elem_size);
}

const void* tmap_binary_search_nokey_bad (const void* data, size_t num_elem, size_t elem_size, int (*above) (const void* elem))
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

const void* tmap_binary_search_nokey_no_fix_possible (const void* data, size_t num_elem, size_t elem_size, int (*above) (const void* elem))
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
        {
            if (lower_idx == mid_idx) // DK: search point is at or above lower_idx but below lower_idx + 1. 
                                      // if it is at the lower_idx, the lower_idx should be returned
                                      // if it is above, the lower_idx+1 should be returned
                                      // if only 'above' function is provided, there is no way to distinguish these cases
                                      // match check is needed. This is a fundamental flaw in this algorithm design. 
                                      // Verdict: Not fixable without changing call signature.
            {
                ++lower_idx;
                break;
            }
            lower_idx = mid_idx;
        }
    }
    return data + (lower_idx * elem_size);
}

const void* tmap_binary_search_nokey (const void* data, size_t num_elem, size_t elem_size, int (*above) (const void* elem))
{
    tmap_failure ("Internal error- the function tmap_binary_search_nokey is called which has no correct implementation");
    return NULL;
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
