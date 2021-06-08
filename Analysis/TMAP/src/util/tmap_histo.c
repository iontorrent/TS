/* Copyright (C) 2020 Ion Torrent Systems, Inc. All Rights Reserved */

#include "tmap_histo.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "tmap_bsearch.h"
#include <stdlib.h>

// initialize histogram bin lower bound array from bin sizes array
void init_ihist_lowerb (size_t* bin_sizes, size_t bin_cnt, int64_t* bin_lower_bounds, int64_t lowest)
{
    size_t i, runsum = lowest;
    for (i = 0; i != bin_cnt; ++i)
    {
        bin_lower_bounds [i] = runsum;
        runsum += bin_sizes [i];
    }
}
void init_dhist_lowerb(double* bin_sizes, int bin_cnt, double* bin_lower_bounds, double lowest)
{
    size_t i;
    double runsum = lowest;
    for (i = 0; i != bin_cnt; ++i)
    {
        bin_lower_bounds [i] = runsum;
        assert (bin_sizes [i] > 0.0);
        runsum += bin_sizes [i];
    }
}
// initialize histogram with counters of arbitrary type
void init_hist (void* hist, size_t counter_type_size, size_t bin_cnt)
{
    memset (hist, 0, counter_type_size * bin_cnt);
}
void init_hist64 (uint64_t* hist, size_t bin_cnt)
{
    init_hist (hist, sizeof (uint64_t), bin_cnt);
}

int int64_cmp (const void* i1, const void* i2)
{
    if (*(int64_t *) i1 < *(int64_t *) i2) return -1;
    else if (*(int64_t *) i1 == *(int64_t *) i2) return 0;
    else return 1;
}

void add_to_hist64i (uint64_t* hist, size_t bin_cnt, int64_t* bin_lower_bounds, int64_t value)
{
    // find the bin: do linear search to save on recurrent calls (and on implementation, as bsearch does not fit :)
    // DK: tmap_binary_search is flawed (enters infine loop when searching for a value equal to first element in the array)
    // const int64_t* bound = tmap_binary_search (&value, bin_lower_bounds, bin_cnt, sizeof (int64_t), lt_int);
    const int64_t* bound = bsearch (&value, bin_lower_bounds, bin_cnt, sizeof (int64_t), int64_cmp);
    // get bin index
    size_t idx = bound - bin_lower_bounds;
    // increment histogram bin value
    ++ hist [idx];
}

void add_to_hist64d (uint64_t* hist, size_t bin_cnt, double* bin_lower_bounds, double value)
{
    // find the bin: do linear search to save on recurrent calls (and on implementation, as bsearch does not fit :)
    const double* bound = tmap_binary_search (&value, bin_lower_bounds, bin_cnt, sizeof (double), lt_double);
    // get bin index
    size_t idx = bound - bin_lower_bounds;
    // increment bin value
    ++ hist [idx];
}

static unsigned digno (int64_t value)
{
    unsigned digno = 0;
    if (value < 0)
        digno += 1, value = -value;
    while (value)
        ++digno, value /= 10;
    if (!digno)
        digno = 1;
    return digno;
}

#define MIN_BAR_LEN 8
void render_hist64i (uint64_t* hist, size_t bin_cnt, int64_t* bin_lower_bounds, FILE* f, unsigned width)
{
    uint64_t maxv = 0;
    unsigned bound_digs = 1;
    size_t idx;
    for (idx = 0; idx != bin_cnt; ++idx) 
    {
        if (maxv < hist [idx])
            maxv = hist [idx];
        unsigned cur_bound_w = digno (bin_lower_bounds [idx]);
        if (bound_digs < cur_bound_w)
            bound_digs = cur_bound_w;
    }
    unsigned max_digs = digno (maxv);
    size_t scale = 1;
    unsigned overhead = MIN_BAR_LEN + 3 + max_digs + bound_digs;
    if (width > overhead)
    {
        width -= overhead;
        if (width < maxv) 
            scale = maxv / width;
    }
    else
        scale = 0;

    for (idx = 0; idx != bin_cnt; ++idx)
    {
        fprintf (f, "%*ld ", bound_digs, bin_lower_bounds [idx]);
        if (scale)
        {
            fputc ('|', f);
            int pos, last = hist [idx] / scale;
            for (pos = 0; pos != last; ++pos)
                fputc ('=', f);
        }
        fprintf (f, "%ld", hist [idx]);
    }
}

void render_hist64d (uint64_t* hist, size_t bin_cnt, double* bin_lower_bounds, FILE* f, unsigned width)
{
    uint64_t maxv = 0;
    unsigned bound_digs = 1;
    size_t idx;
    for (idx = 0; idx != bin_cnt; ++idx) 
    {
        if (maxv < hist [idx])
            maxv = hist [idx];
        unsigned cur_bound_w = digno (bin_lower_bounds [idx]);
        if (bound_digs < cur_bound_w)
            bound_digs = cur_bound_w;
    }
    unsigned max_digs = digno (maxv);
    size_t scale = 1;
    unsigned overhead = MIN_BAR_LEN + 3 + max_digs + bound_digs;
    if (width > overhead)
    {
        width -= overhead;
        if (width < maxv) 
            scale = maxv / width;
    }
    else
        scale = 0;

    for (idx = 0; idx != bin_cnt; ++idx)
    {
        fprintf (f, "%*f ", bound_digs, bin_lower_bounds [idx]);
        if (scale)
        {
            fputc ('|', f);
            int pos, last = hist [idx] / scale;
            for (pos = 0; pos != last; ++pos)
                fputc ('=', f);
        }
        fprintf (f, "%ld", hist [idx]);
    }
}
