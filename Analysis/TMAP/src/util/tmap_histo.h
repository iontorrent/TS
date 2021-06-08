/* Copyright (C) 2020 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef __tmap_histo_h__
#define __tmap_histo_h__

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" 
{
#endif

void init_ihist_lowerb (size_t* bin_sizes, size_t bin_cnt, int64_t* bin_lower_bounds, int64_t lowest);
void init_dhist_lowerb (double* bin_sizes, int bin_cnt, double* bin_lower_bounds, double lowest);
void init_hist (void* hist, size_t counter_type_size, size_t bin_cnt);
void init_hist64 (uint64_t* hist, size_t bin_cnt);
void add_to_hist64i (uint64_t* hist, size_t bin_cnt, int64_t* bin_lower_bounds, int64_t value);
void add_to_hist64d (uint64_t* hist, size_t bin_cnt, double* bin_lower_bounds, double value);
void render_hist64i (uint64_t* hist, size_t bin_cnt, int64_t* bin_lower_bounds, FILE* f, unsigned width);
void render_hist64d (uint64_t* hist, size_t bin_cnt, double* bin_lower_bounds, FILE* f, unsigned width);

#ifdef __cplusplus
extern "C" 
{
#endif

#endif // __tmap_histo_h__
