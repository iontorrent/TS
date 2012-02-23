/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef NUCSTEPCACHE_H
#define NUCSTEPCACHE_H

#include "TimeCompression.h"
#include "RegionParams.h"

class NucStep{
  public:
    // scratch space to cache values that are recomputed by region as they arise
    // buffers for handling the computed nucleotide rise
    // as time is shifted, this is now approximately uniform per well
    float *nuc_rise_coarse_step;
    int i_start_coarse_step[NUMNUC];
    float *nuc_rise_fine_step;
    int i_start_fine_step[NUMNUC];

    float *per_nuc_coarse_step[NUMNUC];
    float *per_nuc_fine_step[NUMNUC];

    int nuc_flow_t;

    NucStep();
    void Alloc(int npts);
    void Delete();
    float *NucFineStep(int NucID);
    float *NucCoarseStep(int NucID);
    void CalculateNucRiseFineStep(reg_params *a_region, TimeCompression &time_c);
    void CalculateNucRiseCoarseStep(reg_params *a_region, TimeCompression &time_c);
};



#endif // NUCSTEPCACHE_H