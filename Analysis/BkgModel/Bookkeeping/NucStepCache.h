/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef NUCSTEPCACHE_H
#define NUCSTEPCACHE_H

#include "TimeCompression.h"
#include "RegionParams.h"
#include "FlowBuffer.h"

class NucStep{
    // These are just locally cached pointers into the nuc_rise_{coarse,fine}_step arrays.
    float **per_flow_coarse_step;
    float **per_flow_fine_step;

    int all_flow_t;
    bool precomputed_step;

  public:
    // scratch space to cache values that are recomputed by region as they arise
    // buffers for handling the computed nucleotide rise
    // as time is shifted, this is now approximately uniform per well
    float *nuc_rise_coarse_step;
    float *nuc_rise_fine_step;

    // Dynamically allocated start arrays.
    int *i_start_coarse_step;
    int *i_start_fine_step;

    // keep records of timing
    float *t_mid_nuc_actual;
    float *t_sigma_actual;

    NucStep();
    ~NucStep();
    void Alloc(int npts, int flow_block_size);
    void Delete();
    float *NucFineStep(int NucID);
    const float *NucCoarseStep(int NucID) const;
    void CalculateNucRiseFineStep(reg_params* a_region, TimeCompression& time_c, FlowBufferInfo& my_flow);
    void CalculateNucRiseFineStep(
        reg_params* a_region, 
        int frames, 
        std::vector<float> &frameNumber, 
        FlowBufferInfo& my_flow);
    void CalculateNucRiseCoarseStep(
        reg_params *a_region, const TimeCompression &time_c, const FlowBufferInfo &my_flow);
    void ForceLockCalculateNucRiseCoarseStep(
        reg_params *a_region, const TimeCompression &time_c, const FlowBufferInfo &my_flow);
    void Unlock();
};



#endif // NUCSTEPCACHE_H
