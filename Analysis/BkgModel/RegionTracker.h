/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONTRACKER_H
#define REGIONTRACKER_H

#include "BkgMagicDefines.h"
#include "RegionParams.h"
#include "TimeCompression.h"
#include "DNTPRiseModel.h"
#include "GlobalDefaultsForBkgModel.h"


class RegionTracker{
  public:
    // current per-region parameters
    // this could be a list
    reg_params rp;
    reg_params rp_high;
    reg_params rp_low;
    
    // this is a regional parameter, even if not obviously so
    float   *dark_matter_compensator;  // compensate for systematic errors in background hydrogen modeling, "dark matter"
    int nuc_flow_t;  // a useful number

    // scratch space to cache values that are recomputed by region as they arise
    // buffers for handling the computed nucleotide rise
    // as time is shifted, this is now approximately uniform per well
    float *nuc_rise_coarse_step;
    int i_start_coarse_step[NUMNUC];
    float *nuc_rise_fine_step;
    int i_start_fine_step[NUMNUC];

    RegionTracker();
    ~RegionTracker();
    void AllocScratch(int npts);
    void CalculateNucRiseFineStep(reg_params *a_region, TimeCompression &time_c);
    void CalculateNucRiseCoarseStep(reg_params *a_region, TimeCompression &time_c);
    void ResetDarkMatter();
    void RestrictRatioDrift();
    void Delete();
    void NormalizeDarkMatter(float *scale_factor, int npts);
    void InitHighRegionParams(float t_mid_nuc_start);
    void InitLowRegionParams(float t_mid_nuc_start);
    void InitModelRegionParams(float t_mid_nuc_start,float sigma_start, float dntp_concentration_in_uM,GlobalDefaultsForBkgModel &global_defaults);
    void InitRegionParams(float t_mid_nuc_start,float sigma_start, float dntp_concentration_in_uM,GlobalDefaultsForBkgModel &global_defaults);
    void DumpDarkMatter(FILE *my_fp, int x, int y);
    void DumpDarkMatterTitle(FILE *my_fp);
};

#endif // REGIONTRACKER_H

