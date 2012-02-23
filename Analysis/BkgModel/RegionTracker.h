/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONTRACKER_H
#define REGIONTRACKER_H

#include "BkgMagicDefines.h"
#include "RegionParams.h"
#include "TimeCompression.h"
#include "NucStepCache.h"
#include "DarkHalo.h"
#include "GlobalDefaultsForBkgModel.h"




class RegionTracker{
  public:
    // current per-region parameters
    // this could be a list
    reg_params rp;
    reg_params rp_high;
    reg_params rp_low;

    NucStep cache_step;
    Halo missing_mass;

    RegionTracker();
    ~RegionTracker();
    void AllocScratch(int npts);
    void RestrictRatioDrift();
    void Delete();
    void InitHighRegionParams(float t_mid_nuc_start);
    void InitLowRegionParams(float t_mid_nuc_start);
    void InitModelRegionParams(float t_mid_nuc_start,float sigma_start, float dntp_concentration_in_uM,GlobalDefaultsForBkgModel &global_defaults);
    void InitRegionParams(float t_mid_nuc_start,float sigma_start, float dntp_concentration_in_uM,GlobalDefaultsForBkgModel &global_defaults);
};

#endif // REGIONTRACKER_H

