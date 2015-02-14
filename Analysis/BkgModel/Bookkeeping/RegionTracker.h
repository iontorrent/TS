/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONTRACKER_H
#define REGIONTRACKER_H

#include "BkgMagicDefines.h"
#include "RegionParams.h"
#include "TimeCompression.h"
#include "NucStepCache.h"
#include "DarkHalo.h"
#include "GlobalDefaultsForBkgModel.h"
#include "DoubleExpSmoothing.h"

class RegionTracker{
  public:
    // current per-region parameters
    // this could be a list
    reg_params rp;
    reg_params rp_high;
    reg_params rp_low;

    NucStep cache_step;
    Halo missing_mass;

    DoubleExpSmoothing tmidnuc_smoother, copy_drift_smoother, ratio_drift_smoother;

    RegionTracker();    // for serializing, only.
    RegionTracker( const class CommandLineOpts * inception_state );
    ~RegionTracker();
    void AllocScratch(int npts, int flow_block_size);
    void RestrictRatioDrift();
    void ResetLocalRegionParams( int flow_block_size );
    void Delete();
    void InitHighRegionParams(float t_mid_nuc_start, int flow_block_size);
    void InitLowRegionParams(float t_mid_nuc_start, int flow_block_size);
    void InitModelRegionParams(float t_mid_nuc_start,float sigma_start, GlobalDefaultsForBkgModel &global_defaults, int flow_block_size);
    void InitRegionParams(float t_mid_nuc_start,float sigma_start, GlobalDefaultsForBkgModel &global_defaults, int flow_block_size);

 private:

    // Boost serialization support:
    bool restart;
    friend class boost::serialization::access;
    template<class Archive>
      void save(Archive& ar, const unsigned int version) const
      {
	// fprintf(stdout, "Serialize: save RegionTracker ... ");
	ar
	   & rp
	   & rp_high
	   & rp_low
           & tmidnuc_smoother
           & copy_drift_smoother
           & ratio_drift_smoother
	  // cache_step  // re-initted every time
	  & missing_mass;
	// fprintf(stdout, "done RegionTracker\n");
      }
    template<class Archive>
      void load(Archive& ar, const unsigned int version)
      {
	// fprintf(stdout, "Serialize: load RegionTracker ... ");
	ar
	   & rp
	   & rp_high
	   & rp_low
           & tmidnuc_smoother
           & copy_drift_smoother
           & ratio_drift_smoother
	  // cache_step  // rebuilt every time by AllocScratch()
	  // AllocScratch() called in AllocFitBuffers()
	  // by the RegionalizedData object that owns this RegionTracker
	  & missing_mass;
	
	restart = true;
	// fprintf(stdout, "done RegionTracker\n");
      }
      BOOST_SERIALIZATION_SPLIT_MEMBER()

};


#endif // REGIONTRACKER_H

