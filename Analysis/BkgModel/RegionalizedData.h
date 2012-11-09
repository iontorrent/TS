/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONALIZEDDATA_H
#define REGIONALIZEDDATA_H


#include <stdio.h>
#include <set>
#include <vector>

#include "Image.h"
#include "Region.h"
#include "FlowBuffer.h"

#include "Utils.h"
#include "BkgMagicDefines.h"
#include "BeadParams.h"
#include "BeadTracker.h"
#include "RegionParams.h"
#include "RegionTracker.h"
#include "EmphasisVector.h"
#include "GlobalDefaultsForBkgModel.h"

#include "TimeCompression.h"

#include "BkgTrace.h"
#include "EmptyTraceTracker.h"
#include "BeadScratch.h"
#include "Serialization.h"


class RegionalizedData
{


  public:
    int fitters_applied;
    // the subregion that contains regionalized data
    Region *region;

    flow_buffer_info my_flow;  // specific to the buffers in my_trace that are filled, also supplies flow order

    TimeCompression time_c; // region specific
// local region emphasis vectors - may vary across chip by trace
    EmphasisClass emphasis_data; // region specific, effectively a parameter

    BkgTrace my_trace;  // initialized and populated by this object
    EmptyTraceTracker *emptyTraceTracker;
    EmptyTrace *emptytrace;

    // The things we're applying optimizers to fit:
    BeadTracker my_beads;
    RegionTracker my_regions;

    // initial guesses for nuc rise parameters
    float   sigma_start;
    float   t_mid_nuc_start;
    // BkgModel wants to rezero and do other things that are
    // detrimental if using sdat data Save a little state here to
    // indicate don't rezero
    bool doDcOffset;

    // space for processing current bead in optimization (multiflow levmar)
    // recycled for use in other optimizers
    // possibly should have more than one instance
    BeadScratchSpace my_scratch; // @TODO: transfer to >fitters<

    // flag for copying data from sdat chunks without interpolation
    bool regionAndTimingMatchSdat;

    void DumpRegionParameters (float cur_avg_resid);
    void DumpTimeAndEmphasisByRegion (FILE *my_fp);
    void LimitedEmphasis();
    void AdaptiveEmphasis();
    void NoData();

    int get_region_col()
    {
      return region->col;
    }
    int get_region_row()
    {
      return region->row;
    }
    int GetNumLiveBeads() const
    {
      return my_beads.numLBeads;
    }

    int GetNumFrames()
    {
      return time_c.npts();
    }


    int GetNumHighPPF() const
    {
      return my_beads.NumHighPPF();
    }

    int GetNumPolyclonal() const
    {
      return my_beads.NumPolyclonal();
    }

    int GetNumBadKey() const
    {
      return my_beads.NumBadKey();
    }
    bead_params& GetParams (int iWell)
    {
      return my_beads.params_nn[iWell];
    }
    void GetParams (int iWell,struct bead_params *pOut)
    {
      if ( (iWell >= 0) && (iWell < my_beads.numLBeads) & (pOut != NULL))
      {
        memcpy (pOut,&my_beads.params_nn[iWell],sizeof (struct bead_params));
      }
    }
    void DumpEmptyTrace (FILE *my_fp);
    void DumpInitValues (FILE *my_fp)
    {
      if (region!=NULL)
        fprintf (my_fp, "%d\t%d\t%f\t%f\n",region->col,region->row, t_mid_nuc_start, sigma_start);
    }
        
    RegionalizedData();
    ~RegionalizedData();

    void AllocTraceBuffers();
    void AllocFitBuffers();
    void SetTimeAndEmphasis (GlobalDefaultsForBkgModel &global_defaults, float tmid, float t0_offset);
    void SetupTimeAndBuffers (GlobalDefaultsForBkgModel &global_defaults,float sigma_guess,
                              float t_mid_nuc_guess,
                              float t0_offset);
     // set states of emphasis for the fit                         
    void SetCrudeEmphasisVectors();
    void SetFineEmphasisVectors();
    
    // don't know if this is correct, again may want coprocessor instances
    // loading the data into the regionalized structure
    void AddOneFlowToBuffer (GlobalDefaultsForBkgModel &global_defaults, int flow);
    void UpdateTracesFromImage (Image *img, int flow);
    void UpdateTracesFromImage (SynchDat &chunk, int flow);

    bool LoadOneFlow (Image *img, GlobalDefaultsForBkgModel &global_defaults, int flow);
    bool LoadOneFlow (SynchDat &data, GlobalDefaultsForBkgModel &global_defaults, int flow);
    void SetTshiftLimitsForSynchDat();
    // technically trivial fitters that fit the dc_offset model
    // so we may think about doing this on a coprocessor
    void RezeroByCurrentTiming();
    void RezeroTraces (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty, int fnum);
    void RezeroTracesAllFlows (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty);
    void PickRepresentativeHighQualityWells (float ssq_filter);

 private:

    // Serialization section
    friend class boost::serialization::access;
    template<typename Archive>
      void save(Archive& ar, const unsigned version) const
      {
	// fprintf(stdout, "Serialization: save RegionalizedData...");
	ar & 
	  region &
	  my_flow &
	  time_c &
	  emphasis_data &
	  my_trace &
	  emptyTraceTracker &
	  emptytrace &
	  my_beads &
	  my_regions &
	  sigma_start &
	  t_mid_nuc_start &
	  doDcOffset &
	  my_scratch &
	  regionAndTimingMatchSdat;
	// fprintf(stdout, "done with RegionalizedData\n");
      }
    template<typename Archive>
      void load(Archive& ar, const unsigned version)
      {
	// fprintf(stdout, "Serialization: load RegionalizedData...");
	ar & 
	  region &
	  my_flow &
	  time_c &
	  emphasis_data &
	  my_trace &
	  emptyTraceTracker &
	  emptytrace &
	  my_beads &
	  my_regions &
	  sigma_start &
	  t_mid_nuc_start &
	  doDcOffset &
	  my_scratch &
	  regionAndTimingMatchSdat;
	
	AllocFitBuffers ();
	AllocTraceBuffers();
	SetCrudeEmphasisVectors();
	
	// fprintf(stdout, "done with RegionalizedData\n");
      }
    
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

#endif // REGIONALIZEDDATA_H
