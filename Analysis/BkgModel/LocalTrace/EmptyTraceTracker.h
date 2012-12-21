/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef EMPTYTRACETRACKER_H
#define EMPTYTRACETRACKER_H

#include <vector>

#include "EmptyTrace.h"
#include "EmptyTraceReplay.h"
#include "Region.h"
#include "GlobalDefaultsForBkgModel.h"
#include "ImageSpecClass.h"
#include "CommandLineOpts.h"
#include "SynchDat.h"
#include "Serialization.h"

// declare save_construct_data
class EmptyTraceTracker;
namespace boost {namespace serialization {
    template<class Archive>
      inline void save_construct_data(Archive& ar, const EmptyTraceTracker * o, const unsigned int version);
    template<class Archive>
      inline void load_construct_data(Archive& ar, EmptyTraceTracker * o, const unsigned int version);
  }}

class EmptyTraceTracker
{
 public:
  EmptyTraceTracker (std::vector<Region> &_regions,
		     std::vector<RegionTiming> &_regiontiming,
		     std::vector<float> &_sep_t0_est,
		     CommandLineOpts &_inception_state);
    
    ~EmptyTraceTracker();
    
    void Allocate(Mask *bfmask, ImageSpecClass &imgSpec);
    void SetEmptyTracesFromImage (Image &img, PinnedInFlow &pinnedInFlow, int flow, Mask *bfmask);
    void SetEmptyTracesFromImage(SynchDat &mesh, PinnedInFlow &pinnedInFlow, int flow, Mask *bfmask);
    void SetEmptyTracesFromImageForRegion(Image &img, PinnedInFlow &pinnedInFlow, int flow, Mask *bfmask, Region& region, float t_mid_nuc);
    void SetEmptyTracesFromImageForRegion(SynchDat &mesh, PinnedInFlow &pinnedInFlow, int flow, Mask *bfmask, Region& region, float t_mid_nuc, float sigma, float t_start,TimeCompression *time_cp);
    EmptyTrace *AllocateEmptyTrace (Region &region, int imgFrames);
    EmptyTrace *GetEmptyTrace (Region &region);

  private:
    std::vector<EmptyTrace *>emptyTracesForBMFitter;
    std::vector<Region> &regions;
    std::vector<RegionTiming> &regionTiming;
    std::vector<float> &sep_t0_est;
    int maxNumRegions;
    std::vector<int> imgFrames;
    std::string outlierDumpFile;

    int MaxNumRegions (std::vector<Region>& regions);
    void InitializeDumpOutlierTracesFile();
    void DumpOutlierTracesPerFlowPerRegion(int flow, Region& region, int nOutliers, int nRef);

    CommandLineOpts& inception_state; // why do I need to know this

    // the time compression variables needed here should be parsed out
    GlobalDefaultsForBkgModel global_defaults;

    EmptyTraceTracker (); //do not use

    // Serialization section
    friend class boost::serialization::access;

    template<typename Archive>
      friend void boost::serialization::save_construct_data(Archive& ar, const EmptyTraceTracker * o, const unsigned int version);
    template<typename Archive>
      friend void boost::serialization::load_construct_data(Archive& ar, EmptyTraceTracker * o, const unsigned int version);

    template<typename Archive>
      void serialize(Archive& ar, const unsigned int version) {
      // fprintf(stdout, "Serialize EmptyTraceTracker ... ");
      ar &
	emptyTracesForBMFitter &
	imgFrames &
	maxNumRegions &
	outlierDumpFile;
      // fprintf(stdout, "done EmptyTraceTracker\n");
    }
};

namespace boost { namespace serialization {
    template<typename Archive>
      inline void save_construct_data(Archive& ar, const EmptyTraceTracker * o, const unsigned int version) {
      // fprintf(stdout, "save_construct_data EmptyTraceTracker ... ");
      std::vector<Region> *regions_ptr = & o->regions;
      std::vector<RegionTiming> *regionTiming_ptr = & o->regionTiming;
      std::vector<float> *sep_t0_est_ptr = & o->sep_t0_est;
      // ar << inception_state_ptr;  later restored as a NULL pointer
      ar << regions_ptr;
      ar << regionTiming_ptr;
      ar << sep_t0_est_ptr;
      // fprintf(stdout, "done save_construct_data\n");
    }

    template<typename Archive>
      inline void load_construct_data(Archive& ar, EmptyTraceTracker * o, const unsigned int version){
      // fprintf(stdout, "load_construct_data EmptyTraceTracker ... ");
        std::vector<Region> *regions_ptr;
	std::vector<RegionTiming> *regionTiming_ptr;
	std::vector<float> *sep_t0_est_ptr;
	CommandLineOpts *inception_state_ptr = NULL;
	// ar >> inception_state_ptr;  after all, we didn't save this either
	ar >> regions_ptr;
	ar >> regionTiming_ptr;
	ar >> sep_t0_est_ptr;
	::new(o) EmptyTraceTracker(*regions_ptr, *regionTiming_ptr,
				   *sep_t0_est_ptr, *inception_state_ptr);
	// fprintf(stdout, "done load_construct_data, inception_state = NULL\n");
   }
  }}

#endif // EMPTYTRACETRACKER_H
