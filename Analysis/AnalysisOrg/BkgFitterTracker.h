/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGFITTERTRACKER_H
#define BKGFITTERTRACKER_H


#include "CommandLineOpts.h"
#include "cudaWrapper.h"
#include "Separator.h"
#include "RegionTimingCalc.h"
#include "SignalProcessingMasterFitter.h"
#include "GlobalDefaultsForBkgModel.h"
#include "FlowBuffer.h"
#include "BkgDataPointers.h"
#include "ImageLoader.h"
#include "RegionTrackerReplay.h"
#include "BkgModelReplay.h"

#include "SlicedPrequel.h"
#include "Serialization.h"

#include "SignalProcessingFitterQueue.h"

#include "BkgModelHdf5.h"

class BkgFitterTracker
{
public:
  std::vector<RegionalizedData *> sliced_chip;
  std::vector<SignalProcessingMasterFitter *> signal_proc_fitters;
  int numFitters;

  GlobalDefaultsForBkgModel global_defaults;  // shared across everything

  EmptyTraceTracker *all_emptytrace_track;  // has to be a pointer because of the excessive construction work

  // shared math cache
  PoissonCDFApproxMemo poiss_cache; // math routines the bkg model needs to do a lot

  // queue object for fitters
  BkgModelWorkInfo *bkinfo;


  // how we're going to fit
  ProcessorQueue analysis_queue;
  ComputationPlanner analysis_compute_plan;

  BkgParamH5 all_params_hdf;
  std::vector<int16_t> washout_flow;

  void ThreadedInitialization ( RawWells &rawWells, CommandLineOpts &inception_state, ComplexMask &a_complex_mask,
                                char *results_folder,ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est,
                                std::vector<Region> &regions,
                                std::vector<RegionTiming> &region_timing,
                                SeqListClass &my_keys,
				bool restart);
  void InitCacheMath();
  void ExecuteFitForFlow ( int flow, ImageTracker &my_img_set, bool last );
  void SetUpTraceTracking ( SlicedPrequel &my_prequel_setup, CommandLineOpts &inception_state, ImageSpecClass &my_image_spec, ComplexMask &cmask );
  void PlanComputation ( BkgModelControlOpts &bkg_control );
  void SpinUp();
  void UnSpinSingleFlowFitGpuThreads();
  void UnSpinMultiFlowFitGpuThreads();
  void SetRegionProcessOrder();
  BkgFitterTracker ( int numRegions );
  void AllocateRegionData(std::size_t numRegions);
  void DeleteFitters();
  ~BkgFitterTracker();
  
  // text based diagnostics in blocks of flows
  void DumpBkgModelRegionInfo ( char *results_folder,int flow,bool last_flow );
  void DumpBkgModelBeadInfo ( char *results_folder, int flow, bool last_flow, bool debug_bead_only );
  void DumpBkgModelBeadParams ( char *results_folder,  int flow, bool debug_bead_only );
  void DumpBkgModelBeadOffset ( char *results_folder, int flow, bool debug_bead_only );
  void DumpBkgModelEmphasisTiming ( char *results_folder, int flow );
  void DumpBkgModelInitVals ( char *results_folder, int flow );
  void DumpBkgModelDarkMatter ( char *results_folder, int flow );
  void DumpBkgModelEmptyTrace ( char *results_folder, int flow );
  void DumpBkgModelRegionParameters ( char *results_folder,int flow );
  // text based diagnostics


  void DetermineMaxLiveBeadsAndFramesAcrossAllRegionsForGpu();

 private:
  
  BkgFitterTracker(){
    all_emptytrace_track = NULL;
    bkinfo = NULL;
    numFitters = 0;
  }

  // Serialization section
  friend class boost::serialization::access;
  template<typename Archive>
    void load(Archive& ar, const unsigned version)
    {
      // fprintf(stdout, "Serialization: save BkgFitterTracker ... ");
      ar & 
	sliced_chip &
	global_defaults &      // serialize out before signal_proc_fitters as ref'd
	// signal_proc_fitters &  // rebuilt in ThreadedInitialization
	numFitters &
        washout_flow &
	all_emptytrace_track;
	// poiss_cache &   // rebuilt in ThreadedInitialization
	// bkinfo;         // rebuilt in ThreadedInitialization

      signal_proc_fitters.resize(numFitters); // see constructor
      // fprintf(stdout, "done BkgFitterTracker\n");
    }
  template<typename Archive>
    void save(Archive& ar, const unsigned version) const
    {
      // fprintf(stdout, "Serialization: save BkgFitterTracker ... ");
      ar & 
	sliced_chip &
	global_defaults &      // serialize out before signal_proc_fitters as ref'd
	// signal_proc_fitters &  // 
	numFitters &
        washout_flow &
	all_emptytrace_track;
	// poiss_cache &
	// bkinfo; 

      // fprintf(stdout, "done BkgFitterTracker\n");
    }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

};

#endif // BKGFITTERTRACKER_H
