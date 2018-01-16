/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGFITTERTRACKER_H
#define BKGFITTERTRACKER_H

#include "RingBuffer.h"
#include "CommandLineOpts.h"
#include "cudaWrapper.h"

#include "RegionTimingCalc.h"
#include "SignalProcessingMasterFitter.h"
#include "GlobalDefaultsForBkgModel.h"
#include "FlowBuffer.h"
#include "BkgDataPointers.h"
#include "ImageLoader.h"

#include "SlicedPrequel.h"
#include "Serialization.h"

#include "SignalProcessingFitterQueue.h"

#include "BkgModelHdf5.h"

//#include <unordered_map>
#include <map>

class BkgFitterTracker
{
private:
  //RingBuffer<float> *ampEstBufferForGPU;
  void InitBeads_BestRegion(const CommandLineOpts &inception_state);
  int numFitters;
public:
  std::vector<RegionalizedData *> sliced_chip;
  std::vector<class SlicedChipExtras>   sliced_chip_extras;
  std::vector<SignalProcessingMasterFitter *> signal_proc_fitters;


  regionProcessOrderVector region_order;


  GlobalDefaultsForBkgModel global_defaults;  // shared across everything

  EmptyTraceTracker *all_emptytrace_track;  // has to be a pointer because of the excessive construction work

  // shared math cache
  PoissonCDFApproxMemo poiss_cache; // math routines the bkg model needs to do a lot

  // queue object for fitters
  BkgModelWorkInfo *bkinfo;


  // how we're going to fit
  //ProcessorQueue analysis_queue;
  //ComputationPlanner analysis_compute_plan;

  // how we're going to fit
  ProcessorQueue CpuQueueControl;  //former analysis_queue
  cudaWrapper GpuQueueControl;



  BkgParamH5 all_params_hdf;
  std::vector<int16_t> washout_flow;

  //void CreateRingBuffer(int numBuffers, int bufSize);
  //RingBuffer<float>* getRingBuffer() const { return ampEstBufferForGPU; }

  BkgFitterTracker ();
  BkgFitterTracker ( int numRegions );

  void SetNumRegions(int numRegions);

  void AllocateRegionData(std::size_t numRegions, const CommandLineOpts * inception_state);
  //void SetUpPipelines ( BkgModelControlOpts &bkg_control);

  void SetUpCpuAndGpuPipelines (BkgModelControlOpts &bkg_control );

  //ToDo: remove this function and do the configuration aty a higher level
  void UpdateGPUPipelineExecutionConfiguration(CommandLineOpts & inception_state);
  void UpdateAndCheckGPUCommandlineOptions(CommandLineOpts & inception_state);

  void SpinnUpGpuThreads();
  void SpinnUpCpuThreads();

  void UnSpinGpuThreads();
  void UnSpinCpuThreads();
  //  void UnSpinMultiFlowFitGpuThreads();

  //checks system state and accordingly switches to different GPU pipeline execution mode
  void checkAndInitGPUPipelineSwitch(
      const CommandLineOpts &inception_state,
      const ImageSpecClass &my_image_spec,
      SemQueue *packQueue,
      SemQueue *writeQueue,
      ChunkyWells *rawWells,
      int flow,
      bool restart);


  void DeleteFitters();
  ~BkgFitterTracker();


  bool useGpuAcceleration() { return GpuQueueControl.useGpuAcceleration(); }
  void ThreadedInitialization ( RawWells &rawWells, const CommandLineOpts &inception_state,
                                const ComplexMask &a_complex_mask,
                                const char *results_folder,const ImageSpecClass &my_image_spec,
                                const std::vector<float> &smooth_t0_est,
                                std::vector<Region> &regions,
                                const std::vector<RegionTiming> &region_timing,
                                const SeqListClass &my_keys,
                                bool restart,
                                int num_flow_blocks );
  void InitCacheMath();
  void ExecuteFitForFlow ( int raw_flow, ImageTracker &my_img_set, bool last,
                           int flow_key, master_fit_type_table *table,
                           const CommandLineOpts * inception_state );
  void ExecuteGPUFlowByFlowSignalProcessing(
      int raw_flow,
      ImageTracker &my_img_set,
      bool last,
      int flow_key,
      const CommandLineOpts *inception_state,
      const std::vector<float> *smooth_t0_est);
  void CollectSampleWellsForGPUFlowByFlowSignalProcessing(
        int raw_flow,
        int flow_block_size,
        ImageTracker &my_img_set,
        bool last,
        int flow_key,
        const CommandLineOpts *inception_state,
        const std::vector<float> *smooth_t0_est);

  void SetUpTraceTracking ( const SlicedPrequel &my_prequel_setup, const CommandLineOpts &inception_state, const ImageSpecClass &my_image_spec, const ComplexMask &cmask, int flow_block_size );


  void PlanComputation ( BkgModelControlOpts &bkg_control);

  void SetRegionProcessOrder(const CommandLineOpts &inception_state);
  int findRegion(int x, int y);



  // text based diagnostics in blocks of flows
  void DumpBkgModelRegionInfo ( char *results_folder,int flow,bool last_flow,
                                FlowBlockSequence::const_iterator flow_block ) const;
  void DumpBkgModelBeadInfo ( char *results_folder, int flow, bool last_flow, bool debug_bead_only,
                              FlowBlockSequence::const_iterator flow_block ) const;
  void DumpBkgModelBeadParams ( char *results_folder,  int flow, bool debug_bead_only, int flow_block_size ) const;
  void DumpBkgModelBeadOffset ( char *results_folder, int flow, bool debug_bead_only ) const;
  void DumpBkgModelEmphasisTiming ( char *results_folder, int flow ) const;
  void DumpBkgModelInitVals ( char *results_folder, int flow ) const;
  void DumpBkgModelDarkMatter ( char *results_folder, int flow ) const;
  void DumpBkgModelEmptyTrace ( char *results_folder, int flow, int flow_block_size ) const;
  void DumpBkgModelRegionParameters ( char *results_folder,int flow, int flow_block_size ) const;

  void SaveRegParamsJson(const string &filename_json);
  void LoadRegParamsFromJson(const string &filename_json);

  void DetermineAndSetGPUAllocationAndKernelParams( BkgModelControlOpts &bkg_control,
                                      int global_max_flow_key, int global_max_flow_max );
  // for beads in the bestRegion
  std::pair<int, int> bestRegion;
  const Region *bestRegion_region;
  void InitBeads_xyflow(const CommandLineOpts &inception_state);

  // xyflow
  //std::vector<XYFlow_class> xyf_hash;
  HashTable_xyflow xyf_hash;

  // sliced_chip_cur_bead_block and sliced_chip_cur_buffer_block are scratch regions that
  // should just match sliced_chip.
  void AllocateSlicedChipScratchSpace( int global_flow_max );

  int getMaxFrames(const ImageSpecClass &my_image_spec, const std::vector<RegionTiming> &region_timing);

  void setWashoutThreshold(float threshold)
  {
	  for(size_t n = 0; n < signal_proc_fitters.size(); ++n)
	  {
          if(signal_proc_fitters[n])
		  {
              signal_proc_fitters[n]->setWashoutThreshold(threshold);
		  }
	  }
  }

  void setWashoutFlowDetection(int detection)
  {
	  for(size_t n = 0; n < signal_proc_fitters.size(); ++n)
	  {
          if(signal_proc_fitters[n])
		  {
              signal_proc_fitters[n]->setWashoutFlowDetection(detection);
		  }
	  }
  }

  void setUpRingBuffer(int numBuffers, int size);

 private:



  // Serialization section
  friend class boost::serialization::access;
  template<typename Archive>
    void load(Archive& ar, const unsigned version)
    {

      ar &
	    sliced_chip &
	    global_defaults &      // serialize out before signal_proc_fitters as ref'd
	    // signal_proc_fitters &  // rebuilt in ThreadedInitialization
	    numFitters &
      washout_flow &
	    all_emptytrace_track &
	    GpuQueueControl;

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
	all_emptytrace_track &
  GpuQueueControl;
	// poiss_cache &
	// bkinfo;

      // fprintf(stdout, "done BkgFitterTracker\n");
    }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

};

#endif // BKGFITTERTRACKER_H
