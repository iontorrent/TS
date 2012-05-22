/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGFITTERTRACKER_H
#define BKGFITTERTRACKER_H


#include "CommandLineOpts.h"
#include "cudaWrapper.h"
#include "Separator.h"
#include "RegionTimingCalc.h"
#include "BkgModel.h"
#include "GlobalDefaultsForBkgModel.h"
#include "FlowBuffer.h"
#include "BkgDataPointers.h"
#include "ImageLoader.h"
#include "RegionTrackerReplay.h"
#include "BkgModelReplay.h"

typedef std::pair<int, int> beadRegion;
typedef std::vector<beadRegion> regionProcessOrderVector;
bool sortregionProcessOrderVector (const beadRegion& r1, const beadRegion& r2);


struct BkgModelWorkInfo
{
  int type;
  BkgModel *bkgObj;
  int flow;
  Image *img;
  bool last;
  bool learning;
};

struct SeparatorWorkInfo
{
  int type;
  char label[10];
  Separator *separator;
  Image *img;
  Region *region;
  Mask *mask;
};

struct ImageInitBkgWorkInfo
{
  int type;
  //AvgKeyIncorporation *kic;
  // possible replacement for kic
  float t_mid_nuc;
  float t_sigma;
  int numRegions;
  Region *regions;
  int r;
  char *experimentName;
  Mask *maskPtr;        // shared, updated at end of background model processing
  PinnedInFlow *pinnedInFlow;  // shared among threads, owned by ImageTracker
  RawWells *rawWells;
  EmptyTraceTracker *emptyTraceTracker;

  BkgDataPointers *ptrs;

  int rows;
  int cols;
  int maxFrames;
  int uncompFrames;
  int *timestamps;
  RawWells *bkgDbg1;
  RawWells *bkgDbg2;
  RawWells *bkgDebugKmult;
  CommandLineOpts *clo;
  SequenceItem *seqList;
  int numSeqListItems;
  std::vector<float> *sep_t0_estimate;
  PoissonCDFApproxMemo *math_poiss;
  BkgModel **BkgModelFitters;
  std::set<int> *sample;
  std::vector<float> *tauB;
  std::vector<float> *tauE;
};

extern void *DynamicBkgFitWorker (void *arg, bool use_gpu);
void* BkgFitWorkerCpu (void *arg);
void* DynamicBkgFitWorkerCpu (void *arg);
extern void *BkgFitWorker (void *arg, bool use_gpu);
bool CheckBkgDbgRegion (Region *r,BkgModelControlOpts &bkg_control);


// allocate compute resources for our analysis
struct ComputationPlanner
{
  int numBkgWorkers;
  int numDynamicBkgWorkers;
  int numBkgWorkers_gpu;
  bool use_gpu_acceleration;
  bool dynamic_gpu_balance;
  float gpu_work_load;
  bool use_all_gpus;
  std::vector<int> valid_devices;
  regionProcessOrderVector region_order;
  // dummy
  int lastRegionToProcess;
};

struct ProcessorQueue
{
  // Create array to hold gpu_info
  std::vector<BkgFitWorkerGpuInfo> gpu_info;
  WorkerInfoQueue *threadWorkQ;
  WorkerInfoQueue *threadWorkQ_gpu;
  DynamicWorkQueueGpuCpu* dynamicthreadWorkQ;
  // our one item instance
  WorkerInfoQueueItem item;
};

void PlanMyComputation (ComputationPlanner &my_compute_plan, BkgModelControlOpts &bkg_control);

void SetRegionProcessOrder (int numRegions, BkgModel** fitters, ComputationPlanner &analysis_compute_plan);
void AllocateProcessorQueue (ProcessorQueue &my_queue,ComputationPlanner &analysis_compute_plan, int numRegions);
void AssignQueueForItem (ProcessorQueue &analysis_queue,ComputationPlanner &analysis_compute_plan, int numRegions, int r);

void WaitForRegionsToFinishProcessing (ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan, int flow);
void SpinDownCPUthreads (ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan);
void SpinUpGPUAndDynamicThreads (ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan);



//lightweight class to handle debugging wells: this & bubbles need to be rethought sometime
class dbgWellTracker
{
  public:
    RawWells *bkgDbg1;
    RawWells *bkgDbg2;
    RawWells *bkgDebugKmult;

    dbgWellTracker();
    void Init (char *experimentName, int rows, int cols, int numFlows, char *flowOrder);
    void Close();
    ~dbgWellTracker();
};

class BkgFitterTracker
{
  public:
    BkgModel **BkgModelFitters;
    int numFitters;
    
    GlobalDefaultsForBkgModel global_defaults;  // shared across everything
    
    // shared math cache
    PoissonCDFApproxMemo poiss_cache; // math routines the bkg model needs to do a lot

    // queue object for fitters
    BkgModelWorkInfo *bkinfo;


    // how we're going to fit
    ProcessorQueue analysis_queue;
    ComputationPlanner analysis_compute_plan;

    // debugging file
    dbgWellTracker my_bkg_dbg_wells;
    
    void ThreadedInitialization (RawWells &rawWells, CommandLineOpts &clo, Mask *maskPtr, PinnedInFlow *pinnedInFlow, char *experimentName,ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, Region *regions, int totalRegions, RegionTiming *region_timing,SeqListClass &my_keys, BkgDataPointers *ptrs,EmptyTraceTracker &emptytracetracker, std::vector<float> *tauB=NULL, std::vector<float> *tauE=NULL);
    void ExecuteFitForFlow (int flow, ImageTracker &my_img_set, bool last);
    void PlanComputation (BkgModelControlOpts &bkg_control);
    void SpinUp();
    void UnSpinGpuThreads();
    void SetRegionProcessOrder();
    BkgFitterTracker (int numRegions);
    void DeleteFitters();
    ~BkgFitterTracker();
    void DumpBkgModelRegionInfo (char *experimentName,int flow,bool last_flow);
    void DumpBkgModelBeadInfo (char *experimentName, int flow, bool last_flow, bool debug_bead_only);
    void DumpBkgModelBeadParams (char *experimentName,  int flow, bool debug_bead_only);
    void DumpBkgModelBeadOffset (char *experimentName, int flow, bool debug_bead_only);
    void DumpBkgModelEmphasisTiming (char *experimentName, int flow);
    void DumpBkgModelInitVals (char *experimentName, int flow);
    void DumpBkgModelDarkMatter (char *experimentName, int flow);
    void DumpBkgModelEmptyTrace (char *experimentName, int flow);
    void DumpBkgModelRegionParameters (char *experimentName,int flow);
};

#endif // BKGFITTERTRACKER_H
