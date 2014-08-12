/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SIGNALPROCESSINGFITTERQUEUE_H
#define SIGNALPROCESSINGFITTERQUEUE_H

#include <boost/serialization/utility.hpp>
#include "cudaWrapper.h"
#include "WorkerInfoQueue.h"
#include "SignalProcessingMasterFitter.h"

typedef std::pair<int, int> beadRegion;
typedef std::vector<beadRegion> regionProcessOrderVector;
bool sortregionProcessOrderVector (const beadRegion& r1, const beadRegion& r2);

struct ProcessorQueue
{

  enum QueueType {
    CPU_QUEUE,
    GPU_QUEUE,
  };

  // our one item instance
  WorkerInfoQueueItem item;

  ProcessorQueue() {
    fitting_queues.resize(2);
    for (unsigned int i=0; i<fitting_queues.size(); ++i) {
      fitting_queues[i] = NULL;
    }
    heterogeneous_computing = false;
    gpuMultiFlowFitting = true;
    gpuSingleFlowFitting = true;
  }
  void SetCpuQueue(WorkerInfoQueue* q) { fitting_queues[CPU_QUEUE] = q;}
  void SetGpuQueue(WorkerInfoQueue* q) { fitting_queues[GPU_QUEUE] = q;}
  void AllocateGpuInfo(int n) { gpu_info.resize(n); }
  void turnOffHeterogeneousComputing() { heterogeneous_computing = false; }
  void turnOnHeterogeneousComputing() { heterogeneous_computing = true; }
  void turnOnGpuMultiFlowFitting() { gpuMultiFlowFitting = true; }
  void turnOffGpuMultiFlowFitting() { gpuMultiFlowFitting = false; }
  void turnOnGpuSingleFlowFitting() { gpuSingleFlowFitting = true; }
  void turnOffGpuSingleFlowFitting() { gpuSingleFlowFitting = false; }
  
  bool useHeterogenousCompute() { return heterogeneous_computing; }
  bool performGpuMultiFlowFitting() { return gpuMultiFlowFitting; }
  bool performGpuSingleFlowFitting() { return gpuSingleFlowFitting; }

  int GetNumQueues() { return fitting_queues.size();}
  std::vector<WorkerInfoQueue*>& GetQueues() { return fitting_queues; }
  std::vector<BkgFitWorkerGpuInfo>& GetGpuInfo() { return gpu_info; }
  void SpinUpGPUThreads( struct ComputationPlanner &analysis_compute_plan );
  static void CreateGpuThreadsForFitType(
    std::vector<BkgFitWorkerGpuInfo> &gpuInfo,
    WorkerInfoQueue* q,
    WorkerInfoQueue* fallbackQ,
    int numWorkers,
    std::vector<int> &gpus
   );
  /*std::vector<BkgFitWorkerGpuInfo>& GetGpuInfo() {
	  std::vector<BkgFitWorkerGpuInfo>::iterator iter;
	  for(iter = gpu_info.begin(); iter!=gpu_info.end(); iter++)
	  {
		  (*iter).queue = GetGpuQueue();
		  (*iter).fallbackQueue = GetCpuQueue();
	  }
	  return gpu_info;
  }*/
  WorkerInfoQueue* GetCpuQueue() { return fitting_queues[CPU_QUEUE]; }
  WorkerInfoQueue* GetGpuQueue() { return fitting_queues[GPU_QUEUE]; }

private:
  // Create array to hold gpu_info
  std::vector<BkgFitWorkerGpuInfo> gpu_info;

  std::vector<WorkerInfoQueue*> fitting_queues;
  bool heterogeneous_computing; // use both gpu and cpu
  bool gpuMultiFlowFitting;
  bool gpuSingleFlowFitting;
};


struct BkgModelWorkInfo
{
  int type;
  SignalProcessingMasterFitter *bkgObj;
  int flow;
  bool doingSdat;
  SynchDat *sdat;
  Image *img;
  bool last;
  ProcessorQueue* pq;
  int flow_key;
  PolyclonalFilterOpts polyclonal_filter_opts;
  master_fit_type_table *table;
  const CommandLineOpts *inception_state;
};


//prototype GPU trace generation whole block on GPU
struct BkgModelImgToTraceInfoGPU
{
  int type;
  int numfitters;
  int regionMaxX;
  int regionMaxY;
  Image * img;
  Mask * bfmask;
  std::vector<float> * smooth_t0_est;
  BkgModelWorkInfo * BkgInfo;
};



struct ImageInitBkgWorkInfo
{
  int type;
  //AvgKeyIncorporation *kic;
  // possible replacement for kic
  float t_mid_nuc;
  float t_sigma;
  float t0_frame;
  int numRegions;
  Region *regions;
  int r;
  const char *results_folder;
  
  Mask *maskPtr;        // shared, updated at end of background model processing
  PinnedInFlow *pinnedInFlow;  // shared among threads, owned by ImageTracker
  
  class RawWells *rawWells;
  EmptyTraceTracker *emptyTraceTracker;

  BkgDataPointers *ptrs;

  int rows;
  int cols;
  int maxFrames;
  int uncompFrames;
  int *timestamps;
  
  const CommandLineOpts *inception_state;

  bool nokey;
  SequenceItem *seqList;
  int numSeqListItems;
  
  const std::vector<float> *sep_t0_estimate;
  PoissonCDFApproxMemo *math_poiss;
  SignalProcessingMasterFitter **signal_proc_fitters;
  RegionalizedData **sliced_chip;
  struct SlicedChipExtras *sliced_chip_extras;
  GlobalDefaultsForBkgModel *global_defaults;
  std::set<int> *sample;
  std::vector<float> *tauB;
  std::vector<float> *tauE;

  bool restart;
  int16_t *washout_flow;
};

void* BkgFitWorkerCpu (void *arg);
bool CheckBkgDbgRegion (const Region *r,const BkgModelControlOpts &bkg_control);


// allocate compute resources for our analysis
struct ComputationPlanner
{
  int numBkgWorkers;
  int numBkgWorkers_gpu;
  int numGpuWorkers;
  bool use_gpu_acceleration;
  float gpu_work_load;
  bool use_all_gpus;
  std::vector<int> valid_devices;
  regionProcessOrderVector region_order;
  // dummy
  int lastRegionToProcess;
  bool use_gpu_only_fitting;
  bool gpu_multiflow_fit;
  bool gpu_singleflow_fit;

  ComputationPlanner() 
  {
    numBkgWorkers = 0;
    numBkgWorkers_gpu = 0;
    numGpuWorkers = 0;
    use_gpu_acceleration = false;
    gpu_work_load = 0;
    use_all_gpus = false;
    lastRegionToProcess = 0;
    use_gpu_only_fitting = true;
    gpu_multiflow_fit = true;
    gpu_singleflow_fit = true;
  }
};

/* // Boost serialization support:
template<class Archive>
void serialize(Archive& ar, ComputationPlanner& p, const unsigned int version)
{
  ar
  & p.numBkgWorkers
  & p.numDynamicBkgWorkers
  & p.numBkgWorkers_gpu
  & p.use_gpu_acceleration
  & p.gpu_work_load
  & p.use_all_gpus
  & p.valid_devices
  & p.region_order
  & p.lastRegionToProcess;
*/

void PlanMyComputation (ComputationPlanner &my_compute_plan, BkgModelControlOpts &bkg_control );
void SetRegionProcessOrder (int numRegions, SignalProcessingMasterFitter** fitters, ComputationPlanner &analysis_compute_plan);
void AllocateProcessorQueue (ProcessorQueue &my_queue,ComputationPlanner &analysis_compute_plan, int numRegions);
void AssignQueueForItem (ProcessorQueue &analysis_queue,ComputationPlanner &analysis_compute_plan);

void WaitForRegionsToFinishProcessing (ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan);
//void SpinDownCPUthreads (ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan);
bool UseGpuAcceleration(float useGpuFlag);

void DoInitialBlockOfFlowsAllBeadFit (WorkerInfoQueueItem &item);
void DoSingleFlowFitAndPostProcessing(WorkerInfoQueueItem &item);



#endif // SIGNALPROCESSINGFITTERQUEUE_H
