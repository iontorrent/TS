/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SIGNALPROCESSINGFITTERQUEUE_H
#define SIGNALPROCESSINGFITTERQUEUE_H

#include <boost/serialization/utility.hpp>
#include "WorkerInfoQueue.h"
#include "RingBuffer.h"
#include "SignalProcessingMasterFitter.h"
#include "RawWells.h"

typedef std::pair<int, int> beadRegion;
typedef std::vector<beadRegion> regionProcessOrderVector;
bool sortregionProcessOrderVector (const beadRegion& r1, const beadRegion& r2);

struct BkgFitWorkerGpuInfo
{
  int gpu_index;
  void* queue;
  void* fallbackQueue;
};

/*
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
*/


class ProcessorQueue
{

    //this queue is owned by this class
    WorkerInfoQueue * workQueue;

    //this is just a handle to the gpu queue so jobs can be handed to this queue if needed
    WorkerInfoQueue * gpuQueue;

    int numWorkers;
    bool heterogeneousComputing; // use both gpu and cpu
    bool gpuMultiFlowFitting;
    bool gpuSingleFlowFitting;

protected:

    void CreateGpuThreadsForFitType(
        std::vector<BkgFitWorkerGpuInfo> &gpuInfo,
        WorkerInfoQueue* q,
        WorkerInfoQueue* fallbackQ,
        int numWorkers,
        std::vector<int> &gpus );

public:

  ProcessorQueue() {

    workQueue = NULL;
    gpuQueue = NULL;
    numWorkers = 6;
    heterogeneousComputing = false;
    gpuMultiFlowFitting = true;
    gpuSingleFlowFitting = true;
  }

  ~ProcessorQueue(){
    destroyWorkQueue();
  }


  void createWorkQueue(int numRegions);
  void destroyWorkQueue();
  void setGpuQueue(WorkerInfoQueue * GpuQ){gpuQueue = GpuQ;}

  void setNumWorkers(int numBkgWorker){ numWorkers = numBkgWorker; }
  int getNumWorkers(){ return numWorkers; }

  void configureQueue(BkgModelControlOpts &bkg_control);

  void turnOffHeterogeneousComputing() { heterogeneousComputing = false; }
  void turnOnHeterogeneousComputing() { heterogeneousComputing = true; }

  void turnOnGpuMultiFlowFitting() { gpuMultiFlowFitting = true; }
  void turnOffGpuMultiFlowFitting() { gpuMultiFlowFitting = false; }
  void turnOnGpuSingleFlowFitting() { gpuSingleFlowFitting = true; }
  void turnOffGpuSingleFlowFitting() { gpuSingleFlowFitting = false; }

  bool useHeterogenousCompute() { return heterogeneousComputing; }
  bool performGpuMultiFlowFitting() { return gpuMultiFlowFitting; }
  bool performGpuSingleFlowFitting() { return gpuSingleFlowFitting; }

  void SpinUpWorkerThreads();

  void UnSpinBkgModelThreads();

  WorkerInfoQueue* GetQueue() { return workQueue; }
  WorkerInfoQueue* GetGpuQueue() { return gpuQueue; }

  void initItem (void * itemData);

  void CreateItemAndAssignItemToQueue(void * itemData);

  void AssignItemToQueue (WorkerInfoQueueItem &item);
  void AssignMultiFLowFitItemToQueue(WorkerInfoQueueItem &item);
  void AssignSingleFLowFitItemToQueue(WorkerInfoQueueItem &item);

  void WaitForRegionsToFinishProcessing ();

  WorkerInfoQueueItem TryGettingFittingJob(WorkerInfoQueue** curQ);

};



struct BkgModelWorkInfo
{
  int type;
  SignalProcessingMasterFitter *bkgObj;
  int flow;
  Image *img;
  bool last;
  ProcessorQueue* QueueControl;
  int flow_key;
  PolyclonalFilterOpts polyclonal_filter_opts;
  master_fit_type_table *table;
  const CommandLineOpts *inception_state;
  const std::vector<float> *smooth_t0_est;
  void ** SampleCollection; // pointer to pointer so all bkinfo objects point to the same dynamically generated sample-collection
  //RingBuffer<float> *gpuAmpEstPerFlow;
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
  const std::vector<float> *smooth_t0_est;
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
  SlicedChipExtras *sliced_chip_extras;
  GlobalDefaultsForBkgModel *global_defaults;
  std::set<int> *sample;
  std::vector<float> *tauB;
  std::vector<float> *tauE;

  bool restart;
  int16_t *washout_flow;
};

// Some information needed by the helper thread which coordinates the handshake 
// between GPU and CPU for writing amplitudes estimates to rawwell buffers
/*struct GPUFlowByFlowPipelineInfo
{
  RingBuffer<float> *ampEstimatesBuf;
  std::vector<SignalProcessingMasterFitter*> *fitters;
  int startingFlow;
  int endingFlow;  
  SemQueue *packQueue;
  SemQueue *writeQueue;
  ChunkyWells *rawWells;
};
*/

void* BkgFitWorkerCpu (void *arg);
bool CheckBkgDbgRegion (const Region *r,const BkgModelControlOpts &bkg_control);


// allocate compute resources for our analysis
/*
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
*/
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

//void PlanMyComputation (ComputationPlanner &my_compute_plan, BkgModelControlOpts &bkg_control );
//void SetRegionProcessOrder (int numRegions, SignalProcessingMasterFitter** fitters, ComputationPlanner &analysis_compute_plan);
//void AllocateProcessorQueue (ProcessorQueue &my_queue,ComputationPlanner &analysis_compute_plan, int numRegions);
//void AssignQueueForItem (ProcessorQueue &analysis_queue,ComputationPlanner &analysis_compute_plan);

//void WaitForRegionsToFinishProcessing (ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan);
//void SpinDownCPUthreads (ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan);
//bool UseGpuAcceleration(float useGpuFlag);

void DoInitialBlockOfFlowsAllBeadFit (WorkerInfoQueueItem &item);
void DoSingleFlowFitAndPostProcessing(WorkerInfoQueueItem &item);



#endif // SIGNALPROCESSINGFITTERQUEUE_H
