/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SignalProcessingFitterQueue.h"
#include "BkgMagicDefines.h"
#include "FlowSequence.h"
#include <iostream>
#include <fstream>
#include "BkgFitterTracker.h"

using namespace std;

static void DoConstructSignalProcessingFitterAndData (WorkerInfoQueueItem &item);
static void DoMultiFlowRegionalFit (WorkerInfoQueueItem &item);
static void DoInitialBlockOfFlowsRemainingRegionalFit (WorkerInfoQueueItem &item);
static void DoPostFitProcessing (WorkerInfoQueueItem &item);
//static WorkerInfoQueueItem TryGettingFittingJobForCpuFromQueue(ProcessorQueue* pq, WorkerInfoQueue** curQ);

bool sortregionProcessOrderVector (const beadRegion& r1, const beadRegion& r2)
{
  return (r1.second > r2.second);
}

// BkgWorkers to be created as threads

void *BkgFitWorkerCpu(void *arg)
{
  ProcessorQueue* pq = static_cast<ProcessorQueue*>(arg);
  assert(pq);

  WorkerInfoQueue* curQ = NULL;
  bool done = false;
  WorkerInfoQueueItem item;
  while (!done)
  {
    //item = TryGettingFittingJobForCpuFromQueue(pq, &curQ);
    item = pq->TryGettingFittingJob(&curQ);
    if (item.finished == true)
    {
      // we are no longer needed...go away!
      done = true;
      curQ->DecrementDone();
      continue;
    }

    int event = * ( (int *) item.private_data);

    if (event == MULTI_FLOW_REGIONAL_FIT)
    {
      DoMultiFlowRegionalFit(item);
    }
    else if (event == INITIAL_FLOW_BLOCK_ALLBEAD_FIT)
    {
      DoInitialBlockOfFlowsAllBeadFit(item);
    }
    else if (event == INITIAL_FLOW_BLOCK_REMAIN_REGIONAL_FIT)
    {
      DoInitialBlockOfFlowsRemainingRegionalFit(item);
    }
    else if (event == SINGLE_FLOW_FIT) {
      DoSingleFlowFitAndPostProcessing(item);
    }
    else if (event == POST_FIT_STEPS) {
      DoPostFitProcessing(item);
    }
    else if (event == imageInitBkgModel)
    {
      DoConstructSignalProcessingFitterAndData (item);
    }

    // indicate we finished that bit of work
    curQ->DecrementDone();
  }

  return (NULL);
}

void DoMultiFlowRegionalFit (WorkerInfoQueueItem &item) {
  BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);
  FlowBlockSequence::const_iterator flowBlock =
    info->inception_state->bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( info->flow );

  info->bkgObj->InitializeFlowBlock( flowBlock->size() );

  info->bkgObj->ProcessImage (info->img, info->flow, info->flow - flowBlock->begin(),
                                flowBlock->size() );
  // execute block if necessary
  if (info->bkgObj->TestAndTriggerComputation (info->last)) 
  {
    info->bkgObj->MultiFlowRegionalFitting (info->flow, info->last,
      info->flow_key, flowBlock->size(), info->table, flowBlock->begin() ); // CPU based regional fit
    if ( info->inception_state->bkg_control.signal_chunks.flow_block_sequence.
          HasFlowInFirstFlowBlock( info->flow ))
    {
      info->type = INITIAL_FLOW_BLOCK_ALLBEAD_FIT; 
      //if (info->pq->GetGpuQueue() && info->pq->performGpuMultiFlowFitting())
      //  info->pq->GetGpuQueue()->PutItem(item);
      //else
      //  info->pq->GetCpuQueue()->PutItem(item);
      info->QueueControl->AssignMultiFLowFitItemToQueue(item);
    }
    else
    {
      info->type = SINGLE_FLOW_FIT;
      //if (info->pq->GetGpuQueue() && info->pq->performGpuSingleFlowFitting())
      //  info->pq->GetGpuQueue()->PutItem(item);
      //else
      //  info->pq->GetCpuQueue()->PutItem(item);
      info->QueueControl->AssignSingleFLowFitItemToQueue(item);
    }
  }
}

void DoInitialBlockOfFlowsAllBeadFit(WorkerInfoQueueItem &item)
{
  BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);
  FlowBlockSequence::const_iterator flowBlock =
    info->inception_state->bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( info->flow );
  info->bkgObj->FitAllBeadsForInitialFlowBlock( KEY_LEN, flowBlock->size(), info->table, flowBlock->begin() );
  info->type = INITIAL_FLOW_BLOCK_REMAIN_REGIONAL_FIT; 
  //info->pq->GetCpuQueue()->PutItem(item);
  info->QueueControl->AssignItemToQueue(item);
}

void DoInitialBlockOfFlowsRemainingRegionalFit(WorkerInfoQueueItem &item)
{
  // printf("=====> Remaining fit steps job on CPU\n");
  BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);
  FlowBlockSequence::const_iterator flowBlock =
    info->inception_state->bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( info->flow );
  info->bkgObj->RemainingFitStepsForInitialFlowBlock( KEY_LEN, flowBlock->size(), info->table, flowBlock->begin() );
  info->type = SINGLE_FLOW_FIT; 
  //if (info->pq->GetGpuQueue() && info->pq->performGpuSingleFlowFitting())
  //  info->pq->GetGpuQueue()->PutItem(item);
  //else
  //  info->pq->GetCpuQueue()->PutItem(item);
  info->QueueControl->AssignSingleFLowFitItemToQueue(item);
}

void DoPostFitProcessing(WorkerInfoQueueItem &item) {
  // printf("=====> Post Processing job on CPU\n");
  BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);
  FlowBlockSequence::const_iterator flowBlock =
    info->inception_state->bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( info->flow );
  bool ewscale_correct = info->img->isEmptyWellAmplitudeAvailable();
  int flow_block_id = info->inception_state->bkg_control.signal_chunks.flow_block_sequence.FlowBlockIndex( info->flow );
  info->bkgObj->PreWellCorrectionFactors( ewscale_correct, flowBlock->size(), flowBlock->begin() ); // correct data for anything needed

  // export to wells,debug, etc, reset for new set of traces
  info->bkgObj->ExportAllAndReset(info->flow, info->last, flowBlock->size(), info->polyclonal_filter_opts, flow_block_id, flowBlock->begin() ); 
}

void DoSingleFlowFitAndPostProcessing(WorkerInfoQueueItem &item) {
  BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);
  FlowBlockSequence::const_iterator flowBlock =
    info->inception_state->bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( info->flow );
  bool ewscale_correct = info->img->isEmptyWellAmplitudeAvailable();
  int flow_block_id = info->inception_state->bkg_control.signal_chunks.flow_block_sequence.FlowBlockIndex( info->flow );
  // info->bkgObj->AllocResDataCube(info->inception_state->loc_context.regionXSize, info->inception_state->loc_context.regionYSize, flowBlock->size());
  info->bkgObj->FitEmbarassinglyParallelRefineFit( flowBlock->size(), flowBlock->begin() );
  info->bkgObj->PreWellCorrectionFactors( ewscale_correct, flowBlock->size(), flowBlock->begin() ); // correct data for anything needed
  
  // export to wells,debug, etc, reset for new set of traces
  info->bkgObj->ExportAllAndReset(info->flow,info->last, flowBlock->size(), info->polyclonal_filter_opts, flow_block_id, flowBlock->begin() ); 
}

void DoConstructSignalProcessingFitterAndData (WorkerInfoQueueItem &item)
{
  ImageInitBkgWorkInfo *info = (ImageInitBkgWorkInfo *) (item.private_data);
  int r = info->r;

  // assign a CPU fitter for this particular patch

  // should we enable debug for this region?
  bool reg_debug_enable;
  reg_debug_enable = CheckBkgDbgRegion (&info->regions[r],info->inception_state->bkg_control);
//@TODO: get rid of >control< options on initializer that don't affect allocation or initial computation
// Sweep all of those into a flag-setting operation across all the fitters, or send some of then to global-defaults
  SignalProcessingMasterFitter *local_fitter = new SignalProcessingMasterFitter (
          info->sliced_chip[r], 
          info->sliced_chip_extras[r],
          *info->global_defaults, 
          info->results_folder, 
          info->maskPtr,
          info->pinnedInFlow, 
          info->rawWells, 
          &info->regions[r], 
          *info->sample, 
          *info->sep_t0_estimate,
          reg_debug_enable, 
          info->inception_state->loc_context.rows, 
          info->inception_state->loc_context.cols,
          info->maxFrames,
          info->uncompFrames,
          info->timestamps, 
          info->emptyTraceTracker,
          info->t_sigma,
          info->t_mid_nuc,
          info->t0_frame,
          info->nokey,
	  			info->seqList,
          info->numSeqListItems,
          info->restart,
          info->washout_flow,
          info->inception_state
          );

  local_fitter->SetPoissonCache (info->math_poiss);
  local_fitter->SetComputeControlFlags (info->inception_state->bkg_control.enable_trace_xtalk_correction);
  local_fitter->SetPointers (info->ptrs);
  local_fitter->writeDebugFiles(info->inception_state->bkg_control.pest_control.bkg_debug_files);


  // now allocate fitters within the bkgmodel object
  // I'm expanding the constructor here so we can split the objects nicely
  //local_fitter->SetUpFitObjects();

  // put this fitter in the list
  info->signal_proc_fitters[r] = local_fitter;
}

bool CheckBkgDbgRegion (const Region *r, const BkgModelControlOpts &bkg_control)
{
  for (unsigned int i=0;i < bkg_control.pest_control.BkgTraceDebugRegions.size();i++)
  {
    const Region *dr = &bkg_control.pest_control.BkgTraceDebugRegions[i];

    if ( (dr->row >= r->row)
         && (dr->row < (r->row+r->h))
         && (dr->col >= r->col)
         && (dr->col < (r->col+r->w)))
    {
      return true;
    }
  }

  return false;
}


/*
void PlanMyComputation(
    ComputationPlanner &my_compute_plan, 
    BkgModelControlOpts &bkg_control)
{
  // -- Tuning parameters --

  // This will override gpuWorkLoad=1 and will only use GPU for chips which are allowed in the following function
  my_compute_plan.use_gpu_acceleration = UseGpuAcceleration(bkg_control.gpuControl.gpuWorkLoad);
  my_compute_plan.gpu_work_load = bkg_control.gpuControl.gpuWorkLoad;
  my_compute_plan.lastRegionToProcess = 0;
  // Option to use all GPUs in system (including display devices). If set to true, will only use the
  // devices with the highest computer version. For example, if you have a system with 4 Fermi compute
  // devices and 1 Quadro (Tesla based) for display, only the 4 Fermi devices will be used.
  my_compute_plan.use_all_gpus = bkg_control.gpuControl.gpuUseAllDevices;

  // force to run on user supplied gpu device id's
  if (bkg_control.gpuControl.gpuDeviceIds.size() > 0) {
    my_compute_plan.valid_devices.clear();
    my_compute_plan.valid_devices = bkg_control.gpuControl.gpuDeviceIds;
  }
  
  if (configureGpu (
          my_compute_plan.use_gpu_acceleration, 
          my_compute_plan.valid_devices,
          my_compute_plan.use_all_gpus,
          my_compute_plan.numBkgWorkers_gpu))
  {
    my_compute_plan.use_gpu_only_fitting = bkg_control.gpuControl.doGpuOnlyFitting;
    my_compute_plan.gpu_multiflow_fit = bkg_control.gpuControl.gpuMultiFlowFit;
    my_compute_plan.gpu_singleflow_fit = bkg_control.gpuControl.gpuSingleFlowFit;
    printf ("use_gpu_acceleration: %d\n", my_compute_plan.use_gpu_acceleration);
  }
  else
  {
    my_compute_plan.use_gpu_acceleration = false;
    my_compute_plan.gpu_work_load = 0;
    bkg_control.gpuControl.gpuFlowByFlowExecution = false;
  }

  if (bkg_control.signal_chunks.numCpuThreads)
  {
    // User specified number of threads:
    my_compute_plan.numBkgWorkers = bkg_control.signal_chunks.numCpuThreads;
  }
  else
  {
    // Limit threads to 1.5 * number of cores, with minimum of 4 threads:
    //my_compute_plan.numBkgWorkers = my_compute_plan.use_gpu_acceleration ? numCores() 
    //                                      : std::max (4, 3 * numCores() / 2);
    my_compute_plan.numBkgWorkers = std::max (4, 3 * numCores() / 2);
  }
}


void AllocateProcessorQueue (ProcessorQueue &my_queue,ComputationPlanner &analysis_compute_plan, int numRegions)
{
  //create queue for passing work to thread pool
 

  my_queue.SetCpuQueue(new WorkerInfoQueue (numRegions*analysis_compute_plan.numBkgWorkers+1));
  if (analysis_compute_plan.use_gpu_acceleration) {
    if(analysis_compute_plan.numBkgWorkers_gpu) {
      my_queue.AllocateGpuInfo(analysis_compute_plan.numBkgWorkers_gpu);
      my_queue.SetGpuQueue(new WorkerInfoQueue (numRegions*analysis_compute_plan.numBkgWorkers_gpu+1));

    }
  }
  
  // decide on whether to use both CPU and GPU for bkg model fitting jobs
  if (analysis_compute_plan.use_gpu_only_fitting) {
    my_queue.turnOffHeterogeneousComputing();
  }

  if (!analysis_compute_plan.gpu_multiflow_fit) {
    my_queue.turnOffGpuMultiFlowFitting();
  }

  if (!analysis_compute_plan.gpu_singleflow_fit) {
    my_queue.turnOffGpuSingleFlowFitting();
  }

  {
    int cworker;
    pthread_t work_thread;

    // spawn threads for doing background correction/fitting work
    for (cworker = 0; cworker < analysis_compute_plan.numBkgWorkers; cworker++)
    {
      int t = pthread_create (&work_thread, NULL, BkgFitWorkerCpu,
                              &my_queue);
      pthread_detach(work_thread);
      if (t)
        fprintf (stderr, "Error starting thread\n");
    }

  }

  fprintf (stdout, "Number of CPU threads for beadfind: %d\n", analysis_compute_plan.numBkgWorkers);
  if (analysis_compute_plan.use_gpu_acceleration)
    fprintf (stdout, "Number of GPU threads for background model: %d\n", analysis_compute_plan.numBkgWorkers_gpu);
  else
    fprintf (stdout, "Number of CPU threads for background model: %d\n", analysis_compute_plan.numBkgWorkers);
}

void WaitForRegionsToFinishProcessing (ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan)
{
  // wait for all of the regions to finish processing before moving on to the next
  // image
  // Need better logic...This is just following the different steps involved in signal processing
  analysis_queue.GetCpuQueue()->WaitTillDone();
  if (analysis_queue.GetGpuQueue())
    analysis_queue.GetGpuQueue()->WaitTillDone();
  analysis_queue.GetCpuQueue()->WaitTillDone();
  if (analysis_queue.GetGpuQueue())
    analysis_queue.GetGpuQueue()->WaitTillDone();
//  if (analysis_queue.GetSingleFitGpuQueue())
//    analysis_queue.GetSingleFitGpuQueue()->WaitTillDone();
  if (analysis_compute_plan.use_gpu_acceleration)
    analysis_queue.GetCpuQueue()->WaitTillDone();
}

*/
/*

void ProcessorQueue::SpinUpGPUThreads(ComputationPlanner &analysis_compute_plan )
{
  if (analysis_compute_plan.use_gpu_acceleration) {
    // create gpu thread for multi flow fit
    CreateGpuThreadsForFitType(GetGpuInfo(),
        GetGpuQueue(),
    		GetCpuQueue(),
     		analysis_compute_plan.numBkgWorkers_gpu,
    		analysis_compute_plan.valid_devices
        );
  }
}



void ProcessorQueue::CreateGpuThreadsForFitType(
    std::vector<BkgFitWorkerGpuInfo> &gpuInfo, 
    WorkerInfoQueue* q,
    WorkerInfoQueue* fallbackQ,
    int numWorkers, 
    std::vector<int> &gpus
  )
{
  int threadsPerDevice = numWorkers / gpus.size();
  for (int i = 0; i < numWorkers; i++)
  {
    pthread_t work_thread;

    int deviceId = i / threadsPerDevice;

    gpuInfo[i].gpu_index = gpus[deviceId];
    gpuInfo[i].queue = (void*) q;
    gpuInfo[i].fallbackQueue = (void*) fallbackQ;

    // Spawn GPU workers pulling items from either the combined queue (dynamic)
    // or a separate GPU queue (static)
    int t = pthread_create (&work_thread, NULL, BkgFitWorkerGpu, &gpuInfo[i]);
    pthread_detach(work_thread);
    if (t)
      fprintf (stderr, "Error starting GPU thread\n");
  }
}


void AssignQueueForItem (ProcessorQueue &analysis_queue,ComputationPlanner &analysis_compute_plan)
{
  (void) analysis_compute_plan;
  analysis_queue.GetCpuQueue()->PutItem(analysis_queue.item);
}

bool UseGpuAcceleration(float useGpuFlag) {
  if (useGpuFlag) {
    if (ChipIdDecoder::BigEnoughForGPU())
      return true;
    else {
      printf("GPU acceleration suppressed on small chips\n");
      return false;
    }
  }
  return false;
}

WorkerInfoQueueItem TryGettingFittingJobForCpuFromQueue(ProcessorQueue* pq, WorkerInfoQueue** curQ)
{
  WorkerInfoQueueItem item;
  if (pq->useHeterogenousCompute()) {
    std::vector<WorkerInfoQueue*>& queues = pq->GetQueues();
    for (unsigned int i=0; i<queues.size(); ++i) {
      if (queues[i]) {
        item = queues[i]->TryGetItem();
        if (item.private_data != NULL) 
        {
          *curQ = queues[i];
          return item;
        }
      }
    }
  }
  item = pq->GetCpuQueue()->GetItem();
  *curQ = pq->GetCpuQueue();
  return item;
}

*/




void ProcessorQueue::WaitForRegionsToFinishProcessing()
{
  // wait for all of the regions to finish processing before moving on to the next
  // image
  // Need better logic...This is just following the different steps involved in signal processing
  GetQueue()->WaitTillDone();
  if (GetGpuQueue())
    GetGpuQueue()->WaitTillDone();
  GetQueue()->WaitTillDone();
  if (GetGpuQueue())
    GetGpuQueue()->WaitTillDone();
//  if (analysis_queue.GetSingleFitGpuQueue())
//    analysis_queue.GetSingleFitGpuQueue()->WaitTillDone();
  if (GetGpuQueue())
    GetQueue()->WaitTillDone();
}



void ProcessorQueue::createWorkQueue(int numRegions)
{
  if(workQueue == NULL)
    workQueue = new WorkerInfoQueue(numRegions*getNumWorkers()+1);
}

void ProcessorQueue::destroyWorkQueue()
{
  if(workQueue)
    delete workQueue;
  workQueue = NULL;
}



void ProcessorQueue::configureQueue(BkgModelControlOpts &bkg_control)
{
  if (bkg_control.signal_chunks.numCpuThreads)
  {
    // User specified number of threads:
    setNumWorkers(bkg_control.signal_chunks.numCpuThreads);
  }
  else
  {    // Limit threads to 1.5 * number of cores, with minimum of 4 threads:
    //my_compute_plan.numBkgWorkers = my_compute_plan.use_gpu_acceleration ? numCores()
    //                                      : std::max (4, 3 * numCores() / 2);
    setNumWorkers(std::max (4, 3 * numCores() / 2) );
  }

  // queue control with regards to gpu jobs
  gpuMultiFlowFitting = bkg_control.gpuControl.gpuMultiFlowFit;  
  gpuSingleFlowFitting = bkg_control.gpuControl.gpuSingleFlowFit;
}



void ProcessorQueue::SpinUpWorkerThreads()
{
    int cworker;
    pthread_t work_thread;

    // spawn threads for doing background correction/fitting work
    for (cworker = 0; cworker < getNumWorkers(); cworker++)
    {
      int t = pthread_create (&work_thread, NULL, BkgFitWorkerCpu, this);
      pthread_detach(work_thread);
      if (t)
        fprintf (stderr, "Error starting thread\n");
    }
}

/*

void ProcessorQueue::CreateGpuThreadsForFitType(
    std::vector<BkgFitWorkerGpuInfo> &gpuInfo,
    WorkerInfoQueue* q,
    WorkerInfoQueue* fallbackQ,
    int numWorkers,
    std::vector<int> &gpus
  )
{
  int threadsPerDevice = numWorkers / gpus.size();
  for (int i = 0; i < numWorkers; i++)
  {
    pthread_t work_thread;

    int deviceId = i / threadsPerDevice;

    gpuInfo[i].gpu_index = gpus[deviceId];
    gpuInfo[i].queue = (void*) q;
    gpuInfo[i].fallbackQueue = (void*) fallbackQ;

    // Spawn GPU workers pulling items from either the combined queue (dynamic)
    // or a separate GPU queue (static)
    int t = pthread_create (&work_thread, NULL, BkgFitWorkerGpu, &gpuInfo[i]);
    pthread_detach(work_thread);
    if (t)
      fprintf (stderr, "Error starting GPU thread\n");
  }
}
*/

void ProcessorQueue::UnSpinBkgModelThreads ()
{
  if (GetQueue())
  {
    WorkerInfoQueueItem item;
    item.finished = true;
    item.private_data = NULL;
    for (int i=0;i < numWorkers;i++)
      GetQueue()->PutItem (item);
    GetQueue()->WaitTillDone();

    destroyWorkQueue();

  }
}


void ProcessorQueue::CreateItemAndAssignItemToQueue(void * privateData){
  WorkerInfoQueueItem item;
  item.finished = false;
  item.private_data = privateData;
  AssignItemToQueue(item);
}


void ProcessorQueue::AssignItemToQueue (WorkerInfoQueueItem &item)
{
  GetQueue()->PutItem(item);
}


void ProcessorQueue::AssignMultiFLowFitItemToQueue(WorkerInfoQueueItem &item)
{
  if (GetGpuQueue() && performGpuMultiFlowFitting())
    GetGpuQueue()->PutItem(item);
  else
    GetQueue()->PutItem(item);
}

void ProcessorQueue::AssignSingleFLowFitItemToQueue(WorkerInfoQueueItem &item)
{
  if (GetGpuQueue() && performGpuSingleFlowFitting())
    GetGpuQueue()->PutItem(item);
  else
    GetQueue()->PutItem(item);
}



/* a little hacky since item dequeue happens after job completion in the caller
 * we have to let the caller know from which Queue you we got the job
 * hence the **curQ to point to the current queue
 */

WorkerInfoQueueItem ProcessorQueue::TryGettingFittingJob(WorkerInfoQueue** curQ)
{
  WorkerInfoQueueItem item;

  /*try to get an item from the work queue*/
  WorkerInfoQueue * pQ = GetQueue();
  assert(pQ);
  item = pQ->TryGetItem();
  if (item.private_data != NULL){
    *curQ = pQ;
    return item;
  }
  /*if heterogeneous execution try GPU Q if cpu queue was empty*/
  if (useHeterogenousCompute())
  {
    pQ = GetGpuQueue();
    assert(pQ);
    item = pQ->TryGetItem();
    if (item.private_data != NULL){
      *curQ = pQ;
      return item;
    }
  }

  /*if both tries came up empty wait on cpu Q */
  pQ = GetQueue();
  assert(pQ);
  *curQ = pQ;
  return pQ->GetItem();

}






