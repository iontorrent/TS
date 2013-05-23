/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SignalProcessingFitterQueue.h"
#include "BkgMagicDefines.h"
#include <iostream>
#include <fstream>

using namespace std;

void DoConstructSignalProcessingFitterAndData (WorkerInfoQueueItem &item);
void DoMultiFlowRegionalFit (WorkerInfoQueueItem &item);
void DoInitialBlockOfFlowsRemainingRegionalFit (WorkerInfoQueueItem &item);
void DoPostFitProcessing (WorkerInfoQueueItem &item);
WorkerInfoQueueItem TryGettingFittingJobForCpuFromQueue(ProcessorQueue* pq, WorkerInfoQueue** curQ);

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
    item = TryGettingFittingJobForCpuFromQueue(pq, &curQ);
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

void *SimpleBkgFitWorkerGpu(void *arg)
{
  WorkerInfoQueue *q = static_cast<WorkerInfoQueue *> (arg);
  assert (q);

  SimpleFitStreamExecutionOnGpu(q);

  return (NULL);
}


void DoMultiFlowRegionalFit (WorkerInfoQueueItem &item) {
  BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);

  if (info->doingSdat)
  {
    info->bkgObj->ProcessImage (* (info->sdat), info->flow);
  }
  else
  {
    info->bkgObj->ProcessImage (info->img, info->flow);
  }
  // execute block if necessary
  if (info->bkgObj->TestAndTriggerComputation (info->last)) 
  {
    info->bkgObj->MultiFlowRegionalFitting (info->flow, info->last); // CPU based regional fit
    if (info->flow < info->bkgObj->region_data->my_flow.numfb)
    {
      info->type = INITIAL_FLOW_BLOCK_ALLBEAD_FIT; 
      if (info->pq->GetGpuQueue() && info->pq->performGpuMultiFlowFitting())
        info->pq->GetGpuQueue()->PutItem(item);
      else
        info->pq->GetCpuQueue()->PutItem(item);
    }
    else
    {
      info->type = SINGLE_FLOW_FIT;
      if (info->pq->GetGpuQueue() && info->pq->performGpuSingleFlowFitting())
        info->pq->GetGpuQueue()->PutItem(item);
      else
        info->pq->GetCpuQueue()->PutItem(item);
    }
  }
}

void DoInitialBlockOfFlowsAllBeadFit(WorkerInfoQueueItem &item)
{
  //printf("=====> All bead fit job on CPU\n");
  BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);
  info->bkgObj->FitAllBeadsForInitialFlowBlock();
  info->type = INITIAL_FLOW_BLOCK_REMAIN_REGIONAL_FIT; 
  info->pq->GetCpuQueue()->PutItem(item);
}

void DoInitialBlockOfFlowsRemainingRegionalFit(WorkerInfoQueueItem &item)
{
  //printf("=====> Remaining fit steps job on CPU\n");
  BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);
  info->bkgObj->RemainingFitStepsForInitialFlowBlock();
  info->type = SINGLE_FLOW_FIT; 
  if (info->pq->GetGpuQueue() && info->pq->performGpuSingleFlowFitting())
    info->pq->GetGpuQueue()->PutItem(item);
  else
    info->pq->GetCpuQueue()->PutItem(item);
}

void DoPostFitProcessing(WorkerInfoQueueItem &item) {
  //printf("=====> Post Processing job on CPU\n");
  BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);
  info->bkgObj->PreWellCorrectionFactors(); // correct data for anything needed
  info->bkgObj->ExportAllAndReset(info->flow,info->last); // export to wells,debug, etc, reset for new set of traces
}

void DoSingleFlowFitAndPostProcessing(WorkerInfoQueueItem &item) {
  //printf("=====> CPU Single flow fit and post Processing job\n");
  BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);
  info->bkgObj->FitEmbarassinglyParallelRefineFit();
  info->bkgObj->PreWellCorrectionFactors(); // correct data for anything needed
  info->bkgObj->ExportAllAndReset(info->flow,info->last); // export to wells,debug, etc, reset for new set of traces
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
  SignalProcessingMasterFitter *local_fitter = new SignalProcessingMasterFitter (info->sliced_chip[r], 
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
                                                                                 info->washout_flow);

  local_fitter->SetPoissonCache (info->math_poiss);
  local_fitter->SetComputeControlFlags (info->inception_state->bkg_control.enableXtalkCorrection);
  local_fitter->SetPointers (info->ptrs);
  local_fitter->writeDebugFiles(info->inception_state->bkg_control.bkg_debug_files);


  // now allocate fitters within the bkgmodel object
  // I'm expanding the constructor here so we can split the objects nicely
  //local_fitter->SetUpFitObjects();

  // put this fitter in the list
  info->signal_proc_fitters[r] = local_fitter;
}

bool CheckBkgDbgRegion (Region *r,BkgModelControlOpts &bkg_control)
{
  for (unsigned int i=0;i < bkg_control.BkgTraceDebugRegions.size();i++)
  {
    Region *dr = &bkg_control.BkgTraceDebugRegions[i];

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


void PlanMyComputation (ComputationPlanner &my_compute_plan, BkgModelControlOpts &bkg_control)
{
  // -- Tuning parameters --

  // This will override gpuWorkLoad=1 and will only use GPU for chips which are allowed in the following function
  my_compute_plan.use_gpu_acceleration = UseGpuAcceleration(bkg_control.gpuControl.gpuWorkLoad);
  my_compute_plan.gpu_work_load = bkg_control.gpuControl.gpuWorkLoad;
  my_compute_plan.lastRegionToProcess = 0;
  // Option to use all GPUs in system (including display devices). If set to true, will only use the
  // devices with the highest computer version. For example, if you have a system with 4 Fermi compute
  // devices and 1 Quadro (Tesla based) for display, only the 4 Fermi devices will be used.
  my_compute_plan.use_all_gpus = false;

  if (configureGpu (my_compute_plan.use_gpu_acceleration, my_compute_plan.valid_devices, my_compute_plan.use_all_gpus,
                    my_compute_plan.numBkgWorkers_gpu))
  {
    my_compute_plan.use_gpu_only_fitting = bkg_control.gpuControl.doGpuOnlyFitting;
    my_compute_plan.gpu_multiflow_fit = bkg_control.gpuControl.gpuMultiFlowFit;
    my_compute_plan.gpu_singleflow_fit = bkg_control.gpuControl.gpuSingleFlowFit;
    printf ("use_gpu_acceleration: %d\n", my_compute_plan.use_gpu_acceleration);

    //pass command line params for Kernel configuration
    configureKernelExecution(bkg_control.gpuControl);
  }
  else
  {
    my_compute_plan.use_gpu_acceleration = false;
    my_compute_plan.gpu_work_load = 0;
  }

  if (bkg_control.numCpuThreads)
  {
    // User specified number of threads:
    my_compute_plan.numBkgWorkers = bkg_control.numCpuThreads;
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

void SpinUpGPUThreads(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan)
{
  if (analysis_compute_plan.use_gpu_acceleration) {
    // create gpu thread for multi flow fit
    CreateGpuThreadsForFitType(analysis_queue.GetGpuInfo(),  
        analysis_compute_plan.numBkgWorkers_gpu, analysis_queue.GetGpuQueue(),
        analysis_compute_plan.valid_devices);
    // create gpu thread for single flow fit
/*    CreateGpuThreadsForFitType(analysis_queue.GetSingleFitGpuInfo(), GPU_SINGLE_FLOW_FIT, 
        analysis_compute_plan.numSingleFlowFitGpuWorkers, analysis_queue.GetSingleFitGpuQueue(),
        analysis_compute_plan.valid_devices);*/
  }
}

void CreateGpuThreadsForFitType(
    std::vector<BkgFitWorkerGpuInfo> &gpuInfo, 
  //  GpuFitType fittype, 
    int numWorkers, 
    WorkerInfoQueue* q,
    std::vector<int> &gpus)
{
  int threadsPerDevice = numWorkers / gpus.size();
  for (int i = 0; i < numWorkers; i++)
  {
    pthread_t work_thread;

    int deviceId = i / threadsPerDevice;

//    gpuInfo[i].type = fittype;
    gpuInfo[i].gpu_index = gpus[deviceId];
    gpuInfo[i].queue = (void*) q;

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
    ChipIdEnum Id = ChipIdDecoder::GetGlobalChipId();
    switch (Id) {
      case ChipId318:
      case ChipId900:
        return true;
      case ChipId314:
      case ChipId316:
      case ChipId316v2:    
      default:
      {
        printf("GPU acceleration turned off\n");
        return false;
      }
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
