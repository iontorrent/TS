/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BkgFitterTracker.h"
#include "BkgDataPointers.h"


bool sortregionProcessOrderVector (const beadRegion& r1, const beadRegion& r2)
{
  return (r1.second > r2.second);
}

dbgWellTracker::dbgWellTracker()
{
  bkgDbg1=NULL;
  bkgDbg2 = NULL;
  bkgDebugKmult = NULL;
}

void dbgWellTracker::Init (char *experimentName, int rows, int cols, int numFlows, char *flowOrder)
{
  bkgDbg1 = new RawWells (experimentName, "1.tau");
  bkgDbg1->CreateEmpty (numFlows, flowOrder, rows, cols);
  bkgDbg1->OpenForWrite();
  bkgDbg2 = new RawWells (experimentName, "1.lmres");
  bkgDbg2->CreateEmpty (numFlows, flowOrder, rows, cols);
  bkgDbg2->OpenForWrite();
  bkgDebugKmult = new RawWells (experimentName, "1.kmult");
  bkgDebugKmult->CreateEmpty (numFlows, flowOrder, rows, cols);
  bkgDebugKmult->OpenForWrite();
}

void dbgWellTracker::Close()
{
  if (bkgDbg1!=NULL)
  {
    bkgDbg1->Close();
    delete bkgDbg1;
  }
  if (bkgDbg2!=NULL)
  {
    bkgDbg2->Close();
    delete bkgDbg2;
  }
  if (bkgDebugKmult!=NULL)
  {
    bkgDebugKmult->Close();
    delete bkgDebugKmult;
  }
}
dbgWellTracker::~dbgWellTracker()
{
  Close();
}



extern void *DynamicBkgFitWorker (void *arg, bool use_gpu)
{
  DynamicWorkQueueGpuCpu *q = static_cast<DynamicWorkQueueGpuCpu*> (arg);
  assert (q);

  bool done = false;

  WorkerInfoQueueItem item;
  while (!done)
  {
    //printf("Waiting for item: %d\n", pthread_self());
    if (use_gpu)
      item = q->GetGpuItem();
    else
      item = q->GetCpuItem();

    if (item.finished == true)
    {
      // we are no longer needed...go away!
      done = true;
      q->DecrementDone();
      continue;
    }

    int event = * ( (int *) item.private_data);

    if (event == bkgWorkE)
    {
      BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);
      info->bkgObj->ProcessImage (info->img, info->flow, info->last,
                                  info->learning, use_gpu);
    }

    q->DecrementDone();
  }
  return (NULL);
}

// BkgWorkers to be created as threads
void* BkgFitWorkerCpu (void *arg)
{
  // Wrapper to create a worker
  return (BkgFitWorker (arg,false));
}

void* DynamicBkgFitWorkerCpu (void *arg)
{
  // Wrapper to create a worker
  return (DynamicBkgFitWorker (arg,false));
}

extern void *BkgFitWorker (void *arg, bool use_gpu)
{
  WorkerInfoQueue *q = static_cast<WorkerInfoQueue *> (arg);
  assert (q);

  bool done = false;

  while (!done)
  {
    WorkerInfoQueueItem item = q->GetItem();

    if (item.finished == true)
    {
      // we are no longer needed...go away!
      done = true;
      q->DecrementDone();
      continue;
    }

    int event = * ( (int *) item.private_data);

    if (event == bkgWorkE)
    {
      BkgModelWorkInfo *info = (BkgModelWorkInfo *) (item.private_data);
      info->bkgObj->ProcessImage (info->img, info->flow, info->last,
                                  info->learning, use_gpu);
    }
    else if (event == imageInitBkgModel)
    {
      ImageInitBkgWorkInfo *info = (ImageInitBkgWorkInfo *) (item.private_data);
      int r = info->r;

      // BkgModelReplay object is passed into BkgModel constructor which is
      // responsible for deleting it.  Constructed here instead of in
      // the BkgModel constructor as command line options are available here
      BkgModelReplay *replay;
      if (info->clo->bkg_control.replayBkgModelData){
  replay = new BkgModelReplayReader(*info->clo, info->regions[r].index);;
      }
      else if (info->clo->bkg_control.recordBkgModelData){
  replay = new BkgModelReplayRecorder(*info->clo, info->regions[r].index);
      }
      else {
  replay = new BkgModelReplay(true);
      }

      // should we enable debug for this region?
      bool reg_debug_enable;
      reg_debug_enable = CheckBkgDbgRegion (&info->regions[r],info->clo->bkg_control);
//@TODO: get rid of >control< options on initializer that don't affect allocation or initial computation
// Sweep all of those into a flag-setting operation across all the fitters, or send some of then to global-defaults
      BkgModel *bkgmodel = new BkgModel (info->experimentName, info->maskPtr,
                                         info->pinnedInFlow, info->rawWells, &info->regions[r], *info->sample, info->sep_t0_estimate,
                                         reg_debug_enable, info->clo->loc_context.rows, info->clo->loc_context.cols,
                                         info->maxFrames,info->uncompFrames,info->timestamps,info->math_poiss, info->emptyTraceTracker,
                                         replay,
                                         info->t_sigma,
                                         info->t_mid_nuc,
                                         info->clo->bkg_control.dntp_uM,
                                         info->clo->bkg_control.enableXtalkCorrection,
                                         info->clo->bkg_control.enableBkgModelClonalFilter,
                                         info->seqList,
                                         info->numSeqListItems
           );

      bkgmodel->SetPointers (info->ptrs);

      // @TODO: this is >absolutely< the wrong place to set these quantities
      // >only< quantities that take computation/ allocation should be set here
      // that is,we shouldn't divide the initialization between sections of the code
      bkgmodel->SetSingleFlowFitProjectionSearch (info->clo->bkg_control.useProjectionSearchForSingleFlowFit);
      if (info->clo->bkg_control.bkgDebugParam)
      {
        bkgmodel->SetParameterDebug (info->bkgDbg1,info->bkgDbg2,info->bkgDebugKmult);
      }

      if (info->clo->bkg_control.vectorize)
      {
        bkgmodel->performVectorization (true);
      }

      bkgmodel->SetAmplLowerLimit (info->clo->bkg_control.AmplLowerLimit);


      // put this fitter in the listh
      info->BkgModelFitters[r] = bkgmodel;

    }

    // indicate we finished that bit of work
    q->DecrementDone();
  }

  return (NULL);
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
  if (bkg_control.numCpuThreads)
  {
    // User specified number of threads:
    my_compute_plan.numBkgWorkers        = bkg_control.numCpuThreads;
    my_compute_plan.numDynamicBkgWorkers = bkg_control.numCpuThreads;
  }
  else
  {
    // Limit threads to 1.5 * number of cores, with minimum of 4 threads:
    my_compute_plan.numBkgWorkers        = max (4, 3 * numCores() / 2);
    my_compute_plan.numDynamicBkgWorkers = numCores();
  }
  //fprintf(stdout, "ProcessImageToWell: bkg_control.numCpuThreads = %d, numCores() = %d\n", bkg_control.numCpuThreads, numCores());
  fprintf (stdout, "ProcessImageToWell: numBkgWorkers = %d, numDynamicBkgWorkers = %d\n", my_compute_plan.numBkgWorkers, my_compute_plan.numDynamicBkgWorkers);
  my_compute_plan.numBkgWorkers_gpu = 0;

  // -- Tuning parameters --

  // Flag to disable CUDA acceleration
  my_compute_plan.use_gpu_acceleration = (bkg_control.gpuWorkLoad > 0) ? true : false;

  // Dynamic balance (set to true) will create one large queue for both the CPUs and GPUs to pull from
  // whenever the resource is free. A static balance (set to false) will create two queues, one for the
  // GPU to pull from and one for the CPU to pull from. This division will be set deterministically so
  // the same answer to achieved each time.
  my_compute_plan.dynamic_gpu_balance = bkg_control.useBothCpuAndGpu;
  // If static balance, what percent of queue items should be added to the GPU queue. Not that there is
  // only one queue for all the GPUs to share. It is not deterministic which GPU each work item will be
  // assigned to.  This will potentially lead to non-deterministic solution on systems with mixed GPU
  // generations where floating point arithmetic is slightly different (i.e. Tesla vs Fermi).
  my_compute_plan.gpu_work_load = bkg_control.gpuWorkLoad;
  my_compute_plan.lastRegionToProcess = 0;
  // Option to use all GPUs in system (including display devices). If set to true, will only use the
  // devices with the highest computer version. For example, if you have a system with 4 Fermi compute
  // devices and 1 Quadro (Tesla based) for display, only the 4 Fermi devices will be used.
  my_compute_plan.use_all_gpus = false;

  if (configureGpu (my_compute_plan.use_gpu_acceleration, my_compute_plan.valid_devices, my_compute_plan.use_all_gpus,
                    bkg_control.numGpuThreads, my_compute_plan.numBkgWorkers_gpu))
  {
    printf ("use_gpu_acceleration: %d\n", my_compute_plan.use_gpu_acceleration);
  }
  else
  {
    my_compute_plan.use_gpu_acceleration = false;
    my_compute_plan.gpu_work_load = 0;
  }
}



void AllocateProcessorQueue (ProcessorQueue &my_queue,ComputationPlanner &analysis_compute_plan, int numRegions)
{
  //create queue for passing work to thread pool
  my_queue.gpu_info.resize (analysis_compute_plan.numBkgWorkers_gpu);
  my_queue.threadWorkQ = NULL;
  my_queue.threadWorkQ_gpu = NULL;
  my_queue.dynamicthreadWorkQ = NULL;

  my_queue.threadWorkQ = new WorkerInfoQueue (numRegions*analysis_compute_plan.numBkgWorkers+1);
  if (analysis_compute_plan.use_gpu_acceleration && analysis_compute_plan.dynamic_gpu_balance)
  {
    // If we are using a dynamic work queue, make the queue large enough for both the CPU and GPU workers
    my_queue.dynamicthreadWorkQ = new DynamicWorkQueueGpuCpu (numRegions);
  }
  else
  {
    if (analysis_compute_plan.numBkgWorkers_gpu)
      my_queue.threadWorkQ_gpu = new WorkerInfoQueue (numRegions*analysis_compute_plan.numBkgWorkers_gpu+1);
  }

  {
    int cworker;
    pthread_t work_thread;

    // spawn threads for doing backgroun correction/fitting work
    for (cworker = 0; cworker < analysis_compute_plan.numBkgWorkers; cworker++)
    {
      int t = pthread_create (&work_thread, NULL, BkgFitWorkerCpu,
                              my_queue.threadWorkQ);
      if (t)
        fprintf (stderr, "Error starting thread\n");
    }

  }

  fprintf (stdout, "Number of CPU threads for beadfind: %d\n", analysis_compute_plan.numBkgWorkers);
  if (analysis_compute_plan.dynamic_gpu_balance)
  {
    fprintf (stdout, "Number of CPU threads for background model: %d\n", analysis_compute_plan.numDynamicBkgWorkers);
    fprintf (stdout, "Number of GPU threads for background model: %d\n", analysis_compute_plan.numBkgWorkers_gpu);
  }
  else
  {
    if (analysis_compute_plan.use_gpu_acceleration)
      fprintf (stdout, "Number of GPU threads for background model: %d\n", analysis_compute_plan.numBkgWorkers_gpu);
    else
      fprintf (stdout, "Number of CPU threads for background model: %d\n", analysis_compute_plan.numBkgWorkers);
  }
}

void WaitForRegionsToFinishProcessing (ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan,  int flow)
{
  // wait for all of the regions to finish processing before moving on to the next
  // image
  if (analysis_compute_plan.use_gpu_acceleration && analysis_compute_plan.dynamic_gpu_balance)
  {
    analysis_queue.dynamicthreadWorkQ->WaitTillDone();
  }
  else
  {
    analysis_queue.threadWorkQ->WaitTillDone();
  }
  if (analysis_queue.threadWorkQ_gpu)
    analysis_queue.threadWorkQ_gpu->WaitTillDone();

  if (analysis_compute_plan.use_gpu_acceleration && analysis_compute_plan.dynamic_gpu_balance)
  {
    int gpuRegions = analysis_queue.dynamicthreadWorkQ->getGpuReadIndex();
    printf ("Job ratio --> Flow: %d GPU: %f CPU: %f\n",
            flow, (float) gpuRegions/ (float) analysis_compute_plan.lastRegionToProcess,
            (1.0 - ( (float) gpuRegions/ (float) analysis_compute_plan.lastRegionToProcess)));
    analysis_queue.dynamicthreadWorkQ->ResetIndices();
  }

}

void SpinDownCPUthreads (ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan)
{
  // tell all the worker threads to exit
  if (analysis_compute_plan.use_gpu_acceleration && analysis_compute_plan.dynamic_gpu_balance)
  {
    analysis_queue.item.finished = true;
    analysis_queue.item.private_data = NULL;
    for (int i=0;i < analysis_compute_plan.numBkgWorkers;i++)
      analysis_queue.threadWorkQ->PutItem (analysis_queue.item);
    analysis_queue.threadWorkQ->WaitTillDone();
  }
}

void SpinUpGPUAndDynamicThreads (ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan)
{
  // create CPU threads for running dynamic gpu load since work items
  // are placed in a different queue
  if (analysis_compute_plan.use_gpu_acceleration && analysis_compute_plan.dynamic_gpu_balance)
  {
    pthread_t work_thread;
    for (int cworker = 0; cworker < analysis_compute_plan.numDynamicBkgWorkers; cworker++)
    {
      int t = pthread_create (&work_thread, NULL, DynamicBkgFitWorkerCpu,
                              analysis_queue.dynamicthreadWorkQ);
      if (t)
        fprintf (stderr, "Error starting dynamic CPU thread\n");
    }

  }

  for (int i = 0; i < analysis_compute_plan.numBkgWorkers_gpu; i++)
  {
    pthread_t work_thread;

    int deviceId = i / analysis_compute_plan.numBkgWorkers_gpu; // @TODO: this should be part of the analysis_compute_plan already, so clo is not needed here.

    analysis_queue.gpu_info[i].dynamic_gpu_load = analysis_compute_plan.dynamic_gpu_balance;
    analysis_queue.gpu_info[i].gpu_index = analysis_compute_plan.valid_devices[deviceId];
    analysis_queue.gpu_info[i].queue = (analysis_compute_plan.dynamic_gpu_balance ? (void*) analysis_queue.dynamicthreadWorkQ :
                                        (void*) analysis_queue.threadWorkQ_gpu);

    // Spawn GPU workers pulling items from either the combined queue (dynamic)
    // or a separate GPU queue (static)
    int t = pthread_create (&work_thread, NULL, BkgFitWorkerGpu, &analysis_queue.gpu_info[i]);
    if (t)
      fprintf (stderr, "Error starting GPU thread\n");
  }
}

void AssignQueueForItem (ProcessorQueue &analysis_queue,ComputationPlanner &analysis_compute_plan, int numRegions, int r)
{

  if (analysis_compute_plan.use_gpu_acceleration && analysis_compute_plan.dynamic_gpu_balance)
  {
    // Add all items to the shared queue
    analysis_queue.dynamicthreadWorkQ->PutItem (analysis_queue.item);
  }
  else
  {
    // Deterministically split items between the CPU
    // and GPU queues
    if (r >= int (analysis_compute_plan.gpu_work_load * float (numRegions)))
    {
      analysis_queue.threadWorkQ->PutItem (analysis_queue.item);
    }
    else
    {
      analysis_queue.threadWorkQ_gpu->PutItem (analysis_queue.item);
    }
  }
}

void BkgFitterTracker::SetRegionProcessOrder ()
{

  analysis_compute_plan.region_order.resize (numFitters);
  int numBeads;
  int zeroRegions = 0;
  for (int i=0; i<numFitters; ++i)
  {
    numBeads = BkgModelFitters[i]->GetNumLiveBeads();
    if (numBeads == 0)
      zeroRegions++;

    analysis_compute_plan.region_order[i] = beadRegion (i, numBeads);
  }
  std::sort (analysis_compute_plan.region_order.begin(), analysis_compute_plan.region_order.end(), sortregionProcessOrderVector);

  int nonZeroRegions = numFitters - zeroRegions;

  if (analysis_compute_plan.gpu_work_load != 0)
  {
    printf ("Number of live bead regions: %d\n", nonZeroRegions);

    int gpuRegions = int (analysis_compute_plan.gpu_work_load * float (nonZeroRegions));
    if (gpuRegions > 0)
      analysis_compute_plan.lastRegionToProcess = gpuRegions;
  }
}


void BkgFitterTracker::UnSpinGpuThreads ()
{

  if (analysis_queue.threadWorkQ_gpu)
  {
    WorkerInfoQueueItem item;
    item.finished = true;
    item.private_data = NULL;
    for (int i=0;i < analysis_compute_plan.numBkgWorkers_gpu;i++)
      analysis_queue.threadWorkQ_gpu->PutItem (item);
    analysis_queue.threadWorkQ_gpu->WaitTillDone();

    delete analysis_queue.threadWorkQ_gpu;
  }
}


BkgFitterTracker::BkgFitterTracker (int numRegions)
{
  numFitters = numRegions;
  BkgModelFitters = new BkgModel * [numRegions];
  bkinfo = NULL;
}

void BkgFitterTracker::DeleteFitters()
{
  for (int r = 0; r < numFitters; r++)
    delete BkgModelFitters[r];
  delete [] BkgModelFitters; // remove pointers
  BkgModelFitters = NULL;
  if (bkinfo!=NULL)
    delete[] bkinfo;
}

BkgFitterTracker::~BkgFitterTracker()
{
  numFitters = 0;
  DeleteFitters();
}

void BkgFitterTracker::PlanComputation (BkgModelControlOpts &bkg_control)
{
  // how are we dividing the computation amongst resources available as directed by command line constraints

  PlanMyComputation (analysis_compute_plan,bkg_control);

  AllocateProcessorQueue (analysis_queue,analysis_compute_plan,numFitters);
}



void BkgFitterTracker::ThreadedInitialization (RawWells &rawWells, CommandLineOpts &clo, Mask *maskPtr, PinnedInFlow *pinnedInFlow, char *experimentName,
     ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, Region *regions, int totalRegions, RegionTiming *region_timing,SeqListClass &my_keys,
                 BkgDataPointers *ptrs,EmptyTraceTracker &emptytracetracker, std::vector<float> *tauB,std::vector<float> *tauE)
{
  // designate a set of reads that will be processed regardless of whether they pass filters
  set<int> randomLibSet;
  MaskSample<int> randomLib (*maskPtr, MaskLib, clo.flt_control.nUnfilteredLib);
  randomLibSet.insert (randomLib.Sample().begin(), randomLib.Sample().end());

  // construct the shared math table
  poiss_cache.Allocate (MAX_HPLEN+1,MAX_POISSON_TABLE_ROW,POISSON_TABLE_STEP);
  poiss_cache.GenerateValues(); // fill out my table
  // make sure the GPU knows about this table
  if (analysis_compute_plan.use_gpu_acceleration)
  {
    InitConstantMemoryOnGpu (poiss_cache);
  }

  if (clo.bkg_control.bkgDebugParam)
    my_bkg_dbg_wells.Init (experimentName,my_image_spec.rows,my_image_spec.cols,clo.flow_context.numTotalFlows,clo.flow_context.flowOrder);  //@TODO: 3rd duplicated code instance

  ImageInitBkgWorkInfo *linfo = new ImageInitBkgWorkInfo[numFitters];
  for (int r = 0; r < numFitters; r++)
  {
    // load up the entire image, and do standard image processing (maybe all this 'standard' stuff could be on one call?)
    linfo[r].type = imageInitBkgModel;
    linfo[r].clo = &clo;
    linfo[r].r = r;
    linfo[r].BkgModelFitters = &BkgModelFitters[0];
    linfo[r].bkgDbg1 = my_bkg_dbg_wells.bkgDbg1;
    linfo[r].bkgDbg2 = my_bkg_dbg_wells.bkgDbg2;
    linfo[r].bkgDebugKmult = my_bkg_dbg_wells.bkgDebugKmult;
    linfo[r].rows = my_image_spec.rows;
    linfo[r].cols = my_image_spec.cols;
    linfo[r].maxFrames = clo.img_control.maxFrames;
    linfo[r].uncompFrames = my_image_spec.uncompFrames;
    linfo[r].timestamps = my_image_spec.timestamps;
    linfo[r].pinnedInFlow = pinnedInFlow;
    linfo[r].rawWells = &rawWells;
    linfo[r].emptyTraceTracker = &emptytracetracker;
    if (clo.bkg_control.bkgModelHdf5Debug)
    {
      linfo[r].ptrs = ptrs;
    }
    else
    {
      linfo[r].ptrs = NULL;
    }
    linfo[r].regions = &regions[0];
    linfo[r].numRegions = totalRegions;
    //linfo[r].kic = keyIncorporation;
    linfo[r].t_mid_nuc = region_timing[r].t_mid_nuc;
    linfo[r].t_sigma = region_timing[r].t_sigma;
    linfo[r].experimentName = experimentName;
    linfo[r].maskPtr = maskPtr;
    linfo[r].sep_t0_estimate = &smooth_t0_est;
    if (tauB != NULL)
      linfo[r].tauB = tauB;
    if (tauE != NULL)
      linfo[r].tauE = tauE;
    linfo[r].math_poiss = &poiss_cache;
    linfo[r].seqList = my_keys.seqList;
    linfo[r].numSeqListItems= my_keys.numSeqListItems;
    linfo[r].sample = &randomLibSet;

    analysis_queue.item.finished = false;
    analysis_queue.item.private_data = (void *) &linfo[r];
    analysis_queue.threadWorkQ->PutItem (analysis_queue.item);

  }
  // wait for all of the images to be loaded and initially processed
  analysis_queue.threadWorkQ->WaitTillDone();

  delete[] linfo;

  SpinDownCPUthreads (analysis_queue,analysis_compute_plan);
  // set up for flow-by-flow fitting
  bkinfo = new BkgModelWorkInfo[numFitters];

}


// call the fitters for each region
void BkgFitterTracker::ExecuteFitForFlow (int flow, ImageTracker &my_img_set, bool last)
{
  for (int r = 0; r < numFitters; r++)
  {
    // these get free'd by the thread that processes them
    bkinfo[r].type = bkgWorkE;
    bkinfo[r].bkgObj = BkgModelFitters[analysis_compute_plan.region_order[r].first];
    bkinfo[r].flow = flow;
    bkinfo[r].img = & (my_img_set.img[flow]);
    bkinfo[r].last = last;
    bkinfo[r].learning = false;

    analysis_queue.item.finished = false;
    analysis_queue.item.private_data = (void *) &bkinfo[r];
    AssignQueueForItem (analysis_queue,analysis_compute_plan,numFitters,r);
  }

  WaitForRegionsToFinishProcessing (analysis_queue,analysis_compute_plan, flow);
}

void BkgFitterTracker::SpinUp()
{
  SpinUpGPUAndDynamicThreads (analysis_queue,analysis_compute_plan);
}


void BkgFitterTracker::DumpBkgModelBeadParams (char *experimentName,  int flow, bool debug_bead_only)
{
  FILE *bkg_mod_bead_dbg = NULL;
  char *bkg_mod_bead_dbg_fname = (char *) malloc (512);
  snprintf (bkg_mod_bead_dbg_fname, 512, "%s/%s.%04d.%s", experimentName, "BkgModelBeadData",flow+1,"txt");
  fopen_s (&bkg_mod_bead_dbg, bkg_mod_bead_dbg_fname, "wt");
  free (bkg_mod_bead_dbg_fname);

  DumpBeadTitle (bkg_mod_bead_dbg);

  for (int r = 0; r < numFitters; r++)
  {
    BkgModelFitters[r]->DumpExemplarBead (bkg_mod_bead_dbg,debug_bead_only);
  }
  fclose (bkg_mod_bead_dbg);
}

void BkgFitterTracker::DumpBkgModelBeadOffset (char *experimentName, int flow, bool debug_bead_only)
{
  FILE *bkg_mod_bead_dbg = NULL;
  char *bkg_mod_bead_dbg_fname = (char *) malloc (512);
  snprintf (bkg_mod_bead_dbg_fname, 512, "%s/%s.%04d.%s", experimentName, "BkgModelBeadDcData",flow+1,"txt");
  fopen_s (&bkg_mod_bead_dbg, bkg_mod_bead_dbg_fname, "wt");
  free (bkg_mod_bead_dbg_fname);



  for (int r = 0; r < numFitters; r++)
  {
    BkgModelFitters[r]->DumpExemplarBeadDcOffset (bkg_mod_bead_dbg,debug_bead_only);
  }
  fclose (bkg_mod_bead_dbg);
}


void BkgFitterTracker::DumpBkgModelBeadInfo (char *experimentName,  int flow, bool last_flow, bool debug_bead_only)
{
  // get some regional data for the entire chip as debug
  // only do this every 20 flows as this is the block
  // should be triggered by bkgmodel
  if (CheckFlowForWrite (flow,last_flow))
  {
    DumpBkgModelBeadParams (experimentName, flow, debug_bead_only);
    DumpBkgModelBeadOffset (experimentName,  flow, debug_bead_only);
  }
}

void BkgFitterTracker::DumpBkgModelEmphasisTiming (char *experimentName, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_time_dbg = NULL;
  char *bkg_mod_time_name = (char *) malloc (512);
  snprintf (bkg_mod_time_name, 512, "%s/%s.%04d.%s", experimentName, "BkgModelEmphasisData",flow+1,"txt");
  fopen_s (&bkg_mod_time_dbg, bkg_mod_time_name, "wt");
  free (bkg_mod_time_name);

  for (int r = 0; r < numFitters; r++)
  {
    BkgModelFitters[r]->DumpTimeAndEmphasisByRegion (bkg_mod_time_dbg);
    BkgModelFitters[r]->DumpTimeAndEmphasisByRegionH5 (r);
  }
  fclose (bkg_mod_time_dbg);
}

void BkgFitterTracker::DumpBkgModelInitVals (char *experimentName, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_init_dbg = NULL;
  char *bkg_mod_init_name = (char *) malloc (512);
  snprintf (bkg_mod_init_name, 512, "%s/%s.%04d.%s", experimentName, "BkgModelInitVals",flow+1,"txt");
  fopen_s (&bkg_mod_init_dbg, bkg_mod_init_name, "wt");
  free (bkg_mod_init_name);

  for (int r = 0; r < numFitters; r++)
  {
    BkgModelFitters[r]->DumpInitValues (bkg_mod_init_dbg);

  }
  fclose (bkg_mod_init_dbg);
}

void BkgFitterTracker::DumpBkgModelDarkMatter (char *experimentName, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_dark_dbg = NULL;
  char *bkg_mod_dark_name = (char *) malloc (512);
  snprintf (bkg_mod_dark_name, 512, "%s/%s.%04d.%s", experimentName, "BkgModelDarkMatterData",flow+1,"txt");
  fopen_s (&bkg_mod_dark_dbg, bkg_mod_dark_name, "wt");
  free (bkg_mod_dark_name);

  BkgModelFitters[0]->DumpDarkMatterTitle (bkg_mod_dark_dbg);

  for (int r = 0; r < numFitters; r++)
  {
    BkgModelFitters[r]->DumpDarkMatter (bkg_mod_dark_dbg);

  }
  fclose (bkg_mod_dark_dbg);
}

void BkgFitterTracker::DumpBkgModelEmptyTrace (char *experimentName, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_mt_dbg = NULL;
  char *bkg_mod_mt_name = (char *) malloc (512);
  snprintf (bkg_mod_mt_name, 512, "%s/%s.%04d.%s", experimentName, "BkgModelEmptyTraceData",flow+1,"txt");
  fopen_s (&bkg_mod_mt_dbg, bkg_mod_mt_name, "wt");
  free (bkg_mod_mt_name);


  for (int r = 0; r < numFitters; r++)
  {
    BkgModelFitters[r]->DumpEmptyTrace (bkg_mod_mt_dbg);

  }
  fclose (bkg_mod_mt_dbg);
}

void BkgFitterTracker::DumpBkgModelRegionParameters (char *experimentName,int flow)
{
  FILE *bkg_mod_reg_dbg = NULL;
  char *bkg_mod_reg_dbg_fname = (char *) malloc (512);
  snprintf (bkg_mod_reg_dbg_fname, 512, "%s/%s.%04d.%s", experimentName, "BkgModelRegionData",flow+1,"txt");
  fopen_s (&bkg_mod_reg_dbg, bkg_mod_reg_dbg_fname, "wt");
  free (bkg_mod_reg_dbg_fname);

  struct reg_params rp;

  DumpRegionParamsTitle (bkg_mod_reg_dbg);

  for (int r = 0; r < numFitters; r++)
  {
    BkgModelFitters[r]->GetRegParams (rp);
    //@TODO this routine should have no knowledge of internal representation of variables
    // Make this a routine to dump an informative line to a selected file from a regional parameter structure/class
    // especially as we use a very similar dumping line a lot in different places.
    // note: t0, rdr, and pdr evolve over time.  It would be nice to also capture how they changed throughout the analysis
    // this only captures the final value of each.
    DumpRegionParamsLine (bkg_mod_reg_dbg, BkgModelFitters[r]->GetRegion()->row,BkgModelFitters[r]->GetRegion()->col, rp);

  }
  fclose (bkg_mod_reg_dbg);
}

void BkgFitterTracker::DumpBkgModelRegionInfo (char *experimentName, int flow, bool last_flow)
{
  // get some regional data for the entire chip as debug
  // only do this every 20 flows as this is the block
  // should be triggered by bkgmodel
  if (CheckFlowForWrite (flow,last_flow))
  {
    DumpBkgModelRegionParameters (experimentName, flow);
    DumpBkgModelDarkMatter (experimentName,  flow);
    DumpBkgModelEmphasisTiming (experimentName, flow);
    DumpBkgModelEmptyTrace (experimentName,flow);
    DumpBkgModelInitVals (experimentName, flow);
  }
}
