/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ProcessImageToWell.h"
#include "WellFileManipulation.h"

bool sortregionProcessOrderVector(const beadRegion& r1, const beadRegion& r2)
{
  return (r1.second > r2.second);
}


void ExportSubRegionSpecsToImage(CommandLineOpts &clo)
{
  // Default analysis mode sets values to 0 and whole-chip processing proceeds.
  // otherwise, command line override (--analysis-region) can define a subchip region.
  Image::chipSubRegion.row = (clo.GetChipRegion()).row;
  Image::chipSubRegion.col = (clo.GetChipRegion()).col;
  Image::chipSubRegion.h = (clo.GetChipRegion()).h;
  Image::chipSubRegion.w = (clo.GetChipRegion()).w;

}
void SetUpWholeChip(Region &wholeChip,int rows, int cols)
{
  //Used later to generate mask statistics for the whole chip
  wholeChip.row = 0;
  wholeChip.col = 0;
  wholeChip.w = cols;
  wholeChip.h = rows;
}

void UpdateBeadFindOutcomes(Mask *maskPtr, Region &wholeChip, char *experimentName, CommandLineOpts &clo, int update_stats)
{
  char maskFileName[2048];
  if (!update_stats)
  {
    sprintf(maskFileName, "%s/bfmask.stats", experimentName);
    maskPtr->DumpStats(wholeChip, maskFileName, !clo.SINGLEBF);
  }
  sprintf(maskFileName, "%s/bfmask.bin", experimentName);
  maskPtr->WriteRaw(maskFileName);
  maskPtr->validateMask();
}

void FixCroppedRegions(CommandLineOpts &clo, ImageSpecClass &my_image_spec)
{
  //If no cropped regions defined on command line, set cropRegions to whole chip
  if (!clo.cropRegions)
  {
    clo.numCropRegions = 1;
    clo.cropRegions = (Region *) malloc(sizeof(Region));
    SetUpWholeChip(clo.cropRegions[0],my_image_spec.rows,my_image_spec.cols);
  }
}



void SetExcludeMask(CommandLineOpts &clo, Mask *maskPtr, char *chipType, int rows, int cols)
{
  //Uncomment next line to revert to old exclusion mask file usage.  Remove once it is passed.
  #define OLDWAY
  
  bool applyExclusionMask = true;
  
  /*
   *  Determine if this is a cropped dataset
   *  3 types:
   *    wholechip image dataset - rows,cols should be == to chip_len_x,chip_len_y
   *    cropped image dataset - above test is false AND chip_offset_x == -1
   *    blocked image dataset - above test is false AND chip_offset_x != -1
   */
  if ((rows == clo.chip_len_y) && (cols == clo.chip_len_x)) {
    //This is a wholechip dataset
    applyExclusionMask = true;
    fprintf (stderr, "This is a wholechip dataset so the exclusion mask will be applied\n");
  }
  else {
    if (clo.chip_offset_x == -1){
      applyExclusionMask = false;
      fprintf (stderr, "This is a cropped dataset so the exclusion mask will not be applied\n");
    }
    else {
    #ifdef OLDWAY
      applyExclusionMask = false;
      fprintf (stderr, "This is a block dataset so the exclusion mask will not be applied\n");
    #else
      applyExclusionMask = true;
      fprintf (stderr, "This is a block dataset so the exclusion mask will be applied\n");
    #endif
    }
  }
  /*
   *  If we get a cropped region definition from the command line, we want the whole chip to be MaskExclude
   *  except for the defined crop region(s) which are marked MaskEmpty.  If no cropRegion defined on command line,
   *  then we proceed with marking the entire chip MaskEmpty
   */
  if (clo.numCropRegions == 0){
    maskPtr->Init(cols, rows, MaskEmpty);
  }
  else {
    maskPtr->Init(cols, rows, MaskExclude);

    // apply one or more cropRegions, mark them MaskEmpty
    for (int q = 0; q < clo.numCropRegions; q++)
    {
      maskPtr->MarkRegion(clo.cropRegions[q], MaskEmpty);
    }
  }
  
  /*
   * Apply exclude mask from file
   */
  clo.exclusionMaskSet = false;
  if (chipType && applyExclusionMask)
  {
    char *exclusionMaskFileName = NULL;
    char filename[64] = { 0 };
  #ifdef OLDWAY
    sprintf(filename, "exclusionMask_%s.bin", chipType);
  #else
    sprintf(filename, "excludeMask_%s", chipType);
  #endif
    exclusionMaskFileName = GetIonConfigFile(filename);
    fprintf(stderr, "Exclusion Mask File = '%s'\n", exclusionMaskFileName);
    if (exclusionMaskFileName)
    {
      clo.exclusionMaskSet = true;
    #ifdef OLDWAY
      Mask excludeMask(1, 1);
      excludeMask.SetMask(exclusionMaskFileName);
      free(exclusionMaskFileName);
      //--- Mark beadfind masks with MaskExclude bits from exclusionMaskFile
      maskPtr->SetThese(&excludeMask, MaskExclude);
    #else
      FILE *excludeFile = NULL;
      excludeFile = fopen (exclusionMaskFileName,"rb");
      assert(excludeFile != NULL);
      uint16_t x = 0;
      uint16_t y = 0;
      while (1){
        if (fread(&x, sizeof(x), 1, excludeFile) != 1) break;
        if (fread(&y, sizeof(y), 1, excludeFile) != 1) break;
        //fprintf (stderr, "Excluding %d %d (%d %d)\n",x,y,(int) x - clo.chip_offset_x,(int) y - clo.chip_offset_y);
        maskPtr->Set((int) x - clo.chip_offset_x,(int) y - clo.chip_offset_y,MaskExclude);
      }
    #endif
    }
    else {
      fprintf (stderr, "WARNING: Exclusion Mask %s not applied\n", filename);
    }
  }
}


// utility function
void MakeSymbolicLinkToOldDirectory(CommandLineOpts &clo, char *experimentName)
{
// Create symbolic link to bfmask.bin and 1.wells in new subdirectory: links are for disc space usage reasons
  char *oldpath = NULL;
  int sz = strlen(clo.wellsFilePath) + strlen(clo.wellsFileName) + 2;
  oldpath = (char *) malloc(sz);
  snprintf(oldpath, sz, "%s/%s", clo.wellsFilePath, clo.wellsFileName);
  char *fullPath = realpath(oldpath, NULL);

  char *newpath = NULL;
  sz = strlen(experimentName) + strlen(clo.wellsFileName) + 2;
  newpath = (char *) malloc(sz);
  snprintf(newpath, sz, "%s/%s", experimentName, clo.wellsFileName);

  int ret = symlink(fullPath, newpath);
  if (ret)
  {
    perror(oldpath);
  }
  free(oldpath);
  free(newpath);
  free(fullPath);
}



// fill the new directory with files needed for report generation
void CopyFilesForReportGeneration(CommandLineOpts &clo, char *experimentName, SequenceItem *seqList)
{
//--- Copy files needed for report generation ---
//--- Copy bfmask.stats ---
  int sz;
  char *newpath = NULL;
  char *oldpath = NULL;
  sz = strlen(clo.wellsFilePath) + strlen("bfmask.stats") + 2;
  oldpath = (char *) malloc(sz);
  snprintf(oldpath, sz, "%s/%s", clo.wellsFilePath, "bfmask.stats");
  sz = strlen(experimentName) + strlen("bfmask.stats") + 2;
  newpath = (char *) malloc(sz);
  snprintf(newpath, sz, "%s/%s", experimentName, "bfmask.stats");
  fprintf(stderr, "%s\n%s\n", oldpath, newpath);
  CopyFile(oldpath, newpath);
  free(oldpath);
  free(newpath);
//--- Copy avgNukeTrace_ATCG.txt and avgNukeTrace_TCAG.txt
  for (int q = 0; q < 2; q++)
  {
    char *filename;
    filename = (char *) malloc(strlen("avgNukeTrace_") + strlen(
                                 seqList[q].seq) + 5);
    sprintf(filename, "avgNukeTrace_%s.txt", seqList[q].seq);

    sz = strlen(clo.wellsFilePath) + strlen(filename) + 2;
    oldpath = (char *) malloc(sz);
    snprintf(oldpath, sz, "%s/%s", clo.wellsFilePath, filename);

    sz = strlen(experimentName) + strlen(filename) + 2;
    newpath = (char *) malloc(sz);
    snprintf(newpath, sz, "%s/%s", experimentName, filename);

    CopyFile(oldpath, newpath);
    free(oldpath);
    free(newpath);
    free(filename);
  }
}

void LoadBeadMaskFromFile(CommandLineOpts &clo, Mask *maskPtr, int &rows, int &cols)
{
  char maskFileName[2048];

  // Load beadmask from file
  sprintf(maskFileName, "%s/%s", clo.wellsFilePath, "./bfmask.bin");

  maskPtr->SetMask(maskFileName);
  if (maskPtr->SetMask(maskFileName))
  {
    exit(EXIT_FAILURE);
  }
  rows = maskPtr->H();
  cols = maskPtr->W();
  clo.rows = rows;
  clo.cols = cols;

  //--- Note that if we specify cropped regions on the command line, we are supposing that the original
  //    analysis was a whole chip analysis.  This is a safe assumption for the most part.
  // TODO: Need to rationalize the cropped region handling.

  clo.regionsX = 1;
  clo.regionsY = 1;

  if (clo.numRegions == 1)
  {
    int rx, ry;
    for (ry = 0; ry < rows; ry++)
    {
      for (rx = 0; rx < cols; rx++)
      {
        if (rx >= clo.regionXOrigin && rx < (clo.regionXOrigin
                                             + clo.regionXSize) && ry >= clo.regionYOrigin && ry
            < (clo.regionYOrigin + clo.regionYSize))
          ;
        else
          (*maskPtr)[rx + ry * cols] = MaskExclude;
      }
    }
  }

  clo.WriteProcessParameters();

  //--- Handle cropped regions defined from command line
  if (clo.numCropRegions > 0)
  {
    maskPtr->CropRegions(clo.cropRegions, clo.numCropRegions, MaskExclude);
  }

}


// distance to NN-smooth the t_zero (not t_mid_nuc!!!) estimate from the separator
#define SEPARATOR_T0_ESTIMATE_SMOOTH_DIST 15


// Brute-force NN averaging used to smooth the t0 estimate from the separator.  This algorithm can be sped-up
// considerably by sharing summations across multiple data points, similar to what is done in the image class for
// neighbor-subtracting images.
void NNSmoothT0Estimate(Mask *mask,int imgRows,int imgCols,std::vector<float> &sep_t0_est,std::vector<float> &output_t0_est)
{
  for (int r=0;r < imgRows;r++)
  {
    for (int c=0;c < imgCols;c++)
    {
      // OK..we're going to compute the Neighbor-average for the well at (r,c)
      float sum = 0.0;
      int nsum = 0;
      int lr = r-SEPARATOR_T0_ESTIMATE_SMOOTH_DIST;
      int ur = r+SEPARATOR_T0_ESTIMATE_SMOOTH_DIST;
      int lc = c-SEPARATOR_T0_ESTIMATE_SMOOTH_DIST;
      int uc = c+SEPARATOR_T0_ESTIMATE_SMOOTH_DIST;
      lr = (lr < 0?0:lr);
      lc = (lc < 0?0:lc);
      ur = (ur >= imgRows?imgRows-1:ur);
      uc = (uc >= imgCols?imgCols-1:uc);

      for (int sr=lr;sr <= ur;sr++)
        for (int sc=lc;sc <= uc;sc++)
          if (!mask->Match(sc,sr,(MaskType)(MaskPinned | MaskIgnore | MaskExclude)))
          {
            sum += sep_t0_est[sc+sr*imgCols];
            nsum++;
          }

      // if there we're no wells to form and average from, just copy the value from the original
      // un-smoothed vector
      if (nsum > 0)
        output_t0_est[c+r*imgCols] = sum / nsum;
      else
        output_t0_est[c+r*imgCols] = sep_t0_est[c+r*imgCols];
    }
  }
}


dbgWellTracker::dbgWellTracker()
{
  bkgDbg1=NULL;
  bkgDbg2 = NULL;
  bkgDebugKmult = NULL;
}

void dbgWellTracker::Init(char *experimentName, int rows, int cols, int numFlows, Flow *flw)
{
  bkgDbg1 = new RawWells(experimentName, "1.tau");
  bkgDbg1->CreateEmpty(numFlows, flw->GetFlowOrder(), rows, cols);
  bkgDbg1->OpenForWrite();
  bkgDbg2 = new RawWells(experimentName, "1.lmres");
  bkgDbg2->CreateEmpty(numFlows, flw->GetFlowOrder(), rows, cols);
  bkgDbg2->OpenForWrite();
  bkgDebugKmult = new RawWells(experimentName, "1.kmult");
  bkgDebugKmult->CreateEmpty(numFlows, flw->GetFlowOrder(), rows, cols);
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




extern void *DynamicBkgFitWorker(void *arg, bool use_gpu)
{
  DynamicWorkQueueGpuCpu *q = static_cast<DynamicWorkQueueGpuCpu*>(arg);
  assert(q);

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

    int event = *((int *) item.private_data);

    if (event == bkgWorkE)
    {
      BkgModelWorkInfo *info = (BkgModelWorkInfo *)(item.private_data);
      info->bkgObj->ProcessImage(info->img, info->flow, info->last,
                                 info->learning, use_gpu);
    }

    q->DecrementDone();
  }
  return (NULL);
}

// BkgWorkers to be created as threads
void* BkgFitWorkerCpu(void *arg)
{
  // Wrapper to create a worker
  return(BkgFitWorker(arg,false));
}

void* DynamicBkgFitWorkerCpu(void *arg)
{
  // Wrapper to create a worker
  return(DynamicBkgFitWorker(arg,false));
}

extern void *BkgFitWorker(void *arg, bool use_gpu)
{
  WorkerInfoQueue *q = static_cast<WorkerInfoQueue *>(arg);
  assert(q);

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

    int event = *((int *) item.private_data);

    if (event == bkgWorkE)
    {
      BkgModelWorkInfo *info = (BkgModelWorkInfo *)(item.private_data);
      info->bkgObj->ProcessImage(info->img, info->flow, info->last,
                                 info->learning, use_gpu);
    }
    else if (event == SeparatorWorkE)
    {
      SeparatorWorkInfo *winfo =
        (SeparatorWorkInfo *)(item.private_data);
      winfo->separator->FindBeads(winfo->img, winfo->region, winfo->mask,
                                  winfo->label);
    }
    else if (event == imageLoadE || event == imageLoadKeyE)
    {
      ImageLoadWorkInfo *info = (ImageLoadWorkInfo *)(item.private_data);
      Mask tMask(info->lmask);

      //Use Image Class object to read datafile, possibly filter data, then get raw data.
      //      info->img->LoadRaw(info->name);
      info->img->FilterForPinned(&tMask, MaskEmpty);
      info->img->Normalize(info->normStart, info->normEnd);

      // correct in-channel electrical cross-talk
      info->img->XTChannelCorrect(info->lmask);

      // Calculate average background for each well
      info->img->BackgroundCorrect(&tMask, MaskBead, MaskEmpty,
                                   info->NNinnerx, info->NNinnery, info->NNouterx,
                                   info->NNoutery, NULL, true);

      if (event == imageLoadKeyE)
      {
        // for each region, apply the signal extraction method (can be integral, pre-trainied vector, etc)
        // the separator methods may be called as threads, I'm thinking the separator object itself can manage this, and supply a 'wait for all threads' call
        for (int r = 0; r < info->numRegions; r++)
        {
          info->separator->CalcSignal(info->img, &info->regions[r],
                                      &tMask, MaskBead, info->flow,
                                      SeparatorSignal_Integral);
        }
        delete info->img;
        info->img = NULL;
      }
      info->finished = true;
    } 
    else if (event == imageInitBkgModel)
    {
      ImageInitBkgWorkInfo *info = (ImageInitBkgWorkInfo *)(item.private_data);
      int r = info->r;

      // should we enable debug for this region?
       bool reg_debug_enable;
     reg_debug_enable = CheckBkgDbgRegion(&info->regions[r],info->clo);

      BkgModel *bkgmodel = new BkgModel(info->experimentName, info->maskPtr,
                                        info->localMask, info->rawWells, &info->regions[r], *info->sample, info->sep_t0_estimate,
                                        reg_debug_enable, info->clo->rows, info->clo->cols,
                                        info->maxFrames,info->uncompFrames,info->timestamps,info->math_poiss,
                                        info->t_sigma,
                                        info->t_mid_nuc,
                                        info->clo->dntp_uM,
                                        info->clo->enableXtalkCorrection,
                                        info->clo->enableBkgModelClonalFilter,
                                        info->seqList,
			                    		info->numSeqListItems);

      if (info->clo->bkgDebugParam)
      {
        bkgmodel->SetParameterDebug(info->bkgDbg1,info->bkgDbg2,info->bkgDebugKmult);
      }

      if (info->clo->vectorize)
      {
        bkgmodel->performVectorization(true);
      }

      bkgmodel->SetAmplLowerLimit(info->clo->AmplLowerLimit);
      bkgmodel->SetKrateConstraintType(info->clo->relaxKrateConstraint);

      // put this fitter in the listh
      info->BkgModelFitters[r] = bkgmodel;

    }

    // indicate we finished that bit of work
    q->DecrementDone();
  }

  return (NULL);
}




bool CheckBkgDbgRegion(Region *r,CommandLineOpts *clo)
{
  for (unsigned int i=0;i < clo->BkgTraceDebugRegions.size();i++)
  {
    Region *dr = &clo->BkgTraceDebugRegions[i];

    if ((dr->row >= r->row)
        && (dr->row < (r->row+r->h))
        && (dr->col >= r->col)
        && (dr->col < (r->col+r->w)))
    {
      return true;
    }
  }

  return false;
}



void PlanComputation(ComputationPlanner &my_compute_plan, CommandLineOpts &clo)
{
  my_compute_plan.numBkgWorkers = numCores()*3;
  my_compute_plan.numBkgWorkers /=2;
  my_compute_plan.numDynamicBkgWorkers = clo.numCpuThreads;
  // Limit threads to half the number of cores with minimum of 2 threads
  my_compute_plan.numBkgWorkers = (my_compute_plan.numBkgWorkers > 4 ? my_compute_plan.numBkgWorkers : 4);
  my_compute_plan.numBkgWorkers_gpu=0;

  // -- Tuning parameters --

  // Flag to disable CUDA acceleration
  my_compute_plan.use_gpu_acceleration = (clo.gpuWorkLoad > 0) ? true : false;

// Dynamic balance (set to true) will create one large queue for both the CPUs and GPUs to pull from
  // whenever the resource is free. A static balance (set to false) will create two queues, one for the
  // GPU to pull from and one for the CPU to pull from. This division will be set deterministically so
  // the same answer to achieved each time.
  my_compute_plan.dynamic_gpu_balance = true;
  // If static balance, what percent of queue items should be added to the GPU queue. Not that there is
  // only one queue for all the GPUs to share. It is not deterministic which GPU each work item will be
  // assigned to.  This will potentially lead to non-deterministic solution on systems with mixed GPU
  // generations where floating point arithmetic is slightly different (i.e. Tesla vs Fermi).
  my_compute_plan.gpu_work_load = clo.gpuWorkLoad;
  my_compute_plan.lastRegionToProcess = 0;
  // Option to use all GPUs in system (including display devices). If set to true, will only use the
  // devices with the highest computer version. For example, if you have a system with 4 Fermi compute
  // devices and 1 Quadro (Tesla based) for display, only the 4 Fermi devices will be used.
  my_compute_plan.use_all_gpus = false;

  if (configureGpu(my_compute_plan.use_gpu_acceleration, my_compute_plan.valid_devices, my_compute_plan.use_all_gpus,
                   clo.numGpuThreads, my_compute_plan.numBkgWorkers_gpu))
  {
    printf("use_gpu_acceleration: %d\n", my_compute_plan.use_gpu_acceleration);
  }
  else
  {
    my_compute_plan.use_gpu_acceleration = false;
    my_compute_plan.gpu_work_load = 0;
  }

}

void SetRegionProcessOrder(int numRegions, BkgModel** fitters, ComputationPlanner &analysis_compute_plan)
{

  analysis_compute_plan.region_order.resize(numRegions);
  int numBeads;
  int zeroRegions = 0;
  for (int i=0; i<numRegions; ++i)
  {
    numBeads = fitters[i]->GetNumLiveBeads();
    if (numBeads == 0)
      zeroRegions++;

    analysis_compute_plan.region_order[i] = beadRegion(i, numBeads);
  }
  std::sort(analysis_compute_plan.region_order.begin(), analysis_compute_plan.region_order.end(), sortregionProcessOrderVector);

  int nonZeroRegions = numRegions - zeroRegions;

  if (analysis_compute_plan.gpu_work_load != 0)
  {
    printf("Number of live bead regions: %d\n", nonZeroRegions);

    int gpuRegions = int(analysis_compute_plan.gpu_work_load * float(nonZeroRegions));
    if (gpuRegions > 0)
      analysis_compute_plan.lastRegionToProcess = gpuRegions;
  }
}



void AllocateProcessorQueue(ProcessorQueue &my_queue,ComputationPlanner &analysis_compute_plan, int numRegions)
{
  //create queue for passing work to thread pool
  my_queue.gpu_info.resize(analysis_compute_plan.numBkgWorkers_gpu);
  my_queue.threadWorkQ = NULL;
  my_queue.threadWorkQ_gpu = NULL;
  my_queue.dynamicthreadWorkQ = NULL;

  my_queue.threadWorkQ = new WorkerInfoQueue(numRegions*analysis_compute_plan.numBkgWorkers+1);
  if (analysis_compute_plan.use_gpu_acceleration && analysis_compute_plan.dynamic_gpu_balance)
  {
    // If we are using a dynamic work queue, make the queue large enough for both the CPU and GPU workers
    my_queue.dynamicthreadWorkQ = new DynamicWorkQueueGpuCpu(numRegions);
  }
  else
  {
    if (analysis_compute_plan.numBkgWorkers_gpu)
      my_queue.threadWorkQ_gpu = new WorkerInfoQueue(numRegions*analysis_compute_plan.numBkgWorkers_gpu+1);
  }

  {
    int cworker;
    pthread_t work_thread;

    // spawn threads for doing backgroun correction/fitting work
    for (cworker = 0; cworker < analysis_compute_plan.numBkgWorkers; cworker++)
    {
      int t = pthread_create(&work_thread, NULL, BkgFitWorkerCpu,
                             my_queue.threadWorkQ);
      if (t)
        fprintf(stderr, "Error starting thread\n");
    }

  }

  fprintf(stdout, "Number of CPU threads for beadfind: %d\n", analysis_compute_plan.numBkgWorkers);
  if (analysis_compute_plan.use_gpu_acceleration)
  {
    fprintf(stdout, "Number of CPU threads for background model: %d\n", analysis_compute_plan.numDynamicBkgWorkers);
    fprintf(stdout, "Number of GPU threads for background model: %d\n", analysis_compute_plan.numBkgWorkers_gpu);
  }
  else
  {
    fprintf(stdout, "Number of CPU threads for background model: %d\n", analysis_compute_plan.numBkgWorkers);
  }
}

void WaitForRegionsToFinishProcessing(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan, CommandLineOpts &clo, int flow)
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
    printf("Job ratio --> Flow: %d GPU: %f CPU: %f\n",
           flow, (float)gpuRegions/(float)analysis_compute_plan.lastRegionToProcess,
           (1.0 - ((float)gpuRegions/(float)analysis_compute_plan.lastRegionToProcess)));
    analysis_queue.dynamicthreadWorkQ->ResetIndices();
  }

}

void SpinDownCPUthreads(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan)
{
  // tell all the worker threads to exit
  if (analysis_compute_plan.use_gpu_acceleration && analysis_compute_plan.dynamic_gpu_balance)
  {
    analysis_queue.item.finished = true;
    analysis_queue.item.private_data = NULL;
    for (int i=0;i < analysis_compute_plan.numBkgWorkers;i++)
      analysis_queue.threadWorkQ->PutItem(analysis_queue.item);
    analysis_queue.threadWorkQ->WaitTillDone();
  }
}

void SpinUpGPUAndDynamicThreads(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan, CommandLineOpts &clo)
{
  // create CPU threads for running dynamic gpu load since work items
  // are placed in a different queue
  if (analysis_compute_plan.use_gpu_acceleration && analysis_compute_plan.dynamic_gpu_balance)
  {
    pthread_t work_thread;
    for (int cworker = 0; cworker < analysis_compute_plan.numDynamicBkgWorkers; cworker++)
    {
      int t = pthread_create(&work_thread, NULL, DynamicBkgFitWorkerCpu,
                             analysis_queue.dynamicthreadWorkQ);
      if (t)
        fprintf(stderr, "Error starting dynamic CPU thread\n");
    }

  }

  for (int i = 0; i < analysis_compute_plan.numBkgWorkers_gpu; i++)
  {
    pthread_t work_thread;

    int deviceId = i / clo.numGpuThreads;

    analysis_queue.gpu_info[i].dynamic_gpu_load = analysis_compute_plan.dynamic_gpu_balance;
    analysis_queue.gpu_info[i].gpu_index = analysis_compute_plan.valid_devices[deviceId];
    analysis_queue.gpu_info[i].queue = (analysis_compute_plan.dynamic_gpu_balance ? (void*)analysis_queue.dynamicthreadWorkQ :
                                        (void*)analysis_queue.threadWorkQ_gpu);

    // Spawn GPU workers pulling items from either the combined queue (dynamic)
    // or a separate GPU queue (static)
    int t = pthread_create(&work_thread, NULL, BkgFitWorkerGpu, &analysis_queue.gpu_info[i]);
    if (t)
      fprintf(stderr, "Error starting GPU thread\n");
  }
}

void UnSpinGpuThreads(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan)
{

  if (analysis_queue.threadWorkQ_gpu)
  {
    WorkerInfoQueueItem item;
    item.finished = true;
    item.private_data = NULL;
    for (int i=0;i < analysis_compute_plan.numBkgWorkers_gpu;i++)
      analysis_queue.threadWorkQ_gpu->PutItem(item);
    analysis_queue.threadWorkQ_gpu->WaitTillDone();

    delete analysis_queue.threadWorkQ_gpu;
  }
}

void AssignQueueForItem(ProcessorQueue &analysis_queue,ComputationPlanner &analysis_compute_plan, CommandLineOpts &clo, int r)
{

  if (analysis_compute_plan.use_gpu_acceleration && analysis_compute_plan.dynamic_gpu_balance)
  {
    // Add all items to the shared queue
    analysis_queue.dynamicthreadWorkQ->PutItem(analysis_queue.item);
  }
  else
  {
    // Deterministically split items between the CPU
    // and GPU queues
    if (r >= int(analysis_compute_plan.gpu_work_load * float(clo.numRegions)))
    {
      analysis_queue.threadWorkQ->PutItem(analysis_queue.item);
    }
    else
    {
      analysis_queue.threadWorkQ_gpu->PutItem(analysis_queue.item);
    }
  }
}

void ReadOptimizedDefaultsForBkgModel(CommandLineOpts &clo, char *chipType,char *experimentName)
{
  //optionally set optimized parameters from GeneticOptimizer runs
  if (!clo.gopt)
  {
    //create default param file name
    char filename[64];
    sprintf(filename, "gopt_%s.param", chipType);
    clo.gopt = GetIonConfigFile(filename);
  } // set defaults if nothing set at all
  if (clo.gopt)
  {
    if (strcmp(clo.gopt, "opt") == 0)
      GlobalDefaultsForBkgModel::ReadEmphasisVectorFromFile(experimentName); //GeneticOptimizer run - load its vector
    else if (strcmp(clo.gopt, "disable") != 0) //load gopt defaults unless disabled
      GlobalDefaultsForBkgModel::SetGoptDefaults(clo.gopt);
  }
}

void OverrideDefaultsForBkgModel(CommandLineOpts &clo, char *chipType,char *experimentName)
{
  // set global parameter defaults from command line values if necessary
  if (clo.krate[0] > 0)
    GlobalDefaultsForBkgModel::SetKrateDefaults(clo.krate);

  if (clo.diff_rate[0] > 0)
    GlobalDefaultsForBkgModel::SetDDefaults(clo.diff_rate);

  if (clo.kmax[0] > 0)
    GlobalDefaultsForBkgModel::SetKmaxDefaults(clo.kmax);

  if (clo.no_rdr_fit_first_20_flows)
    GlobalDefaultsForBkgModel::FixRdrInFirst20Flows(true);
  
  if (clo.damp_kmult>0)
    GlobalDefaultsForBkgModel::SetDampenKmult(clo.damp_kmult);
}

void SetupXtalkParametersForBkgModel(CommandLineOpts &clo, char *chipType,char *experimentName)
{
   // search for config file for chip type
  if (!clo.xtalk)
  {
    //create default param file name
    char filename[64];
    sprintf(filename, "xtalk_%s.param", chipType);
    clo.xtalk = GetIonConfigFile(filename);
    if (!clo.xtalk)
        GlobalDefaultsForBkgModel::ReadXtalk(""); // nothing found
  } // set defaults if nothing set at all
  if (clo.xtalk)
  {
    if (strcmp(clo.xtalk, "local") == 0){
      char my_file[2048];
      sprintf(my_file,"%s/my_xtalk.txt",experimentName);
      GlobalDefaultsForBkgModel::ReadXtalk(my_file); //rerunning in local directory for optimization purposes
    }else if (strcmp(clo.xtalk, "disable") != 0) //disabled = don't load
      GlobalDefaultsForBkgModel::ReadXtalk(clo.xtalk);
    else GlobalDefaultsForBkgModel::ReadXtalk(""); // must be non-null to be happy
  } // isolated function

}

void SetBkgModelGlobalDefaults(CommandLineOpts &clo, char *chipType,char *experimentName)
{
  // @TODO: Bad coding style to use static variables as shared global state
  GlobalDefaultsForBkgModel::SetChipType(chipType); // bad, but need to know this before the first Image arrives
  // better to have global object pointed to by individual entities to make this maneuver very clear

  ReadOptimizedDefaultsForBkgModel(clo,chipType,experimentName);
  OverrideDefaultsForBkgModel(clo,chipType,experimentName); // after we read from the file so we can tweak
  
  SetupXtalkParametersForBkgModel(clo,chipType,experimentName); 

}


void DoThreadedBackgroundModel(RawWells &rawWells, CommandLineOpts &clo, Mask *maskPtr, Flow *flw, char *experimentName, int numFlows, char *chipType,
                               ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, Region *regions, RegionTiming *region_timing,SequenceItem* seqList,int numSeqListItems)
{
 
  MemUsage("StartingBackground");
  Mask localMask(maskPtr);
  localMask.CalculateLiveNeighbors();
  
  PoissonCDFApproxMemo poiss_cache; // math routines the bkg model needs to do a lot
  poiss_cache.Allocate(MAX_HPLEN+1,MAX_POISSON_TABLE_ROW,POISSON_TABLE_STEP);
  poiss_cache.GenerateValues(); // fill out my table
  int my_nuc_block[NUMFB];
  // TODO: Objects should be isolated!!!!
  GlobalDefaultsForBkgModel::SetFlowOrder(flw->GetFlowOrder());
  GlobalDefaultsForBkgModel::GetFlowOrderBlock(my_nuc_block,0,NUMFB);
  // @TODO these matrices never get updated for later blocks of nucs and should be updated during iterations
  InitializeLevMarSparseMatrices(my_nuc_block);
  
  dbgWellTracker my_bkg_dbg_wells;
  if (clo.bkgDebugParam)
    my_bkg_dbg_wells.Init(experimentName,my_image_spec.rows,my_image_spec.cols,numFlows,flw);

  SetBkgModelGlobalDefaults(clo,chipType,experimentName);

  // designate a set of reads that will be processed regardless of whether they pass filters
  set<int> randomLibSet;
  MaskSample<int> randomLib(*maskPtr, MaskLib, clo.nUnfilteredLib);
  randomLibSet.insert(randomLib.Sample().begin(), randomLib.Sample().end());
      
  clo.NormalizeZeros = false;
  // how are we dividing the computation amongst resources available as directed by command line constraints
  ComputationPlanner analysis_compute_plan;
  PlanComputation(analysis_compute_plan,clo);

  if (analysis_compute_plan.use_gpu_acceleration) {
    InitConstantMemoryOnGpu(poiss_cache);
  }

  ProcessorQueue analysis_queue;
  AllocateProcessorQueue(analysis_queue,analysis_compute_plan,clo.numRegions);

  BkgModel *BkgModelFitters[clo.numRegions];
  MemUsage("BeforeBgInitialization");
  // Threaded initialization of all the regional fitters
  {
    ImageInitBkgWorkInfo *linfo =
      new ImageInitBkgWorkInfo[clo.numRegions];
    for (int r = 0; r < clo.numRegions; r++)
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
      linfo[r].maxFrames = clo.maxFrames;
      linfo[r].uncompFrames = my_image_spec.uncompFrames;
      linfo[r].timestamps = my_image_spec.timestamps;
      linfo[r].localMask = &localMask;
      linfo[r].rawWells = &rawWells;
      linfo[r].regions = &regions[0];
      linfo[r].numRegions = clo.numRegions;
      //linfo[r].kic = keyIncorporation;
      linfo[r].t_mid_nuc = region_timing[r].t_mid_nuc;
      linfo[r].t_sigma = region_timing[r].t_sigma;
      linfo[r].experimentName = experimentName;
      linfo[r].maskPtr = maskPtr;
      linfo[r].sep_t0_estimate = &smooth_t0_est;
      linfo[r].math_poiss = &poiss_cache;
      linfo[r].seqList =seqList;
      linfo[r].numSeqListItems=numSeqListItems;
      linfo[r].sample = &randomLibSet;

      analysis_queue.item.finished = false;
      analysis_queue.item.private_data = (void *) &linfo[r];
      analysis_queue.threadWorkQ->PutItem(analysis_queue.item);

    }
    // wait for all of the images to be loaded and initially processed
    analysis_queue.threadWorkQ->WaitTillDone();

    delete[] linfo;
  }
  
  MemUsage("AfterBgInitialization");
  SpinDownCPUthreads(analysis_queue,analysis_compute_plan);

  SpinUpGPUAndDynamicThreads(analysis_queue,analysis_compute_plan,clo);

  SetRegionProcessOrder(clo.numRegions, BkgModelFitters, analysis_compute_plan);

  // Image Loading thread setup to grab flows in the background
  ImageTracker my_img_set(numFlows,clo.ignoreChecksumErrors);
  bubbleTracker my_bubble_tracker;
  if (clo.filterBubbles == 1)
    my_bubble_tracker.Init(experimentName,my_image_spec.rows,my_image_spec.cols,numFlows,flw);
  ImageLoadWorkInfo glinfo;
  SetUpImageLoaderInfo(glinfo,clo,localMask, my_img_set, my_image_spec,my_bubble_tracker,numFlows);

  pthread_t loaderThread;
  pthread_create(&loaderThread, NULL, FileLoader, &glinfo);

  // Now do threaded solving, going through all the flows
  // activating each regional fitter for each flow
  int saveWellsFrequency = 3; // Save every 3 blocks of flows in case we want to change block size
  BkgModelWorkInfo *bkinfo = new BkgModelWorkInfo[clo.numRegions];
  // process all flows...
  time_t flow_start;
  time_t flow_end;

  for (int flow = 0; flow < numFlows; flow++)
  {
    time(&flow_start);
    my_img_set.WaitForFlowToLoad(flow);
    // call the fitter for each region
    {
      for (int r = 0; r < clo.numRegions; r++)
      {
        // these get free'd by the thread that processes them
        bkinfo[r].type = bkgWorkE;
        bkinfo[r].bkgObj = BkgModelFitters[analysis_compute_plan.region_order[r].first];
        bkinfo[r].flow = flow;
        bkinfo[r].img = &(my_img_set.img[flow]);
        bkinfo[r].last = ((flow) == (numFlows - 1));
        bkinfo[r].learning = false;

        analysis_queue.item.finished = false;
        analysis_queue.item.private_data = (void *) &bkinfo[r];
        AssignQueueForItem(analysis_queue,analysis_compute_plan,clo,r);
      }

      WaitForRegionsToFinishProcessing(analysis_queue,analysis_compute_plan, clo, flow);
      MemUsage("Memory_Flow: " + ToStr(flow));
      time(&flow_end);
      fprintf(stdout, "ProcessImage compute time for flow %d: %0.1lf sec.\n",
              flow, difftime(flow_end, flow_start));
    }
    // capture the regional parameters every 20 flows, plus one bead per region at "random"
    // @TODO replace with clean hdf5 interface for sampling beads and region parameters
      DumpBkgModelRegionInfo(experimentName,BkgModelFitters,clo.numRegions,flow,false);
      DumpBkgModelBeadInfo(experimentName,BkgModelFitters,clo.numRegions,flow,false);

    my_img_set.FinishFlow(flow);
    IncrementalWriteWells(rawWells,flow,false,saveWellsFrequency,numFlows);
    // individual regions output a '.' when they finish...terminate them all with a \n to keep the
    // display clean
    printf("\n");
  }
  rawWells.WriteRanks();
  rawWells.WriteInfo();
  rawWells.Close();

  // before we delete the BkgModel objects, lets capture the region-wide parameters
  // from them into a debug file
  DumpBkgModelRegionInfo(experimentName,BkgModelFitters,clo.numRegions,numFlows-1, true);
  DumpBkgModelBeadInfo(experimentName,BkgModelFitters,clo.numRegions,numFlows-1, true);

  for (int r = 0; r < clo.numRegions; r++)
    delete BkgModelFitters[r];

  UnSpinGpuThreads(analysis_queue,analysis_compute_plan);

  CleanupLevMarSparseMatrices();
  GlobalDefaultsForBkgModel::StaticCleanup();
  pthread_join(loaderThread, NULL);
}

void IncrementalWriteWells(RawWells &rawWells,int flow, bool last_flow,int saveWellsFrequency,int numFlows)
{
    int testWellFrequency = saveWellsFrequency*NUMFB; // block size
    if (((flow+1) % (saveWellsFrequency*NUMFB) == 0 && (flow != 0))  || (flow+1) >= numFlows || last_flow) {
      fprintf(stdout, "Writing incremental wells at flow: %d\n", flow);
      MemUsage("BeforeWrite");
      rawWells.WriteWells();
      rawWells.SetChunk(0, rawWells.NumRows(), 0, rawWells.NumCols(), flow+1, min(testWellFrequency,numFlows-(flow+1)));
      MemUsage("AfterWrite");
    }
}

void DumpBkgModelBeadInfo(char *experimentName, BkgModel *BkgModelFitters[], int numRegions, int flow, bool last_flow)
{
  // get some regional data for the entire chip as debug
  // only do this every 20 flows as this is the block
  // should be triggered by bkgmodel
  if ((flow+1) % NUMFB ==0 || last_flow)
  {
    FILE *bkg_mod_bead_dbg = NULL;
    char *bkg_mod_bead_dbg_fname = (char *) malloc(512);
    snprintf(bkg_mod_bead_dbg_fname, 512, "%s/%s.%04d.%s", experimentName, "BkgModelBeadData",flow+1,"txt");
    fopen_s(&bkg_mod_bead_dbg, bkg_mod_bead_dbg_fname, "wt");
    free(bkg_mod_bead_dbg_fname);

    DumpBeadTitle(bkg_mod_bead_dbg);

    for (int r = 0; r < numRegions; r++)
    {
      BkgModelFitters[r]->DumpExemplarBead(bkg_mod_bead_dbg,true);
    }
    fclose(bkg_mod_bead_dbg);
  }
}

void DumpBkgModelEmphasisTiming(char *experimentName,BkgModel *BkgModelFitters[],int numRegions, int flow)
{
    // dump the dark matter, which is a fitted compensation term
    FILE *bkg_mod_time_dbg = NULL;
    char *bkg_mod_time_name = (char * )malloc(512);
    snprintf(bkg_mod_time_name, 512, "%s/%s.%04d.%s", experimentName, "BkgModelEmphasisData",flow+1,"txt");
    fopen_s(&bkg_mod_time_dbg, bkg_mod_time_name, "wt");
    free(bkg_mod_time_name);
        
    for (int r = 0; r < numRegions; r++)
    {
      BkgModelFitters[r]->DumpTimeAndEmphasisByRegion(bkg_mod_time_dbg);

    }
    fclose(bkg_mod_time_dbg);
}

void DumpBkgModelDarkMatter(char *experimentName,BkgModel *BkgModelFitters[],int numRegions, int flow)
{
    // dump the dark matter, which is a fitted compensation term
    FILE *bkg_mod_dark_dbg = NULL;
    char *bkg_mod_dark_name = (char * )malloc(512);
    snprintf(bkg_mod_dark_name, 512, "%s/%s.%04d.%s", experimentName, "BkgModelDarkMatterData",flow+1,"txt");
    fopen_s(&bkg_mod_dark_dbg, bkg_mod_dark_name, "wt");
    free(bkg_mod_dark_name);
    
    BkgModelFitters[0]->DumpDarkMatterTitle(bkg_mod_dark_dbg);
    
    for (int r = 0; r < numRegions; r++)
    {
      BkgModelFitters[r]->DumpDarkMatter(bkg_mod_dark_dbg);

    }
    fclose(bkg_mod_dark_dbg);
}

void DumpBkgModelRegionParameters(char *experimentName,BkgModel *BkgModelFitters[],int numRegions, int flow)
{
    FILE *bkg_mod_reg_dbg = NULL;
    char *bkg_mod_reg_dbg_fname = (char *) malloc(512);
    snprintf(bkg_mod_reg_dbg_fname, 512, "%s/%s.%04d.%s", experimentName, "BkgModelRegionData",flow+1,"txt");
    fopen_s(&bkg_mod_reg_dbg, bkg_mod_reg_dbg_fname, "wt");
    free(bkg_mod_reg_dbg_fname);

    struct reg_params rp;

    DumpRegionParamsTitle(bkg_mod_reg_dbg);

    for (int r = 0; r < numRegions; r++)
    {
      BkgModelFitters[r]->GetRegParams(&rp);
      //@TODO this routine should have no knowledge of internal representation of variables
      // Make this a routine to dump an informative line to a selected file from a regional parameter structure/class
      // especially as we use a very similar dumping line a lot in different places.
      // note: t0, rdr, and pdr evolve over time.  It would be nice to also capture how they changed throughout the analysis
      // this only captures the final value of each.
      DumpRegionParamsLine(bkg_mod_reg_dbg, BkgModelFitters[r]->GetRegion()->row,BkgModelFitters[r]->GetRegion()->col, rp);

    }
    fclose(bkg_mod_reg_dbg);
}

void DumpBkgModelRegionInfo(char *experimentName,BkgModel *BkgModelFitters[],int numRegions, int flow, bool last_flow)
{
  // get some regional data for the entire chip as debug
  // only do this every 20 flows as this is the block
  // should be triggered by bkgmodel
  if ((flow+1) % NUMFB ==0 || last_flow)
  {
    DumpBkgModelRegionParameters(experimentName, BkgModelFitters, numRegions, flow);
    DumpBkgModelDarkMatter(experimentName, BkgModelFitters, numRegions, flow);
    DumpBkgModelEmphasisTiming(experimentName,BkgModelFitters, numRegions, flow);
  }
}



void SetUpRegions(Region *regions, int rows, int cols, int xinc, int yinc)
{
  int i,x,y;

  for (i = 0, x = 0; x < cols; x += xinc)
  {
    for (y = 0; y < rows; y += yinc)
    {
      regions[i].col = x;
      regions[i].row = y;
      regions[i].w = xinc;
      regions[i].h = yinc;
      if (regions[i].col + regions[i].w > cols) // technically I don't think these ever hit since I'm truncating to calc xinc * yinc
        regions[i].w = cols - regions[i].col; // but better to be safe!
      if (regions[i].row + regions[i].h > rows)
        regions[i].h = rows - regions[i].row;
      i++;
    }
  }
}


void SetUpRegionDivisions(CommandLineOpts &clo, int rows, int cols)
{
  int xinc, yinc;

  clo.regionsX = 1;
  clo.regionsY = 1;

  // fixed region size
  xinc = clo.regionXSize;
  yinc = clo.regionYSize;
  clo.regionsX = cols / xinc;
  clo.regionsY = rows / yinc;
  // make sure we cover the edges in case rows/yinc or cols/xinc not exactly divisible
  if (((double) cols / (double) xinc) != clo.regionsX)
    clo.regionsX++;
  if (((double) rows / (double) yinc) != clo.regionsY)
    clo.regionsY++;
  clo.numRegions = clo.regionsX * clo.regionsY;
}



void DoDiffSeparatorFromCLO(DifferentialSeparator *diffSeparator, CommandLineOpts &clo, Mask *maskPtr, string &analysisLocation, Flow *flw, SequenceItem *seqList, int numSeqListItems)
{
  DifSepOpt opts;

  opts.bfType = clo.bfType;
  opts.bfDat = clo.bfDat;
  opts.bfBgDat = clo.bfBgDat;
  opts.resultsDir = clo.dirExt;
  opts.outData = analysisLocation;
  opts.analysisDir =  analysisLocation;
  opts.ignoreChecksumErrors = clo.ignoreChecksumErrors;
  opts.noduds = clo.noduds;

  opts.minRatioLiveWell = clo.bfMinLiveRatio;
  opts.libFilterQuantile = clo.bfLibFilterQuantile;
  opts.tfFilterQuantile = clo.bfTfFilterQuantile;
  opts.useProjectedCurve = clo.bfUseProj > 0;
  opts.doRecoverSdFilter = clo.skipBeadfindSdRecover == 0;
  opts.flowOrder = flw->GetFlowOrder();
  if (!opts.outData.empty())
  {
    string sep = "";
    if (*opts.outData.rbegin() != '/')
    {
      sep = "/";
    }
    opts.outData = opts.outData + sep + "separator";
  }
  else
  {
    opts.outData = "separator";
  }
  opts.mask = maskPtr;
  cout << "Out Data: " << opts.outData << endl;
  cout << "Analysis location: " << opts.analysisDir << endl;
  diffSeparator->SetKeys(seqList, numSeqListItems, clo.bfMinLiveLibSnr);
  if (clo.beadfindLagOneFilt > 0) {
    opts.filterLagOneSD = true;
  }
  if (clo.beadfindThumbnail == 1) {
    opts.t0MeshStep = clo.regionXSize;
    opts.bfMeshStep = clo.regionXSize;
    opts.tauEEstimateStep = clo.regionXSize;
    opts.useMeshNeighbors = 0;
    opts.regionXSize = clo.regionXSize;
    opts.regionYSize = clo.regionYSize;
  }
  diffSeparator->Run(opts);
}


void SetUpToProcessImages(ImageSpecClass &my_image_spec, CommandLineOpts &clo, char *experimentName, TrackProgress &my_progress)
{
  // set up to process images aka 'dat' files.
  ExportSubRegionSpecsToImage(clo);

  // make sure we're using XTCorrection at the right offset for cropped regions
  Image::SetCroppedRegionOrigin(clo.cropped_region_x_offset,clo.cropped_region_y_offset);
  Image::CalibrateChannelXTCorrection(clo.dirExt,"lsrowimage.dat");

  my_image_spec.DeriveSpecsFromDat(clo,1,experimentName); // dummy - only reads 1 dat file
  fprintf(my_progress.fpLog, "VFR = %s\n", my_image_spec.vfr_enabled ? "enabled":"disabled");  // always enabled these days, useless?
}

void SetUpRegionsForAnalysis(ImageSpecClass &my_image_spec, CommandLineOpts &clo, Region &wholeChip)
{

  FixCroppedRegions(clo, my_image_spec);

  SetUpRegionDivisions(clo,my_image_spec.rows,my_image_spec.cols);
  SetUpWholeChip(wholeChip,my_image_spec.rows,my_image_spec.cols);

}

  
void SetupForBkgModelTiming(DifferentialSeparator *diffSeparator, std::vector<float> &smooth_t0_est, RegionTiming *region_timing, 
                            Region *region_list, int numRegions, ImageSpecClass &my_image_spec, Mask *maskPtr, bool doSmoothing){
      // compute timing information
    AvgKeyIncorporation *keyIncorporation = NULL;

      //Create a mask that tracks the pinned pixels discovered in each image
      maskPtr->CalculateLiveNeighbors();

      // Setup t0 estimation from beadfind to pass to background model
      std::vector<float> sep_t0_est;
      sep_t0_est = diffSeparator->GetT0();
      smooth_t0_est = sep_t0_est;
      if (doSmoothing) {
	printf("smoothing t0 estimate from separator.......");
	NNSmoothT0Estimate(maskPtr,my_image_spec.rows,my_image_spec.cols,sep_t0_est,smooth_t0_est);
	printf("done.\n");
      }	
      // do some incorporation signal modeling
      keyIncorporation = diffSeparator;
      //FillRegionalTimingParameters(region_timing, region_list, numRegions, keyIncorporation);
      threadedFillRegionalTimingParameters(region_timing,region_list,numRegions,keyIncorporation);
      keyIncorporation = NULL;
}

void CreateWellsFileForWriting(RawWells &rawWells, Mask *maskPtr,
                               CommandLineOpts &clo,
                               Flow *flw, int numFlows,
                               int numRows, int numCols,
                               const char *chipType) {
  // set up wells data structure
  MemUsage("BeforeWells");
  rawWells.SetCompression(3);
  rawWells.SetRows(numRows);
  rawWells.SetCols(numCols);
  rawWells.SetFlows(numFlows);
  rawWells.SetFlowOrder(flw->GetFlowOrder());
  SetWellsToLiveBeadsOnly(rawWells,maskPtr);
  // any model outputs a wells file of this nature
  GetMetaDataForWells(clo.dirExt,rawWells,chipType);
  rawWells.SetChunk(0, rawWells.NumRows(), 0, rawWells.NumCols(), 0, min(60, numFlows));
  rawWells.OpenForWrite();
  MemUsage("AfterWells");
}
      
  
// output from this are a functioning wells file and a beadfind mask
// images are only known internally to this.
void RealImagesToWells(RawWells &rawWells, Mask *maskPtr,
                       CommandLineOpts &clo,
                       char *experimentName, string &analysisLocation,
                       Flow *flw, int numFlows,
                       SequenceItem *seqList, int numSeqListItems,
                       TrackProgress &my_progress, Region &wholeChip,
                       int &well_rows, int &well_cols)
{

  char *chipType = GetChipId(clo.dirExt);
  ChipIdDecoder::SetGlobalChipId(chipType);

  ImageSpecClass my_image_spec;
  SetUpToProcessImages(my_image_spec, clo, experimentName, my_progress);

  SetUpRegionsForAnalysis(my_image_spec, clo, wholeChip);

  SetExcludeMask(clo,maskPtr,chipType,my_image_spec.rows,my_image_spec.cols);

  // Write processParameters.parse file now that processing is about to begin
  clo.WriteProcessParameters();

  // region definitions, in theory shared between background model and beadfind
  Region region_list[clo.numRegions];
  SetUpRegions(region_list,my_image_spec.rows,my_image_spec.cols,clo.regionXSize,clo.regionYSize);
  // need crude nuc rise timing information for bkg model currently
  RegionTiming region_timing[clo.numRegions];
  // global timing estimates - need to have to pass to bkgmodel
    std::vector<float> smooth_t0_est;

  /*********************************************************************
  // Beadfind Section
   *********************************************************************/

  if (clo.beadfindType == "differential")
  {
    DifferentialSeparator *diffSeparator=NULL;
    
    diffSeparator = new DifferentialSeparator();
    DoDiffSeparatorFromCLO(diffSeparator, clo, maskPtr, analysisLocation, flw, seqList,numSeqListItems);
    // now actually set up the mask I want
    maskPtr->Copy(diffSeparator->GetMask());
    my_progress.ReportState("Beadfind Complete");
    
    SetupForBkgModelTiming(diffSeparator, smooth_t0_est, region_timing, 
			   region_list, clo.numRegions, my_image_spec, maskPtr, clo.beadfindThumbnail == 0);
      // cleanup scope
      if (diffSeparator !=NULL){
        delete diffSeparator;
        diffSeparator = NULL;
      }
    // Update progress bar status file: well proc complete/image proc started
    updateProgress(WELL_TO_IMAGE);
  }
  else
  {
    fprintf(stderr, "Don't recognize --beadfind-type %s\n", clo.beadfindType.c_str());
    exit(EXIT_FAILURE);
  }

  UpdateBeadFindOutcomes(maskPtr, wholeChip, experimentName, clo, 0);
  my_progress.ReportState("Bead Categorization Complete");

  if (clo.BEADFIND_ONLY)
  {
    // Remove temporary wells file
    if (clo.LOCAL_WELLS_FILE)
      unlink(clo.tmpWellsFile);
    fprintf(stdout,
            "Beadfind Only Mode has completed successfully\n");
    exit(EXIT_SUCCESS);
  }


  /********************************************************************
   *
   *  Background Modelling Process
   *
   *******************************************************************/

  CreateWellsFileForWriting(rawWells, maskPtr, clo, flw, 
                            numFlows, my_image_spec.rows, my_image_spec.cols,
                            chipType);

  // we might use some other model here
  if (clo.USE_BKGMODEL)
  {
    // Here's the point where we really do the background model and not just setup for it
    DoThreadedBackgroundModel(rawWells, clo, maskPtr, flw, experimentName, numFlows, chipType, my_image_spec, smooth_t0_est, region_list, region_timing,seqList, numSeqListItems);

  } // end BKG_MODEL

  // whatever model we run, copy the signals to permanent location
  CopyTmpWellFileToPermanent(clo, experimentName);
  // mask file may be updated by model processing
  UpdateBeadFindOutcomes(maskPtr, wholeChip, experimentName, clo, 0);
  

  my_progress.ReportState("Raw flowgrams complete");
  // we deal only with wells from now on
  well_rows=my_image_spec.rows;
  well_cols=my_image_spec.cols;

  free(chipType);
}


// output from this are a functioning wells file and a beadfind mask
// images are only known internally to this.
void GetFromImagesToWells(RawWells &rawWells, Mask *maskPtr,
                          CommandLineOpts &clo,
                          char *experimentName, string &analysisLocation,
                          Flow *flw, int numFlows,
                          SequenceItem *seqList, int numSeqListItems,
                          TrackProgress &my_progress, Region &wholeChip,
                          int &well_rows, int &well_cols)
{

  /*
   *  Two ways to proceed.  If this is raw data analysis, continue from here.
   *  If this is a re-basecall of an existing wells file, jump to basecalling
   *  section.
   */
  if (!clo.USE_RAWWELLS)
  {
    // do beadfind and signal processing
    RealImagesToWells(rawWells, maskPtr,clo,experimentName,analysisLocation,
                      flw,numFlows,seqList,numSeqListItems,my_progress,
                      wholeChip,well_rows,well_cols);

  } // end image processing
  else    // Processing --from-wells - no images are ever used
  {
    // Update progress bar file: well find is complete
    updateProgress(WELL_TO_IMAGE);

    // grab the previous beadfind
    LoadBeadMaskFromFile(clo, maskPtr,well_rows,well_cols);

    // copy/link files from old directory for the report
    MakeSymbolicLinkToOldDirectory(clo, experimentName);
    CopyFilesForReportGeneration(clo, experimentName,seqList);
    SetChipTypeFromWells(rawWells);
  }
}
