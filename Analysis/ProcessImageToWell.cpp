/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <deque>
#include <fstream>
#include "ProcessImageToWell.h"
#include "WellFileManipulation.h"
#include "mixed.h"
#include "DataCube.h"
#include "H5File.h"

using namespace std;

bool sortregionProcessOrderVector (const beadRegion& r1, const beadRegion& r2)
{
  return (r1.second > r2.second);
}

void ExportSubRegionSpecsToImage (CommandLineOpts &clo)
{
  // Default analysis mode sets values to 0 and whole-chip processing proceeds.
  // otherwise, command line override (--analysis-region) can define a subchip region.
  Image::chipSubRegion.row = (clo.GetChipRegion()).row;
  Image::chipSubRegion.col = (clo.GetChipRegion()).col;
  Image::chipSubRegion.h = (clo.GetChipRegion()).h;
  Image::chipSubRegion.w = (clo.GetChipRegion()).w;

}

void SetUpWholeChip (Region &wholeChip,int rows, int cols)
{
  //Used later to generate mask statistics for the whole chip
  wholeChip.row = 0;
  wholeChip.col = 0;
  wholeChip.w = cols;
  wholeChip.h = rows;
}

void UpdateBeadFindOutcomes (Mask *maskPtr, Region &wholeChip, char *experimentName, CommandLineOpts &clo, int update_stats)
{
  char maskFileName[2048];
  if (!update_stats)
  {
    sprintf (maskFileName, "%s/bfmask.stats", experimentName);
    maskPtr->DumpStats (wholeChip, maskFileName, !clo.bfd_control.SINGLEBF);
  }
  sprintf (maskFileName, "%s/bfmask.bin", experimentName);
  maskPtr->WriteRaw (maskFileName);
  maskPtr->validateMask();
}

void FixCroppedRegions (CommandLineOpts &clo, ImageSpecClass &my_image_spec)
{
  //If no cropped regions defined on command line, set cropRegions to whole chip
  if (!clo.loc_context.cropRegions)
  {
    clo.loc_context.numCropRegions = 1;
    clo.loc_context.cropRegions = (Region *) malloc (sizeof (Region));
    SetUpWholeChip (clo.loc_context.cropRegions[0],my_image_spec.rows,my_image_spec.cols);
  }
}



void SetExcludeMask (CommandLineOpts &clo, Mask *maskPtr, char *chipType, int rows, int cols)
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
  if ( (rows == clo.loc_context.chip_len_y) && (cols == clo.loc_context.chip_len_x))
  {
    //This is a wholechip dataset
    applyExclusionMask = true;
    fprintf (stderr, "This is a wholechip dataset so the exclusion mask will be applied\n");
  }
  else
  {
    if (clo.loc_context.chip_offset_x == -1)
    {
      applyExclusionMask = false;
      fprintf (stderr, "This is a cropped dataset so the exclusion mask will not be applied\n");
    }
    else
    {
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
  if (clo.loc_context.numCropRegions == 0)
  {
    maskPtr->Init (cols, rows, MaskEmpty);
  }
  else
  {
    maskPtr->Init (cols, rows, MaskExclude);

    // apply one or more cropRegions, mark them MaskEmpty
    for (int q = 0; q < clo.loc_context.numCropRegions; q++)
    {
      maskPtr->MarkRegion (clo.loc_context.cropRegions[q], MaskEmpty);
    }
  }

  /*
   * Apply exclude mask from file
   */
  clo.loc_context.exclusionMaskSet = false;
  if (chipType && applyExclusionMask)
  {
    char *exclusionMaskFileName = NULL;
    char filename[64] = { 0 };
#ifdef OLDWAY
    sprintf (filename, "exclusionMask_%s.bin", chipType);
#else
    sprintf (filename, "excludeMask_%s", chipType);
#endif
    exclusionMaskFileName = GetIonConfigFile (filename);
    fprintf (stderr, "Exclusion Mask File = '%s'\n", exclusionMaskFileName);
    if (exclusionMaskFileName)
    {
      clo.loc_context.exclusionMaskSet = true;
#ifdef OLDWAY
      Mask excludeMask (1, 1);
      excludeMask.SetMask (exclusionMaskFileName);
      free (exclusionMaskFileName);
      //--- Mark beadfind masks with MaskExclude bits from exclusionMaskFile
      maskPtr->SetThese (&excludeMask, MaskExclude);
#else
      FILE *excludeFile = NULL;
      excludeFile = fopen (exclusionMaskFileName,"rb");
      assert (excludeFile != NULL);
      uint16_t x = 0;
      uint16_t y = 0;
      while (1)
      {
        if (fread (&x, sizeof (x), 1, excludeFile) != 1) break;
        if (fread (&y, sizeof (y), 1, excludeFile) != 1) break;
        //fprintf (stderr, "Excluding %d %d (%d %d)\n",x,y,(int) x - clo.chip_offset_x,(int) y - clo.chip_offset_y);
        maskPtr->Set ( (int) x - clo.loc_context.chip_offset_x, (int) y - clo.loc_context.chip_offset_y,MaskExclude);
      }
#endif
    }
    else
    {
      fprintf (stderr, "WARNING: Exclusion Mask %s not applied\n", filename);
    }
  }
}


// utility function
void MakeSymbolicLinkToOldDirectory (CommandLineOpts &clo, char *experimentName)
{
// Create symbolic link to bfmask.bin and 1.wells in new subdirectory: links are for disc space usage reasons
  char *oldpath = NULL;
  int sz = strlen (clo.sys_context.wellsFilePath) + strlen (clo.sys_context.wellsFileName) + 2;
  oldpath = (char *) malloc (sz);
  snprintf (oldpath, sz, "%s/%s", clo.sys_context.wellsFilePath, clo.sys_context.wellsFileName);
  char *fullPath = realpath (oldpath, NULL);

  char *newpath = NULL;
  sz = strlen (experimentName) + strlen (clo.sys_context.wellsFileName) + 2;
  newpath = (char *) malloc (sz);
  snprintf (newpath, sz, "%s/%s", experimentName, clo.sys_context.wellsFileName);

  int ret = symlink (fullPath, newpath);
  if (ret)
  {
    perror (oldpath);
  }
  free (oldpath);
  free (newpath);
  free (fullPath);
}



// fill the new directory with files needed for report generation
void CopyFilesForReportGeneration (CommandLineOpts &clo, char *experimentName, SeqListClass &my_keys)
{
//--- Copy files needed for report generation ---
//--- Copy bfmask.stats ---
  int sz;
  char *newpath = NULL;
  char *oldpath = NULL;
  sz = strlen (clo.sys_context.wellsFilePath) + strlen ("bfmask.stats") + 2;
  oldpath = (char *) malloc (sz);
  snprintf (oldpath, sz, "%s/%s", clo.sys_context.wellsFilePath, "bfmask.stats");
  sz = strlen (experimentName) + strlen ("bfmask.stats") + 2;
  newpath = (char *) malloc (sz);
  snprintf (newpath, sz, "%s/%s", experimentName, "bfmask.stats");
  fprintf (stderr, "%s\n%s\n", oldpath, newpath);
  CopyFile (oldpath, newpath);
  free (oldpath);
  free (newpath);
//--- Copy avgNukeTrace_ATCG.txt and avgNukeTrace_TCAG.txt
//@TODO:  Is this really compatible with 3 keys?
  for (int q = 0; q < my_keys.numSeqListItems; q++)
  {
    char *filename;
    filename = (char *) malloc (strlen ("avgNukeTrace_") + strlen (
                                  my_keys.seqList[q].seq) + 5);
    sprintf (filename, "avgNukeTrace_%s.txt", my_keys.seqList[q].seq);

    sz = strlen (clo.sys_context.wellsFilePath) + strlen (filename) + 2;
    oldpath = (char *) malloc (sz);
    snprintf (oldpath, sz, "%s/%s", clo.sys_context.wellsFilePath, filename);

    sz = strlen (experimentName) + strlen (filename) + 2;
    newpath = (char *) malloc (sz);
    snprintf (newpath, sz, "%s/%s", experimentName, filename);

    CopyFile (oldpath, newpath);
    free (oldpath);
    free (newpath);
    free (filename);
  }
}

void LoadBeadMaskFromFile (CommandLineOpts &clo, Mask *maskPtr, int &rows, int &cols)
{
  char maskFileName[2048];

  // Load beadmask from file
  sprintf (maskFileName, "%s/%s", clo.sys_context.wellsFilePath, "./bfmask.bin");

  maskPtr->SetMask (maskFileName);
  if (maskPtr->SetMask (maskFileName))
  {
    exit (EXIT_FAILURE);
  }
  rows = maskPtr->H();
  cols = maskPtr->W();
  clo.loc_context.rows = rows;
  clo.loc_context.cols = cols;

  //--- Note that if we specify cropped regions on the command line, we are supposing that the original
  //    analysis was a whole chip analysis.  This is a safe assumption for the most part.
  // TODO: Need to rationalize the cropped region handling.

  clo.loc_context.regionsX = 1;
  clo.loc_context.regionsY = 1;

  if (clo.loc_context.numRegions == 1)
  {
    int rx, ry;
    for (ry = 0; ry < rows; ry++)
    {
      for (rx = 0; rx < cols; rx++)
      {
        if (rx >= clo.loc_context.regionXOrigin && rx < (clo.loc_context.regionXOrigin
            + clo.loc_context.regionXSize) && ry >= clo.loc_context.regionYOrigin && ry
            < (clo.loc_context.regionYOrigin + clo.loc_context.regionYSize))
          ;
        else
          (*maskPtr) [rx + ry * cols] = MaskExclude;
      }
    }
  }

  clo.WriteProcessParameters();

  //--- Handle cropped regions defined from command line
  if (clo.loc_context.numCropRegions > 0)
  {
    maskPtr->CropRegions (clo.loc_context.cropRegions, clo.loc_context.numCropRegions, MaskExclude);
  }

}


// distance to NN-smooth the t_zero (not t_mid_nuc!!!) estimate from the separator
#define SEPARATOR_T0_ESTIMATE_SMOOTH_DIST 15


// Brute-force NN averaging used to smooth the t0 estimate from the separator.  This algorithm can be sped-up
// considerably by sharing summations across multiple data points, similar to what is done in the image class for
// neighbor-subtracting images.
void NNSmoothT0Estimate (Mask *mask,int imgRows,int imgCols,std::vector<float> &sep_t0_est,std::vector<float> &output_t0_est)
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
          if (!mask->Match (sc,sr, (MaskType) (MaskPinned | MaskIgnore | MaskExclude)))
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
    else if (event == SeparatorWorkE)
    {
      SeparatorWorkInfo *winfo =
        (SeparatorWorkInfo *) (item.private_data);
      winfo->separator->FindBeads (winfo->img, winfo->region, winfo->mask,
                                   winfo->label);
    }
    else if (event == imageLoadE || event == imageLoadKeyE)
    {
      ImageLoadWorkInfo *info = (ImageLoadWorkInfo *) (item.private_data);
      Mask tMask (info->lmask);

      //Use Image Class object to read datafile, possibly filter data, then get raw data.
      //      info->img->LoadRaw(info->name);
      info->img->FilterForPinned (&tMask, MaskEmpty);
      info->img->Normalize (info->normStart, info->normEnd);

      // correct in-channel electrical cross-talk
      info->img->XTChannelCorrect (info->lmask);

      // Calculate average background for each well
      info->img->BackgroundCorrect (&tMask, MaskBead, MaskEmpty,
                                    info->NNinnerx, info->NNinnery, info->NNouterx,
                                    info->NNoutery, NULL, true);

      if (event == imageLoadKeyE)
      {
        // for each region, apply the signal extraction method (can be integral, pre-trainied vector, etc)
        // the separator methods may be called as threads, I'm thinking the separator object itself can manage this, and supply a 'wait for all threads' call
        for (int r = 0; r < info->numRegions; r++)
        {
          info->separator->CalcSignal (info->img, &info->regions[r],
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
      ImageInitBkgWorkInfo *info = (ImageInitBkgWorkInfo *) (item.private_data);
      int r = info->r;

      // should we enable debug for this region?
      bool reg_debug_enable;
      reg_debug_enable = CheckBkgDbgRegion (&info->regions[r],info->clo);

      BkgModel *bkgmodel = new BkgModel (info->experimentName, info->maskPtr,
                                         info->localMask, info->emptyInFlow, info->rawWells, &info->regions[r], *info->sample, info->sep_t0_estimate,
                                         reg_debug_enable, info->clo->loc_context.rows, info->clo->loc_context.cols,
                                         info->maxFrames,info->uncompFrames,info->timestamps,info->math_poiss,
                                         info->t_sigma,
                                         info->t_mid_nuc,
                                         info->clo->bkg_control.dntp_uM,
                                         info->clo->bkg_control.enableXtalkCorrection,
                                         info->clo->bkg_control.enableBkgModelClonalFilter,
                                         info->seqList,
                                         info->numSeqListItems);

      bkgmodel->SetResError(info->resError);
      bkgmodel->SetKRateMult(info->kMult);
      bkgmodel->SetBeadOnceParam(info->beadOnceParam);
      if (info->clo->bkg_control.bkgDebugParam)
      {
        bkgmodel->SetParameterDebug (info->bkgDbg1,info->bkgDbg2,info->bkgDebugKmult);
      }

      if (info->clo->bkg_control.vectorize)
      {
        bkgmodel->performVectorization (true);
      }

      bkgmodel->SetAmplLowerLimit (info->clo->bkg_control.AmplLowerLimit);
      bkgmodel->SetKrateConstraintType (info->clo->bkg_control.relaxKrateConstraint);

      // put this fitter in the listh
      info->BkgModelFitters[r] = bkgmodel;

    }

    // indicate we finished that bit of work
    q->DecrementDone();
  }

  return (NULL);
}




bool CheckBkgDbgRegion (Region *r,CommandLineOpts *clo)
{
  for (unsigned int i=0;i < clo->bkg_control.BkgTraceDebugRegions.size();i++)
  {
    Region *dr = &clo->bkg_control.BkgTraceDebugRegions[i];

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
  fprintf(stdout, "ProcessImageToWell: numBkgWorkers = %d, numDynamicBkgWorkers = %d\n", my_compute_plan.numBkgWorkers, my_compute_plan.numDynamicBkgWorkers);
  my_compute_plan.numBkgWorkers_gpu = 0;

  // -- Tuning parameters --

  // Flag to disable CUDA acceleration
  my_compute_plan.use_gpu_acceleration = (bkg_control.gpuWorkLoad > 0) ? true : false;

  // Dynamic balance (set to true) will create one large queue for both the CPUs and GPUs to pull from
  // whenever the resource is free. A static balance (set to false) will create two queues, one for the
  // GPU to pull from and one for the CPU to pull from. This division will be set deterministically so
  // the same answer to achieved each time.
  my_compute_plan.dynamic_gpu_balance = true;
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
  if (analysis_compute_plan.use_gpu_acceleration)
  {
    fprintf (stdout, "Number of CPU threads for background model: %d\n", analysis_compute_plan.numDynamicBkgWorkers);
    fprintf (stdout, "Number of GPU threads for background model: %d\n", analysis_compute_plan.numBkgWorkers_gpu);
  }
  else
  {
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

void ReadOptimizedDefaultsForBkgModel (CommandLineOpts &clo, char *chipType,char *experimentName)
{
  //optionally set optimized parameters from GeneticOptimizer runs
  if (!clo.bkg_control.gopt)
  {
    //create default param file name
    char filename[64];
    sprintf (filename, "gopt_%s.param", chipType);
    clo.bkg_control.gopt = GetIonConfigFile (filename);
  } // set defaults if nothing set at all
  if (clo.bkg_control.gopt)
  {
    if (strcmp (clo.bkg_control.gopt, "opt") == 0)
      GlobalDefaultsForBkgModel::ReadEmphasisVectorFromFile (experimentName);   //GeneticOptimizer run - load its vector
    else if (strcmp (clo.bkg_control.gopt, "disable") != 0)     //load gopt defaults unless disabled
      GlobalDefaultsForBkgModel::SetGoptDefaults (clo.bkg_control.gopt);
  }
}

void OverrideDefaultsForBkgModel (CommandLineOpts &clo, char *chipType,char *experimentName)
{
  // set global parameter defaults from command line values if necessary
  if (clo.bkg_control.krate[0] > 0)
    GlobalDefaultsForBkgModel::SetKrateDefaults (clo.bkg_control.krate);

  if (clo.bkg_control.diff_rate[0] > 0)
    GlobalDefaultsForBkgModel::SetDDefaults (clo.bkg_control.diff_rate);

  if (clo.bkg_control.kmax[0] > 0)
    GlobalDefaultsForBkgModel::SetKmaxDefaults (clo.bkg_control.kmax);

  if (clo.bkg_control.no_rdr_fit_first_20_flows)
    GlobalDefaultsForBkgModel::FixRdrInFirst20Flows (true);

  if (clo.bkg_control.damp_kmult>0)
    GlobalDefaultsForBkgModel::SetDampenKmult (clo.bkg_control.damp_kmult);
  if (clo.bkg_control.generic_test_flag>0)
    GlobalDefaultsForBkgModel::SetGenericTestFlag (true);
  if (clo.bkg_control.var_kmult_only>0)
    GlobalDefaultsForBkgModel::SetVarKmultControl (true);
}

void SetupXtalkParametersForBkgModel (CommandLineOpts &clo, char *chipType,char *experimentName)
{
  // search for config file for chip type
  if (!clo.bkg_control.xtalk)
  {
    //create default param file name
    char filename[64];
    sprintf (filename, "xtalk_%s.param", chipType);
    clo.bkg_control.xtalk = GetIonConfigFile (filename);
    if (!clo.bkg_control.xtalk)
      GlobalDefaultsForBkgModel::ReadXtalk ("");   // nothing found
  } // set defaults if nothing set at all
  if (clo.bkg_control.xtalk)
  {
    if (strcmp (clo.bkg_control.xtalk, "local") == 0)
    {
      char my_file[2048];
      sprintf (my_file,"%s/my_xtalk.txt",experimentName);
      GlobalDefaultsForBkgModel::ReadXtalk (my_file);   //rerunning in local directory for optimization purposes
    }
    else if (strcmp (clo.bkg_control.xtalk, "disable") != 0)     //disabled = don't load
      GlobalDefaultsForBkgModel::ReadXtalk (clo.bkg_control.xtalk);
    else GlobalDefaultsForBkgModel::ReadXtalk ("");   // must be non-null to be happy
  } // isolated function

}

//@TODO: should be BkgModelControlOpts
void SetBkgModelGlobalDefaults (CommandLineOpts &clo, char *chipType,char *experimentName)
{
  // @TODO: Bad coding style to use static variables as shared global state
  GlobalDefaultsForBkgModel::SetChipType (chipType);   // bad, but need to know this before the first Image arrives
  // better to have global object pointed to by individual entities to make this maneuver very clear

  ReadOptimizedDefaultsForBkgModel (clo,chipType,experimentName);
  OverrideDefaultsForBkgModel (clo,chipType,experimentName);   // after we read from the file so we can tweak

  SetupXtalkParametersForBkgModel (clo,chipType,experimentName);

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

void BkgFitterTracker::PlanComputation(BkgModelControlOpts &bkg_control)
{
  // how are we dividing the computation amongst resources available as directed by command line constraints

  PlanMyComputation (analysis_compute_plan,bkg_control);

  AllocateProcessorQueue (analysis_queue,analysis_compute_plan,numFitters);
}



void BkgFitterTracker::ThreadedInitialization (RawWells &rawWells, CommandLineOpts &clo, Mask *maskPtr, Mask &localMask, short *emptyInFlow, char *experimentName,
                                               ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, Region *regions, int totalRegions, RegionTiming *region_timing,SeqListClass &my_keys, DataCube<float> *bgResidualError, DataCube<float> *kMult, DataCube<float> *beadOnceParam)
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

  ImageInitBkgWorkInfo *linfo =
    new ImageInitBkgWorkInfo[numFitters];
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
    linfo[r].localMask = &localMask;
    linfo[r].emptyInFlow = emptyInFlow;
    linfo[r].rawWells = &rawWells;
    if (clo.bkg_control.bkgModelHdf5Debug) {
      linfo[r].resError = bgResidualError;
      linfo[r].kMult = kMult;
      linfo[r].beadOnceParam = beadOnceParam;
    }
    else {
      linfo[r].resError = NULL;
      linfo[r].kMult = NULL;
      linfo[r].beadOnceParam = NULL;
    }
    linfo[r].regions = &regions[0];
    linfo[r].numRegions = totalRegions;
    //linfo[r].kic = keyIncorporation;
    linfo[r].t_mid_nuc = region_timing[r].t_mid_nuc;
    linfo[r].t_sigma = region_timing[r].t_sigma;
    linfo[r].experimentName = experimentName;
    linfo[r].maskPtr = maskPtr;
    linfo[r].sep_t0_estimate = &smooth_t0_est;
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

void InitializeEmptyInFlow(short *emptyInFlow, Mask *maskPtr)
{
  // wells marked as -1 are valid empty wells
  // wells that become non-empty as flows load are set to that flow value
  // initial non-empty wells are set to flow zero using the beadfind mask
  int w = maskPtr->W();
  int h = maskPtr->H();

 // all empty wells set to -1
  int nonempty = 0;
  for (int y=0; y<h; y++){
    for (int x=0; x<w; x++){
      int i = maskPtr->ToIndex(y,x);
      if (maskPtr->Match (x,y,MaskEmpty)){  // empty
	emptyInFlow[i] = -1;
      }
      else { // not ( empty )
	emptyInFlow[i] = 0;  // not-empty coming into flow 0
	nonempty++;
      }
    }
  }
  fprintf(stdout, "InitializeEmptyInFlow: %d non-empty wells prior to flow 0\n", nonempty);
}
	   
void DoThreadedBackgroundModel (RawWells &rawWells, CommandLineOpts &clo, Mask *maskPtr, char *experimentName, int numFlows, char *chipType,
                                ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, Region *regions, int totalRegions, RegionTiming *region_timing,SeqListClass &my_keys)
{

  MemUsage ("StartingBackground");

  Mask pinnedMask (maskPtr); // @TODO -- trace usage
  pinnedMask.CalculateLiveNeighbors();

  // this array tracks "reasonable" empty wells in each flow as we load it.
  short* emptyInFlow = new short[maskPtr->W()*maskPtr->H()];
  InitializeEmptyInFlow(emptyInFlow, maskPtr);

  int my_nuc_block[NUMFB];
  // TODO: Objects should be isolated!!!!
  GlobalDefaultsForBkgModel::SetFlowOrder (clo.flow_context.flowOrder); // @TODO: 2nd duplicated code instance
  GlobalDefaultsForBkgModel::GetFlowOrderBlock (my_nuc_block,0,NUMFB);
  // @TODO these matrices never get updated for later blocks of nucs and should be updated during iterations
  InitializeLevMarSparseMatrices (my_nuc_block);

  SetBkgModelGlobalDefaults (clo,chipType,experimentName);

  BkgFitterTracker GlobalFitter (totalRegions);
// plan
  GlobalFitter.PlanComputation(clo.bkg_control);
  // Setup the output flows for background parameters. @todoThis should be wrapped up in an object                              BgkParamH5 bgParamH5;                     
  MemUsage ("BeforeBgInitialization");

  BkgParamH5 bgParamH5;
  if (clo.bkg_control.bkgModelHdf5Debug) {
    bgParamH5.Init(clo, numFlows);
  }
  GlobalFitter.ThreadedInitialization (rawWells, clo, maskPtr, pinnedMask, emptyInFlow, experimentName, my_image_spec,smooth_t0_est,regions,totalRegions, region_timing, my_keys, &bgParamH5.bgResidualError, &bgParamH5.kRateMultiplier, &bgParamH5.beadOnceParam);
  MemUsage ("AfterBgInitialization");

  // Image Loading thread setup to grab flows in the background
  ImageTracker my_img_set (numFlows,clo.img_control.ignoreChecksumErrors);
  bubbleTracker my_bubble_tracker;
  if (clo.bkg_control.filterBubbles == 1)
    my_bubble_tracker.Init (experimentName,my_image_spec.rows,my_image_spec.cols,numFlows,clo.flow_context.flowOrder); //@TODO: 4th duplicated code instance
  ImageLoadWorkInfo glinfo;
  SetUpImageLoaderInfo (glinfo,clo,pinnedMask, emptyInFlow, my_img_set, my_image_spec,my_bubble_tracker,numFlows);

  pthread_t loaderThread;
  pthread_create (&loaderThread, NULL, FileLoader, &glinfo);

  // Now do threaded solving, going through all the flows

  GlobalFitter.SpinUp();
// need to have initialized the regions for this
  GlobalFitter.SetRegionProcessOrder ();
  
  // process all flows...
  for (int flow = 0; flow < numFlows; flow++)
  {
    time_t flow_start;
    time_t flow_end;
    time (&flow_start);

    // coordinate with the ImageLoader threads for this flow to be read in
    // the loop guarantees all flows up this one have been read in
    // needed so emptyInFlow is determistic and data block is fully loaded
    my_img_set.WaitForFlowToLoad (flow);
    bool last = ( (flow) == (numFlows - 1));

    GlobalFitter.ExecuteFitForFlow (flow,my_img_set,last); // isolate this object so it can carry out actions in any order it chooses.

    MemUsage ("Memory_Flow: " + ToStr (flow));
    time (&flow_end);
    fprintf (stdout, "ProcessImage compute time for flow %d: %0.1lf sec.\n",
             flow, difftime (flow_end, flow_start));

    // capture the regional parameters every 20 flows, plus one bead per region at "random"
    // @TODO replace with clean hdf5 interface for sampling beads and region parameters
    DumpBkgModelRegionInfo (experimentName,GlobalFitter.BkgModelFitters,totalRegions,flow,last);
    DumpBkgModelBeadInfo (experimentName,GlobalFitter.BkgModelFitters,totalRegions,flow,last, clo.bkg_control.debug_bead_only>0);

    //@TODO:  no CLO use here - use GlobalDefaultsForBKGMODEL instead
    ApplyClonalFilter (experimentName, GlobalFitter.BkgModelFitters, totalRegions, clo.bkg_control.enableBkgModelClonalFilter, flow);

    // coordinate with the ImageLoader threads that this flow is done with
    // and release resources associated with this image
    my_img_set.FinishFlow (flow);

    IncrementalWriteWells (rawWells,flow,false,clo.bkg_control.saveWellsFrequency,numFlows);
    bgParamH5.IncrementalWrite(GlobalFitter, clo, flow, numFlows);
    DumpBkgModelRegionInfo(experimentName,GlobalFitter.BkgModelFitters,clo.loc_context.numRegions,numFlows-1, true);
    // individual regions output a '.' when they finish...terminate them all with a \n to keep the
    // display clean
    printf ("\n");
  }
  rawWells.WriteRanks();
  rawWells.WriteInfo();
  rawWells.Close();
  bgParamH5.Close();
  GlobalFitter.UnSpinGpuThreads ();

  CleanupLevMarSparseMatrices();
  GlobalDefaultsForBkgModel::StaticCleanup();
  pthread_join (loaderThread, NULL);

  delete[] emptyInFlow;
  emptyInFlow = NULL;
}

void IncrementalWriteWells (RawWells &rawWells,int flow, bool last_flow,int saveWellsFrequency,int numFlows)
{
  int testWellFrequency = saveWellsFrequency*NUMFB; // block size
  if ( ( (flow+1) % (saveWellsFrequency*NUMFB) == 0 && (flow != 0))  || (flow+1) >= numFlows || last_flow)
  {
    fprintf (stdout, "Writing incremental wells at flow: %d\n", flow);
    MemUsage ("BeforeWrite");
    rawWells.WriteWells();
    rawWells.SetChunk (0, rawWells.NumRows(), 0, rawWells.NumCols(), flow+1, min (testWellFrequency,numFlows- (flow+1)));
    MemUsage ("AfterWrite");
  }
}

void ApplyClonalFilter (const char* experimentName, BkgModel *BkgModelFitters[], int numRegions, bool doClonalFilter, int flow)
{
  if (flow == 80 and doClonalFilter)
  {
    deque<float> ppf;
    deque<float> ssq;
    GetFilterTrainingSample (ppf, ssq, BkgModelFitters, numRegions);
    DumpFilterInfo (experimentName, ppf, ssq);
    ApplyClonalFilter (BkgModelFitters, numRegions, ppf, ssq);
  }
}

void ApplyClonalFilter (BkgModel *BkgModelFitters[], int numRegions, const deque<float>& ppf, const deque<float>& ssq)
{
  clonal_filter filter;
  filter_counts counts;
  make_filter (filter, counts, ppf, ssq);

  for (int r=0; r<numRegions; ++r)
  {
    int numWells = BkgModelFitters[r]->GetNumLiveBeads();
    for (int well=0; well<numWells; ++well)
    {
      bead_params& bead = BkgModelFitters[r]->GetParams (well);
      bead.my_state.clonal_read  = filter.is_clonal (bead.my_state.ppf, bead.my_state.ssq);
    }
  }
}

void GetFilterTrainingSample (deque<float>& ppf, deque<float>& ssq, BkgModel *BkgModelFitters[], int numRegions)
{
  for (int r=0; r<numRegions; ++r)
  {
    int numWells = BkgModelFitters[r]->GetNumLiveBeads();
    for (int well=0; well<numWells; ++well)
    {
      bead_params bead;
      BkgModelFitters[r]->GetParams (well, &bead);
      if (bead.my_state.random_samp and not bead.my_state.bad_read)
      {
        ppf.push_back (bead.my_state.ppf);
        ssq.push_back (bead.my_state.ssq);
      }
    }
  }
}

void DumpFilterInfo (const char* experimentName, const deque<float>& ppf, const deque<float>& ssq)
{
  string fname = string (experimentName) + "/BkgModelFilterData.txt";
  ofstream out (fname.c_str());
  assert (out);

  deque<float>::const_iterator p = ppf.begin();
  deque<float>::const_iterator s = ssq.begin();
  for (; p!=ppf.end(); ++p, ++s)
  {
    out << setw (12) << setprecision (2) << scientific << *p
    << setw (12) << setprecision (2) << scientific << *s
    << endl;
  }
}

void DumpBkgModelBeadParams (char *experimentName, BkgModel *BkgModelFitters[], int numRegions, int flow, bool debug_bead_only)
{
  FILE *bkg_mod_bead_dbg = NULL;
  char *bkg_mod_bead_dbg_fname = (char *) malloc (512);
  snprintf (bkg_mod_bead_dbg_fname, 512, "%s/%s.%04d.%s", experimentName, "BkgModelBeadData",flow+1,"txt");
  fopen_s (&bkg_mod_bead_dbg, bkg_mod_bead_dbg_fname, "wt");
  free (bkg_mod_bead_dbg_fname);

  DumpBeadTitle (bkg_mod_bead_dbg);

  for (int r = 0; r < numRegions; r++)
  {
    BkgModelFitters[r]->DumpExemplarBead (bkg_mod_bead_dbg,debug_bead_only);
  }
  fclose (bkg_mod_bead_dbg);
}

void DumpBkgModelBeadOffset (char *experimentName, BkgModel *BkgModelFitters[], int numRegions, int flow, bool debug_bead_only)
{
  FILE *bkg_mod_bead_dbg = NULL;
  char *bkg_mod_bead_dbg_fname = (char *) malloc (512);
  snprintf (bkg_mod_bead_dbg_fname, 512, "%s/%s.%04d.%s", experimentName, "BkgModelBeadDcData",flow+1,"txt");
  fopen_s (&bkg_mod_bead_dbg, bkg_mod_bead_dbg_fname, "wt");
  free (bkg_mod_bead_dbg_fname);



  for (int r = 0; r < numRegions; r++)
  {
    BkgModelFitters[r]->DumpExemplarBeadDcOffset (bkg_mod_bead_dbg,debug_bead_only);
  }
  fclose (bkg_mod_bead_dbg);
}


void DumpBkgModelBeadInfo (char *experimentName, BkgModel *BkgModelFitters[], int numRegions, int flow, bool last_flow, bool debug_bead_only)
{
  // get some regional data for the entire chip as debug
  // only do this every 20 flows as this is the block
  // should be triggered by bkgmodel
  if ( (flow+1) % NUMFB ==0 || last_flow)
  {
    DumpBkgModelBeadParams (experimentName, BkgModelFitters, numRegions, flow, debug_bead_only);
    DumpBkgModelBeadOffset (experimentName, BkgModelFitters, numRegions, flow, debug_bead_only);
  }
}

void DumpBkgModelEmphasisTiming (char *experimentName,BkgModel *BkgModelFitters[],int numRegions, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_time_dbg = NULL;
  char *bkg_mod_time_name = (char *) malloc (512);
  snprintf (bkg_mod_time_name, 512, "%s/%s.%04d.%s", experimentName, "BkgModelEmphasisData",flow+1,"txt");
  fopen_s (&bkg_mod_time_dbg, bkg_mod_time_name, "wt");
  free (bkg_mod_time_name);

  for (int r = 0; r < numRegions; r++)
  {
    BkgModelFitters[r]->DumpTimeAndEmphasisByRegion (bkg_mod_time_dbg);

  }
  fclose (bkg_mod_time_dbg);
}

void DumpBkgModelInitVals (char *experimentName,BkgModel *BkgModelFitters[],int numRegions, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_init_dbg = NULL;
  char *bkg_mod_init_name = (char *) malloc (512);
  snprintf (bkg_mod_init_name, 512, "%s/%s.%04d.%s", experimentName, "BkgModelInitVals",flow+1,"txt");
  fopen_s (&bkg_mod_init_dbg, bkg_mod_init_name, "wt");
  free (bkg_mod_init_name);

  for (int r = 0; r < numRegions; r++)
  {
    BkgModelFitters[r]->DumpInitValues (bkg_mod_init_dbg);

  }
  fclose (bkg_mod_init_dbg);
}

void DumpBkgModelDarkMatter (char *experimentName,BkgModel *BkgModelFitters[],int numRegions, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_dark_dbg = NULL;
  char *bkg_mod_dark_name = (char *) malloc (512);
  snprintf (bkg_mod_dark_name, 512, "%s/%s.%04d.%s", experimentName, "BkgModelDarkMatterData",flow+1,"txt");
  fopen_s (&bkg_mod_dark_dbg, bkg_mod_dark_name, "wt");
  free (bkg_mod_dark_name);

  BkgModelFitters[0]->DumpDarkMatterTitle (bkg_mod_dark_dbg);

  for (int r = 0; r < numRegions; r++)
  {
    BkgModelFitters[r]->DumpDarkMatter (bkg_mod_dark_dbg);

  }
  fclose (bkg_mod_dark_dbg);
}

void DumpBkgModelEmptyTrace (char *experimentName,BkgModel *BkgModelFitters[],int numRegions, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_mt_dbg = NULL;
  char *bkg_mod_mt_name = (char *) malloc (512);
  snprintf (bkg_mod_mt_name, 512, "%s/%s.%04d.%s", experimentName, "BkgModelEmptyTraceData",flow+1,"txt");
  fopen_s (&bkg_mod_mt_dbg, bkg_mod_mt_name, "wt");
  free (bkg_mod_mt_name);


  for (int r = 0; r < numRegions; r++)
  {
    BkgModelFitters[r]->DumpEmptyTrace (bkg_mod_mt_dbg);

  }
  fclose (bkg_mod_mt_dbg);
}

void DumpBkgModelRegionParameters (char *experimentName,BkgModel *BkgModelFitters[],int numRegions, int flow)
{
  FILE *bkg_mod_reg_dbg = NULL;
  char *bkg_mod_reg_dbg_fname = (char *) malloc (512);
  snprintf (bkg_mod_reg_dbg_fname, 512, "%s/%s.%04d.%s", experimentName, "BkgModelRegionData",flow+1,"txt");
  fopen_s (&bkg_mod_reg_dbg, bkg_mod_reg_dbg_fname, "wt");
  free (bkg_mod_reg_dbg_fname);

  struct reg_params rp;

  DumpRegionParamsTitle (bkg_mod_reg_dbg);

  for (int r = 0; r < numRegions; r++)
  {
    BkgModelFitters[r]->GetRegParams (&rp);
    //@TODO this routine should have no knowledge of internal representation of variables
    // Make this a routine to dump an informative line to a selected file from a regional parameter structure/class
    // especially as we use a very similar dumping line a lot in different places.
    // note: t0, rdr, and pdr evolve over time.  It would be nice to also capture how they changed throughout the analysis
    // this only captures the final value of each.
    DumpRegionParamsLine (bkg_mod_reg_dbg, BkgModelFitters[r]->GetRegion()->row,BkgModelFitters[r]->GetRegion()->col, rp);

  }
  fclose (bkg_mod_reg_dbg);
}

void DumpBkgModelRegionInfo (char *experimentName,BkgModel *BkgModelFitters[],int numRegions, int flow, bool last_flow)
{
  // get some regional data for the entire chip as debug
  // only do this every 20 flows as this is the block
  // should be triggered by bkgmodel
  if ( (flow+1) % NUMFB ==0 || last_flow)
  {
    DumpBkgModelRegionParameters (experimentName, BkgModelFitters, numRegions, flow);
    DumpBkgModelDarkMatter (experimentName, BkgModelFitters, numRegions, flow);
    DumpBkgModelEmphasisTiming (experimentName,BkgModelFitters, numRegions, flow);
    DumpBkgModelEmptyTrace (experimentName,BkgModelFitters,numRegions,flow);
    DumpBkgModelInitVals (experimentName, BkgModelFitters,numRegions,flow);
  }
}



void SetUpRegions (Region *regions, int rows, int cols, int xinc, int yinc)
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
      if (regions[i].col + regions[i].w > cols)   // technically I don't think these ever hit since I'm truncating to calc xinc * yinc
        regions[i].w = cols - regions[i].col; // but better to be safe!
      if (regions[i].row + regions[i].h > rows)
        regions[i].h = rows - regions[i].row;
      i++;
    }
  }
}


void SetUpRegionDivisions (CommandLineOpts &clo, int rows, int cols)
{
  int xinc, yinc;

  clo.loc_context.regionsX = 1;
  clo.loc_context.regionsY = 1;

  // fixed region size
  xinc = clo.loc_context.regionXSize;
  yinc = clo.loc_context.regionYSize;
  clo.loc_context.regionsX = cols / xinc;
  clo.loc_context.regionsY = rows / yinc;
  // make sure we cover the edges in case rows/yinc or cols/xinc not exactly divisible
  if ( ( (double) cols / (double) xinc) != clo.loc_context.regionsX)
    clo.loc_context.regionsX++;
  if ( ( (double) rows / (double) yinc) != clo.loc_context.regionsY)
    clo.loc_context.regionsY++;
  clo.loc_context.numRegions = clo.loc_context.regionsX * clo.loc_context.regionsY;
}



void DoDiffSeparatorFromCLO (DifferentialSeparator *diffSeparator, CommandLineOpts &clo, Mask *maskPtr, string &analysisLocation, SequenceItem *seqList, int numSeqListItems)
{
  DifSepOpt opts;

  opts.bfType = clo.bfd_control.bfType;
  opts.bfDat = clo.bfd_control.bfDat;
  opts.bfBgDat = clo.bfd_control.bfBgDat;
  opts.resultsDir = clo.sys_context.dat_source_directory;
  opts.outData = analysisLocation;
  opts.analysisDir =  analysisLocation;
  opts.ignoreChecksumErrors = clo.img_control.ignoreChecksumErrors;
  opts.noduds = clo.bfd_control.noduds;

  opts.minRatioLiveWell = clo.bfd_control.bfMinLiveRatio;
  opts.doRecoverSdFilter = clo.bfd_control.skipBeadfindSdRecover == 0;

  opts.tfFilterQuantile = clo.bfd_control.bfTfFilterQuantile;
  opts.libFilterQuantile = clo.bfd_control.bfLibFilterQuantile;

  opts.flowOrder = clo.flow_context.flowOrder; // 5th duplicated code instance of translating flow order to nucs
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
  diffSeparator->SetKeys (seqList, numSeqListItems, clo.bfd_control.bfMinLiveLibSnr, clo.bfd_control.bfMinLiveTfSnr);
  if (clo.bfd_control.beadfindLagOneFilt > 0)
  {
    opts.filterLagOneSD = true;
  }
  if (clo.bfd_control.beadfindThumbnail == 1)
  {
    opts.t0MeshStep = clo.loc_context.regionXSize;
    opts.bfMeshStep = clo.loc_context.regionXSize;
    opts.tauEEstimateStep = clo.loc_context.regionXSize;
    opts.useMeshNeighbors = 0;
    opts.regionXSize = clo.loc_context.regionXSize;
    opts.regionYSize = clo.loc_context.regionYSize;
  }
  diffSeparator->Run (opts);
}


void SetUpToProcessImages (ImageSpecClass &my_image_spec, CommandLineOpts &clo, char *experimentName, TrackProgress &my_progress)
{
  // set up to process images aka 'dat' files.
  ExportSubRegionSpecsToImage (clo);

  // make sure we're using XTCorrection at the right offset for cropped regions
  Image::SetCroppedRegionOrigin (clo.loc_context.cropped_region_x_offset,clo.loc_context.cropped_region_y_offset);
  Image::CalibrateChannelXTCorrection (clo.sys_context.dat_source_directory,"lsrowimage.dat");

  my_image_spec.DeriveSpecsFromDat (clo,1,experimentName);   // dummy - only reads 1 dat file
  fprintf (my_progress.fpLog, "VFR = %s\n", my_image_spec.vfr_enabled ? "enabled":"disabled");    // always enabled these days, useless?
}

void SetUpRegionsForAnalysis (ImageSpecClass &my_image_spec, CommandLineOpts &clo, Region &wholeChip)
{

  FixCroppedRegions (clo, my_image_spec);

  SetUpRegionDivisions (clo,my_image_spec.rows,my_image_spec.cols);
  SetUpWholeChip (wholeChip,my_image_spec.rows,my_image_spec.cols);

}


void SetupForBkgModelTiming (DifferentialSeparator *diffSeparator, std::vector<float> &smooth_t0_est, RegionTiming *region_timing,
                             Region *region_list, int numRegions, ImageSpecClass &my_image_spec, Mask *maskPtr, bool doSmoothing)
{
  // compute timing information
  AvgKeyIncorporation *keyIncorporation = NULL;

  //Create a mask that tracks the pinned pixels discovered in each image
  maskPtr->CalculateLiveNeighbors();

  // Setup t0 estimation from beadfind to pass to background model
  std::vector<float> sep_t0_est;
  sep_t0_est = diffSeparator->GetT0();
  smooth_t0_est = sep_t0_est;
  if (doSmoothing)
  {
    printf ("smoothing t0 estimate from separator.......");
    NNSmoothT0Estimate (maskPtr,my_image_spec.rows,my_image_spec.cols,sep_t0_est,smooth_t0_est);
    printf ("done.\n");
  }
  // do some incorporation signal modeling
  keyIncorporation = diffSeparator;
  //FillRegionalTimingParameters(region_timing, region_list, numRegions, keyIncorporation);
  threadedFillRegionalTimingParameters (region_timing,region_list,numRegions,keyIncorporation);
  keyIncorporation = NULL;
}

void CreateWellsFileForWriting (RawWells &rawWells, Mask *maskPtr,
                                CommandLineOpts &clo,
                                int numFlows,
                                int numRows, int numCols,
                                const char *chipType)
{
  // set up wells data structure
  MemUsage ("BeforeWells");
   int flowChunk = min(clo.bkg_control.saveWellsFrequency*NUMFB, numFlows);
   //rawWells.SetFlowChunkSize(flowChunk);
  rawWells.SetCompression (3);
  rawWells.SetRows (numRows);
  rawWells.SetCols (numCols);
  rawWells.SetFlows (numFlows);
  rawWells.SetFlowOrder (clo.flow_context.flowOrder); // 6th duplicated code
  SetWellsToLiveBeadsOnly (rawWells,maskPtr);
  // any model outputs a wells file of this nature
  GetMetaDataForWells (clo.sys_context.dat_source_directory,rawWells,chipType);
  rawWells.SetChunk (0, rawWells.NumRows(), 0, rawWells.NumCols(), 0, flowChunk);
  rawWells.OpenForWrite();
  MemUsage ("AfterWells");
}


// output from this are a functioning wells file and a beadfind mask
// images are only known internally to this.
void RealImagesToWells (RawWells &rawWells, Mask *maskPtr,
                        CommandLineOpts &clo,
                        char *experimentName, string &analysisLocation,
                        int numFlows,
                        SeqListClass &my_keys,
                        TrackProgress &my_progress, Region &wholeChip,
                        int &well_rows, int &well_cols)
{

  char *chipType = GetChipId (clo.sys_context.dat_source_directory);
  ChipIdDecoder::SetGlobalChipId (chipType);

  ImageSpecClass my_image_spec;
  SetUpToProcessImages (my_image_spec, clo, experimentName, my_progress);

  SetUpRegionsForAnalysis (my_image_spec, clo, wholeChip);

  SetExcludeMask (clo,maskPtr,chipType,my_image_spec.rows,my_image_spec.cols);

  // Write processParameters.parse file now that processing is about to begin
  clo.WriteProcessParameters();

  // region definitions, in theory shared between background model and beadfind
  int totalRegions = clo.loc_context.numRegions;  // we may omit some regions due to exclusion or other ideas
  Region region_list[totalRegions];
  SetUpRegions (region_list,my_image_spec.rows,my_image_spec.cols,clo.loc_context.regionXSize,clo.loc_context.regionYSize);
  // need crude nuc rise timing information for bkg model currently
  RegionTiming region_timing[totalRegions];
  // global timing estimates - need to have to pass to bkgmodel
  std::vector<float> smooth_t0_est;

  /*********************************************************************
  // Beadfind Section
   *********************************************************************/

  if (clo.bfd_control.beadfindType == "differential")
  {
    DifferentialSeparator *diffSeparator=NULL;

    diffSeparator = new DifferentialSeparator();
    DoDiffSeparatorFromCLO (diffSeparator, clo, maskPtr, analysisLocation, my_keys.seqList,my_keys.numSeqListItems);
    // now actually set up the mask I want
    maskPtr->Copy (diffSeparator->GetMask());
    my_progress.ReportState ("Beadfind Complete");

    SetupForBkgModelTiming (diffSeparator, smooth_t0_est, region_timing,
                            region_list, totalRegions, my_image_spec, maskPtr, clo.bfd_control.beadfindThumbnail == 0);
    // cleanup scope
    if (diffSeparator !=NULL)
    {
      delete diffSeparator;
      diffSeparator = NULL;
    }
    // Update progress bar status file: well proc complete/image proc started
    updateProgress (WELL_TO_IMAGE);
  }
  else
  {
    fprintf (stderr, "Don't recognize --beadfind-type %s\n", clo.bfd_control.beadfindType.c_str());
    exit (EXIT_FAILURE);
  }

  UpdateBeadFindOutcomes (maskPtr, wholeChip, experimentName, clo, 0);
  my_progress.ReportState ("Bead Categorization Complete");

  if (clo.mod_control.BEADFIND_ONLY)
  {
    // Remove temporary wells file
    if (clo.sys_context.LOCAL_WELLS_FILE)
      unlink (clo.sys_context.tmpWellsFile);
    fprintf (stdout,
             "Beadfind Only Mode has completed successfully\n");
    exit (EXIT_SUCCESS);
  }


  /********************************************************************
   *
   *  Background Modelling Process
   *
   *******************************************************************/

  CreateWellsFileForWriting (rawWells, maskPtr, clo,
                             numFlows, my_image_spec.rows, my_image_spec.cols,
                             chipType);

  // we might use some other model here
  if (clo.mod_control.USE_BKGMODEL)
  {
    // Here's the point where we really do the background model and not just setup for it
    // @TODO: isolate bkg_control variables here - we should >not< use clo as a garbage variable.
    DoThreadedBackgroundModel (rawWells, clo, maskPtr, experimentName, numFlows, chipType, my_image_spec, smooth_t0_est, region_list, totalRegions, region_timing,my_keys);

  } // end BKG_MODEL

  // whatever model we run, copy the signals to permanent location
  CopyTmpWellFileToPermanent (clo, experimentName);
  // mask file may be updated by model processing
  UpdateBeadFindOutcomes (maskPtr, wholeChip, experimentName, clo, 0);


  my_progress.ReportState ("Raw flowgrams complete");
  // we deal only with wells from now on
  well_rows=my_image_spec.rows;
  well_cols=my_image_spec.cols;

  free (chipType);
}


// output from this are a functioning wells file and a beadfind mask
// images are only known internally to this.
void GetFromImagesToWells (RawWells &rawWells, Mask *maskPtr,
                           CommandLineOpts &clo,
                           char *experimentName, string &analysisLocation,
                           int numFlows,
                           SeqListClass &my_keys,
                           TrackProgress &my_progress, Region &wholeChip,
                           int &well_rows, int &well_cols)
{

  /*
   *  Two ways to proceed.  If this is raw data analysis, continue from here.
   *  If this is a re-basecall of an existing wells file, jump to basecalling
   *  section.
   */
  if (!clo.mod_control.USE_RAWWELLS)
  {
    // do beadfind and signal processing
    RealImagesToWells (rawWells, maskPtr,clo,experimentName,analysisLocation,
                       numFlows,my_keys,my_progress,
                       wholeChip,well_rows,well_cols);

  } // end image processing
  else    // Processing --from-wells - no images are ever used
  {
    // Update progress bar file: well find is complete
    updateProgress (WELL_TO_IMAGE);

    // grab the previous beadfind
    LoadBeadMaskFromFile (clo, maskPtr,well_rows,well_cols);

    // copy/link files from old directory for the report
    MakeSymbolicLinkToOldDirectory (clo, experimentName);
    CopyFilesForReportGeneration (clo, experimentName,my_keys);
    SetChipTypeFromWells (rawWells);
  }
}
