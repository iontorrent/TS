/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ImageLoader.h"
#include "Utils.h"


void *FileLoadWorker (void *arg)
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
    ImageLoadWorkInfo *info = (ImageLoadWorkInfo *) item.private_data;

    time_t start;
    time_t end;
    time (&start);

    // dump dc offset information before we do any normalization
    DumpDcOffset (info);

    info->img[info->flow].Normalize(info->normStart, info->normEnd);

    // correct in-channel electrical cross-talk
    info->img[info->flow].XTChannelCorrect();

    // dump pH step debug info to file (its really fast, don't worry)
    DumpStep (info);

    if (info->doRawBkgSubtract)
    {
      info->img[info->flow].BackgroundCorrect (info->mask, MaskBead, MaskEmpty, info->NNinnerx,
          info->NNinnery,info->NNouterx,info->NNoutery, NULL, false, true, true);
      if (info->flow==0)
        printf ("Notify user: NN empty subtraction in effect.  No further warnings will be given. %d \n",info->flow);
    }

    // Calculate average background for each well
    info->emptyTraceTracker->SetEmptyTracesFromImage (info->img[info->flow], *(info->pinnedInFlow), info->flow, info->mask);

    // just a note: no guarantee that prior flows have finished yet
    ( (int volatile *) info->CurRead) [info->flow] = 1;
    time (&end);

    fprintf (stdout, "FileLoadWorker: ImageProcessing time for flow %d: %0.1lf sec\n", info->flow, difftime (end,start));

    q->DecrementDone();
  }

  return (NULL);
}

void *FileLoader (void *arg)
{
  ImageLoadWorkInfo *info = (ImageLoadWorkInfo *) arg;
  int flow = 0;

  WorkerInfoQueue *loadWorkQ = new WorkerInfoQueue (info->flow);
  ImageLoadWorkInfo *ninfo = new ImageLoadWorkInfo[info->flow];

  for (flow = 0; flow < info->flow; flow++)
  {
    memcpy (&ninfo[flow], info, sizeof (*info));
    ninfo[flow].flow = flow;
  }
  WorkerInfoQueueItem item;
  int numWorkers = numCores() /2; // @TODO - this should be subject to clo options
  // int numWorkers = 1;
  numWorkers = (numWorkers < 1 ? 1:numWorkers);
  fprintf (stdout, "FileLoader: numWorkers threads = %d\n", numWorkers);
  {
    int cworker;
    pthread_t work_thread;
    // spawn threads for doing background correction/fitting work
    for (cworker = 0; cworker < numWorkers; cworker++)
    {
      int t = pthread_create (&work_thread, NULL, FileLoadWorker,
                              loadWorkQ);
      if (t)
        fprintf (stderr, "Error starting thread\n");
    }
  }

  time_t start, end;
  int numFlows = info->flow;
  for (flow = 0; flow < numFlows; flow++)
  {
    if (flow%NUMFB == 0)
      time (&start);

    sprintf (info->name, "%s/%s%04d.dat", info->dat_source_directory, info->acqPrefix,
             (flow + (flow / info->numFlowsPerCycle) * info->hasWashFlow));

    // don't read ahead too far of BkgModel regional fitter threads
    while (flow > info->lead &&
           ! ( (int volatile *) info->CurProcessed) [flow - info->lead])
      sleep (1);

    if (!info->img[flow].LoadRaw (info->name))
    {
      exit (EXIT_FAILURE);
    }
    // pinning updates only need to be complete to remove indeterminacy
    // in values dumped in DumpStep in FileLoadWorker
    info->pinnedInFlow->Update (flow, &info->img[flow]);

    item.finished = false;
    item.private_data = &ninfo[flow];
    loadWorkQ->PutItem (item);

    // wait for the the BkgModel regional fitter threads to finish this block
    while ( ( (flow+1) %NUMFB == 0) && ! ( (int volatile *) info->CurProcessed[flow]))
      sleep (1);

    if ( (flow+1) %NUMFB == 0)
    {
      time (&end);
      fprintf (stdout, "FileLoader: Total time taken from flow %d to %d: %0.1lf sec\n", ( (flow+1) - NUMFB),
               flow, difftime (end, start));
    }
  }

  // wait for all of the images to be processed
  loadWorkQ->WaitTillDone();

  for (int cworker = 0; cworker < numWorkers; cworker++)
  {
    item.private_data = NULL;
    item.finished = true;
    loadWorkQ->PutItem (item);
  }
  // wait for all workers to exit
  loadWorkQ->WaitTillDone();
  delete loadWorkQ;
  delete[] ninfo;

  info->finished = true;

  return NULL;
}

void ImageTracker::WaitForFlowToLoad (int flow)
{
  // disable compiler optimizing this loop away
  while (! ( (int volatile *) CurRead) [flow])
  {
    sleep (1); // wait for the load worker to load the current image
  }
}

ImageTracker::ImageTracker (int _numFlows, int ignoreChecksumErrors, Mask *maskPtr)
{
  numFlows = _numFlows;
  img = new Image[numFlows];

  for (int n = 0; n < numFlows; n++)
  {
    img[n].SetImgLoadImmediate (false);
    img[n].SetNumAcqFiles (numFlows);
    img[n].SetIgnoreChecksumErrors (ignoreChecksumErrors);
  }
  CurRead = new unsigned int [numFlows];
  CurProcessed = new unsigned int [numFlows];
  memset (CurRead, 0, numFlows*sizeof (unsigned int));
  memset (CurProcessed, 0, numFlows*sizeof (unsigned int));

  // this class tracks pinned wells in each flow as we load it
  pinnedInFlow = NULL;
}

void ImageTracker::FinishFlow (int flow)
{
  img[flow].Close();
  ( (int volatile *) CurProcessed) [flow] = 1;
}

ImageTracker::~ImageTracker()
{
  if (img!=NULL) delete[] img;
  if (CurRead !=NULL) delete[] CurRead;
  if (CurProcessed!=NULL) delete[] CurProcessed;
  if (pinnedInFlow != NULL) delete pinnedInFlow;

}

void ImageTracker::InitPinnedInFlow(int numFlows, Mask *maskPtr, CommandLineOpts& clo)
{
  assert(pinnedInFlow == NULL);  // there is only one, just in case called again...
  if (pinnedInFlow == NULL)
  {
    if (clo.bkg_control.replayBkgModelData)
      pinnedInFlow = new PinnedInFlowReader(maskPtr, numFlows, clo);
    else if (clo.bkg_control.recordBkgModelData)
      pinnedInFlow = new PinnedInFlowRecorder(maskPtr, numFlows, clo);
    else
      pinnedInFlow = new PinnedInFlow(maskPtr, numFlows);
    
    pinnedInFlow->Initialize(maskPtr);
  }
}

void SetUpImageLoaderInfo (ImageLoadWorkInfo &glinfo, CommandLineOpts &clo,
                           Mask *maskPtr, ImageTracker &my_img_set,
                           ImageSpecClass &my_image_spec,
                           EmptyTraceTracker &emptytracetracker,
                           int numFlows)
{

  int normStart = 5;
  int normEnd = 20;

  // do this here as my_img_set is split into pieces
  my_img_set.InitPinnedInFlow(numFlows, maskPtr, clo);

  glinfo.type = imageLoadAllE;
  glinfo.flow = numFlows;
  glinfo.img = my_img_set.img;  // just use ImageTracker object instead?
  glinfo.pinnedInFlow = my_img_set.pinnedInFlow;
  glinfo.mask = maskPtr;
  glinfo.normStart = normStart;
  glinfo.normEnd = normEnd;
  glinfo.NNinnerx = clo.img_control.NNinnerx;
  glinfo.NNinnery = clo.img_control.NNinnery;
  glinfo.NNouterx = clo.img_control.NNouterx;
  glinfo.NNoutery = clo.img_control.NNoutery;
  glinfo.CurRead = & (my_img_set.CurRead[0]);
  glinfo.CurProcessed = & (my_img_set.CurProcessed[0]);
  glinfo.dat_source_directory = clo.sys_context.dat_source_directory;
  glinfo.acqPrefix = my_image_spec.acqPrefix;
  glinfo.numFlowsPerCycle = clo.flow_context.numFlowsPerCycle; // @TODO: really?  is this even used correctly
  glinfo.hasWashFlow = clo.GetWashFlow();
  glinfo.finished = false;
  glinfo.lead = (clo.bkg_control.readaheadDat != 0) ? clo.bkg_control.readaheadDat : my_image_spec.LeadTimeForChipSize();
  glinfo.emptyTraceTracker = &emptytracetracker;
  glinfo.clo = &clo;
  printf("Subtract Empties: %d\n", clo.img_control.nn_subtract_empties);
  glinfo.doRawBkgSubtract = (clo.img_control.nn_subtract_empties>0);
}


//@TODO: why is this not a method of info?
void DumpStep (ImageLoadWorkInfo *info)
{
//@TODO:  if only there were some object, say a flow context, that had already converted the flowOrder into the string of nucleotides to the length of all the flows
// we wouldn't need to write code like this...but that would be crazy talk.
  char nucChar = info->clo->flow_context.flowOrder[info->flow % strlen (info->clo->flow_context.flowOrder) ];  // @TODO: 12th time this is done independently in the code(!!!!!)
  string nucStepDir = string (info->clo->GetExperimentName()) + string ("/NucStep");

  // Make output directory or quit
  if (mkdir (nucStepDir.c_str(), 0777) && (errno != EEXIST))
  {
    perror (nucStepDir.c_str());
    return;
  }

  // Set region width & height
  int rWidth=50;
  int rHeight=50;

  // Lower left corner of the region should equal 0 modulus xModulus or yModulus
  int xModulus=50;
  int yModulus=50;

  vector<string> regionName;
  vector<int> regionStartCol;
  vector<int> regionStartRow;
  vector<int> regionWidth;
  vector<int> regionHeight;

  const RawImage *rawImg = info->img[info->flow].GetImage();

  ChipIdEnum chipId = ChipIdDecoder::GetGlobalChipId();
  if (chipId == ChipId900)
  {
    // Proton chips
    regionName.push_back ("inlet");
    regionStartCol.push_back ( xModulus * floor(0.1 * (float) rawImg->cols / (float) xModulus) );
    regionStartRow.push_back ( yModulus * floor(0.5 * (float) rawImg->rows / (float) yModulus) );
    regionWidth.push_back (rWidth);
    regionHeight.push_back(rHeight);

    regionName.push_back ("middle");
    regionStartCol.push_back ( xModulus * floor(0.5 * (float) rawImg->cols / (float) xModulus) );
    regionStartRow.push_back ( yModulus * floor(0.5 * (float) rawImg->rows / (float) yModulus) );
    regionWidth.push_back (rWidth);
    regionHeight.push_back(rHeight);

    regionName.push_back ("outlet");
    regionStartCol.push_back ( xModulus * floor(0.9 * (float) rawImg->cols / (float) xModulus) );
    regionStartRow.push_back ( yModulus * floor(0.5 * (float) rawImg->rows / (float) yModulus) );
    regionWidth.push_back (rWidth);
    regionHeight.push_back(rHeight);
  }
  else
  {
    // PGM chips
    regionName.push_back ("inlet");
    regionStartCol.push_back ( xModulus * floor(0.9 * (float) rawImg->cols / (float) xModulus) );
    regionStartRow.push_back ( yModulus * floor(0.9 * (float) rawImg->rows / (float) yModulus) );
    regionWidth.push_back (rWidth);
    regionHeight.push_back(rHeight);

    regionName.push_back ("middle");
    regionStartCol.push_back ( xModulus * floor(0.5 * (float) rawImg->cols / (float) xModulus) );
    regionStartRow.push_back ( yModulus * floor(0.5 * (float) rawImg->rows / (float) yModulus) );
    regionWidth.push_back (rWidth);
    regionHeight.push_back(rHeight);

    regionName.push_back ("outlet");
    regionStartCol.push_back ( xModulus * floor(0.1 * (float) rawImg->cols / (float) xModulus) );
    regionStartRow.push_back ( yModulus * floor(0.1 * (float) rawImg->rows / (float) yModulus) );
    regionWidth.push_back (rWidth);
    regionHeight.push_back(rHeight);
  }

  for (unsigned int iRegion=0; iRegion<regionStartRow.size(); iRegion++)
  {
    info->img[info->flow].DumpStep (
      regionStartCol[iRegion],
      regionStartRow[iRegion],
      regionWidth[iRegion],
      regionHeight[iRegion],
      regionName[iRegion],
      nucChar,
      nucStepDir,
      info->mask,
      info->pinnedInFlow,
      info->flow
    );
  }
}

void DumpDcOffset (ImageLoadWorkInfo *info)
{
  char nucChar = info->clo->flow_context.flowOrder[info->flow % strlen (info->clo->flow_context.flowOrder) ];  // @TODO: 13th time this is done independently in the code(!!!!!)
  string dcOffsetDir = string (info->clo->GetExperimentName()) + string ("/dcOffset");

  // Make output directory or quit
  if (mkdir (dcOffsetDir.c_str(), 0777) && (errno != EEXIST))
  {
    perror (dcOffsetDir.c_str());
    return;
  }

  int nSample=100000;
  info->img[info->flow].DumpDcOffset (
    nSample,
    dcOffsetDir,
    nucChar,
    info->flow
  );
}
