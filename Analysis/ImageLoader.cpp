/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ImageLoader.h"


void *FileLoadWorker(void *arg)
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
    ImageLoadWorkInfo *info = (ImageLoadWorkInfo *) item.private_data;

    time_t start;
    time_t end;
    time(&start);
    if (info->bubbleWells != NULL)
    {
      FilterBubbleWells(&info->img[info->flow], info->flow, info->lmask,
                        info->bubbleWells, info->bubbleSdWells);
    }

    info->img[info->flow].FilterForPinned(info->lmask, MaskEmpty); //@TODO non-deterministic, does that matter?
    UpdateEmptyWells(info->flow, &info->img[info->flow], info->emptyInFlow);
    info->img[info->flow].Normalize(info->normStart, info->normEnd);

    // correct in-channel electrical cross-talk
    info->img[info->flow].XTChannelCorrect(info->lmask);

    // dump pH step debug info to file (its really fast, don't worry)
    char nucChar = info->clo->flow_context.flowOrder[info->flow % strlen(info->clo->flow_context.flowOrder)]; // @TODO: first duplicated code event
    char nucStepFileName[256];
    double step;
    const RawImage *rawImg = info->img[info->flow].GetImage();
    sprintf(nucStepFileName, "%s/NucStep_outlet_%c.txt", info->clo->GetExperimentName(), nucChar);
    step = info->img[info->flow].DumpStep((int)(rawImg->rows*0.1), (int)(rawImg->cols*0.1), 10, 10, 2.0, nucStepFileName, info->lmask, info->flow);
    sprintf(nucStepFileName, "%s/NucStep_midlet_%c.txt", info->clo->GetExperimentName(), nucChar);
    step = info->img[info->flow].DumpStep((int)(rawImg->rows*0.4), (int)(rawImg->cols*0.4), 10, 10, 2.0, nucStepFileName, info->lmask, info->flow);
    sprintf(nucStepFileName, "%s/NucStep_inlet_%c.txt", info->clo->GetExperimentName(), nucChar);
    step = info->img[info->flow].DumpStep((int)(rawImg->rows*0.9), (int)(rawImg->cols*0.9), 10, 10, 2.0, nucStepFileName, info->lmask, info->flow);

    // Calculate average background for each well
#ifndef SINGLE_BKG_TRACE
    info->img[info->flow].BackgroundCorrect(info->lmask, MaskBead, MaskEmpty, info->NNinnerx,
                                            info->NNinnery,info->NNouterx,info->NNoutery, NULL, false, true, true);
#endif
    // just a note: no guarantee that prior flows have finished yet
    info->CurRead[info->flow] = 1;
    time(&end);

    fprintf(stdout, "FileLoadWorker: ImageProcessing time for flow %d: %0.1lf sec\n", info->flow, difftime(end,start));

    q->DecrementDone();
  }

  return (NULL);
}

void *FileLoader(void *arg)
{
  ImageLoadWorkInfo *info = (ImageLoadWorkInfo *) arg;
  int flow = 0;

  WorkerInfoQueue *loadWorkQ = new WorkerInfoQueue(info->flow);
  ImageLoadWorkInfo *ninfo = new ImageLoadWorkInfo[info->flow];

  for (flow = 0; flow < info->flow; flow++)
  {
    memcpy(&ninfo[flow], info, sizeof(*info));
    ninfo[flow].flow = flow;
  }
  WorkerInfoQueueItem item;
  int numWorkers = numCores()/2;  // @TODO - this should be subject to clo options
  // int numWorkers = 1;
  numWorkers = (numWorkers < 1 ? 1:numWorkers);
  fprintf(stdout, "FileLoader: numWorkers threads = %d\n", numWorkers);
  {
    int cworker;
    pthread_t work_thread;
    // spawn threads for doing background correction/fitting work
    for (cworker = 0; cworker < numWorkers; cworker++)
    {
      int t = pthread_create(&work_thread, NULL, FileLoadWorker,
                             loadWorkQ);
      if (t)
        fprintf(stderr, "Error starting thread\n");
    }
  }

  time_t start, end;
  int numFlows = info->flow;
  for (flow = 0; flow < numFlows; flow++)
  {
    if (flow%NUMFB == 0)
        time(&start);

    sprintf(info->name, "%s/%s%04d.dat", info->dat_source_directory, info->acqPrefix,
            (flow + (flow / info->numFlowsPerCycle) * info->hasWashFlow));

    // don't read ahead too far of BkgModel regional fitter threads
    while (flow > info->lead && !info->CurProcessed[flow - info->lead])
        sleep(1);

    if (!info->img[flow].LoadRaw(info->name))
    {
      exit(EXIT_FAILURE);
    }

    item.finished = false;
    item.private_data = &ninfo[flow];
    loadWorkQ->PutItem(item);
    
    // wait for the the BkgModel regional fitter threads to finish this block
    while (((flow+1)%NUMFB == 0) && !info->CurProcessed[flow])
        sleep(1);

    if ((flow+1)%NUMFB == 0) {
        time(&end);
        fprintf(stdout, "FileLoader: Total time taken from flow %d to %d: %0.1lf sec\n", ((flow+1) - NUMFB), 
            flow, difftime(end, start));
    }
  }

  // wait for all of the images to be processed
  loadWorkQ->WaitTillDone();

  for (int cworker = 0; cworker < numWorkers; cworker++)
  {
    item.private_data = NULL;
    item.finished = true;
    loadWorkQ->PutItem(item);
  }
  // wait for all workers to exit
  loadWorkQ->WaitTillDone();
  delete loadWorkQ;
  delete[] ninfo;

  info->finished = true;

  return NULL;
}


void FilterBubbleWells(Image *img, int flowIx, Mask *mask,
                       RawWells *bubbleWells, RawWells *bubbleSdWells)
{
  const RawImage *raw = img->GetImage();
  int rows = raw->rows;
  int cols = raw->cols;
  if (rows <= 0 || cols <= 0)
  {
    cout << "Why bad row/cols for flow: " << flowIx << " rows: " << rows << " cols: " << cols << endl;
    exit(EXIT_FAILURE);
  }
  vector<float> wellMean(raw->rows * raw->cols, 0);
  vector<float> wellSd(raw->rows * raw->cols, 0);
  SampleStats<float> wellStat;
  SampleQuantiles<float> sampMeanQuantiles(1000);
  SampleQuantiles<float> sampSdQuantiles(1000);
  for (int rowIx = 0; rowIx < raw->rows; rowIx++)
  {
    for (int colIx = 0; colIx < raw->cols; colIx++)
    {
      int idx = rowIx * raw->cols + colIx;
      if ((*mask)[idx] & MaskExclude || (*mask)[idx] & MaskPinned)
      {
        continue;
      }
      assert((size_t)idx < wellMean.size());
      wellStat.Clear();
      for (int frameIx = 0; frameIx < raw->frames; frameIx++)
      {
        float val = raw->image[frameIx * raw->frameStride + colIx + rowIx * raw->cols] - raw->image[ colIx + rowIx * raw->cols];
        wellStat.AddValue(val);
      }
      wellMean[idx] = wellStat.GetMean();
      wellSd[idx] = wellStat.GetSD();
      sampMeanQuantiles.AddValue(wellMean[idx]);
      sampSdQuantiles.AddValue(wellSd[idx]);
    }
  }
  double meanThreshold = 0;
  double sdThreshold = 0;
  if (sampSdQuantiles.GetNumSeen() > 0)
  {
    meanThreshold = sampMeanQuantiles.GetMedian() - (5 * (sampMeanQuantiles.GetQuantile(.75) - sampMeanQuantiles.GetQuantile(.25))/2);;
    sdThreshold = sampSdQuantiles.GetMedian() - (5 * (sampSdQuantiles.GetQuantile(.75) - sampSdQuantiles.GetQuantile(.25))/2);
  }
  int meanFlagged = 0;
  int sdFlagged = 0;
  for (size_t i = 0; i < wellMean.size(); i++)
  {
    float val = 0;
    if ((*mask)[i] & MaskExclude || (*mask)[i] & MaskPinned)
    {
      continue;
    }
    if (wellMean[i] < meanThreshold)
    {
      val = 1.0;
      meanFlagged++;
      bubbleWells->WriteFlowgram(flowIx, i  % cols, i / cols, 1.0);
    }
    else if (wellSd[i] < sdThreshold)
    {
      sdFlagged++;
      bubbleSdWells->WriteFlowgram(flowIx, i  % cols, i / cols, 1.0);
    }
  }
  cout << "Flow : " << flowIx << " Flagged mean: " << meanFlagged << " and sd: " << sdFlagged << endl;
}

#ifdef __INTEL_COMPILER
#pragma optimize ( "", off )
#endif
void ImageTracker::WaitForFlowToLoad(int flow)
{
  // disable compiler optimizating this loop away
  // while (! ((int volatile *)CurRead) [flow]){ // @TODO 
  while (!CurRead[flow]){
    sleep(1); // wait for the load worker to load the current image
  }
}
#ifdef __INTEL_COMPILER
#pragma optimize ( "", on )
#endif

ImageTracker::ImageTracker(int _numFlows, int ignoreChecksumErrors)
{
  numFlows = _numFlows;
  img = new Image[numFlows];
  for (int n = 0; n < numFlows; n++)
  {
    img[n].SetImgLoadImmediate(false);
    img[n].SetNumAcqFiles(numFlows);
    img[n].SetIgnoreChecksumErrors(ignoreChecksumErrors);
  }
  CurRead = new unsigned int [numFlows];
  CurProcessed = new unsigned int [numFlows];
  memset(CurRead, 0, numFlows*sizeof(unsigned int));
  memset(CurProcessed, 0, numFlows*sizeof(unsigned int));
}
void ImageTracker::FinishFlow(int flow)
{
  img[flow].Close();
  CurProcessed[flow] = 1;
}

ImageTracker::~ImageTracker()
{
  if (img!=NULL) delete[] img;
  if (CurRead !=NULL) delete[] CurRead;
  if (CurProcessed!=NULL) delete[] CurProcessed;
}


bubbleTracker::bubbleTracker()
{
  bubbleWells=NULL;
  bubbleSdWells = NULL;
}

void bubbleTracker::Init(char *experimentName, int rows, int cols, int numFlows, char *flowOrder)
{
  cout << "Filtering bubbles..." << numFlows << " " << flowOrder << " " << rows << " " << cols << endl;
  bubbleWells = new RawWells(experimentName, "bubbleMean.wells");
  bubbleWells->CreateEmpty(numFlows, flowOrder, rows, cols);
  bubbleWells->OpenForWrite();

  bubbleSdWells = new RawWells(experimentName, "bubbleSd.wells");
  bubbleSdWells->CreateEmpty(numFlows, flowOrder, rows, cols);
  bubbleSdWells->OpenForWrite();
}
void bubbleTracker::Close()
{
  if (bubbleWells != NULL)
  {
    bubbleWells->Close();
    delete bubbleWells;
  }
  if (bubbleSdWells != NULL)
  {
    bubbleSdWells->Close();
    delete bubbleSdWells;
  }
}
bubbleTracker::~bubbleTracker()
{
  Close();
}


void SetUpImageLoaderInfo(ImageLoadWorkInfo &glinfo, CommandLineOpts &clo, Mask &localMask, short *emptyInFlow,
                          ImageTracker &my_img_set, ImageSpecClass &my_image_spec, bubbleTracker &my_bubble_tracker, int numFlows)
{

  int normStart = 5;
  int normEnd = 20;

  glinfo.type = imageLoadAllE;
  glinfo.flow = numFlows;
  glinfo.img = my_img_set.img;
  glinfo.lmask = &localMask;
  glinfo.emptyInFlow = emptyInFlow;
  glinfo.normStart = normStart;
  glinfo.normEnd = normEnd;
  glinfo.NNinnerx = clo.img_control.NNinnerx;
  glinfo.NNinnery = clo.img_control.NNinnery;
  glinfo.NNouterx = clo.img_control.NNouterx;
  glinfo.NNoutery = clo.img_control.NNoutery;
  glinfo.CurRead = &(my_img_set.CurRead[0]);
  glinfo.CurProcessed = &(my_img_set.CurProcessed[0]);
  glinfo.dat_source_directory = clo.sys_context.dat_source_directory;
  glinfo.acqPrefix = my_image_spec.acqPrefix;
  glinfo.numFlowsPerCycle = clo.flow_context.numFlowsPerCycle; // @TODO: really?  is this even used correctly
  glinfo.hasWashFlow = clo.GetWashFlow();
  glinfo.finished = false;
  glinfo.bubbleWells = my_bubble_tracker.bubbleWells;
  glinfo.bubbleSdWells = my_bubble_tracker.bubbleSdWells;
  glinfo.lead = (clo.bkg_control.readaheadDat != 0) ? clo.bkg_control.readaheadDat : my_image_spec.LeadTimeForChipSize();
  glinfo.clo = &clo;
}

int UpdateEmptyWells(int flow, Image *img, short *emptyInFlow)
{
  const RawImage *raw = img->GetImage();
  int rows = raw->rows;
  int cols = raw->cols;
  int x, y, frame;
  int pinnedCount = 0;
  int i = 0;

  if (rows <= 0 || cols <= 0)
  {
    cout << "Why bad row/cols for flow: " << flow << " rows: " << rows << " cols: " << cols << endl;
    exit(EXIT_FAILURE);
  }
  char s[256];
  int n = sprintf (s, "Filtering for pinned pixels... ");
  
  for (y=0;y<rows;y++)
  {
    for (x=0;x<cols;x++)
    {
      short currFlow = emptyInFlow[i];
      if ( (currFlow < 0) | (currFlow > flow) ){
	// check for pinned pixels in this flow
        for (frame=0;frame<raw->frames;frame++)
        {
          if (raw->image[frame*raw->frameStride + i] == 0 ||
              raw->image[frame*raw->frameStride + i] == 0x3fff)
          {
	    // pixel is pinned high or low
	    currFlow = flow;
	    emptyInFlow[i] = flow;
	    pinnedCount++;
	    break;
	  }
	} // end frame loop
      } // end if
      while (((short volatile *)emptyInFlow)[i] > currFlow){
	// race condition, a later flow already updated this well, keep trying
	((short volatile *)emptyInFlow)[i] = currFlow;
      }
      i++;
    }  // end x loop
  } // end y loop
  n += sprintf(&s[n],  "UpdateEmptyWells: found %d pinned in flow %d\n", pinnedCount, flow);
  assert(n<255);
  fprintf (stdout,"%s", s);
  return pinnedCount;
}
