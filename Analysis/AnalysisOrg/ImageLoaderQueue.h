/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGELOADERQUEUE_H
#define IMAGELOADERQUEUE_H

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>
#include <limits.h>
#include <signal.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <armadillo>


#include "Image.h"
#include "Region.h"
#include "Mask.h"
#include "ComplexMask.h"

#include "WorkerInfoQueue.h"
#include "Stats.h"
#include "SampleStats.h"
#include "ReservoirSample.h"
#include "SampleQuantiles.h"
#include "CommandLineOpts.h"
#include "ImageSpecClass.h"
#include "SpecialDataTypes.h"
#include "RawWells.h"
#include "EmptyTraceTracker.h"
#include "TikhonovSmoother.h"
#include "PinnedInFlow.h"
#include "PinnedInFlowReplay.h"
#include "SynchDat.h"

// queuing system may change for coprocessor environment
// make separate module to clarify code

struct ImageLoadWorkInfo
{
  int type;
  
  int flow;  // the absolute flow value we are working on
  int cur_buffer; // the buffer where the absolute flow value is located
  
  int flow_buffer_size; // the size of the buffer
  int startingFlow; // the flow we are starting the count from
  
  char name[512];

  // these are >sized< to flow_buffer_size
  // cur_buffer must be within these bounds
  // array of Image objects shared with ImageTracker object
  Image *img;  // shared across ImageLoader worker threads
  SynchDat *sdat;
  // the following two arrays also are shared with the ImageTracker object
  // these arrays are used to coordinate ongoing flow status across threads
  unsigned int *CurRead;      // CurRead[flow] is the image read in?
  unsigned int *CurProcessed; // CurReadProcessed[flow], bkg model done?
  
  Mask *mask; // shared across  all ImageLoader and BkgModelFitter threads
// pinnedInFlow shared across all ImageLoader and BkgModelFitter threads
  PinnedInFlow *pinnedInFlow; // handled by ImageTracker object and shared
  
  int normStart;
  int normEnd;
  int NNinnerx, NNinnery, NNouterx, NNoutery;
  int smooth_span;
  
  char *dat_source_directory;
  char *acqPrefix;
  Region *regions;

  int numRegions;
  int numFlowsPerCycle;
  int hasWashFlow;
  int lead;
  bool doingSdat;
  bool finished;
  bool doRawBkgSubtract;
  bool doEmptyWellNormalization;
  CommandLineOpts *inception_state;
};


void *FileLoadWorker (void *arg);
void *FileLoader (void *arg);
void *FileSDatLoader (void *arg);


void DumpStep (ImageLoadWorkInfo *info);
void DumpDcOffset (ImageLoadWorkInfo *info);

#endif // IMAGELOADERQUEUE_H
