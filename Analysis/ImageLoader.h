/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGELOADER_H
#define IMAGELOADER_H

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

#include "cudaWrapper.h"
#include "Flow.h"
#include "Image.h"
#include "Region.h"
#include "Mask.h"
#include "Separator.h"
#include "WorkerInfoQueue.h"
#include "Stats.h"
#include "SampleStats.h"
#include "ReservoirSample.h"
#include "SampleQuantiles.h"
#include "CommandLineOpts.h"
#include "ImageSpecClass.h"
#include "SpecialDataTypes.h"
#include "RawWells.h"

// comment out the line below if the background model is configured to use a per-well
// background trace
#define SINGLE_BKG_TRACE

struct ImageLoadWorkInfo
{
  int type;
  int flow;
  char name[512];
  Image *img;
  Mask *lmask;
  int normStart;
  int normEnd;
  int NNinnerx, NNinnery, NNouterx, NNoutery;
  unsigned int *CurRead;
  unsigned int *CurProcessed;
  char *dirExt;
  char *acqPrefix;
  Region *regions;
  Separator *separator;
  int numRegions;
  int numFlowsPerCycle;
  int hasWashFlow;
  int lead;
  bool finished;
  RawWells *bubbleWells;
  RawWells *bubbleSdWells;
  CommandLineOpts *clo;
};


void *FileLoadWorker(void *arg);
void *FileLoader(void *arg);

void FilterBubbleWells(Image *img, int flowIx, Mask *mask,
                       RawWells *bubbleWells, RawWells *bubbleSdWells);


class ImageTracker
{
public:
  Image *img;
  unsigned int *CurRead;
  unsigned int *CurProcessed;
  int numFlows;

  ImageTracker(int _numFlows, int ignoreChecksumErrors);
  void FinishFlow(int flow);
  void WaitForFlowToLoad(int flow);
  ~ImageTracker();
};

//lightweight class to handle bubbles
class bubbleTracker
{
public:
  RawWells *bubbleWells;
  RawWells *bubbleSdWells;

  bubbleTracker();
  void Init(char *experimentName, int rows, int cols, int numFlows, Flow *flw);
  void Close();
  ~bubbleTracker();
};

void SetUpImageLoaderInfo(ImageLoadWorkInfo &glinfo, CommandLineOpts &clo, Mask &localMask,
                          ImageTracker &my_img_set, ImageSpecClass &my_image_spec, bubbleTracker &my_bubble_tracker, int numFlows);


#endif // IMAGELOADER_H
