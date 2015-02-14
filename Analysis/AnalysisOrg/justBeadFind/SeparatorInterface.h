/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SEPARATORINTERFACE_H
#define SEPARATORINTERFACE_H

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
#include <set>
#include <algorithm>
#include <limits>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <armadillo>

#include "DifferentialSeparator.h"
#include "Mask.h"
#include "Region.h"
#include "SeqList.h"
#include "ImageSpecClass.h"
#include "RegionTimingCalc.h"
#include "SlicedPrequel.h"
#include "MaskFunctions.h"
#include "ProgramState.h"
#include "CaptureImageState.h"
#include "TrackProgress.h"


//@TODO: call the beadfind, essentially as a separate program
void DoDiffSeparatorFromCLO(DifferentialSeparator *diffSeparator, CommandLineOpts &inception_state, Mask *maskPtr, string &analysisLocation,
                            SequenceItem *seqList, int numSeqListItems);

//@TODO: and extract the information that needs passing downstream to the background model
void NNSmoothT0Estimate(Mask *mask,int imgRows,int imgCols,std::vector<float> &sep_t0_est,std::vector<float> &output_t0_est);
void NNSmoothT0EstimateFast (Mask *mask,int imgRows,int imgCols,std::vector<float> &sep_t0_est,
                             std::vector<float> &output_t0_est, int x_clip, int y_clip);


void getTausFromSeparator(Mask *maskPtr, DifferentialSeparator *diffSeparator, std::vector<float> &tauB, std::vector<float> &tauE);
void SetupForBkgModelTiming (DifferentialSeparator *diffSeparator, std::vector<float> &smooth_t0_est, 
                             std::vector<RegionTiming> &region_timing, std::vector<Region>& region_list, 
                             ImageSpecClass &my_image_spec, Mask *maskPtr, int numThreads, int x_clip, int y_clip);

//void SetupForBkgModelTiming (DifferentialSeparator *diffSeparator, std::vector<float> &smooth_t0_est, std::vector<RegionTiming>& region_timing, std::vector<Region>& region_list, ImageSpecClass &my_image_spec, Mask *maskPtr, bool doSmoothing, int numThreads);

void IsolatedBeadFind (SlicedPrequel &my_prequel_setup, ImageSpecClass &my_image_spec, Region &wholeChip, CommandLineOpts &inception_state, char *results_folder, string &analysisLocation, SeqListClass &my_keys, TrackProgress &my_progress);


#endif // SEPARATORINTERFACE_H
