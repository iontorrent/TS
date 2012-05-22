/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PROCESSIMAGETOWELL_H
#define PROCESSIMAGETOWELL_H

#include <string>
#include <vector>
#include "CommandLineOpts.h"
#include "Mask.h"
#include "RawWells.h"
#include "Region.h"
#include "SeqList.h"
#include "TrackProgress.h"

void DoThreadedBackgroundModel(RawWells &rawWells, CommandLineOpts &clo, Mask *maskPtr, char *experimentName, int numFlows, char *chipType,ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, Region *regions,int totalRegions, RegionTiming *region_timing, SeqListClass &my_keys);
void DoThreadedBackgroundModel(RawWells &rawWells, CommandLineOpts &clo, Mask *maskPtr, char *experimentName, int numFlows, char *chipType,ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, Region *regions, int totalRegions, RegionTiming *region_timing, SeqListClass &my_keys, std::vector<float> *tauB, std::vector<float> *tauE);

void GetFromImagesToWells(RawWells &rawWells, Mask *maskPtr,
                          CommandLineOpts &clo,
                          char *experimentName, std::string &analysisLocation,
                          int numFlows,
                          SeqListClass &my_keys,
                          TrackProgress &my_progress, Region &wholeChip,
                          int &well_rows, int &well_cols);

#endif // PROCESSIMAGETOWELL_H
