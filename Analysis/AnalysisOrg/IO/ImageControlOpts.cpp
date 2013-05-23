/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ImageControlOpts.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace std;

void ImageControlOpts::DefaultImageOpts()
{
  maxFrames = 0;    // Set later from the first raw image header.
  totalFrames = 0;
  nn_subtract_empties = 0; // do >not< subtract nearest-neighbor empties
  NNinnerx = 1;
  NNinnery = 1;
  NNouterx = 12;
  NNoutery = 8;
  ignoreChecksumErrors = 0;
  hilowPixFilter = 0;   // default is disabled
  flowTimeOffset = 1000;
  col_flicker_correct = false; //default to turn on
  col_flicker_correct_verbose = false;
  aggressive_cnc = false;
  gain_correct_images = false;
  gain_debug_output = false;
  has_wash_flow = 0;
  // image diagnostics
  outputPinnedWells = 0;
  tikSmoothingFile[0] = '\000';   // (APB)
  tikSmoothingInternal[0] = '\000'; // (APB)
  doSdat = false; // Look for synchronized dat (sdat) files instead of usual dats.
  total_timeout = 0; // 0 means use whatever the image class has set as default
  sdatSuffix = "sdat";
  //if (acqPrefix != NULL) free (acqPrefix);
  acqPrefix = strdup("acq_");
  threaded_file_access = 1;
}

ImageControlOpts::~ImageControlOpts()
{
  if (acqPrefix!=NULL) {
    free( acqPrefix);
    acqPrefix = NULL;
  }
}

