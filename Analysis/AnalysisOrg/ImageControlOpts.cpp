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

  // image diagnostics
  outputPinnedWells = 0;

}

