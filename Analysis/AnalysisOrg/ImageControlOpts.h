/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGECONTROLOPTS_H
#define IMAGECONTROLOPTS_H

#include <string>
#include "ChipIdDecoder.h"


// control options on loading dat files
class ImageControlOpts{
 public:
  int totalFrames;
  int maxFrames; // Set later from the first raw image header.
  int nn_subtract_empties;
  int NNinnerx;
  int NNinnery;
  int NNouterx;
  int NNoutery;
  int hilowPixFilter;
  int ignoreChecksumErrors; // set to true to force corrupt checksum files to load anyway - beware!
  int flowTimeOffset;

  // do diagnostics?
  int outputPinnedWells;

  void DefaultImageOpts(void);
};

#endif // IMAGECONTROLOPTS_H
