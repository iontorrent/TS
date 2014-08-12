/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGECONTROLOPTS_H
#define IMAGECONTROLOPTS_H

#include <string>
#include "ChipIdDecoder.h"
#include "HandleExpLog.h"
#include "OptBase.h"

// control options on loading dat files
class ImageControlOpts{
 public:
  int totalFrames;
  int maxFrames; // Set later from the first raw image header.
  bool nn_subtract_empties;
  int NNinnerx;
  int NNinnery;
  int NNouterx;
  int NNoutery;
  bool hilowPixFilter;
  int ignoreChecksumErrors; // set to true to force corrupt checksum files to load anyway - beware!
  int flowTimeOffset;
  bool gain_correct_images; // use beadfind file to calculate pixel gain and correct all images
  bool gain_debug_output;   // output text file with gain measurement for each pixel
  bool col_flicker_correct;
  bool col_flicker_correct_verbose;
  bool aggressive_cnc;
  bool col_pair_pixel_xtalk_correct;
  float pair_xtalk_fraction;
  int readaheadDat;  // should this be img_control instead???

  // do diagnostics?
  bool outputPinnedWells;
  char PCATest[128];
  char tikSmoothingFile[512];  // file holding data smoothing coeffcients (APB)
  char tikSmoothingInternal[32];  // parameter for internal smoothing matrix (APB)
  bool doSdat;
  int total_timeout; // optional arg for image class, when set will cause the image class to wait this many seconds before giving up
  bool threaded_file_access; // read DAT files for signal processing in image processing threads
  std::string sdatSuffix;
    char *acqPrefix;
    int has_wash_flow;

    
  void DefaultImageOpts();
  void ImageControlForProton(bool default_aggressive);
  ~ImageControlOpts();
  // try to parse explog once and only once
  void SetWashFlow(char *explog_path) {
        int hasWashFlow = HasWashFlow(explog_path);
        has_wash_flow = (hasWashFlow < 0 ? 0 : hasWashFlow);
  };
  void PrintHelp();
  void SetOpts(OptArgs &opts, Json::Value& json_params);
};

#endif // IMAGECONTROLOPTS_H
