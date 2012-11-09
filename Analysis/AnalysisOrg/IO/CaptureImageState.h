/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CAPTUREIMAGESTATE_H
#define CAPTUREIMAGESTATE_H

#include <string>
#include <vector>

#include "ImageSpecClass.h"

// save/load image transformations 
class CaptureImageState{
  public:
    CaptureImageState(std::string analysisLocation);
    
    void CleanUpOldFile();
  
    std::string h5file;    
    void WriteImageGainCorrection(int rows, int cols);
    void LoadImageGainCorrection(int rows, int cols);
    
    void WriteXTCorrection();
    void LoadXTCorrection();
    
    void WriteImageSpec(ImageSpecClass &my_image_spec, int frames);
    void LoadImageSpec(ImageSpecClass &my_image_spec);
  
};

#endif // CAPTUREIMAGESTATE_H
