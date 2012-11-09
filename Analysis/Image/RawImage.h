/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef RAWIMAGE_H
#define RAWIMAGE_H

#include <stdio.h>
#include <stdlib.h>


struct RawImage
{
  int rows, cols, frames;   // vars for image size
  int chip_offset_x, chip_offset_y;   // offset of image origin relative to chip origin
  int uncompFrames,compFrames;
  int channels;    // number of channels in this image
  int interlaceType;   // 0 is not interlaced, 1 and up are various interlace schemes
  int frameStride;   // rows * cols
  short *image;    // raw image (loaded in as unsigned short, and byte-swapped, masked with 0x3fff, and promoted to short)
  int *timestamps;
  short *compImage;
  int *compTimestamp;
  int baseFrameRate;
  int *interpolatedFrames;
  float *interpolatedMult;
  RawImage()
  {
    rows = 0;
    cols = 0;
    frames = 0;
    chip_offset_x = 0;
    chip_offset_y = 0;
    channels = 0;
    interlaceType = 0;
    frameStride = 0;
    image = NULL;
    timestamps = NULL;
    compImage=NULL;
    compTimestamp=NULL;
    interpolatedFrames=NULL;
    interpolatedMult=NULL;
    uncompFrames=0;
    compFrames=0;
    baseFrameRate=0;
  }
};


#endif // RAWIMAGE_H