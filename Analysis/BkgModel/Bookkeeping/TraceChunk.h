/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACECHUNK_H
#define TRACECHUNK_H

#include "TimeCompression.h"

class TraceChunk {

 public:
  void SetDimensions(size_t row, size_t height, size_t col, size_t width, size_t frameStart, size_t depth) {
    mRowStart = row;
    mHeight = height;
    mColStart = col;
    mWidth = width;
    mFrameStart = frameStart;
    mDepth = depth;
  }

  size_t mRowStart, mColStart, mFrameStart;
  size_t mHeight, mWidth, mDepth; // row, col, frame
  TimeCompression mTime;
  std::vector<float> mData;
  
};

#endif // TRACECHUNK_H
