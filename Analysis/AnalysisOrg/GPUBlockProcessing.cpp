/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#include "GPUBlockProcessing.h"

#ifdef ION_COMPILE_CUDA
#include "LayoutTester.h"
#endif

bool ProcessProtonBlockImageOnGPU(
   BkgModelWorkInfo* fitterInfo, 
   int flowBlockSize)
{
#ifdef ION_COMPILE_CUDA
  return blockLevelSignalProcessing(fitterInfo, flowBlockSize);
#endif
  return false;
}
