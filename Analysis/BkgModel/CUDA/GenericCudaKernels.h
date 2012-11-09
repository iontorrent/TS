/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef GENERICCUDAKERNELS_H
#define GENERICCUDAKERNELS_H


#include "BkgMagicDefines.h"
#include "CudaConstDeclare.h" 


__global__ void transposeData(float *dest, float *source, int width, int height);


///////// Transpose Kernel
__global__ void transposeDataToFloat(float *dest, FG_BUFFER_TYPE *source, int width, int height);


#endif // GENERICCUDAKERNELS_H
