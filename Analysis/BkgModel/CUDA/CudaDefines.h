/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef CUDADEFINES_H
#define CUDADEFINES_H

#include "BeadParams.h" 

#define USE_CUDA_ERF
#define USE_CUDA_EXP


//single fit memory config
#define FVAL_L1   //  FVAL buffers in local memory 
//#define JAC_L1 // JAC buffers in local memory  




// Restrict number of iteration of single flow fit
#ifndef ITER
#define ITER 8
#endif

// A macro to (shudder) correctly index in to bead params
#define BEAD_OFFSET(a) (offsetof(BeadParams, a)/sizeof(float))

#define PARAM1_STEPIDX_SHIFT 26
#define PARAM2_STEPIDX_SHIFT 20

#define MAX_ALLOWED_NUM_STREAMS 2

#define MAX_XTALK_NEIGHBOURS 6

#define MAX_NUM_DEVICES 4

// new poisson table layout
#define POISS_FLOAT4


#endif // CUDADEFINES_H

