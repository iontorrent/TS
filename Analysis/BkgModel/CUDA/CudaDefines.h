/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef CUDADEFINES_H
#define CUDADEFINES_H

#include "BeadParams.h" 

#define USE_CUDA_ERF
#define USE_CUDA_EXP

// Don't use lev mar algorithm for single flow fit
#define NOLEVMAR


//single fit memory config
//#define FVAL_L1   //  FVAL buffers in Local Memeory 
//#define JAC_L1 // JAC euffers in Local Memor  //  FVAL buffers in Local Memeory 




// Restrict number of iteration of single flow fit
#ifndef ITER
#define ITER 8
#endif

// Some macros to correctly index in to bead params
#define BEAD_AMPL(a) (offsetof(bead_params, a)/sizeof(float))
#define BEAD_COPIES(c) (offsetof(bead_params, c)/sizeof(float))
#define BEAD_DMULT(d) (offsetof(bead_params, d)/sizeof(float))
#define BEAD_KMULT(k) (offsetof(bead_params, k)/sizeof(float))
#define BEAD_R(r) (offsetof(bead_params, r)/sizeof(float))
#define BEAD_GAIN(g) (offsetof(bead_params, g)/sizeof(float))
#define BEAD_PCA(g) (offsetof(bead_params, g)/sizeof(float))

#define PARAM1_STEPIDX_SHIFT 26
#define PARAM2_STEPIDX_SHIFT 20

#define MAX_NUM_STREAMS 2

#define MAX_XTALK_NEIGHBOURS 6

#define MAX_NUM_DEVICES 4


#endif // CUDADEFINES_H

