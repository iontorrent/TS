/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CUDACONSTDECLARE_H 
#define CUDACONSTDECLARE_H 

#include "BkgMagicDefines.h"
#include "ParamStructs.h"

#include "CudaDefines.h"

#define USE_CUDA_ERF
#define USE_CUDA_EXP

#define USE_GLOBAL_POISSTABLE


#ifndef USE_CUDA_ERF
__constant__ static float ERF_APPROX_TABLE_CUDA[sizeof (ERF_APPROX_TABLE) ];
#endif

#ifndef USE_GLOBAL_POISSTABLE
__constant__ static float POISS_0_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
__constant__ static float POISS_1_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
__constant__ static float POISS_2_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
__constant__ static float POISS_3_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
__constant__ static float POISS_4_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
__constant__ static float POISS_5_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
__constant__ static float POISS_6_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
__constant__ static float POISS_7_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
__constant__ static float POISS_8_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
__constant__ static float POISS_9_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
__constant__ static float POISS_10_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
__constant__ static float POISS_11_APPROX_TABLE_CUDA[MAX_POISSON_TABLE_ROW];
#else
__constant__ static float * POISS_APPROX_TABLE_CUDA_BASE;
__constant__ static float4 * POISS_APPROX_LUT_CUDA_BASE;

#endif
//__constant__ ConstParams CP[NUM_CUDA_FIT_STREAMS];

__constant__ ConstParams CP_MULTIFLOWFIT[MAX_NUM_STREAMS];
__constant__ ConstParams CP_SINGLEFLOWFIT[MAX_NUM_STREAMS];






#endif // CUDACONSTDECLARE_H
