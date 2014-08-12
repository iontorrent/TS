/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CUDACONSTDECLARE_H 
#define CUDACONSTDECLARE_H 

#include "BkgMagicDefines.h"
#include "ParamStructs.h"

#include "CudaDefines.h"

#define USE_CUDA_ERF
#define USE_CUDA_EXP


#ifndef USE_CUDA_ERF
__constant__ static float ERF_APPROX_TABLE_CUDA[sizeof (ERF_APPROX_TABLE) ];
#endif

__constant__ static float * POISS_APPROX_TABLE_CUDA_BASE;
__constant__ static float4 * POISS_APPROX_LUT_CUDA_BASE;


__constant__ ConstParams CP[MAX_ALLOWED_NUM_STREAMS];
__constant__ ConstXtalkParams CP_XTALKPARAMS[MAX_ALLOWED_NUM_STREAMS];





#endif // CUDACONSTDECLARE_H
