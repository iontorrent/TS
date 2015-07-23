/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * MasterKernel.cu
 *
 *  Created on: Mar 5, 2014
 *      Author: jakob
 *
 *This file inlcudes all the *.cu from IncludeKernels to allow more modular code
 *This but not conflicts with being limited to only one compilation unit
 *
 */

#include "MasterKernel.h"

///////////////////////////////////
//INCLUDE .cu files for single compilation unit.

#include "DeviceSymbolCopy.cu"
#include "UtilKernels.cu"
#include "FittingHelpers.cu"
#include "GenerateBeadTraceKernels.cu"
#include "SingleFlowFitKernels.cu"
#include "RegionalFittingKernels.cu"
#include "PostFitKernels.cu"






